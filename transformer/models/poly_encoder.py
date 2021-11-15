import torch
from tqdm import tqdm
from torch import nn, Tensor
from typing import Optional, Dict, List, Tuple, Any
from transformers import ElectraModel
from transformer.models.interface import ModelInterface
from transformer.models.utils import compute_hits
from transformer.layers.attention import CodeAttention
from transformer.layers.embedding import EmbeddingAggregation
from transformer.layers.head import PolyEncoderHead
from transformer.utils.common import set_device, convert_to_tensor, convert_to_numpy
from transformer.models.utils import save_candidates, load_candidates
from transformer.utils.metrics import *


class PolyEncoder(nn.modules.Module, ModelInterface):
    __name__ = "poly_encoder"

    def __init__(self, encoder_type: str = "electra", vocab_size: int = 30000, m_code: int = 64, aggregation_method: str = "first", name_or_path:str = "monologg/koelectra-base-discriminator"):
        # init nn.modules.Module
        nn.modules.Module.__init__(self)
        ModelInterface.__init__(self)

        # define encoder
        if encoder_type in ["electra"]:
            self.left_encoder = ElectraModel.from_pretrained(pretrained_model_name_or_path=name_or_path)
            self.right_encoder = ElectraModel.from_pretrained(pretrained_model_name_or_path=name_or_path)
        self.encoder_type = encoder_type

        # hyper parameters
        self.m_code = m_code
        self.aggregation_method = aggregation_method

        self.left_encoder_config = self.left_encoder.config.to_dict()
        if self.left_encoder_config["vocab_size"] < vocab_size:
            print("Resizeing left_encoder vocab_size to {vocab_size}...".format(vocab_size=vocab_size))
            self.left_encoder.resize_token_embeddings(vocab_size)
            self.left_encoder_config = self.left_encoder.config.to_dict()
        self.left_d_model = self.left_encoder_config["hidden_size"]

        self.right_encoder_config = self.right_encoder.config.to_dict()
        if self.right_encoder_config["vocab_size"] < vocab_size:
            print("Resizeing right_encoder vocab_size to {vocab_size}...".format(vocab_size=vocab_size))
            self.right_encoder.resize_token_embeddings(vocab_size)
            self.right_encoder_config = self.right_encoder.config.to_dict()
        self.right_d_model = self.right_encoder_config["hidden_size"]

        # layers
        self.code_embedding_layer = CodeAttention(m=m_code, d_model=self.left_d_model)
        self.aggregation_layer = EmbeddingAggregation(method=aggregation_method)
        self.poly_encoder_head = PolyEncoderHead(d_model=self.left_d_model)
        self.criterion = nn.NLLLoss()


    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        left_input_ids = inputs["left_input_ids"]
        left_token_type_ids = None
        if "left_token_type_ids" in inputs: left_token_type_ids = inputs["left_token_type_ids"]
        left_attention_mask = None
        if "left_attention_mask" in inputs: left_attention_mask = inputs["left_attention_mask"]
        right_input_ids = inputs["right_input_ids"]
        right_token_type_ids = None
        if "right_token_type_ids" in inputs: right_token_type_ids = inputs["right_token_type_ids"]
        right_attention_mask = None
        if "right_attention_mask" in inputs: right_attention_mask = inputs["right_attention_mask"]
        labels = None
        if "labels" in inputs: labels = inputs["labels"].to(torch.long)

        # left_encoder
        left_outputs = self.left_encoder(input_ids=left_input_ids, attention_mask=left_attention_mask, token_type_ids=left_token_type_ids)
        left_encoder_output = left_outputs["last_hidden_state"]
        # left_encoder_output = left_encoder_output.to(torch.double)
        context_code_embeds, _context_code_embeds_weight = self.code_embedding_layer(left_encoder_output)

        # right_encoder
        right_outputs = self.left_encoder(input_ids=right_input_ids, attention_mask=right_attention_mask, token_type_ids=right_token_type_ids)
        right_encoder_output = right_outputs["last_hidden_state"]
        candidate_embeds = self.aggregation_layer(right_encoder_output)

        outputs = dict()
        poly_encoder_output = self.poly_encoder_head(context_embeds=context_code_embeds, candidate_embeds=candidate_embeds)
        if labels is not None:
            loss = self.criterion(poly_encoder_output, labels)
            outputs["loss"] = loss # loss for backpropagation
            outputs["ce_loss"] = loss # loss for verbose # not be used to update gradient
            outputs["cls_acc"] = get_classification_accuracy(predictions=poly_encoder_output, targets=labels)
        outputs["logits"] = poly_encoder_output
        outputs["left_encoder_logits"] = context_code_embeds
        outputs["right_encoder_logits"] = candidate_embeds
        return outputs

    def forward_left(self, inputs: Dict[str, Tensor], candidate_embeds: torch.Tensor) -> Tensor:
        # candidate_embeds: (candidate_batch_size, right_d_model)
        left_input_ids = inputs["left_input_ids"]
        left_token_type_ids = None
        if "left_token_type_ids" in inputs: left_token_type_ids = inputs["left_token_type_ids"]
        left_attention_mask = None
        if "left_attention_mask" in inputs: left_attention_mask = inputs["left_attention_mask"]

        # left_encoder
        left_outputs = self.left_encoder(input_ids=left_input_ids, attention_mask=left_attention_mask, token_type_ids=left_token_type_ids)
        left_encoder_output = left_outputs["last_hidden_state"]
        # left_encoder_output = left_encoder_output.to(torch.double)
        context_code_embeds, _context_code_embeds_weight = self.code_embedding_layer(left_encoder_output)

        outputs = dict()
        poly_encoder_output = self.poly_encoder_head(context_embeds=context_code_embeds, candidate_embeds=candidate_embeds)
        outputs["logits"] = poly_encoder_output
        outputs["left_encoder_logits"] = context_code_embeds
        outputs["right_encoder_logits"] = candidate_embeds
        return outputs

    def forward_right(self, inputs: Dict[str, Tensor]) -> Tensor:
        right_input_ids = inputs["right_input_ids"]
        right_token_type_ids = None
        if "right_token_type_ids" in inputs: right_token_type_ids = inputs["right_token_type_ids"]
        right_attention_mask = None
        if "right_attention_mask" in inputs: right_attention_mask = inputs["right_attention_mask"]

        # right_encoder
        right_outputs = self.left_encoder(input_ids=right_input_ids, attention_mask=right_attention_mask, token_type_ids=right_token_type_ids)
        right_encoder_output = right_outputs["last_hidden_state"]
        candidate_embeds = self.aggregation_layer(right_encoder_output)

        outputs = dict()
        outputs["right_encoder_logits"] = candidate_embeds
        return outputs

    def iteration_batch(self, batch, device):
        self.assert_contain_elements(required=["labels"], target=batch)
        inputs = {k: convert_to_tensor(v, device=device) for k, v in batch.items()}
        if "labels" in inputs: inputs["labels"] = inputs["labels"].to(torch.long)
        outputs = self.forward(inputs=inputs)
        return outputs

    def get_metric_inputs(self, tokenizer, data_loader, device, hits_k: List[int], **kwargs):
        required_parameters = ["candidates", "candidate_embeds"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
        candidates = kwargs.pop("candidates")
        right_sentences = np.array(candidates)
        candidate_embeds = kwargs.pop("candidate_embeds")

        top_n = max(hits_k)
        candidates = []
        labels = []
        candidate_embeds = convert_to_tensor(candidate_embeds, device=device)
        data_iter = tqdm(data_loader, initial=0, total=len(data_loader), desc="Computing scores")
        for batch_idx, batch in enumerate(data_iter):
            batch = {k: convert_to_tensor(v, device=device) for k, v in batch.items()}
            left_outputs = self.forward_left(inputs=batch, candidate_embeds=candidate_embeds)
            _candidates_indice = left_outputs["logits"]
            _candidates_indice = torch.argsort(_candidates_indice, dim=-1, descending=True)
            _candidates_indice = _candidates_indice[:, :top_n]

            _candidates_indice = convert_to_numpy(tensor=_candidates_indice)
            _candidates = right_sentences[_candidates_indice]
            candidates += _candidates.tolist()
            _labels = tokenizer.batch_decode(batch["right_input_ids"].tolist(), skip_special_tokens=True)
            labels += _labels

        candidates = [[_candidate.strip() for _candidate in candidate] for candidate in candidates]
        labels = [label.strip() for label in labels]
        return candidates, labels

    def _compute_scores(self, metrics, tokenizer, data_loader, device, **kwargs):
        return self._compute_discrimitive_scores(metrics, tokenizer, data_loader, device, **kwargs)

    def get_optimizer(self, lr=5e-5):
        optimizer = ModelInterface.get_optimizer(self=self, optimizer="adam_w", initial_learning_rate=lr)
        return optimizer

    def save(self, path, optimizer, tokenizer, history=None, candidates=None):
        ModelInterface.save(self=self, path=path, optimizer=optimizer, tokenizer=tokenizer, history=history)
        if candidates is not None:
            save_candidates(path=path, data=candidates)
            message = "Saved (candidates, candidate_embeds) into {path}".format(path=path)
            print(message)
        return path

    def load(self, path, load_candidates: bool = False):
        output = ModelInterface.load(self=self, path=path, model_type=self.encoder_type)
        if load_candidates:
            data = load_candidates(path=path)
            candidates, candidate_embeds = data
            self.assert_equal_length(a=candidates, b=candidate_embeds)
            output["candidates"] = candidates
            output["candidate_embeds"] = candidate_embeds
        return output