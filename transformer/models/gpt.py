import torch
from tqdm import tqdm
from torch import nn, Tensor
from typing import Dict, List, Tuple
from transformer.models.interface import ModelInterface
from transformer.models.utils import compute_bleu, compute_meteor, compute_rouge
from transformers import GPT2LMHeadModel
from transformer.utils.common import set_device, convert_to_tensor, convert_to_numpy, get_last_index
from transformer.utils.metrics import get_language_modeling_accuracy, get_perplexity, get_bleu_score

class Gpt2(nn.modules.Module, ModelInterface):
    __name__ = "gpt2"

    def __init__(self, vocab_size: int = 51200, name_or_path:str = "skt/kogpt2-base-v2"):
        # init nn.modules.Module
        nn.modules.Module.__init__(self)
        ModelInterface.__init__(self)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=name_or_path)
        # hyper parameters
        self.config = self.gpt2.config.to_dict()
        if self.config["vocab_size"] < vocab_size:
            print("Resizeing vocab_size to {vocab_size}...".format(vocab_size=vocab_size))
            self.gpt2.resize_token_embeddings(vocab_size)
            self.config = self.gpt2.config.to_dict()

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        input_ids = inputs["input_ids"]
        token_type_ids = None
        if "token_type_ids" in inputs: token_type_ids = inputs["token_type_ids"]
        attention_mask = None
        if "attention_mask" in inputs: attention_mask = inputs["attention_mask"]
        labels = None
        if "labels" in inputs: labels = inputs["labels"]

        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)
        if labels is not None:
            outputs["lm_loss"] = outputs["loss"] # loss for verbose # not be used to update gradient
            outputs["lm_acc"] = get_language_modeling_accuracy(predictions=outputs["logits"], targets=labels, ignore_index=-100)
            outputs["ppl"] = torch.exp(outputs["lm_loss"])
        return outputs

    def iteration_batch(self, batch, device):
        self.assert_contain_elements(required=["labels"], target=batch)
        inputs = {k: convert_to_tensor(v, device=device) for k, v in batch.items()}
        if "labels" in inputs: inputs["labels"] = inputs["labels"].to(torch.long)
        outputs = self.forward(inputs=inputs)
        return outputs

    def generate(self, input_ids, **kwargs):
        return self.gpt2.generate(input_ids, **kwargs)

    def get_metric_inputs(self, tokenizer, data_loader, device, timesteps, decoding_method, **kwargs):
        candidates = []
        labels = []
        data_iter = tqdm(data_loader, initial=0, total=len(data_loader), desc="Computing scores")
        for batch_idx, batch in enumerate(data_iter):
            batch = {k: convert_to_tensor(v, device=device) for k, v in batch.items()}

            _candidates = []
            for input_id_row in batch["input_ids"]:
                begin_idx = get_last_index(obj=input_id_row, value=tokenizer.bos_token_id)
                input_id_row = input_id_row[:begin_idx + 1]
                input_id_row = torch.unsqueeze(input_id_row, axis=0).to(torch.long)
                candidate = self.generate(input_ids=input_id_row, max_length=timesteps, **self.default_generate_params[decoding_method])
                candidate = candidate[0, begin_idx:]
                candidate = tokenizer.decode(candidate, skip_special_tokens=True)
                _candidates.append(candidate)
            candidates += _candidates

            _labels = batch["labels"]
            _labels = [[token_id for token_id in label if token_id >= 0] for label in _labels]
            _labels = tokenizer.batch_decode(_labels, skip_special_tokens=True)
            labels += _labels

        candidates = [candidate.strip() for candidate in candidates]
        labels = [label.strip() for label in labels]
        return candidates, labels

    def _compute_scores(self, metrics, tokenizer, data_loader, device, **kwargs):
        return self._compute_generative_scores(metrics, tokenizer, data_loader, device, **kwargs)

    def get_optimizer(self, lr=5e-5):
        optimizer = ModelInterface.get_optimizer(self=self, optimizer="adam_w", initial_learning_rate=lr)
        return optimizer

    def load(self, path):
        output = ModelInterface.load(self=self, path=path, model_type=self.__name__)
        return output