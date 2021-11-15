import torch
from tqdm import tqdm
from torch import nn, Tensor
from typing import Optional, Dict
from transformers import BartForConditionalGeneration
from transformer.models.interface import ModelInterface
from transformer.models.utils import compute_bleu, compute_meteor, compute_rouge
from transformer.utils.common import set_device, convert_to_tensor, convert_to_numpy
from transformer.utils.metrics import get_language_modeling_accuracy, get_perplexity, get_bleu_score


class Bart(nn.modules.Module, ModelInterface):
    __name__ = "bart"

    def __init__(self, vocab_size: int = 30000, name_or_path:str = "hyunwoongko/kobart"):
        # init nn.modules.Module
        nn.modules.Module.__init__(self)
        ModelInterface.__init__(self)
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=name_or_path)
        # hyper parameters
        self.config = self.bart.config.to_dict()
        if self.config["vocab_size"] < vocab_size:
            print("Resizeing vocab_size to {vocab_size}...".format(vocab_size=vocab_size))
            self.bart.resize_token_embeddings(vocab_size)
            self.config = self.bart.config.to_dict()


    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        input_ids = inputs["input_ids"]
        token_type_ids = None
        if "token_type_ids" in inputs: token_type_ids = inputs["token_type_ids"]
        attention_mask = None
        if "attention_mask" in inputs: attention_mask = inputs["attention_mask"]
        decoder_input_ids = inputs["decoder_input_ids"]
        decoder_token_type_ids = None
        if "decoder_token_type_ids" in inputs: decoder_token_type_ids = inputs["decoder_token_type_ids"]
        decoder_attention_mask = None
        if "decoder_attention_mask" in inputs: decoder_attention_mask = inputs["decoder_attention_mask"]
        labels = None
        if "labels" in inputs: labels = inputs["labels"]

        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                            labels=labels)

        if labels is not None:
            outputs["lm_loss"] = outputs["loss"]  # loss for verbose # not be used to update gradient
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
        output = self.bart.generate(input_ids, **kwargs)
        return output

    def get_metric_inputs(self, tokenizer, data_loader, device, timesteps, decoding_method, **kwargs):
        candidates = []
        labels = []
        data_iter = tqdm(data_loader, initial=0, total=len(data_loader), desc="Computing scores")
        for batch_idx, batch in enumerate(data_iter):
            batch = {k: convert_to_tensor(v, device=device) for k, v in batch.items()}

            batch_size = len(batch["input_ids"])
            input_ids = batch["input_ids"]
            decoder_inputs_ids = [[tokenizer.eos_token_id, tokenizer.bos_token_id]] * batch_size
            _candidates = self.generate(input_ids=input_ids, decoder_inputs_ids=decoder_inputs_ids, max_length=timesteps, **self.default_generate_params[decoding_method])
            _candidates = tokenizer.batch_decode(_candidates.tolist(), skip_special_tokens=True)
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

    def get_optimizer(self, lr=5e-5, beta_1=0.9, beta_2=0.99, optimizer_epsilon=1e-5):
        optimizer = ModelInterface.get_optimizer(self=self, optimizer="adam", initial_learning_rate=lr, beta_1=beta_1, beta_2=beta_2, optimizer_epsilon=optimizer_epsilon)
        return optimizer

    def load(self, path):
        output = ModelInterface.load(self=self, path=path, model_type=self.__name__)
        return output