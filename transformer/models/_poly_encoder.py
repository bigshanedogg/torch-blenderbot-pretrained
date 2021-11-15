import torch
from torch import nn, Tensor
import torchinfo
from torch.cuda.amp import autocast
from typing import Optional, Dict

from transformer.layers.attention import CodeAttention
from transformer.layers.embedding import EmbeddingAggregation
from transformer.layers.head import PolyEncoderHead
from transformer.layers.utils import dot_attention
from transformer.models.interface import ModelInterface
from transformer.models.bert import Bert


class PolyEncoder(nn.modules.Module, ModelInterface):
    __name__ = "poly_encoder"

    def __init__(self, context_encoder, candidate_encoder, m_code, aggregation_method: str = "first"):
        nn.modules.Module.__init__(self)
        # assert
        self.assert_equal(a=context_encoder.d_model, b=candidate_encoder.d_model)
        self.assert_equal_or_lesser(value=m_code, criteria=context_encoder.timesteps-1) # m_code must be lesser than context_timesteps
        # hyper parameters
        self.m_code = m_code
        self.aggregation_method = aggregation_method
        self.context_timesteps = context_encoder.timesteps
        self.candidate_timesteps = candidate_encoder.timesteps
        self.context_embedding_dict = context_encoder.embedding_dict
        self.candidate_embedding_dict = candidate_encoder.embedding_dict
        self.d_model = context_encoder.d_model
        # layers
        self.context_encoder = context_encoder
        self.candidate_encoder = candidate_encoder
        self.code_embedding_layer = CodeAttention(m=m_code, d_model=self.d_model)
        self.aggregation_layer = EmbeddingAggregation(method=aggregation_method)
        self.poly_encoder_head = PolyEncoderHead(d_model=self.d_model)

    @autocast()
    def forward(self, context_inputs: Dict[str, Tensor], candidate_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # context_encoder
        # context_inputs: {"token":(context_batch_size, context_timesteps, context_d_model), ...}
        with torch.autograd.profiler.record_function("context_encoder"):
            context_output = self.context_encoder.forward(inputs=context_inputs)
            context_encoder_output = context_output["encoder"]
            context_code_embeds, _context_code_embeds_weight = self.code_embedding_layer(context_encoder_output)

        # candidate_encoder
        # candidate_inputs: {"token":(candidate_batch_size, candidate_timesteps, candidate_d_model), ...}
        with torch.autograd.profiler.record_function("candidate_encoder"):
            candidate_output = self.candidate_encoder.forward(inputs=candidate_inputs)
            candidate_encoder_output = candidate_output["encoder"]
            candidate_embed = self.aggregation_layer(candidate_encoder_output)

        # head
        # poly_encoder_output: (context_batch_size, candidate_batch_size)
        with torch.autograd.profiler.record_function("poly_encoder_head"):
            poly_encoder_output = self.poly_encoder_head(context_embed=context_code_embeds, candidate_embed=candidate_embed)

        output = dict()
        output["ce"] = poly_encoder_output
        return output

    def forward_context_encoder(self, context_inputs: Dict[str, Tensor], candidate_embed: Tensor) -> Tensor:
        context_output = self.context_encoder.forward(inputs=context_inputs)
        context_encoder_output = context_output["encoder"]
        # context_embed: (context_batch_size, m_codes, d_model)
        context_code_embeds, _context_code_embeds_weight = self.code_embedding_layer(context_encoder_output)

        context_batch_size = context_code_embeds.shape[0]
        candidate_batch_size = candidate_embed.shape[0]
        # candidate_embed_query: (context_batch_size, candidate_batch_size, d_model)
        candidate_embed_query = torch.unsqueeze(candidate_embed, dim=0).expand(context_batch_size, candidate_batch_size, self.d_model)
        # context_embed: (context_batch_size, candidate_batch_size, d_model)
        context_embed, context_attention_weights = dot_attention(query=candidate_embed_query, key=context_code_embeds, value=context_code_embeds)

        output = dict()
        output["context_encoder"] = context_encoder_output
        output["context_embed"] = context_embed
        return output

    def forward_candidate_encoder(self, candidate_inputs: Dict[str, Tensor]) -> Tensor:
        candidate_output = self.candidate_encoder.forward(inputs=candidate_inputs)
        candidate_encoder_output = candidate_output["encoder"]
        candidate_embed = self.aggregation_layer(candidate_encoder_output)

        output = dict()
        output["candidate_encoder"] = candidate_encoder_output
        output["candidate_embed"] = candidate_embed
        return output

    def summary(self, batch_size=8, col_names=["kernel_size", "output_size", "num_params"]):
        context_inputs = dict()
        context_inputs["token"] = torch.zeros((batch_size, self.context_encoder.timesteps)).type(torch.int)
        for k, v in self.context_embedding_dict.items():
            context_inputs[k] = torch.zeros((batch_size, self.context_encoder.timesteps)).type(torch.int)

        candidate_inputs = dict()
        candidate_inputs["token"] = torch.zeros((batch_size, self.candidate_encoder.timesteps)).type(torch.int)
        for k, v in self.candidate_embedding_dict.items():
            candidate_inputs[k] = torch.zeros((batch_size, self.candidate_encoder.timesteps)).type(torch.int)
        summary = torchinfo.summary(self, input_data={"context_inputs": context_inputs, "candidate_inputs":candidate_inputs}, depth=4, col_names=col_names, verbose=0)
        print(summary)