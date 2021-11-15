import torch
from torch import nn, Tensor
from typing import Optional, List, Tuple
from transformer.assertions.object_assertion import ModelAssertion
from transformer.layers.utils import get_positional_encoding_matrix

class EmbeddingModule(nn.modules.Module):
    def __init__(self, timesteps, d_model, embedding_size, return_embedding_weights=True):
        super(EmbeddingModule, self).__init__()
        self.timesteps = timesteps
        self.d_model = d_model
        self.embedding_size = embedding_size
        self.return_embedding_weights = return_embedding_weights
        # parameters
        self.embedding_layer = nn.Embedding(embedding_size, d_model)
        self.embedding_layer.weight.requires_grad = True

    def forward(self, ids: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        '''
        :param ids: (batch_size, timesteps)
        :return:
        '''
        output = self.embedding_layer(ids)

        if self.return_embedding_weights:
            return output, self.embedding_layer.weight
        else:
            return output

class EmbeddingAggregation(nn.modules.Module, ModelAssertion):
    def __init__(self, method):
        super(EmbeddingAggregation, self).__init__()
        self.method = method

    def forward(self, inputs: Tensor) -> Tensor:
        self.assert_equal(a=len(inputs.shape), b=3) # inputs must be 3D-tensor
        # inputs: (batch_size, timesteps, d_model)
        # output: (batch_size, d_model)
        output = None
        if self.method == "first": output = inputs[:, 0, :]
        elif self.method == "last": output = inputs[:, -1, :]
        elif self.method == "sum": output = torch.sum(inputs, axis=1)
        elif self.method == "average": output = torch.mean(inputs, axis=1)
        return output

class EncoderEmbedding(nn.modules.Module):
    def __init__(self, timesteps, d_model, dropout=0.1):
        super(EncoderEmbedding, self).__init__()
        self.timesteps = timesteps
        self.d_model = d_model
        self.dropout = dropout
        self.embed_dropout = nn.modules.dropout.Dropout(dropout)
        positional_encoding_matrix = get_positional_encoding_matrix(timesteps=timesteps, d_model=d_model)
        self.register_buffer("positional_encoding_matrix", positional_encoding_matrix)

    def forward(self, embeds: List[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        '''
        :param embeds: List of embed
        embed: (batch_size, timesteps, d_model)
        :return:
        '''
        output = torch.zeros_like(embeds[0])
        for embed in embeds:
            output = output + embed
        output = output + self.positional_encoding_matrix

        if self.dropout is not None:
            output = self.embed_dropout(output)
        return output