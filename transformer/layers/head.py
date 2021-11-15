import torch
from torch import nn, Tensor
from transformer.layers.utils import get_activation_func, dot_attention

class LanguageModelingHead(nn.modules.Module):
    def __init__(self, d_model, vocab_size, shared_embedding, activation="gelu", layer_norm_epsilon=1e-5, initialization="normal"):
        super(LanguageModelingHead, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.shared_embedding = shared_embedding
        self.initialization = initialization

        if self.shared_embedding:
            self.linear_weight = nn.parameter.Parameter(torch.empty(d_model, d_model, dtype=torch.double))
            self.linear_bias = nn.parameter.Parameter(torch.empty(d_model, dtype=torch.double))
            self.register_parameter('softmax_weight', None)
            self.activation = activation
            self.activation_func = get_activation_func(activation)
            self.layer_normalization = nn.modules.normalization.LayerNorm(d_model, eps=layer_norm_epsilon).double()
        else:
            self.softmax_weight = nn.parameter.Parameter(torch.empty(vocab_size, d_model, dtype=torch.double))
            self.register_parameter('linear_weight', None)
            self.register_parameter('linear_bias', None)
        self.softmax_bias = nn.parameter.Parameter(torch.empty(vocab_size, dtype=torch.double))
        self._reset_parameters()

    def _reset_parameters(self):
        if self.initialization=="normal":
            if self.linear_weight is not None:
                nn.init.normal_(self.linear_weight, mean=0.0, std=0.02)
            if self.softmax_weight is not None:
                nn.init.normal_(self.softmax_weight, mean=0.0, std=0.02)
        elif self.initialization=="xavier_uniform":
            if self.linear_weight is not None:
                nn.init.xavier_uniform_(self.linear_weight)
            if self.softmax_weight is not None:
                nn.init.xavier_uniform_(self.softmax_weight)
        if self.linear_bias is not None:
            nn.init.constant_(self.linear_bias, 0.)
        if self.softmax_bias is not None:
            nn.init.constant_(self.softmax_bias, 0.)

    def forward(self, inputs: Tensor, token_embed_weight: Tensor = None) -> Tensor:
        logits = None
        if self.shared_embedding:
            assert token_embed_weight is not None, "token_embed_weight should not be None when shared_embedding is True"
            logits = self.activation_func(nn.functional.linear(inputs, self.linear_weight, self.linear_bias))
            logits = self.layer_normalization(logits)
            logits = nn.functional.linear(logits, token_embed_weight.double(), self.softmax_bias)
        else:
            assert token_embed_weight is None, "token_embed_weight should be None when shared_embedding is False"
            logits = nn.functional.linear(inputs, self.softmax_weight, self.softmax_bias)
        output = nn.functional.log_softmax(input=logits, dim=-1)
        return output

class NextSentencePredictionHead(nn.modules.Module):
    def __init__(self, d_model, units=2, activation="tanh", initialization="normal"):
        super(NextSentencePredictionHead, self).__init__()
        self.d_model = d_model
        self.units = units
        self.initialization = initialization
        # layers
        self.linear_weight = nn.parameter.Parameter(torch.empty(d_model, d_model, dtype=torch.double))
        self.linear_bias = nn.parameter.Parameter(torch.empty(d_model, dtype=torch.double))
        self.softmax_weight = nn.parameter.Parameter(torch.empty(units, d_model, dtype=torch.double))
        self.softmax_bias = nn.parameter.Parameter(torch.empty(units, dtype=torch.double))
        self.activation = activation
        self.activation_func = get_activation_func(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.initialization=="normal":
            if self.linear_weight is not None:
                nn.init.normal_(self.linear_weight, mean=0.0, std=0.02)
            if self.softmax_weight is not None:
                nn.init.normal_(self.softmax_weight, mean=0.0, std=0.02)
        elif self.initialization=="xavier_uniform":
            if self.linear_weight is not None:
                nn.init.xavier_uniform_(self.linear_weight)
            if self.softmax_weight is not None:
                nn.init.xavier_uniform_(self.softmax_weight)
        if self.linear_bias is not None:
            nn.init.constant_(self.linear_bias, 0.)
        if self.softmax_bias is not None:
            nn.init.constant_(self.softmax_bias, 0.)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch_size, timesteps, d_model)
        # logtis: (batch_size, d_model) #
        logits = inputs[:, 0, :] # cls_token
        logits = nn.functional.linear(logits, self.linear_weight, self.linear_bias)
        logits = self.activation_func(logits)
        logits = nn.functional.linear(logits, self.softmax_weight, self.softmax_bias)
        output = nn.functional.log_softmax(input=logits, dim=-1)
        return output

class PolyEncoderHead(nn.modules.Module):
    def __init__(self, d_model):
        super(PolyEncoderHead, self).__init__()
        self.d_model = d_model

    def forward(self, context_embeds: Tensor, candidate_embeds: Tensor = None) -> Tensor:
        # context_embed: (context_batch_size, m, d_model)
        # candidate_embed: (candidate_batch_size, d_model)
        context_batch_size = context_embeds.shape[0]
        candidate_batch_size = candidate_embeds.shape[0]
        # candidate_embed_query: (context_batch_size, candidate_batch_size, d_model)
        candidate_embed_query = torch.unsqueeze(candidate_embeds, dim=0).expand(context_batch_size, candidate_batch_size, self.d_model)
        # context_embed: (context_batch_size, candidate_batch_size, d_model)
        context_embeds, context_attention_weights = dot_attention(query=candidate_embed_query, key=context_embeds, value=context_embeds)
        # output: (context_batch_size, candidate_batch_size)
        output = torch.sum(context_embeds * candidate_embeds, dim=-1)
        output = nn.functional.log_softmax(output, dim=-1)
        return output