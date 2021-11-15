import copy
import numpy as np
import torch
from torch import nn

def get_activation_func(activation):
    activation_func = None
    if activation == "relu":
        activation_func = nn.functional.relu
    elif activation == "gelu":
        activation_func = nn.functional.gelu
    elif activation == "tanh":
        activation_func = torch.tanh
    elif activation == "sigmoid":
        activation_func = nn.functional.sigmoid
    elif activation == "softmax":
        activation_func = nn.functional.softmax
    elif activation == "log_softmax":
        activation_func = nn.functional.log_softmax
    return activation_func

def get_clones(module, N):
    return nn.modules.container.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_positional_encoding_matrix(timesteps, d_model):
    def _get_angles(timesteps, d_model, i):
        default_value = 1e+4
        rates = 1 / np.power(default_value, (2 * 1) / np.float32(d_model))
        return rates * timesteps

    pos_array = np.expand_dims(np.arange(timesteps), axis=1)
    i_array = np.expand_dims(np.arange(d_model), axis=0) / 2
    positional_encoding_matrix = _get_angles(pos_array, d_model, i_array)
    positional_encoding_matrix[:, 0::2] = np.sin(positional_encoding_matrix[:, 0::2])
    positional_encoding_matrix[:, 1::2] = np.sin(positional_encoding_matrix[:, 1::2])
    positional_encoding_matrix = torch.from_numpy(positional_encoding_matrix)
    return positional_encoding_matrix

def get_pad_mask(inputs: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    assert isinstance(inputs, torch.Tensor), "Inputs must be instance of torch.Tensor"
    assert len(inputs.shape) == 2, "Inputs must be matrix"
    mask = inputs == pad_token_id
    return mask

def get_sub_mask(inputs: torch.Tensor) -> torch.Tensor:
    assert isinstance(inputs, torch.Tensor), "Inputs must be instance of torch.Tensor"
    assert len(inputs.shape) == 2, "Inputs must be matrix"
    sequence_length = inputs.shape[1]
    mask = torch.ones(sequence_length, sequence_length).to(inputs.device)
    mask = torch.triu(mask).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask

def dot_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    # query: (batch_size, m, d_model)
    # key, value: (batch_size, timesteps, d_model)
    # attention_weight: (batch_size, m, sequence_length)
    attention_weight = torch.matmul(query, key.transpose(2, 1))
    attention_weight = nn.functional.softmax(attention_weight, dim=-1)
    # output: (batch_size, m, d_model)
    output = torch.matmul(attention_weight, value)
    return output, attention_weight