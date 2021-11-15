from torch import nn, Tensor
from typing import Optional, Tuple
from .attention import MultiheadAttention, PositionwiseFeedForward


class EncoderLayer(nn.modules.Module):
    def __init__(self, d_model, d_ff, num_heads, pwff_activation="gelu", dropout=0.1, bias=True, layer_norm_epsilon=1e-5, initialization="normal", return_attention_weights=False):
        super(EncoderLayer, self).__init__()
        self.return_attention_weights = return_attention_weights
        self.mha_layer = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, bias=bias, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, initialization=initialization).double()
        self.mha_dropout_layer = nn.modules.dropout.Dropout(dropout).double()
        self.layer_normalization = nn.modules.normalization.LayerNorm(d_model, eps=layer_norm_epsilon).double()
        self.pwff_layer = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, activation=pwff_activation, dropout=dropout, bias=bias, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization).double()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.gelu
        super(EncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        mha_output, mha_weight = self.mha_layer(query=src, key=src, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        mha_output = src + self.mha_dropout_layer(mha_output)
        mha_output = self.layer_normalization(mha_output)
        output = self.pwff_layer(inputs=mha_output)
        if self.return_attention_weights:
            return output, mha_weight
        else:
            return output


class DecoderLayer(nn.modules.Module):
    def __init__(self, d_model, d_ff, num_heads, pwff_activation="gelu", dropout=0.1, bias=True, layer_norm_epsilon=1e-5, initialization="normal", return_attention_weights=False):
        super(DecoderLayer, self).__init__()
        self.return_attention_weights = return_attention_weights
        self.self_mha_layer = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, bias=bias, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, initialization=initialization).double()
        self.self_mha_dropout_layer = nn.modules.dropout.Dropout(dropout).double()
        self.self_mha_layer_normalization = nn.modules.normalization.LayerNorm(d_model, eps=layer_norm_epsilon).double()
        self.memory_mha_layer = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, bias=bias, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, initialization=initialization).double()
        self.memory_mha_dropout_layer = nn.modules.dropout.Dropout(dropout).double()
        self.memory_mha_layer_normalization = nn.modules.normalization.LayerNorm(d_model, eps=layer_norm_epsilon).double()
        self.pwff_layer = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, activation=pwff_activation, dropout=dropout, bias=bias, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization).double()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.gelu
        super(DecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        self_mha_output, self_mha_weight = self.self_mha_layer(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        self_mha_output = tgt + self.self_mha_dropout_layer(self_mha_output)
        self_mha_output = self.self_mha_layer_normalization(self_mha_output)
        memory_mha_output, memory_mha_weight = self.memory_mha_layer(query=self_mha_output, key=memory, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        memory_mha_output = tgt + self.memory_mha_dropout_layer(memory_mha_output)
        memory_mha_output = self.memory_mha_layer_normalization(memory_mha_output)
        output = self.pwff_layer(inputs=memory_mha_output)

        if self.return_attention_weights:
            return output, self_mha_weight
        else:
            return output