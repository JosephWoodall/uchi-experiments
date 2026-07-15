"""Ported from uchi/uchi/flux/bitnet.py, unchanged -- moved to its own
module (rather than living inside model.py) so both model.py (attention
path) and rwkv_model.py (TimeMixing path) can use it without a circular
import between the two.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightQuantizer(torch.autograd.Function):
    """1.58-bit (ternary {-1,0,1}) weight quantization, straight-through
    estimator. Exists in uchi to shrink FLUX's weight storage 20x; here
    it's applied to Ducky's own attention/RWKV projections so parameter
    count grows without storage growing proportionally.
    """

    @staticmethod
    def forward(ctx, weight):
        gamma = weight.abs().mean()
        quantized = torch.round(weight / (gamma + 1e-8))
        quantized = torch.clamp(quantized, -1.0, 1.0)
        return quantized * gamma

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ActivationQuantizer(torch.autograd.Function):
    """8-bit activation quantization, straight-through estimator."""

    @staticmethod
    def forward(ctx, x):
        gamma = x.abs().max(dim=-1, keepdim=True).values
        scale = 127.0 / (gamma + 1e-8)
        quantized = torch.round(x * scale)
        quantized = torch.clamp(quantized, -128.0, 127.0)
        return quantized / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitLinear(nn.Linear):
    """Ternary-weight, 8-bit-activation linear layer. Ported from uchi's
    BitLinear (uchi/uchi/flux/bitnet.py) unchanged."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.layer_norm = nn.LayerNorm(in_features)
        self.quantize = True

    def forward(self, x):
        x_norm = self.layer_norm(x)
        if self.quantize:
            x_quant = ActivationQuantizer.apply(x_norm)
            w_quant = WeightQuantizer.apply(self.weight)
            return F.linear(x_quant, w_quant, self.bias)
        return F.linear(x_norm, self.weight, self.bias)
