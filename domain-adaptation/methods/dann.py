from __future__ import annotations

import math
import torch


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def gradient_reverse(x, strength):
    return _GradientReverse.apply(x, float(strength))


def grl_schedule(progress, maximum=1.0):
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * float(progress))) - 1.0)
