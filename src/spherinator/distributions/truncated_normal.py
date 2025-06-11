import math

import torch
from torch.distributions.kl import register_kl

_EPS = 1e-7


class TruncatedNormal(torch.distributions.Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }

    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        super().__init__()

    def log_prob(self, value):
        return self.log_normalizer() + self.scale * torch.log1p((self.loc * value).sum(-1))

    def log_normalizer(self):
        alpha = self.base_dist.marginal_t.base_dist.concentration1
        beta = self.base_dist.marginal_t.base_dist.concentration0
        return -(
            (alpha + beta) * math.log(2) + torch.lgamma(alpha) - torch.lgamma(alpha + beta) + beta * math.log(math.pi)
        )

    def entropy(self):
        alpha = self.base_dist.marginal_t.base_dist.concentration1
        beta = self.base_dist.marginal_t.base_dist.concentration0
        return -(
            self.log_normalizer() + self.scale * (math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + beta))
        )

    @property
    def mean(self):
        return self.loc * self.base_dist.marginal_t.mean

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        alpha = self.base_dist.marginal_t.base_dist.concentration1
        beta = self.base_dist.marginal_t.base_dist.concentration0
        ratio = (alpha + beta) / (2 * beta)
        return self.base_dist.marginal_t.variance * (
            (1 - ratio) * self.loc.unsqueeze(-1) @ self.loc.unsqueeze(-2) + ratio * torch.eye(self.loc.shape[-1])
        )


@register_kl(TruncatedNormal, TruncatedNormal)
def _kl_truncatednormal_truncatednormal(p, q):
    return -p.entropy() + q.entropy()
