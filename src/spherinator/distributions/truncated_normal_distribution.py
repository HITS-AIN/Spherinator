import torch


def truncated_normal_distribution(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    a: float,
    b: float,
) -> torch.Tensor:
    """Compute the probability density function of a truncated normal distribution.
    Args:
        x (torch.Tensor): The value at which to evaluate the density function.
        mu (torch.Tensor): The mean of the normal distribution.
        sigma (torch.Tensor): The standard deviation of the normal distribution.
        a (float): The lower bound of the truncation interval.
        b (float): The upper bound of the truncation interval.
    Returns:
        torch.Tensor: The probability density evaluated at x.
    """

    if a >= b:
        raise ValueError("The lower bound must be less than the upper bound")
    if not torch.all(sigma > 0):
        raise ValueError("Every standard deviation must be positive")
    if x.shape != mu.shape or x.shape != sigma.shape:
        raise ValueError("x, mu, and sigma must have the same shape")

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    xi = (x - mu) / sigma

    normal = torch.distributions.normal.Normal(0, 1)
    alpha_normal_cdf = normal.cdf(alpha)
    beta_normal_cdf = normal.cdf(beta)

    normalizer = (beta_normal_cdf - alpha_normal_cdf).clamp_min(torch.finfo(x.dtype).tiny)
    probability = sigma.reciprocal() * normal.log_prob(xi).exp() / normalizer

    probability = torch.where((x < a) | (x > b), torch.zeros_like(probability), probability)

    return probability
