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
        torch.Tensor: The density function evaluated at x.
    """
    normal = torch.distributions.normal.Normal(0, 1)

    assert a < b, "The lower bound must be less than the upper bound."
    assert sigma > 0, "The standard deviation must be positive."

    if x < a or x > b:
        return torch.zeros_like(x) + 1e-5

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    xi = (x - mu) / sigma
    alpha_normal_cdf = normal.cdf(alpha)
    beta_normal_cdf = normal.cdf(beta)

    return (
        sigma.reciprocal()
        * 10 ** normal.log_prob(xi)
        * (beta_normal_cdf - alpha_normal_cdf).reciprocal()
    )
