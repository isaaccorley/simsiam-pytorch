import torch
import torch.nn.functional as F


def negative_cosine_similarity(
    p: torch.Tensor,
    z: torch.Tensor
) -> torch.Tensor:
    """ D(p, z) = -(p*z).sum(dim=1).mean() """
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
