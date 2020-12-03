import torch
import torch.nn as nn
import torchvision


class SimSiam(nn.Module):

    def __init__(
        self,
        backbone: str = "resnet18",
        hidden_dim: int = 512,
        pretrained: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:

        super().__init__()

        # Encoder network
        resnet = getattr(backbone, torchvision.models)(pretrained=pretrained)
        embedding_dim = latent_dim = resnet.fc.in_features
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Predictor (mlp) network
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze()

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)