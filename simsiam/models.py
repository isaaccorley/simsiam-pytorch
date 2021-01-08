import torch
import torch.nn as nn
import torchvision


class SimSiam(nn.Module):

    def __init__(
        self,
        backbone: str,
        latent_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        pretrained: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:

        super().__init__()

        # Encoder network
        resnet = getattr(torchvision.models, backbone)(pretrained=pretrained)
        emb_dim = resnet.fc.in_features
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Projection (mlp) network
        self.projection_mlp = nn.Sequential(
            nn.Linear(emb_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        # Predictor network (h)
        self.predictor_mlp = nn.Sequential(
            nn.Linear(latent_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, latent_dim)
        )

        self.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze()

    def project(self, e: torch.Tensor) -> torch.Tensor:
        return self.projection_mlp(e)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_mlp(z)


class MLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
