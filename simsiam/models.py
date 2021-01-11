import torch
import torch.nn as nn
import torchvision


class LinearClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(                                                                                                           
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ProjectionMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PredictorMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):

    def __init__(
        self,
        backbone: str,
        pretrained: bool
    ):
        super().__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.emb_dim = resnet.fc.in_features
        self.model = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()


class SimSiam(nn.Module):

    def __init__(
        self,
        backbone: str,
        latent_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = Encoder(backbone=backbone, pretrained=False)

        # Projection (mlp) network
        self.projection_mlp = ProjectionMLP(
            input_dim=self.encoder.emb_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=latent_dim
        )

        # Predictor network (h)
        self.predictor_mlp = PredictorMLP(
            input_dim=latent_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=latent_dim
        )

    def forward(self, x: torch.Tensor):
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, e: torch.Tensor) -> torch.Tensor:
        return self.projection_mlp(e)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_mlp(z)


class ResNet(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        freeze: bool
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = Encoder(backbone=backbone, pretrained=pretrained)

        if freeze:
            for param in self.encoder.parameters():
                param.requres_grad = False

        # Linear classifier
        self.classifier = LinearClassifier(self.encoder.emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        return self.classifier(e)
