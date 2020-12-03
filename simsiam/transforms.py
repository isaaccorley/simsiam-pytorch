from typing import Tuple, Dict

import kornia
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomGaussianBlur2D(kornia.augmentation.AugmentationBase2D):

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = "reflect",
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.1
    ) -> None:
        super(RandomGaussianBlur2D, self).__init__(
            p=p,
            return_transform=return_transform,
            same_on_batch=same_on_batch
        )

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return None

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return kornia.filters.gaussian_blur2d(
            input=input,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            border_type=self.border_type
        )


def augment_transforms(
    s: float,
    input_shape: Tuple[int, int],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Sequential:

    augs = nn.Sequential(
        kornia.augmentation.ColorJitter(
            brightness=0.8*s,
            contrast=0.8*s,
            saturation=0.8*s,
            hue=0.8*2,
            p=0.8
        ),
        kornia.augmentation.RandomGrayscale(p=0.2),
        RandomGaussianBlur2D(
            kernel_size=(3, 3),
            sigma=(1.5, 1.5),
            p=0.5
        ),
        kornia.augmentation.RandomResizedCrop(
            size=input_shape,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.33),
            interpolation="bilinear",
            p=1.0
        ),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.Normalize(
            mean=torch.tensor(IMAGENET_MEAN),
            std=torch.tensor(IMAGENET_STD)
        )
    )
    augs = augs.to(device)
    return augs

def load_transforms(input_shape: Tuple[int, int]) -> T.Compose:
    return T.Compose([
        T.Resize(size=input_shape, interpolation=Image.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
