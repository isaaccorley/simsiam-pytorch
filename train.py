import os
import json
import argparse
from types import SimpleNamespace

import torch
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from simsiam.models import ResNet, LinearClassifier
from simsiam.transforms import load_transforms, augment_transforms


def main(cfg: SimpleNamespace) -> None:

    model = ResNet(
        backbone=cfg.model.backbone,
        num_classes=cfg.data.num_classes,
        pretrained=False,
        freeze=cfg.model.freeze
    )

    if cfg.model.weights_path:
        model.encoder.load_state_dict(torch.load(cfg.model.weights_path))

    model = model.to(cfg.device)

    opt = torch.optim.SGD(
        params=model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )
    loss_func = torch.nn.CrossEntropyLoss()

    dataset = torchvision.datasets.STL10(
        root=cfg.data.path,
        split="train",
        transform=load_transforms(input_shape=cfg.data.input_shape),
        download=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=torch.multiprocessing.cpu_count()
    )

    transforms = augment_transforms(
        input_shape=cfg.data.input_shape,
        device=cfg.device
    )

    writer = SummaryWriter()

    n_iter = 0
    for epoch in range(cfg.train.epochs):

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch, (x, y) in pbar:

            opt.zero_grad()

            x, y = x.to(cfg.device), y.to(cfg.device)
            x = transforms(x)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            opt.step()

            pbar.set_description("Epoch {}, Loss: {:.4f}".format(epoch, float(loss)))

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(tag="loss", scalar_value=float(loss), global_step=n_iter)

            n_iter += 1

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(writer.log_dir, cfg.model.name + ".pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    main(cfg)
