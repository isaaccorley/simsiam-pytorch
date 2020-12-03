import os
import multiprocessing as mp

import hydra
import torch
import torchvision
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from simsiam.transforms import load_transforms, augment_transforms
from simsiam.models import SimSiam
from simsiam.losses import simsiam_loss


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:

    print(dict(cfg))

    model = SimSiam(
        backbone=cfg.model.backbone,
        hidden_dim=cfg.model.hidden_dim,
        pretrained=cfg.model.pretrained,
        device=cfg.device
    )
    model.train()

    opt = torch.optim.SGD(
        params=model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )

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
        num_workers=mp.cpu_count()
    )

    transforms = augment_transforms(
        s=cfg.data.s,
        input_shape=cfg.data.input_shape,
        device=cfg.device
    )

    writer = SummaryWriter()

    n_iter = 0
    for epoch in range(cfg.train.epochs):

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch, (x, _) in pbar:

            opt.zero_grad()

            x = x.to(cfg.device)

            # augment
            x1, x2 = transforms(x), transforms(x)

            # encode
            z1, z2 = model.encode(x1), model.encode(x2)
            
            # project
            p1, p2 = model.project(z1), model.project(z2)

            # compute loss
            loss = simsiam_loss(z1, z2, p1, p2)
            loss.backward()
            opt.step()

            pbar.set_description("Epoch {}, Loss: {:.4f}".format(epoch, float(loss)))

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(tag="loss", scalar_value=float(loss), global_step=n_iter)

            n_iter += 1

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(writer.log_dir, "model.pt"))


if __name__ == "__main__":
    main()