import os
import multiprocessing as mp
from datetime import datetime

import hydra
import torch
import torchvision
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from simsiam.models import SimSiam
from simsiam.losses import negative_cosine_similarity
from simsiam.transforms import load_transforms, augment_transforms


@hydra.main(config_name="configs/config")
def main(cfg: DictConfig) -> None:

    cfg.data.path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.path)
    cfg.train.log_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.train.log_dir)

    model = SimSiam(
        backbone=cfg.model.backbone,
        latent_dim=cfg.model.latent_dim,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        pred_hidden_dim=cfg.model.pred_hidden_dim,
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
        input_shape=cfg.data.input_shape,
        device=cfg.device
    )

    log_dir = os.path.join(
        cfg.train.log_dir,
        "pretrain_" + datetime.now().strftime('%b%d_%H-%M-%S')
    )
    writer = SummaryWriter(log_dir=log_dir)

    n_iter = 0
    for epoch in range(cfg.train.epochs):

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch, (x, _) in pbar:

            opt.zero_grad()

            x = x.to(cfg.device)

            # augment
            x1, x2 = transforms(x), transforms(x)

            # encode
            e1, e2 = model.encode(x1), model.encode(x2)

            # project
            z1, z2 = model.project(e1), model.project(e2)

            # predict
            p1, p2 = model.project(z1), model.project(z2)

            # compute loss
            loss1 = negative_cosine_similarity(p1, z1)
            loss2 = negative_cosine_similarity(p2, z2)
            loss = loss1/2 + loss2/2
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