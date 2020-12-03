# simsiam-pytorch
Minimal PyTorch Implementation of SimSiam from ["Exploring Simple Siamese Representation Learning" by Chen et al.](https://arxiv.org/abs/2011.10566)

<p align="center"><img src="assets/models.png" width="480"\></p>


### Load and train on a custom dataset

```python
from simsiam import SimSiam
from simsiam.losses import negative_cosine_similarity

model = SimSiam(
    backbone="resnet50",    # encoder network
    latent_dim=2048,        # predictor network output size
    proj_hidden_dim=2048    # projection mlp hidden layer size
    pred_hidden_dim=512     # predictor mlp hidden layer size
    device="cuda"           # use all the parallels
)

dataset = ...
dataloader = ...
opt = ...

for epoch in range(epochs):
    for batch, (x, y) in enumerate(dataloader):
        opt.zero_grad()

        x1, x2 = transforms(x), transforms(x)           # Augment
        e1, e2 = model.encode(x1), model.encode(x2)     # Encode
        z1, z2 = model.project(e1), model.project(e2)   # Project
        p1, p2 = model.project(z1), model.project(z2)   # Predict

        # Compute loss
        loss1 = negative_cosine_similarity(p1, z1)
        loss2 = negative_cosine_similarity(p2, z2)
        loss = loss1/2 + loss2/2
        loss.backward()
        opt.step()

```

### Install dependencies

```bash
pip install -r requirements.txt

```

### Train on STL-10 dataset

Modify config.yaml to your liking and run

```python
python pretrain.py

```

### View logs in tensorboard

```python
tensorboard --logdir=logs

```
