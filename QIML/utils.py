import torch
from pathlib import Path


install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("data/raw"),
    "processed": top.joinpath("data/processed"),
    "models": top.joinpath("models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts")
}


def get_encoded_size(data, model):
    data.setup()
    # Loop to generate a batch of data taken from dataset
    for i in range(0, 12):
        if i == 0:
            X, _ = data.train_set.__getitem__(0)
            X = X.unsqueeze(0)
        else:
            Xtemp, _ = data.train_set.__getitem__(0)
            Xtemp = Xtemp.unsqueeze(0)
            X = torch.cat((X, Xtemp), dim=0)

    # some shape tests before trying to actually train
    z, res = model.encoder(X.unsqueeze(1))
    # out = model(X)[0]
    return z, res
