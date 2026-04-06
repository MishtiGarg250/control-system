import torch
import numpy as np
from model import Net
import os

X = torch.tensor(np.load("data/X.npy"), dtype=torch.float32)
y = torch.tensor(np.load("data/y.npy"), dtype=torch.float32).view(-1,1)

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for e in range(1500):
    loss = loss_fn(model(X), y)
    opt.zero_grad()
    loss.backward()
    opt.step()


os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/model.pt")
