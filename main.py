import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from config import model_config
from model import MusicVAE
from dataset import get_dataset, col_fn
from loss import loss_fn
from utils import get_beta, get_epsilon

def main():
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    
    model = MusicVAE(**model_config)
    dataset = get_dataset()
    dl = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=col_fn) # b 512 but
    opt = Adam(model.parameters(), lr=1e-3)
    lr_sch = ExponentialLR(opt, gamma=0.9999) 
    lr_sch.last_epoch = 46049 # 5.1. 3. 1 (0.9999**n==0.01)
    
    model = model.to(device)
    idx = 0
    for e in range(100):
        model.train()
        for x in dl:
            opt.zero_grad()
            x = x.to(device)
            eps = get_epsilon(idx)
            x_hat, mu, sigma = model(x, mode="train", eps=eps)

            beta = get_beta(idx)
            loss = loss_fn(x_hat, x, mu, sigma, beta)
            loss.backward()
            opt.step()
            lr_sch.step()
            idx += 1
            print(f"\riter: {idx:,} / loss = {loss.item():.3f}", end='')
    torch.save(model.state_dict(), "checkpoint.ckpt")

if __name__ == "__main__":
    main()