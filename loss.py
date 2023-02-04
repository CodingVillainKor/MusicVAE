import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


def recon_loss(x_hat, x):
    return F.nll_loss(torch.log(x_hat).permute(0,2,1), x)

def kl_loss(mu, sigma, beta, free_bits): # 5.2. 1. 2
    device = mu.device
    mu_p  = torch.tensor([0.], device=device)
    sig_p = torch.tensor([1.], device=device)
    p_dist = Normal(mu_p, sig_p)
    q_dist = Normal(mu, sigma)

    kl_div = kl_divergence(q_dist, p_dist)
    free_bits_tensor = torch.tensor([free_bits], device=device)
    zero = torch.tensor([0.], device=device)
    kl_loss = - beta * torch.max(torch.mean(kl_div)-free_bits, zero)

    return kl_loss

def loss_fn(x_hat, x, mu, sigma, beta, free_bits=48.):
    reconstruction_loss = recon_loss(x_hat, x)
    kl_div_loss = kl_loss(mu, sigma, beta, free_bits)

    return reconstruction_loss + kl_div_loss