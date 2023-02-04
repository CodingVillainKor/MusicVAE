import math
# beta for beta-VAE 
def get_beta(idx):
    return 2. - 2.0 * 0.99999 ** idx # 5.2. 1. 2

# epsilon for scheduled sampling coefficient
def get_epsilon(idx, k_rate=2000):
    # inverse sigmoid
    eps = k_rate / (k_rate+math.exp(idx/k_rate))
    return eps