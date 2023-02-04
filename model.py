import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, lstm_first, lstm_second, latent_Wmu, latent_Wsig):
        super().__init__()
        self.lstm_first = nn.LSTM(**lstm_first)
        self.lstm_second = nn.LSTM(**lstm_second)
        self.latent_Wmu = nn.Linear(**latent_Wmu)
        self.latent_Wsig = nn.Linear(**latent_Wsig)
    
    def forward(self, x):
        B = x.shape[0]
        o, (h_1, c) = self.lstm_first(x)
        o, (h, c) = self.lstm_second(o)
        h = h.permute(1, 0, 2).reshape(B, -1) # 2 B H > B 2 H > B (2H)
        mu = self.latent_Wmu(h)
        sig = F.softplus(self.latent_Wsig(h), threshold=5) # (7)

        return mu, sig

class Decoder(nn.Module):
    def __init__(self, 
                 W_cond_init_state, lstm_conductor, 
                 W_dec_init_state, lstm_decoder,
                 num_conductor_out, num_decoder_out,
                 token_embedding, W_dist_out,
                 sampling_strategy = "argmax",
                 training_strategy = "scheduled"):
        super().__init__()
        self.W_cond_init_state = nn.Sequential(nn.Linear(**W_cond_init_state), nn.Tanh())
        self.lstm_conductor = nn.LSTM(**lstm_conductor)
        self.W_dec_init_state = nn.Sequential(nn.Linear(**W_dec_init_state), nn.Tanh())
        self.lstm_decoder = nn.LSTM(**lstm_decoder)
        self.embedding = nn.Embedding(**token_embedding)
        self.dist_out_layer = nn.Sequential(nn.Linear(**W_dist_out), nn.Softmax())

        self.conductor_hidden_size = lstm_conductor["hidden_size"]
        self.num_conductor_out = num_conductor_out
        self.num_decoder_out = num_decoder_out
        self.sampling_strategy = sampling_strategy
        self.training_strategy = training_strategy

    def forward(self, z, x=None, eps=0.):
        conductor_in = self.W_cond_init_state(z)
        conductor_out = self.get_conductor_out(conductor_in)
        states = self.W_dec_init_state(conductor_out)
        if x is not None: # teacher forcing
            if self.training_strategy == "scheduled":
                out = self.decoder_rnn_ss(states, x, eps) # distributions
            else: # teacher forcing
                out = self.decoder_rnn_tf(states, x) # distributions
        else: # feedback generation
            out = self.decoder_rnn_fg(states) # indices
        return out

    def get_conductor_out(self, conductor_in):
        B, D = conductor_in.shape
        h = conductor_in[:, :D//2]
        h1, h2 = h[:, :D//4], h[:, D//4:]
        h = torch.stack((h1, h2), 0)
        c = conductor_in[:, D//2:]
        c1, c2 = c[:, :D//4], c[:, D//4:]
        c = torch.stack((c1, c2), 0)
        outs = []
        start = torch.zeros((B, 1, D//4), 
                            dtype=conductor_in.dtype,
                            device=conductor_in.device)
        o = start
        for i in range(self.num_conductor_out):
            o, (h, c) = self.lstm_conductor(o, (h, c))
            outs.append(o)
        out = torch.cat(outs, 1)
        return out
    
    def decoder_rnn_tf(self, decoder_in, x):
        B, U, D = decoder_in.shape # U is the number of bars
        Bx, L = x.shape 
        assert D % 4 == 0
        assert L % U == 0
        decoder_in = decoder_in.view(-1, D) # B U D > (BU) D
        x = x.view(Bx*U, L//U) # B L > (BU) (L/U) # (B*4) (64/4)
        h = decoder_in[:, :D//2]
        h1, h2 = h[:, :D//4], h[:, D//4:]
        h = torch.stack((h1, h2), 0)
        c = decoder_in[:, D//2:]
        c1, c2 = c[:, :D//4], c[:, D//4:]
        c = torch.stack((c1, c2), 0)
        outs = []
        emb_init = torch.zeros([Bx*U, 1, D//4],
                               dtype=decoder_in.dtype,
                               device=decoder_in.device)
        emb = self.embedding(x)
        emb = torch.cat((emb_init, emb[:, :-1]), 1)
        h_init = h.clone().sum(0)[:, None] # 2 (BU) D > (BU) 1 D # no info about sum
        emb = torch.cat((emb, h_init.repeat(1, emb.shape[1], 1)), -1)
        o, (h, c) = self.lstm_decoder(emb, (h, c))
        o = self.dist_out_layer(o)

        o_unfolded = o.reshape(B, L, -1)
        
        return o_unfolded

    def decoder_rnn_fg(self, decoder_in):
        B, U, D = decoder_in.shape # U is the number of bars
        assert D % 4 == 0
        decoder_in = decoder_in.view(-1, D) # B U D > (BU) D
        h = decoder_in[:, :D//2]
        h1, h2 = h[:, :D//4], h[:, D//4:]
        h = torch.stack((h1, h2), 0)
        c = decoder_in[:, D//2:]
        c1, c2 = c[:, :D//4], c[:, D//4:]
        c = torch.stack((c1, c2), 0)
        outs = []
        emb = torch.zeros([B*U, 1, D//4], 
                          dtype=decoder_in.dtype,
                          device=decoder_in.device)
        h_init = h.clone().sum(0)[:, None] # 2 (BU) D > (BU) 1 D # no info about sum
        for i in range(self.num_decoder_out):
            o_in = torch.cat((emb, h_init), -1)
            o, (h, c) = self.lstm_decoder(o_in, (h, c))
            o = self.dist_out_layer(o)
            emb, indices = self.sample_from_dist(o)
            outs.append(indices)
        out = torch.cat(outs, 1)
        out_unfolded = out.view(B, U*self.num_decoder_out, -1)
        return out_unfolded

    def decoder_rnn_ss(self, decoder_in, x, eps):
        # 1) with torch.no_grad(), get x_hat indices
        # 2) random mixing
        # 3) decoder_rnn_tf with mixed sample
        with torch.no_grad():
            hat_dist = self.decoder_rnn_tf(decoder_in, x)
            _, hat_indices = self.sample_from_dist(hat_dist)
            mixed_x = self.scheduled_sampling(x, hat_indices, eps)

        out = self.decoder_rnn_tf(decoder_in, mixed_x)
        return out

    def scheduled_sampling(self, x, x_hat, eps):
        B, L = x.shape
        device = x.device
        candidate = torch.cat((x, x_hat), -1) # 0: True, 1: hat
        coins = torch.empty(B, L, device=device).fill_(eps) # eps==0: all-True, eps==1: all-hat
        result = torch.bernoulli(coins).to(int)
        
        sampled = torch.gather(candidate, -1, result)
        return sampled

    def sample_from_dist(self, dist):
        if self.sampling_strategy == "argmax":
            indices = torch.argmax(dist, dim=-1)
            emb = self.embedding(indices)
            return emb, indices
        elif self.sampling_strategy == "multinomial":
            B, L, D = dist.shape
            dist_folded = dist.view(B*L, D)
            indices = torch.multinomial(dist_folded, 1) # (BL) 1
            indices = indices.view(B, L)
            emb = self.embdding(indices)
            return emb, indices


class MusicVAE(nn.Module):
    def __init__(self, embedding, encoder, decoder):
        super().__init__()
        self.emb = nn.Embedding(**embedding)
        self.enc = Encoder(**encoder)
        self.dec = Decoder(**decoder)
    
    def forward(self, x=None, z=None, mode="train", eps=0.5):
        if mode == "train":
            embed_out = self.emb(x)
            mu, sig = self.enc(embed_out)
            z_reparam = self.z_reparam(mu, sig)
            x_hat = self.dec(z_reparam, x=x) # x_hat: dist
        elif mode == "generate":
            x_hat = self.dec(z) # x_hat: indices(sampled from dist)
        else: raise TypeError("Unknown mode")
        
        out = x_hat if mode == "generate" else (x_hat, mu, sig)
        return out

    def z_reparam(self, mu, sig):
        eps = torch.randn_like(mu, requires_grad=False)
        
        z_reparam = mu + sig*eps
        return z_reparam
        
if __name__ == "__main__":
    from config import model_config
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    test_x = torch.randint(128, [10, 256], device=device)
    test_z = torch.randn([10, 512], device=device)
    model = MusicVAE(**model_config).to(device)
    out_tf, mu, sig = model(x = test_x)
    out_fg = model(z = test_z, mode="generate")
    
    breakpoint()