import torch
import torch.nn as nn
import math 


class MultiheadSelfAttention(nn.Module):

    def __init__(self, d_model, nheads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)

    def forward(self, x):
        # x: [B, 900, d]
        out, _ = self.attn(x, x, x)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model, nheads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)

    def forward(self, x, c):

        out, _ = self.attn(x, c, c)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nheads):
        super().__init__()
        self.cross_attn = MultiheadCrossAttention(d_model, nheads)
        self.self_attn = MultiheadSelfAttention(d_model, nheads)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x, c):
        x = x + self.cross_attn(self.ln1(x), c)
        x = x + self.self_attn(self.ln2(x))
        x = x + self.mlp(self.ln3(x))
        return x

class Diffusion_Transformer(nn.Module):

    def __init__(self, fom_dim, rom_dim,
                 d_model, nheads, nlayers):

        super().__init__()

        self.fom_embed = nn.Linear(1, d_model)
        self.rom_embed = nn.Linear(1, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nheads)
            for _ in range(nlayers)
        ])

        self.output_proj = nn.Linear(d_model, 1)

    def get_sinusoidal_pos_emb(self, seq_len, dim, device):
        
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        i = torch.arange(dim, device=device).unsqueeze(0)
        angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
        angle_rads = pos * angle_rates
        pos_emb = torch.zeros(seq_len, dim, device=device)
        pos_emb[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_emb[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        
        return pos_emb

    def forward(self, x_t, rom):
        """
        x_t : [B, 900]  noisy FOM
        rom : [B, 375]  ROM prior
        t   : [B]       timestep
        """

        B = x_t.shape[0]

        # tokenize modes
        x = self.fom_embed(x_t.unsqueeze(-1))   # [B,900,d]
        c = self.rom_embed(rom.unsqueeze(-1))   # [B,375,d]

        x = x + self.get_sinusoidal_pos_emb(x.size(1), x.size(2), x.device)
        c = c + self.get_sinusoidal_pos_emb(c.size(1), c.size(2), c.device)

        for block in self.blocks:
            x = block(x, c)

        eps = self.output_proj(x).squeeze(-1)   # [B,900]
        return eps