import ast 

import torch
import torch.nn as nn 
import torch.functional as F 

from dataclasses import dataclass

"""
Inspired from https://github.com/karpathy/minGPT
"""
# ------------------------ MODEL ------------------------
@dataclass
class ViTConfig:
    block_size: int
    max_blocks: int

    num_layers: int
    num_heads: int
    embed_dim: int

    attention: str

    mlp_size: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.block_size * self.max_blocks


class Block(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.num_heads, dropout=config.attn_pdrop)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.embed_dim, 4 * config.embed_dim),
            c_proj  = nn.Linear(4 * config.embed_dim, config.embed_dim),
            act     = nn.GELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class TransformerEncoder(nn.Module): 
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embeddings = nn.Sequential(
            nn.Linear(config.block_size, config.embed_dim),
            nn.Dropout(config.embed_pdrop)
        )
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x


class ViT(nn.Module):
    def __init__(self, config: ViTConfig, num_classes: int):
        super().__init__()
        self.patch_size = config.block_size
        self.num_patches = config.max_blocks

        self.patch_embedding = nn.Conv2d(3, config.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        # add single class token of 1x1xembed_dim (From BERT paper)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.embed_dim))

        self.transformer = TransformerEncoder(config)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.mlp_size),
            nn.Linear(config.mlp_size, num_classes),
            nn.GELU()
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding

        x = self.transformer(x)
        x = x[:, 0]  # Only take the output of the class token
        x = self.mlp_head(x)
        return x

if __name__ == "__main__": 
    # test for same number of params in paper
    config = ViTConfig(
        block_size=9,
        max_blocks=9,
        num_layers=12,
        num_heads=12,
        embed_dim=768,
        attention='scaled_dot_product',
        mlp_size=768,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1
    )

    num_classes = 10  # Example number of classes for classification
    model = ViT(config=config, num_classes=num_classes)

    # Calculate the number of parameters in the model
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {model_params}")
    # for ViT base: 
    # Model params: 85,859,338
    # paper params 86M

