import torch
import torch.nn as nn
import timm
from .moe_lora import MoE_LoRA_MLP

class StudentViTS(nn.Module):
    def __init__(self, num_experts=3, lora_rank=32, lora_alpha=1, dense_moe=False, embed_dim=512):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.hidden_size = self.vit.embed_dim
        for i, blk in enumerate(self.vit.blocks):
            blk.mlp = MoE_LoRA_MLP(blk.mlp, num_experts, lora_rank, lora_alpha, dense_moe)
        self.norm = self.vit.norm
        self.project = nn.Linear(self.hidden_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed[:, : x.size(1)]
        x = self.vit.pos_drop(x)
        routings = []
        for blk in self.vit.blocks:
            x_attn = blk.norm1(x)
            x_attn = blk.attn(x_attn)
            x = x + x_attn
            x_ffn = blk.norm2(x)
            mlp_out = blk.mlp(x_ffn)
            if isinstance(mlp_out, tuple):
                x_ffn, moe_routing = mlp_out
                routings.append(moe_routing)
            else:
                x_ffn = mlp_out
            x = x + x_ffn
        x = self.norm(x)
        cls = x[:, 0]
        img_emb = nn.functional.normalize(self.project(cls), dim=-1)
        tokens = x[:, 1:]
        return img_emb, tokens, routings