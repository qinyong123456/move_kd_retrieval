import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
import timm

class CLIPViTBaseTeacher(nn.Module):
    def __init__(self, name='openai/clip-vit-base-patch16'):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(name)
        self.processor = CLIPImageProcessor.from_pretrained(name)
        self.hidden_size = self.vision.config.hidden_size

    def forward(self, images):
        out = self.vision(images, output_hidden_states=True, output_attentions=True)
        hs = out.hidden_states[-2]
        attn = out.attentions[-2].mean(1)
        tokens = hs[:, 1:]
        cls = hs[:, 0]
        token_weight = torch.softmax(attn[:, 0, 1:], dim=-1)
        return tokens, cls, token_weight

class EVA02Teacher(nn.Module):
    def __init__(self, name='eva02_large_patch14_448'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        self.hidden_size = self.model.embed_dim if hasattr(self.model, 'embed_dim') else self.model.num_features

    def forward(self, images):
        x = self.model.patch_embed(images)
        cls_token = self.model.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed[:, : x.size(1)]
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = x + blk.attn(blk.norm1(x))
            x = x + blk.mlp(blk.norm2(x))
        x = self.model.norm(x)
        cls = x[:, 0]
        tokens = x[:, 1:]
        return tokens, cls

class ConvNeXtTeacher(nn.Module):
    def __init__(self, name='convnext_large.fb_in1k'): 
        super().__init__()
        self.model = timm.create_model(name, pretrained=True, features_only=True)
        self.hidden_size = self.model.feature_info[-1]['num_chs']
        self.proj = nn.Identity()

    def forward(self, images):
        feats = self.model(images)[-1]
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)
        cls = tokens.mean(dim=1)
        return tokens, cls