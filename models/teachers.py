import torch
import torch.nn as nn
import torch.nn.functional as F
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
        tokens = hs[:, 1:].to(images.device)
        cls = hs[:, 0].to(images.device)
        token_weight = torch.softmax(attn[:, 0, 1:], dim=-1).to(images.device)
        return tokens, cls, token_weight

class EVA02Teacher(nn.Module):
    def __init__(self, name='eva02_large_patch14_448'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        self.hidden_size = self.model.embed_dim if hasattr(self.model, 'embed_dim') else self.model.num_features
        img_size = getattr(self.model.patch_embed, 'img_size', 448)
        if isinstance(img_size, (tuple, list)):
            self.expected_hw = (img_size[0], img_size[1])
        else:
            self.expected_hw = (img_size, img_size)

    def forward(self, images):
        if images.shape[-2:] != self.expected_hw:
            images = F.interpolate(images, size=self.expected_hw, mode='bilinear', align_corners=False)
        x = self.model.patch_embed(images)
        cls_token = self.model.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed[:, : x.size(1)]
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = x + blk.attn(blk.norm1(x))
            x = x + blk.mlp(blk.norm2(x))
        x = self.model.norm(x)
        cls = x[:, 0].to(images.device)
        tokens = x[:, 1:].to(images.device)
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
        tokens = feats.flatten(2).transpose(1, 2).to(images.device)
        cls = tokens.mean(dim=1).to(images.device)
        return tokens, cls