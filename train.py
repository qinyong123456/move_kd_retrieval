import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
try:
    from move_kd_retrieval.models.student_vit import StudentViTS
    from move_kd_retrieval.models.teachers import CLIPViTBaseTeacher, EVA02Teacher, ConvNeXtTeacher
    from move_kd_retrieval.models.adapters import TeacherAdapters
    from move_kd_retrieval.losses import ContrastiveLoss, weighted_mse
    from move_kd_retrieval.data.flickr30k import Flickr30kDataset
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.student_vit import StudentViTS
    from models.teachers import CLIPViTBaseTeacher, EVA02Teacher, ConvNeXtTeacher
    from models.adapters import TeacherAdapters
    from losses import ContrastiveLoss, weighted_mse
    from data.flickr30k import Flickr30kDataset

def build_text_encoder(name='openai/clip-vit-base-patch16'):
    tok = CLIPTokenizer.from_pretrained(name)
    txt = CLIPTextModel.from_pretrained(name)
    return tok, txt

def encode_text(tokenizer, text_model, texts):
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
    out = text_model(**{k: v.to(text_model.device) for k, v in inputs.items()})
    emb = out.last_hidden_state[:, 0]
    emb = nn.functional.normalize(emb, dim=-1)
    return emb

def train(root=None, epochs=1, batch_size=32, lr=1e-4, kd_w=0.5, moe_balance_w=0.01, device='cuda', images_dir=None, captions_txt=None):
    ds = Flickr30kDataset(root=root, split='train', images_dir=images_dir, captions_txt=captions_txt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    student = StudentViTS(num_experts=3, lora_rank=32, lora_alpha=1, dense_moe=False, embed_dim=512).to(device)
    clip_t = CLIPViTBaseTeacher().to(device)
    eva_t = EVA02Teacher().to(device)
    conv_t = ConvNeXtTeacher().to(device)
    adapters = TeacherAdapters(student_hidden=student.hidden_size).to(device)
    adapters.add_adapter('CLIP', clip_t.hidden_size)
    adapters.add_adapter('EVA', eva_t.hidden_size)
    adapters.add_adapter('ConvNeXt', conv_t.hidden_size)
    tokenizer, text_model = build_text_encoder()
    text_model = text_model.to(device)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad] + list(adapters.parameters()), lr=lr)
    ctr = ContrastiveLoss()
    for epoch in range(epochs):
        for imgs, captions in dl:
            imgs = imgs.to(device)
            img_emb, v_tokens, routings = student(imgs)
            with torch.no_grad():
                clip_images = clip_t.processor(images=[(imgs[i].mul(0.5).add(0.5).clamp(0,1).permute(1,2,0).cpu().numpy()) for i in range(imgs.size(0))], return_tensors='pt')['pixel_values'].to(device)
                clip_tokens, clip_cls, token_weight = clip_t(clip_images)
                eva_tokens, eva_cls = eva_t(imgs)
                conv_tokens, conv_cls = conv_t(imgs)
            clip_tokens_s = adapters('CLIP', clip_tokens)
            eva_tokens_s = adapters('EVA', eva_tokens)
            conv_tokens_s = adapters('ConvNeXt', conv_tokens)
            kd_clip = weighted_mse(v_tokens, clip_tokens_s, token_weight)
            kd_eva = weighted_mse(v_tokens, eva_tokens_s)
            kd_conv = weighted_mse(v_tokens, conv_tokens_s)
            teacher_weight_cls = torch.stack([
                (img_emb @ nn.functional.normalize(clip_cls, dim=-1).t()).diag(),
                (img_emb @ nn.functional.normalize(eva_cls, dim=-1).t()).diag(),
                (img_emb @ nn.functional.normalize(conv_cls, dim=-1).t()).diag(),
            ], dim=1)
            tw = torch.softmax(teacher_weight_cls, dim=1)
            kd_loss = (tw[:,0]*kd_clip + tw[:,1]*kd_eva + tw[:,2]*kd_conv).mean()
            txt_emb = encode_text(tokenizer, text_model, captions)
            loss_align = ctr(img_emb, txt_emb)
            balance = 0.0
            if len(routings) > 0:
                probs = torch.stack([r[0] for r in routings], dim=0)
                idxes = torch.stack([r[1] for r in routings], dim=0).detach()
                bsz = probs.shape[1]
                bal = 0.0
                for i in range(bsz):
                    p_i = probs[:, i, :].reshape(-1, probs.size(-1))
                    x_i = idxes[:, i, :].reshape(-1, idxes.size(-1))
                    bal = bal + (p_i.mean(0) * x_i.mean(0)).sum()
                balance = bal / bsz
            loss = loss_align + kd_w * kd_loss + moe_balance_w * balance
            opt.zero_grad()
            loss.backward()
            opt.step()
    return student

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default=None)
    p.add_argument('--images_dir', type=str, default=None)
    p.add_argument('--captions_txt', type=str, default=None)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--kd_w', type=float, default=0.5)
    p.add_argument('--moe_balance_w', type=float, default=0.01)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()
    train(args.root, args.epochs, args.batch_size, args.lr, args.kd_w, args.moe_balance_w, args.device, args.images_dir, args.captions_txt)