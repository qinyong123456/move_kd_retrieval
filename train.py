import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import math
import torch.nn.functional as F
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

def train(root=None, epochs=1, batch_size=32, lr=1e-4, kd_w=0.5, moe_balance_w=0.01, device='cuda', images_dir=None, captions_txt=None, output_dir=None, save_every=0):
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
    total_steps = len(dl)
    for epoch in range(epochs):
        step_idx = 0
        epoch_loss_sum = 0.0
        epoch_kd_sum = 0.0
        epoch_align_sum = 0.0
        for imgs, captions in dl:
            imgs = imgs.to(device)
            img_emb, v_tokens, routings, stu_cls = student(imgs)
            tgt_len = v_tokens.shape[1]
            tgt_hw = (int(math.sqrt(tgt_len)), int(math.sqrt(tgt_len)))
            def resize_seq(tokens, src_hw, dst_hw):
                B, L, C = tokens.shape
                h, w = src_hw
                if h*w != L:
                    h = int(math.sqrt(L))
                    w = h
                x = tokens.transpose(1, 2).reshape(B, C, h, w)
                x = F.interpolate(x, size=dst_hw, mode='bilinear', align_corners=False)
                x = x.flatten(2).transpose(1, 2)
                return x
            with torch.no_grad():
                denorm = imgs.clone()
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=denorm.device).view(1,3,1,1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=denorm.device).view(1,3,1,1)
                denorm = denorm * std + mean
                denorm = denorm.clamp(0, 1)
                pil_batch = [Image.fromarray((denorm[i].permute(1,2,0).cpu().numpy()*255).astype('uint8')) for i in range(denorm.size(0))]
                clip_images = clip_t.processor(images=pil_batch, return_tensors='pt')['pixel_values'].to(device)
                clip_tokens, clip_cls, token_weight = clip_t(clip_images)
                eva_tokens, eva_cls, eva_hw = eva_t(imgs)
                conv_tokens, conv_cls, conv_hw = conv_t(imgs)
            clip_tokens_s = adapters('CLIP', clip_tokens.to(device))
            eva_tokens_s = adapters('EVA', eva_tokens.to(device))
            conv_tokens_s = adapters('ConvNeXt', conv_tokens.to(device))
            clip_len = clip_tokens_s.shape[1]
            if clip_len != tgt_len:
                src_hw = (int(math.sqrt(clip_len)), int(math.sqrt(clip_len)))
                clip_tokens_s = resize_seq(clip_tokens_s, src_hw, tgt_hw)
                token_weight = token_weight.to(device)
                tw_map = token_weight.reshape(token_weight.size(0), int(math.sqrt(token_weight.size(1))), int(math.sqrt(token_weight.size(1))))
                tw_map = F.interpolate(tw_map.unsqueeze(1), size=tgt_hw, mode='nearest').squeeze(1)
                token_weight = tw_map.flatten(1)
            eva_tokens_s = resize_seq(eva_tokens_s, eva_hw, tgt_hw)
            conv_tokens_s = resize_seq(conv_tokens_s, conv_hw, tgt_hw)
            kd_clip = weighted_mse(v_tokens, clip_tokens_s, token_weight)
            kd_eva = weighted_mse(v_tokens, eva_tokens_s)
            kd_conv = weighted_mse(v_tokens, conv_tokens_s)
            m_clip = (stu_cls.unsqueeze(1) @ clip_tokens_s.transpose(1, 2)).mean(dim=(1, 2))
            m_eva = (stu_cls.unsqueeze(1) @ eva_tokens_s.transpose(1, 2)).mean(dim=(1, 2))
            m_conv = (stu_cls.unsqueeze(1) @ conv_tokens_s.transpose(1, 2)).mean(dim=(1, 2))
            teacher_weight = torch.stack([m_clip, m_eva, m_conv], dim=1)
            tw = torch.softmax(teacher_weight, dim=1)
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
            step_idx += 1
            epoch_loss_sum += float(loss.detach().cpu())
            epoch_kd_sum += float(kd_loss.detach().cpu())
            epoch_align_sum += float(loss_align.detach().cpu())
            if hasattr(train, 'log_interval'):
                log_interval = train.log_interval
            else:
                log_interval = 10
            if step_idx % log_interval == 0:
                print(f"epoch {epoch+1}/{epochs} step {step_idx}/{total_steps} loss {epoch_loss_sum/step_idx:.4f} kd {epoch_kd_sum/step_idx:.4f} align {epoch_align_sum/step_idx:.4f}")
        print(f"epoch {epoch+1} done avg_loss {epoch_loss_sum/total_steps:.4f} avg_kd {epoch_kd_sum/total_steps:.4f} avg_align {epoch_align_sum/total_steps:.4f}")
        if output_dir and save_every and ((epoch + 1) % save_every == 0):
            os.makedirs(output_dir, exist_ok=True)
            torch.save(student.state_dict(), os.path.join(output_dir, f"student_epoch{epoch+1}.pt"))
            print(f"saved {os.path.join(output_dir, f'student_epoch{epoch+1}.pt')}\n")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(student.state_dict(), os.path.join(output_dir, "student_final.pt"))
        print(f"saved {os.path.join(output_dir, 'student_final.pt')}\n")
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
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--save_every', type=int, default=0)
    args = p.parse_args()
    train.log_interval = args.log_interval
    train(args.root, args.epochs, args.batch_size, args.lr, args.kd_w, args.moe_balance_w, args.device, args.images_dir, args.captions_txt, args.output_dir, args.save_every)