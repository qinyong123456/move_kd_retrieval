import torch
from torch.utils.data import DataLoader
try:
    from move_kd_retrieval.data.flickr30k import Flickr30kDataset
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data.flickr30k import Flickr30kDataset
from transformers import CLIPTokenizer, CLIPTextModel

def encode_texts(tokenizer, model, texts):
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
    out = model(**{k: v.to(model.device) for k, v in inputs.items()})
    emb = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(emb, dim=-1)

def evaluate(student, root=None, batch_size=64, device='cuda', images_dir=None, captions_txt=None):
    ds = Flickr30kDataset(root=root, split='val', images_dir=images_dir, captions_txt=captions_txt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
    imgs = []
    img_embs = []
    caps = []
    for img, cap in dl:
        img = img.to(device)
        with torch.no_grad():
            emb, _, _ = student(img)
        img_embs.append(emb.cpu())
        caps.extend(list(cap))
    img_embs = torch.cat(img_embs, dim=0)
    txt_embs = []
    for i in range(0, len(caps), batch_size):
        batch_caps = caps[i:i+batch_size]
        with torch.no_grad():
            txt_embs.append(encode_texts(tokenizer, text_model, batch_caps).cpu())
    txt_embs = torch.cat(txt_embs, dim=0)
    sims = img_embs @ txt_embs.t()
    ranks = torch.argsort(sims, dim=1, descending=True)
    gt = torch.arange(len(img_embs)).unsqueeze(1)
    r1 = (ranks[:, :1] == gt).any(dim=1).float().mean().item()
    r5 = (ranks[:, :5] == gt).any(dim=1).float().mean().item()
    r10 = (ranks[:, :10] == gt).any(dim=1).float().mean().item()
    return {"R@1": r1, "R@5": r5, "R@10": r10}