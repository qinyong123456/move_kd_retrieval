import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb, txt_emb):
        logits = img_emb @ txt_emb.t() / self.temperature
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        loss_i = nn.CrossEntropyLoss()(logits, labels)
        loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
        return (loss_i + loss_t) / 2

def weighted_mse(student_tokens, teacher_tokens, token_weight=None):
    loss = (student_tokens - teacher_tokens) ** 2
    if token_weight is not None:
        loss = loss.mean(-1)
        loss = (loss * token_weight).sum(-1).mean()
    else:
        loss = loss.mean()
    return loss