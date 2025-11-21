import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_channel, out_channel, rank=32, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        self.lora_A = nn.Linear(in_channel, rank)
        self.lora_B = nn.Linear(rank, out_channel)
        self.lora_alpha = lora_alpha
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lora_B.weight, nonlinearity='relu')
        if self.lora_A.bias is not None:
            nn.init.zeros_(self.lora_A.bias)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

class MoE_LoRA_MLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, num_experts: int = 3, rank: int = 32, alpha: int = 1, dense_moe: bool = False):
        super().__init__()
        self.original = original_mlp
        self.num_experts = num_experts
        self.dense_moe = dense_moe
        in_features = original_mlp.fc1.in_features
        out_features = original_mlp.fc1.out_features
        self.moe_down = nn.ModuleList([LoRALayer(in_features, out_features, rank, 0.05, alpha) for _ in range(num_experts)])
        self.moe_up = nn.ModuleList([LoRALayer(out_features, in_features, rank, 0.05, alpha) for _ in range(num_experts)])
        self.original.fc1.weight.requires_grad = False
        self.original.fc2.weight.requires_grad = False
        self.router = nn.Linear(in_features, num_experts)

    def forward_dense(self, x, original_proj, routing, moe):
        o = original_proj(x)
        l = torch.stack([m(x) for m in moe], 2)
        l = (l * routing[:, :, :, None]).sum(2)
        return o + l

    def forward_sparse(self, x, original_proj, index, moe):
        o = original_proj(x)
        l = torch.zeros_like(o)
        for i in range(self.num_experts):
            id1, id2, _ = torch.where(index == i)
            if id1.numel() > 0:
                l[id1, id2] = moe[i](x[id1, id2])
        return o + l

    def forward(self, x):
        logits = self.router(x)
        routing = F.softmax(logits, dim=-1)
        index = routing.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        expert_choice = y_hard - routing.detach() + routing
        if self.dense_moe:
            down = self.forward_dense(x, self.original.fc1, routing, self.moe_down)
        else:
            down = self.forward_sparse(x, self.original.fc1, index, self.moe_down)
        if hasattr(self.original, 'act'):
            x = self.original.act(down)
        elif hasattr(self.original, 'activation'):
            x = self.original.activation(down)
        elif hasattr(self.original, 'act_fn'):
            x = self.original.act_fn(down)
        else:
            x = F.gelu(down)
        if hasattr(self.original, 'drop'):
            x = self.original.drop(x)
        if self.dense_moe:
            x = self.forward_dense(x, self.original.fc2, routing, self.moe_up)
        else:
            x = self.forward_sparse(x, self.original.fc2, index, self.moe_up)
        if hasattr(self.original, 'drop'):
            x = self.original.drop(x)
        return x, (routing, expert_choice)