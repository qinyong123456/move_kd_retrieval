import torch
import torch.nn as nn

class TeacherAdapters(nn.Module):
    def __init__(self, student_hidden):
        super().__init__()
        self.student_hidden = student_hidden
        self.adapters = nn.ModuleDict()

    def add_adapter(self, name, in_dim, mid_dim=None):
        if name in self.adapters:
            return
        mid = mid_dim or self.student_hidden
        self.adapters[name] = nn.Sequential(nn.Linear(in_dim, mid), nn.GELU(), nn.Linear(mid, self.student_hidden))

    def forward(self, name, x):
        return self.adapters[name](x)