import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class DynamicAttention(nn.Module):  
    def __init__(self, dim):
        super().__init__()
        config = Config.get_dynamic_attn_config()
        num_heads = config['num_heads']
        qk_scale = config['qk_scale']
        attn_drop = config['attn_drop']
        sr_ratio = config['sr_ratio']
        
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio+3,
                         stride=sr_ratio, padding=(sr_ratio+3)//2,
                         groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1,
                         groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C//self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:], 
                                               mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)

class DynamicConv2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        config = Config.get_dynamic_conv_config()
        kernel_size = config['kernel_size']
        reduction_ratio = config['reduction_ratio']
        num_groups = config['num_groups']
        bias = config['bias']
        
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.dim = dim
        self.bias_type = bias
        
        self.weight = nn.Parameter(torch.empty(num_groups, dim, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim//reduction_ratio, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim//reduction_ratio),
            nn.GELU(),
            nn.Conv2d(dim//reduction_ratio, num_groups, kernel_size=1)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        
        scale = self.proj(self.pool(x))
        scale = scale.view(B, self.num_groups, 1, 1, 1, 1)
        scale = torch.softmax(scale, dim=1)
        
        weight = self.weight.unsqueeze(0)
        weight = (weight * scale).sum(dim=1)
        
        x = x.view(B, C, H, W)
        out = F.conv2d(x, weight[0], 
                      bias=self.bias if self.bias is not None else None,
                      padding=self.K//2)
        
        return out