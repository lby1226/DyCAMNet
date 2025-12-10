import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicConv2d(nn.Module):
    def __init__(self, dim, kernel_size, reduction_ratio=4, num_groups=4, bias=True):
        super().__init__()
        
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

class MultiScaleDynamicConv(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        

        self.conv7x7 = DynamicConv2d(dim, kernel_size=7, reduction_ratio=reduction_ratio)
        self.conv5x5 = DynamicConv2d(dim, kernel_size=5, reduction_ratio=reduction_ratio)
        self.conv3x3 = DynamicConv2d(dim, kernel_size=3, reduction_ratio=reduction_ratio)
        self.conv1x1 = DynamicConv2d(dim, kernel_size=1, reduction_ratio=reduction_ratio)
        

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )
        

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim)
        )
        
        self.gelu = nn.GELU()
        
    def forward(self, x):

        identity = self.residual_conv(x)
        

        feat7x7 = self.conv7x7(x)
        feat5x5 = self.conv5x5(x)
        feat3x3 = self.conv3x3(x)
        feat1x1 = self.conv1x1(x)
        

        multi_scale_feats = torch.cat([feat7x7, feat5x5, feat3x3, feat1x1], dim=1)


        out = self.fusion(multi_scale_feats)

        out = out + identity
        out = self.gelu(out)
        
        return out

if __name__ == '__main__':
    model = MultiScaleDynamicConv(dim=64)
    x = torch.randn(1, 64, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 