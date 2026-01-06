import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Type, Callable, List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dynamic_block import DynamicConv2d, DynamicAttention
from multi_scale_dynamic_conv import MultiScaleDynamicConv

# -----------------------------------------------------------------------------
# Abstraction Layer: Interfaces and Configuration
# -----------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """
    Configuration object for the network architecture.
    Encapsulates hyperparameters to avoid hardcoding and allow dependency injection.
    """
    layers: List[int]
    num_classes: int = 2
    zero_init_residual: bool = False
    groups: int = 1
    width_per_group: int = 64
    replace_stride_with_dilation: Optional[List[bool]] = None
    inplanes: int = 64
    fusion_reduction: int = 4
    multi_scale_reduction: int = 4

    def __post_init__(self):
        if self.replace_stride_with_dilation is None:
            self.replace_stride_with_dilation = [False, False, False]

class IModule(nn.Module, ABC):
    """Abstract base class for all modules in the architecture."""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class IFusionStrategy(IModule):
    """Interface for feature fusion strategies."""
    @abstractmethod
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        pass

# -----------------------------------------------------------------------------
# Component Implementations
# -----------------------------------------------------------------------------

class GatedFeatureFusion(IFusionStrategy):
    """
    Implements a gated mechanism to fuse two feature maps.
    Previously FeatureFusionModule.
    """
    def __init__(self, channels: int, reduction_ratio: int):
        super().__init__()
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels * 2, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, conv_feat: torch.Tensor, attn_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate features along channel dimension
        cat_feat = torch.cat([conv_feat, attn_feat], dim=1)
        
        # Compute attention weights
        weights = self.attention_gate(cat_feat)
        
        # Split weights for each branch
        split_idx = conv_feat.size(1)
        w_conv = weights[:, :split_idx]
        w_attn = weights[:, split_idx:]
        
        # Apply weights
        weighted_conv = conv_feat * w_conv
        weighted_attn = attn_feat * w_attn
        
        # Fuse weighted features
        fused = self.fusion_conv(torch.cat([weighted_conv, weighted_attn], dim=1))
        return fused

class ParallelExtractionBlock(IModule):
    """
    Encapsulates the parallel processing branches:
    1. Dynamic Convolution
    2. Dynamic Attention
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dynamic_conv_branch = nn.Sequential(
            DynamicConv2d(dim=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.dynamic_attn_branch = nn.Sequential(
            DynamicAttention(dim=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dynamic_conv_branch(x), self.dynamic_attn_branch(x)

class DyCAMBlock(IModule):
    """
    The main building block of the network.
    Orchestrates dimensionality reduction, parallel extraction, fusion, and expansion.
    Previously Res2DWBlock.
    """
    def __init__(self, 
                 config: NetworkConfig,
                 in_channels: int, 
                 mid_channels: int,
                 out_channels: int,
                 stride: int = 1):
        super().__init__()
        self.config = config
        
        # 1x1 Compression
        self.compression = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Stride adaptation
        self.stride_adapter = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if stride > 1 else nn.Identity()
        
        # Parallel Feature Extraction
        self.parallel_extractor = ParallelExtractionBlock(mid_channels)
        
        # Feature Fusion Strategy
        self.fusion_strategy = GatedFeatureFusion(mid_channels, config.fusion_reduction)
        
        # Multi-scale Context Aggregation
        self.context_aggregator = MultiScaleDynamicConv(
            dim=mid_channels,
            reduction_ratio=config.multi_scale_reduction
        )
        
        # 1x1 Expansion
        self.expansion = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual Connection (Shortcut)
        self.shortcut = self._build_shortcut(in_channels, out_channels, stride)
        
        self.final_activation = nn.ReLU(inplace=True)

    def _build_shortcut(self, in_c: int, out_c: int, stride: int) -> Optional[nn.Module]:
        if stride != 1 or in_c != out_c:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Compression
        out = self.compression(x)
        
        # Spatial Downsampling if needed
        out = self.stride_adapter(out)
        
        # Parallel Extraction
        conv_feat, attn_feat = self.parallel_extractor(out)
        
        # Fusion
        fused_feat = self.fusion_strategy(conv_feat, attn_feat)
        
        # Context Aggregation
        out = self.context_aggregator(fused_feat)
        
        # Expansion
        out = self.expansion(out)
        
        # Residual Connection
        if self.shortcut is not None:
            identity = self.shortcut(x)
            
        out += identity
        return self.final_activation(out)

# -----------------------------------------------------------------------------
# Network Architecture
# -----------------------------------------------------------------------------

class DyCAMNet(IModule):
    """
    Main Network Architecture.
    Constructed using a configuration object and composed of DyCAMBlocks.
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.inplanes = config.inplanes
        
        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stages
        self.stage1 = self._make_stage(64, config.layers[0])
        self.stage2 = self._make_stage(128, config.layers[1], stride=2)
        self.stage3 = self._make_stage(256, config.layers[2], stride=2)
        self.stage4 = self._make_stage(512, config.layers[3], stride=2)
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * 4, config.num_classes)
        
        self._initialize_weights()

    def _make_stage(self, channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        # First block in stage handles stride and channel expansion
        layers.append(DyCAMBlock(
            self.config,
            self.inplanes, 
            channels, 
            channels * 4, 
            stride=stride
        ))
        self.inplanes = channels * 4
        
        # Subsequent blocks
        for _ in range(1, num_blocks):
            layers.append(DyCAMBlock(
                self.config,
                self.inplanes, 
                channels, 
                channels * 4
            ))
            
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if self.config.zero_init_residual:
            for m in self.modules():
                if isinstance(m, DyCAMBlock):
                    nn.init.constant_(m.expansion[1].weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

def dycam_50(pretrained: bool = False, **kwargs: Any) -> DyCAMNet:
    """
    Factory function to construct a DyCAMNet-50 model.
    """
    config = NetworkConfig(
        layers=[3, 4, 6, 3],
        **kwargs
    )
    model = DyCAMNet(config)
    
    # Pretrained loading logic has been removed as per request.
    if pretrained:
        pass # Placeholder if future implementation is needed
        
    return model

if __name__ == '__main__':
    # Test stub
    model = dycam_50()
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 