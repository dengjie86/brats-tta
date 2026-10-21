"""3D U-Net model for BraTS segmentation. Supports variable input sizes with sliding window inference."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """3D 卷积块"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for BraTS 分割（支持可变输入尺寸）
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: tuple = (32, 64, 128, 256),
        dropout: float = 0.3
    ):
        super(UNet3D, self).__init__()
        
        self.features = features
        
        # ========== 编码器 ==========
        self.enc1 = ConvBlock(in_channels, features[0], dropout=dropout)
        self.enc2 = ConvBlock(features[0], features[1], dropout=dropout)
        self.enc3 = ConvBlock(features[1], features[2], dropout=dropout)
        self.enc4 = ConvBlock(features[2], features[3], dropout=dropout)
        
        # 下采样
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # ========== 瓶颈 ==========
        self.bottleneck = ConvBlock(features[3], features[3] * 2, dropout=dropout)
        
        # ========== 解码器 ==========
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(features[3] * 2, features[3], kernel_size=1)
        )
        self.dec4 = ConvBlock(features[3] * 2, features[3], dropout=dropout)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(features[3], features[2], kernel_size=1)
        )
        self.dec3 = ConvBlock(features[2] * 2, features[2], dropout=dropout)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(features[2], features[1], kernel_size=1)
        )
        self.dec2 = ConvBlock(features[1] * 2, features[1], dropout=dropout)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(features[1], features[0], kernel_size=1)
        )
        self.dec1 = ConvBlock(features[0] * 2, features[0], dropout=dropout)
        
        # ========== 输出层 ==========
        self.conv_final = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """前向传播（确保输出与输入尺寸一致）

        Returns:
            logits [B, num_classes, D, H, W]
        """
        logits, _ = self.forward_feat(x)
        return logits

    def forward_feat(self, x):
        """前向传播，同时返回 bottleneck 特征（用于 FADA 域判别器）。

        Returns:
            logits : [B, num_classes, D, H, W]
            feat   : [B, features[-1]*2, D/16, H/16, W/16] bottleneck 特征
        """
        # 保存原始输入尺寸
        input_size = x.shape[2:]  # [D, H, W]
        
        # ========== 编码器 ==========
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # ========== 瓶颈 ==========
        b = self.bottleneck(self.pool(e4))
        
        # ========== 解码器 ==========
        d4 = self.up4(b)
        e4_cropped = self._crop_to_match(d4, e4)
        d4 = torch.cat([d4, e4_cropped], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_cropped = self._crop_to_match(d3, e3)
        d3 = torch.cat([d3, e3_cropped], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_cropped = self._crop_to_match(d2, e2)
        d2 = torch.cat([d2, e2_cropped], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_cropped = self._crop_to_match(d1, e1)
        d1 = torch.cat([d1, e1_cropped], dim=1)
        d1 = self.dec1(d1)
        
        # ========== 输出 ==========
        out = self.conv_final(d1)
        
        # 关键修复：插值输出以匹配原始输入尺寸
        out = F.interpolate(out, size=input_size, mode='trilinear', align_corners=False)
        
        return out, b
    
    def _crop_to_match(self, target, reference):
        """
        裁剪 reference 以匹配 target 的空间尺寸
        
        Args:
            target: 上采样后的特征图（尺寸较小）
            reference: 编码器特征图（尺寸较大）
        
        Returns:
            裁剪后的 reference
        """
        _, _, td, th, tw = target.shape
        _, _, rd, rh, rw = reference.shape
        
        # 计算需要裁剪的量
        d_diff = rd - td
        h_diff = rh - th
        w_diff = rw - tw
        
        # 从中心裁剪
        d_start = d_diff // 2
        h_start = h_diff // 2
        w_start = w_diff // 2
        
        return reference[
            :, :,
            d_start:d_start + td,
            h_start:h_start + th,
            w_start:w_start + tw
        ]
