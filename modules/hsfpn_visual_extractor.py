import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .scsa import SCSA


class FrequencyAttentionModule(nn.Module):
    """
    从输入图像生成多尺度的频域注意力图。
    使用简化的架构，直接从频域特征生成注意力图。
    """
    def __init__(self, in_channels=3, freq_ratio=0.25):
        super().__init__()
        self.freq_ratio = freq_ratio
        
        # 处理拼接后的高低频特征 (6 channels -> 64 channels)
        self.feature_processor = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 为不同尺度生成注意力图 (统一输入64通道，输出1通道)
        self.attn_c2 = self._make_attn_layer(64)
        self.attn_c3 = self._make_attn_layer(64)
        self.attn_c4 = self._make_attn_layer(64)
        self.attn_c5 = self._make_attn_layer(64)
    
    def _make_attn_layer(self, in_dim):
        """生成单通道注意力图"""
        return nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        _, _, H, W = x.shape
        
        # 1. FFT变换到频域
        x_freq = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 创建高通和低通滤波器
        # 低频在四个角落，高频在中心
        mask = torch.zeros((H, W // 2 + 1), device=x.device)
        h_cutoff = int(H * self.freq_ratio)
        w_cutoff = int((W // 2 + 1) * self.freq_ratio)
        
        # 低频mask（四个角）
        mask[:h_cutoff, :w_cutoff] = 1.0
        mask[-h_cutoff:, :w_cutoff] = 1.0
        
        low_pass_mask = mask
        high_pass_mask = 1.0 - mask
        
        # 3. 应用滤波器并逆FFT回空间域
        low_freq_comp = torch.fft.irfft2(x_freq * low_pass_mask, s=(H, W), norm='ortho')
        high_freq_comp = torch.fft.irfft2(x_freq * high_pass_mask, s=(H, W), norm='ortho')
        
        # 4. 拼接高低频特征 (3+3=6 channels)
        freq_features = torch.cat([low_freq_comp, high_freq_comp], dim=1)
        
        # 5. 提取统一的特征表示 (6 -> 64 channels)
        processed_features = self.feature_processor(freq_features)
        
        # 6. 为不同ResNet层生成对应尺度的注意力图
        # C2: stride=4  (H/4, W/4)
        # C3: stride=8  (H/8, W/8)
        # C4: stride=16 (H/16, W/16)
        # C5: stride=32 (H/32, W/32)
        
        attn_c2 = self.attn_c2(F.avg_pool2d(processed_features, 4))  # -> (H/4, W/4)
        attn_c3 = self.attn_c3(F.avg_pool2d(processed_features, 8))  # -> (H/8, W/8)
        attn_c4 = self.attn_c4(F.avg_pool2d(processed_features, 16)) # -> (H/16, W/16)
        attn_c5 = self.attn_c5(F.avg_pool2d(processed_features, 32)) # -> (H/32, W/32)
        
        return {
            'c2': attn_c2,
            'c3': attn_c3,
            'c4': attn_c4,
            'c5': attn_c5
        }


class DctSpatialInteraction(nn.Module):
    """
    Spatial Path of HFP (High Frequency Perception Module)
    使用FFT替代DCT进行高频特征提取
    """
    def __init__(self, in_channels, ratio, isdct=True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
                nn.GroupNorm(1, 1)
            )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        
        # 使用FFT替代DCT
        x_freq = torch.fft.rfft2(x, norm='ortho')
        
        # 生成高通滤波器mask
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        # FFT的频率布局：低频在四个角，需要调整mask
        weight_fft = self._convert_to_fft_mask(weight, h0, w0)
        weight_fft = weight_fft.unsqueeze(0).unsqueeze(0).expand_as(x_freq.real)
        
        # 应用高通滤波
        x_freq_filtered = x_freq * weight_fft
        
        # 逆FFT
        dct_ = torch.fft.irfft2(x_freq_filtered, s=(h0, w0), norm='ortho')
        
        return x * torch.sigmoid(dct_)

    def _compute_weight(self, h, w, ratio):
        """生成DCT风格的高通滤波器（左上角为低频）"""
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0  # 过滤左上角的低频
        return weight
    
    def _convert_to_fft_mask(self, dct_mask, h, w):
        """将DCT风格的mask转换为FFT风格（低频在中心）"""
        # FFT的频率分布：低频在四个角
        # 我们需要将DCT的左上角低频转换为FFT的四角低频
        fft_mask = torch.ones((h, w // 2 + 1), device=dct_mask.device)
        
        # 简化处理：直接使用ratio过滤低频区域
        h0 = int(h * self.ratio[0])
        w0 = int((w // 2 + 1) * self.ratio[1])
        
        # 过滤左上角和左下角的低频
        fft_mask[:h0, :w0] = 0
        fft_mask[-h0:, :w0] = 0
        
        return fft_mask


class DctChannelInteraction(nn.Module):
    """
    Channel Path of HFP (High Frequency Perception Module)
    """
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        
        # 确保groups能整除in_channels
        groups = min(32, in_channels)
        while in_channels % groups != 0:
            groups -= 1
        
        self.channel1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False),
            nn.GroupNorm(groups, in_channels)
        )
        self.channel2x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False),
            nn.GroupNorm(groups, in_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        
        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        # 使用FFT进行高频提取
        x_freq = torch.fft.rfft2(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight_fft = self._convert_to_fft_mask(weight, h, w)
        weight_fft = weight_fft.unsqueeze(0).unsqueeze(0).expand_as(x_freq.real)
        
        x_freq_filtered = x_freq * weight_fft
        dct_ = torch.fft.irfft2(x_freq_filtered, s=(h, w), norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))
        
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight
    
    def _convert_to_fft_mask(self, dct_mask, h, w):
        fft_mask = torch.ones((h, w // 2 + 1), device=dct_mask.device)
        h0 = int(h * self.ratio[0])
        w0 = int((w // 2 + 1) * self.ratio[1])
        fft_mask[:h0, :w0] = 0
        fft_mask[-h0:, :w0] = 0
        return fft_mask


class HFP(nn.Module):
    """
    High Frequency Perception Module
    结合空间路径和通道路径提取高频特征
    """
    def __init__(self, in_channels, ratio, patch=(8, 8), isdct=True):
        super(HFP, self).__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        
        groups = min(32, in_channels)
        while in_channels % groups != 0:
            groups -= 1
            
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, in_channels)
        )
        
    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)


# ============================================================================
# 原始 SDP 模块（已弃用，保留用于参考）
# 使用交叉注意力机制融合不同尺度的特征
# 缺点：需要两个输入，计算复杂度高 O(N²)
# ============================================================================
# class SDP(nn.Module):
#     """
#     Spatial Dependency Perception Module
#     使用交叉注意力机制融合不同尺度的特征
#     """
#     def __init__(self, dim=256, inter_dim=None):
#         super(SDP, self).__init__()
#         self.inter_dim = inter_dim if inter_dim is not None else dim
#         self.in_dim = dim
#         
#         groups = min(32, self.inter_dim)
#         while self.inter_dim % groups != 0:
#             groups -= 1
#         
#         self.conv_q = nn.Sequential(
#             nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
#             nn.GroupNorm(groups, self.inter_dim)
#         )
#         self.conv_k = nn.Sequential(
#             nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
#             nn.GroupNorm(groups, self.inter_dim)
#         )
#         self.conv_out = nn.Conv2d(self.inter_dim, dim, 1, padding=0, bias=False)
#         self.softmax = nn.Softmax(dim=-1)
#         
#     def forward(self, x_low, x_high, patch_size):
#         """
#         x_low: 低层特征（高分辨率）
#         x_high: 高层特征（低分辨率，已上采样到与x_low相同尺寸）
#         patch_size: 用于重排的patch大小
#         """
#         _, _, h_, w_ = x_low.size()
#         
#         # 确保尺寸能被patch_size整除
#         if h_ % patch_size[0] != 0 or w_ % patch_size[1] != 0:
#             # 如果不能整除，使用自适应池化调整尺寸
#             new_h = (h_ // patch_size[0]) * patch_size[0]
#             new_w = (w_ // patch_size[1]) * patch_size[1]
#             x_low = F.adaptive_avg_pool2d(x_low, (new_h, new_w))
#             x_high = F.adaptive_avg_pool2d(x_high, (new_h, new_w))
#             h_, w_ = new_h, new_w
#         
#         # 生成query和key
#         q = self.conv_q(x_low)  # (b, inter_dim, h, w)
#         k = self.conv_k(x_high)  # (b, inter_dim, h, w)
#         
#         # 重排为patch
#         q = self._rearrange_to_patches(q, patch_size)  # (b*num_patches, patch_h*patch_w, inter_dim)
#         k = self._rearrange_to_patches(k, patch_size)  # (b*num_patches, patch_h*patch_w, inter_dim)
#         
#         # 计算注意力
#         # q: (b*num_patches, patch_h*patch_w, inter_dim)
#         # k: (b*num_patches, patch_h*patch_w, inter_dim)
#         attn = torch.matmul(q, k.transpose(1, 2))  # (b*num_patches, patch_h*patch_w, patch_h*patch_w)
#         attn = attn / np.power(self.inter_dim, 0.5)
#         attn = self.softmax(attn)
#         
#         # 应用注意力
#         v = k  # (b*num_patches, patch_h*patch_w, inter_dim)
#         output = torch.matmul(attn, v)  # (b*num_patches, patch_h*patch_w, inter_dim)
#         
#         # 重排回原始形状
#         output = self._rearrange_from_patches(output.transpose(1, 2), patch_size, h_, w_)
#         
#         # 投影回原始通道维度
#         output = self.conv_out(output)
#         
#         return output + x_low
#     
#     def _rearrange_to_patches(self, x, patch_size):
#         """将特征图重排为patches"""
#         b, c, h, w = x.size()
#         p1, p2 = patch_size
#         
#         # (b, c, h, w) -> (b, c, h//p1, p1, w//p2, p2)
#         x = x.view(b, c, h // p1, p1, w // p2, p2)
#         # -> (b, h//p1, w//p2, c, p1, p2)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
#         # -> (b*h//p1*w//p2, c, p1*p2)
#         x = x.view(b * (h // p1) * (w // p2), c, p1 * p2)
#         # -> (b*num_patches, p1*p2, c)
#         x = x.transpose(1, 2)
#         return x
#     
#     def _rearrange_from_patches(self, x, patch_size, h, w):
#         """将patches重排回特征图"""
#         p1, p2 = patch_size
#         num_patches_h = h // p1
#         num_patches_w = w // p2
#         
#         # x: (b*num_patches, c, p1*p2)
#         b_times_patches, c, _ = x.size()
#         b = b_times_patches // (num_patches_h * num_patches_w)
#         
#         # -> (b, num_patches_h, num_patches_w, c, p1, p2)
#         x = x.view(b, num_patches_h, num_patches_w, c, p1, p2)
#         # -> (b, c, num_patches_h, p1, num_patches_w, p2)
#         x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
#         # -> (b, c, h, w)
#         x = x.view(b, c, h, w)
#         return x


class HSFPN_VisualExtractor(nn.Module):
    """
    HS-FPN增强的视觉特征提取器
    集成HFP和SDP模块，专注于微小病灶检测
    """
    def __init__(self, args):
        super(HSFPN_VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        
        # 获取HS-FPN相关参数
        self.use_hsfpn = getattr(args, 'use_hsfpn', True)
        self.hsfpn_ratio = getattr(args, 'hsfpn_ratio', (0.2, 0.2))
        self.hsfpn_output_layer = getattr(args, 'hsfpn_output_layer', 'P3')  # P2或P3
        
        # 加载预训练的ResNet
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        
        # 提取多尺度特征
        # ResNet结构: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        
        self.layer1 = model.layer1  # stride=4,  channels=256  (C2)
        self.layer2 = model.layer2  # stride=8,  channels=512  (C3)
        self.layer3 = model.layer3  # stride=16, channels=1024 (C4)
        self.layer4 = model.layer4  # stride=32, channels=2048 (C5)
        
        if not self.use_hsfpn:
            # 如果不使用HS-FPN，保持原始行为
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        else:
            # 频域注意力模块
            self.freq_attention = FrequencyAttentionModule(in_channels=3, freq_ratio=0.25)
            
            # 1x1卷积降维到256
            self.lateral_c2 = nn.Conv2d(256, 256, kernel_size=1)
            self.lateral_c3 = nn.Conv2d(512, 256, kernel_size=1)
            self.lateral_c4 = nn.Conv2d(1024, 256, kernel_size=1)
            self.lateral_c5 = nn.Conv2d(2048, 256, kernel_size=1)
            
            # HFP模块
            self.hfp_p5 = HFP(256, ratio=None, isdct=False)
            self.hfp_p4 = HFP(256, ratio=None, isdct=False)
            self.hfp_p3 = HFP(256, ratio=self.hsfpn_ratio, patch=(8, 8), isdct=True)
            self.hfp_p2 = HFP(256, ratio=self.hsfpn_ratio, patch=(16, 16), isdct=True)
            
            # ============ 修改点：使用SCSA替换SDP ============
            # 原始: SDP需要两个输入 (x_low, x_high, patch_size)
            # 新方案: SCSA只需单输入，先融合再自注意力
            # 优势: 更简洁，计算量更低，参数量更少
            self.scsa_p3 = SCSA(
                dim=256,
                head_num=8,
                window_size=7,
                group_kernel_sizes=[3, 5, 7, 9],
                qkv_bias=False,
                attn_drop_ratio=0.0,
                gate_layer='sigmoid'
            )
            self.scsa_p2 = SCSA(
                dim=256,
                head_num=8,
                window_size=7,  # P2分辨率更高，可以考虑用14
                group_kernel_sizes=[3, 5, 7, 9],
                qkv_bias=False,
                attn_drop_ratio=0.0,
                gate_layer='sigmoid'
            )
            
            # 输出投影层：256 -> 2048
            self.output_proj = nn.Sequential(
                nn.Conv2d(256, 2048, kernel_size=1),
                nn.GroupNorm(32, 2048),
                nn.ReLU()
            )
            
            # 根据输出层选择合适的池化
            if self.hsfpn_output_layer == 'P2':
                # P2输出尺寸约为56x56（对于224输入）或128x128（对于512输入）
                # 需要池化到合理的patch数量
                self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # 输出196个patches
            else:  # P3
                # P3输出尺寸约为28x28（对于224输入）或64x64（对于512输入）
                self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 输出49个patches，与原始一致
            
            self.avg_fnt = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        if not self.use_hsfpn:
            # 原始行为：只使用C5
            x = self.conv1(images)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)  # C5: (B, 2048, 7, 7)
            
            patch_feats = x
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            return patch_feats, avg_feats
        
        # HS-FPN增强的特征提取
        # 1. 生成频域注意力图
        freq_attns = self.freq_attention(images)
        
        # 2. 提取多尺度特征
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)   # (B, 256, 56, 56) for 224 input
        c3 = self.layer2(c2)  # (B, 512, 28, 28)
        c4 = self.layer3(c3)  # (B, 1024, 14, 14)
        c5 = self.layer4(c4)  # (B, 2048, 7, 7)
        
        # 3. 应用频域注意力增强
        # 调整注意力图尺寸以匹配特征图
        attn_c2 = F.interpolate(freq_attns['c2'], size=c2.shape[2:], mode='bilinear', align_corners=False)
        attn_c3 = F.interpolate(freq_attns['c3'], size=c3.shape[2:], mode='bilinear', align_corners=False)
        attn_c4 = F.interpolate(freq_attns['c4'], size=c4.shape[2:], mode='bilinear', align_corners=False)
        attn_c5 = F.interpolate(freq_attns['c5'], size=c5.shape[2:], mode='bilinear', align_corners=False)
        
        # 逐元素相乘进行特征增强
        c2 = c2 * (1 + attn_c2)  # 使用 1 + attn 保证基础特征不被完全抑制
        c3 = c3 * (1 + attn_c3)
        c4 = c4 * (1 + attn_c4)
        c5 = c5 * (1 + attn_c5)
        
        # 4. 降维到256
        p5 = self.lateral_c5(c5)  # (B, 256, 7, 7)
        p4 = self.lateral_c4(c4)  # (B, 256, 14, 14)
        p3 = self.lateral_c3(c3)  # (B, 256, 28, 28)
        p2 = self.lateral_c2(c2)  # (B, 256, 56, 56)
        
        # 应用HFP
        p5 = self.hfp_p5(p5)
        p4 = self.hfp_p4(p4)
        p3 = self.hfp_p3(p3)
        p2 = self.hfp_p2(p2)
        
        # ============ 修改点：自顶向下融合流程 ============
        # P5 -> P4: 简单上采样+相加（保持不变）
        _, _, h4, w4 = p4.size()
        p4_up = F.interpolate(p5, size=(h4, w4), mode='nearest')
        p4 = p4 + p4_up
        
        # P4 -> P3: 使用SCSA替换SDP
        # 原始: p3 = self.sdp_p3(p3, p3_up, patch_size_p3)
        # 新方案: 先融合，再用SCSA增强
        _, _, h3, w3 = p3.size()
        p3_up = F.interpolate(p4, size=(h3, w3), mode='nearest')
        p3 = p3 + p3_up  # 先相加融合
        p3 = self.scsa_p3(p3)  # 再用SCSA自注意力增强
        
        # P3 -> P2: 使用SCSA替换SDP
        # 原始: p2 = self.sdp_p2(p2, p2_up, patch_size_p2)
        # 新方案: 先融合，再用SCSA增强
        _, _, h2, w2 = p2.size()
        p2_up = F.interpolate(p3, size=(h2, w2), mode='nearest')
        p2 = p2 + p2_up  # 先相加融合
        p2 = self.scsa_p2(p2)  # 再用SCSA自注意力增强
        
        # 选择输出层
        if self.hsfpn_output_layer == 'P2':
            output = p2
        else:  # P3
            output = p3
        
        # 池化到固定尺寸
        output = self.adaptive_pool(output)
        
        # 投影到2048维
        output = self.output_proj(output)  # (B, 2048, H, W)
        
        # 生成patch_feats和avg_feats
        avg_feats = self.avg_fnt(output).squeeze().reshape(-1, output.size(1))  # (B, 2048)
        batch_size, feat_size, _, _ = output.shape
        patch_feats = output.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # (B, num_patches, 2048)
        
        return patch_feats, avg_feats

