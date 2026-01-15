from collections import OrderedDict
from distutils.fancy_getopt import FancyGetopt
from re import M
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import csv
from . import my_globals
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


layer_scale = False
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, D, H, W )
        return x        


class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(head_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(middle_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class VariablePatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding for variable input channels
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=4, embed_dim=768, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size
        else:
            stride = stride

        self.proj_1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv3d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.conv_1 = nn.Conv3d(64*4, 64, kernel_size=1, stride=1)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        #### first channel
        x_chan_0 = x[:,0:1,:,:,:]
        conv_x_sum = self.proj_1(x_chan_0)
        
        for chan in range(1,C):
            x_chan = x[:,chan:chan+1,:,:,:]
            x_tmp = self.proj_1(x_chan)
            conv_x_sum = torch.cat((conv_x_sum, x_tmp), 1)
        
        x = self.conv_1(conv_x_sum)

        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class RadiomicsGuidedChannelAttention3D(nn.Module):
    """
    Improved Channel Attention Module - Utilizing Radiomics Features to Guide Channel Attention

    - Radiomics features are projected to generate channel attention weights
    - Combining statistical information from image features themselves (average pooling, max pooling)
    - Using a gating mechanism to balance the influence of radiomics features and image features
    """
    def __init__(self, channels, radiomics_dim, reduction=16, attention_type='guided'):
        super().__init__()
        self.attention_type = attention_type
        
        self.radiomics_proj = nn.Sequential(
            nn.Linear(radiomics_dim, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        hidden_dim = max(channels // reduction, 1)
        
        self.img_fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False)
        )

        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, radiomics_feat):
        """
        Args:
            x: (B, C, D, H, W) Input feature map
            radiomics_feat: (B, radiomics_dim) Radiomics features (preprocessed)
        Returns:
            x: (B, C, D, H, W) Weighted feature map
        """
        B, C, D, H, W = x.shape
        
        radiomics_attn = self.radiomics_proj(radiomics_feat)  # (B, C)
        radiomics_attn = self.sigmoid(radiomics_attn).view(B, C, 1, 1, 1)
    
        avg_out = self.img_fc(self.avg_pool(x).view(B, C))  # (B, C)
        max_out = self.img_fc(self.max_pool(x).view(B, C))  # (B, C)
        img_attn = self.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        
        combined = torch.cat([
            radiomics_attn.view(B, C),
            img_attn.view(B, C)
        ], dim=1)  # (B, 2*C)
        
        gate_weights = self.gate(combined).view(B, C, 1, 1, 1)  # (B, C, 1, 1, 1)
    
        final_attn = gate_weights * radiomics_attn + (1 - gate_weights) * img_attn
        return x * final_attn


class CrossModalFusion(nn.Module):
    def __init__(self, img_dim, radiomics_dim, fusion_dim=None):
        """
        Args:
            img_dim: Image feature dimension  
            radiomics_dim: Radiomics feature dimension  
            fusion_dim: Dimension after fusion, defaults to img_dim
        """
        super().__init__()
        fusion_dim = fusion_dim or img_dim
        
        self.radiomics_to_img = nn.Sequential(
            nn.Linear(radiomics_dim, fusion_dim, bias=False),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.img_to_radiomics = nn.Sequential(
            nn.Linear(img_dim, radiomics_dim, bias=False),
            nn.LayerNorm(radiomics_dim),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + radiomics_dim, fusion_dim, bias=False),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, img_feat, radiomics_feat):
        enhanced_img = img_feat + self.radiomics_to_img(radiomics_feat)
        enhanced_radiomics = radiomics_feat + self.img_to_radiomics(img_feat)

        combined = torch.cat([enhanced_img, enhanced_radiomics], dim=1)
        fused = self.fusion(combined)
        
        return fused


class RA_CMFormer_(nn.Module):
    def __init__(self, depth=[5, 8, 20, 7], img_size=160, in_chans=4, num_classes=4, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, conv_stem=False,
                 channel_attn_type='guided', use_multi_stage_fusion=True):
        """
            1. Channel attention module now utilizes radiomics features to guide attention computation
            2. Supports multi-stage fusion (integrating radiomics features at multiple stages)
            3. Employs cross-modal fusion mechanism to enable mutual enhancement between image and radiomics features
            4. Enhanced feature fusion strategy (gating mechanism, residual connections)

            Args:
                use_multi_stage_fusion: Whether to perform fusion at multiple stages (True) or only at the final stage (False)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_multi_stage_fusion = use_multi_stage_fusion
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        if conv_stem:
            self.patch_embed1 = head_embedding(in_channels=in_chans, out_channels=embed_dim[0])
            self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3], stride=(1, 2, 2))
        else:
            self.patch_embed1 = VariablePatchEmbed(
                img_size=img_size, patch_size=2, in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], stride=(1, 2, 2))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dim]
        
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
        for i in range(depth[3])])
        
        self.feat_embed_dim = embed_dim[-1]
        self.feat_embed = nn.Sequential(
            nn.Linear(1, self.feat_embed_dim // 4),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(self.feat_embed_dim // 4, self.feat_embed_dim // 2)
        )
        self.feat_attention = nn.Sequential(
            nn.Linear(self.feat_embed_dim // 2, self.feat_embed_dim // 8),
            nn.GELU(),
            nn.Linear(self.feat_embed_dim // 8, 1)
        )
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(self.feat_embed_dim // 2),
            nn.Linear(self.feat_embed_dim // 2, embed_dim[-1]),
            nn.Dropout(drop_rate * 0.5)
        )

        self.channel_attn1 = RadiomicsGuidedChannelAttention3D(
            channels=embed_dim[0],
            radiomics_dim=embed_dim[-1],
            reduction=16
        )
        self.channel_attn2 = RadiomicsGuidedChannelAttention3D(
            channels=embed_dim[1],
            radiomics_dim=embed_dim[-1],
            reduction=16
        )
        self.channel_attn3 = RadiomicsGuidedChannelAttention3D(
            channels=embed_dim[2],
            radiomics_dim=embed_dim[-1],
            reduction=16
        )
        self.channel_attn4 = RadiomicsGuidedChannelAttention3D(
            channels=embed_dim[3],
            radiomics_dim=embed_dim[-1],
            reduction=16
        )
        
        self.norm = nn.BatchNorm3d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.cross_modal_fusion = CrossModalFusion(
            img_dim=embed_dim[-1],
            radiomics_dim=embed_dim[-1],
            fusion_dim=embed_dim[-1]
        )
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim[-1] * 2, embed_dim[-1], bias=False),
            nn.Sigmoid()
        )
        self.final_proj = nn.Sequential(
            nn.LayerNorm(embed_dim[-1]),
            nn.Linear(embed_dim[-1], embed_dim[-1] // 2, bias=False),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim[-1] // 2, num_classes, bias=False)
        ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.final_proj

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.final_proj = nn.Linear(self.embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def feat_fc(self, feat):
        """
        Process variable-length radiomics features
        """
        B, n_features = feat.shape
        feat = feat.unsqueeze(-1)  # (B, n_features, 1)

        feat_embedded = self.feat_embed(feat)  # (B, n_features, feat_embed_dim//2)

        attn_weights = self.feat_attention(feat_embedded)  # (B, n_features, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        feat_aggregated = (feat_embedded * attn_weights).sum(dim=1)  # (B, feat_embed_dim//2)
        feat_output = self.feat_proj(feat_aggregated)  # (B, embed_dim[-1])
        
        return feat_output

    def forward_features(self, x, radiomics_feat=None):
        # Stage 1
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.channel_attn1(x, radiomics_feat)
        
        # Stage 2
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.channel_attn2(x, radiomics_feat)
        
        # Stage 3
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.channel_attn3(x, radiomics_feat)
        
        # Stage 4
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.channel_attn4(x, radiomics_feat)

        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x, feat):
        radiomics_feat = self.feat_fc(feat)  # (B, embed_dim[-1])
        x = self.forward_features(x, radiomics_feat if self.use_multi_stage_fusion else None)
        x = x.flatten(2).mean(-1)   # (B, embed_dim[-1])

        fused_feat = self.cross_modal_fusion(x, radiomics_feat)

        combined = torch.cat([x, radiomics_feat], dim=1)
        gate_weights = self.gate(combined)
        final_feat = gate_weights * fused_feat + (1 - gate_weights) * x
        
        x = self.final_proj(final_feat)

        tsne_plot = False
        if tsne_plot == True:
            return x, radiomics_feat
        else:
            return x

