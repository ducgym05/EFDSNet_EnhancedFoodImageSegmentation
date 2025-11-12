from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm import create_model
import math
from typing import Callable, Union

# ==============================================================================
# PHẦN 1: CÁC LỚP TIỆN ÍCH VÀ NHÁNH NÔNG
# ==============================================================================
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.deformable_groups = deformable_groups

        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

        self.conv_mask = nn.Conv2d(self.in_channels, self.deformable_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
                                   kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_mask()

        self.dcn = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation, groups=1, bias=True)

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        if self.conv_offset.bias is not None:
            self.conv_offset.bias.data.zero_()

    def init_mask(self):
        self.conv_mask.weight.data.zero_()
        if self.conv_mask.bias is not None:
            self.conv_mask.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))

        try:
            return torch.ops.torchvision.deform_conv2d(
                x, self.dcn.weight, offset, mask, self.dcn.bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1],
                1, self.deformable_groups, True
            )
        except AttributeError:
            print("Warning: torch.ops.torchvision.deform_conv2d not found or API changed.")
            return self.dcn(x)

class ShallowBranchConvNeXtDCN(nn.Module):
    def __init__(self, in_chans=3, pretrained=True):
        super().__init__()
        self.backbone = create_model(
            'convnext_small', pretrained=pretrained, in_chans=in_chans, features_only=True
        )
        self.dcn_s8 = DCNv2(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.dcn_s16 = DCNv2(in_channels=384, out_channels=384, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.backbone(x)
        feat_s8 = features[1]
        feat_s16 = features[2]
        refined_feat_s8 = self.dcn_s8(feat_s8)
        refined_feat_s16 = self.dcn_s16(feat_s16)
        return refined_feat_s8, refined_feat_s16

# ======================= CẢI TIẾN MRF VỚI MULTI-SCALE INTERACTIVE FUSION (MSIF) =======================
class MRF(nn.Module):
    """Multi-Resolution Fusion with Multi-Scale Interactive Fusion (MSIF)"""
    def __init__(self, xbc, xsc):
        super(MRF, self).__init__()
        self.conv_align = nn.Sequential(
            nn.Conv2d(xsc, xbc, kernel_size=1, bias=False),
            nn.BatchNorm2d(xbc),
            nn.ReLU(inplace=True)
        )
        # Multi-scale interaction branches
        self.interact_1x1 = nn.Conv2d(xbc, xbc, kernel_size=1, bias=False)  # Channel interaction
        self.interact_3x3 = nn.Conv2d(xbc, xbc, kernel_size=3, padding=1, groups=xbc, bias=False)  # Spatial interaction (depthwise)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(xbc, xbc, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv_refine = nn.Sequential(
            nn.Conv2d(xbc, xbc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(xbc),
            nn.ReLU(inplace=True)
        )

    def forward(self, xb, xs):
        xb_residual = xb
        xs_aligned = self.conv_align(xs)
        xs_aligned = F.interpolate(xs_aligned, size=xb.shape[-2:], mode='bilinear', align_corners=False)
        
        # Multi-scale interaction: Apply on xs to interact with xb
        interact_channel = self.interact_1x1(xs_aligned) * xb
        interact_spatial = self.interact_3x3(xs_aligned) * xb
        
        # Fuse interactions
        fused_interact = interact_channel + interact_spatial
        
        # Gate from xs to control fused
        gate_weights = self.gate(xs_aligned)
        gated_fused = fused_interact * gate_weights
        
        # Refine and residual
        refined = self.conv_refine(gated_fused + xs_aligned)  # Add xs for semantic
        output = refined + xb_residual
        return output

# ==============================================================================
# PHẦN 2: KIẾN TRÚC CHÍNH
# ==============================================================================
class PUPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PUPHead, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.cls_seg = nn.Conv2d(128, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.cls_seg(x)
        return x

class Backbone1(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Backbone1, self).__init__()
        self.cp = ShallowBranchConvNeXtDCN(in_chans=6, pretrained=pretrained)
        self.deepNet = create_model(
            'swinv2_small_window16_256', pretrained=pretrained, features_only=True
        )
        self.inputdown2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Use improved MRF with MSIF for all
        self.MRF0 = MRF(xbc=384, xsc=768)
        self.MRF1 = MRF(xbc=384, xsc=384)
        self.MRF2 = MRF(xbc=192, xsc=384)
        self.PuPHead = PUPHead(in_channels=192, num_classes=num_classes)
        
        # Thêm aux head cho classification
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Avg Pool
            nn.Flatten(),
            nn.Linear(192, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Multi-label
        )


    def forward(self, x):
        original_image, laplacian_image = x
        shallow_feat8, shallow_feat16 = self.cp(laplacian_image)
        deep_input = self.inputdown2(original_image)
        deep_input_resized = F.interpolate(deep_input, size=(256, 256), mode='bilinear', align_corners=False)
        deep_features = self.deepNet(deep_input_resized)
        _, _, deep_feat3, deep_feat4 = [feat.permute(0, 3, 1, 2) for feat in deep_features]
        mrf_out0 = self.MRF0(deep_feat3, deep_feat4)
        mrf_out1 = self.MRF1(shallow_feat16, mrf_out0)
        result = self.MRF2(shallow_feat8, mrf_out1)
        
        # Aux output: Classification từ fused_feat
        aux_out = self.aux_head(result)

        result = self.PuPHead(result)
        result = F.interpolate(result, size=original_image.shape[-2:], mode='bilinear', align_corners=False)
        return {"out": result, "aux": aux_out}

def FDSNet(num_classes: int = 104):
    model = Backbone1(num_classes=num_classes)
    return model