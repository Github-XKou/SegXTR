from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch import nn, cat, add
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from networks.SwinTransformer import SwinTransformer
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F
import logging

class conv_bias(nn.Module):
    def __init__(self, in_ch, out_ch, bias_size=1):
        super(conv_bias, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.merge = nn.Conv3d(out_ch, bias_size, 1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x_bias = self.merge(x)
        return x_bias, x

class LinearAttention(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, num_attention_heads=8):
        super(LinearAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size1 // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size1, self.all_head_size)
        self.key = nn.Linear(hidden_size2, self.all_head_size)
        self.value = nn.Linear(hidden_size2, self.all_head_size)
        self.out = nn.Linear(self.all_head_size, hidden_size1)

        self.layernorm = nn.LayerNorm(hidden_size1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        batch_size, _, height, width, depth = x1.size()

        # Flatten the spatial dimensions (H, W, D) into the sequence dimension
        x1 = x1.view(batch_size, -1, height * width * depth).permute(0, 2, 1)
        x2 = x2.view(batch_size, -1, height * width * depth).permute(0, 2, 1)

        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_layer = query_layer / (self.attention_head_size ** 0.25)
        key_layer = key_layer / (self.attention_head_size ** 0.25)

        # Linear attention mechanism
        key_layer_softmax = F.softmax(key_layer, dim=-2)
        kv = torch.einsum("bhld,bhle->bhde", key_layer_softmax, value_layer)
        context_layer = torch.einsum("bhld,bhde->bhle", query_layer, kv)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)

        # Residual connection and LayerNorm
        attention_output += x1  # Residual connection
        attention_output = self.layernorm(attention_output)

        # Reshape back to original spatial dimensions
        attention_output = attention_output.permute(0, 2, 1).view(batch_size, -1, height, width, depth)

        return attention_output


class FDTR_base(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        depth=(16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16), bias=1
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        # 分割编码器
        self.swintr = SwinTransformer(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.encoder1 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size, 
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2, 
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4, 
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.depth = depth
        
        self.conv0 = conv_bias(in_channels, depth[0], bias_size=bias)
        self.conv1 = conv_bias(depth[0], depth[1], bias_size=bias)

        in_chan = bias
        self.conv2 = conv_bias(depth[1]+ in_chan, depth[2], bias_size=bias)
        in_chan = in_chan + bias
        self.conv3 = conv_bias(depth[2] + in_chan, depth[3], bias_size=bias)
        in_chan = in_chan + bias
        self.conv4 = conv_bias(depth[3] + in_chan, depth[4], bias_size=bias)
        in_chan = in_chan + bias
        self.conv5 = conv_bias(depth[4] + in_chan, depth[5], bias_size=bias)
        in_chan = in_chan + bias
        self.conv6 = conv_bias(depth[5] + in_chan, depth[6], bias_size=bias)
        in_chan = in_chan + bias
        self.conv7 = conv_bias(depth[6] + in_chan, depth[7], bias_size=bias)
        in_chan = in_chan + bias
        self.conv8 = conv_bias(depth[7] + in_chan, depth[8], bias_size=bias)
        in_chan = in_chan + bias
        self.conv9 = conv_bias(depth[8] + in_chan, depth[9], bias_size=bias)
        in_chan = in_chan + bias
        self.conv10 = conv_bias(depth[9]+ in_chan, depth[10], bias_size=bias)
        in_chan = in_chan + bias
        self.conv11 = conv_bias(depth[10]+ in_chan, depth[11], bias_size=bias)
        in_chan = in_chan + bias
        self.conv12 = conv_bias(depth[11] + in_chan, depth[12], bias_size=bias)
        in_chan = in_chan + bias
        self.conv13 = conv_bias(depth[12] + in_chan, depth[13], bias_size=bias)
        in_chan = in_chan + bias
        self.conv14 = conv_bias(depth[13] + in_chan, depth[14], bias_size=bias)
        in_chan = in_chan + bias
        self.conv15 = conv_bias(depth[14] + in_chan, depth[15], bias_size=bias)
        in_chan = in_chan + bias
        self.conv16 = conv_bias(depth[15] + in_chan, depth[16], bias_size=bias)
        in_chan = in_chan + bias
        self.conv17 = conv_bias(depth[16] + in_chan, depth[17], bias_size=bias)


        self.swintr_mim = SwinTransformer(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.mim_encoder1 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.mim_encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.mim_encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.mim_encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.mim_encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.mim_conv0 = conv_bias(in_channels, depth[0], bias_size=bias)
        self.mim_conv1 = conv_bias(depth[0], depth[1], bias_size=bias)

        inm_chan = bias
        self.mim_conv2 = conv_bias(depth[1]+ inm_chan, depth[2], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv3 = conv_bias(depth[2] + inm_chan, depth[3], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv4 = conv_bias(depth[3] + inm_chan, depth[4], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv5 = conv_bias(depth[4] + inm_chan, depth[5], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv6 = conv_bias(depth[5] + inm_chan, depth[6], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv7 = conv_bias(depth[6] + inm_chan, depth[7], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv8 = conv_bias(depth[7] + inm_chan, depth[8], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv9 = conv_bias(depth[8] + inm_chan, depth[9], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv10 = conv_bias(depth[9]+ inm_chan, depth[10], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv11 = conv_bias(depth[10]+ inm_chan, depth[11], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv12 = conv_bias(depth[11] + inm_chan, depth[12], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv13 = conv_bias(depth[12] + inm_chan, depth[13], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv14 = conv_bias(depth[13] + inm_chan, depth[14], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv15 = conv_bias(depth[14] + inm_chan, depth[15], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv16 = conv_bias(depth[15] + inm_chan, depth[16], bias_size=bias)
        inm_chan = inm_chan + bias
        self.mim_conv17 = conv_bias(depth[16] + inm_chan, depth[17], bias_size=bias)

        self.up_1_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)

        self.mim_up_1_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_1_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_0_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_1_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_4_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_2_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_3_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_3_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_2_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_1_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.mim_up_1_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)


        self.down_0_0_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_3_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_3_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.mim_down_0_0_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_0_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_0_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_0_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_0_1_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_0_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_0_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_0_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_1_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_1_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_1_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_1_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_1_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_1_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_2_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_2_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_2_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_2_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_3_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.mim_down_3_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.linear_attention = LinearAttention(hidden_size1=256, hidden_size2=256)

        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]


        self.decoder_segmentation = nn.Sequential(
            nn.Conv3d(depth[-1], 4, kernel_size=1),
            nn.Softmax(dim=1)
        )
        

        self.decoder_mim = nn.Sequential(
            nn.Conv3d(depth[-1], in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.metric_loss = nn.MSELoss()

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in, xmim_in = None):

        xall, hidden_states_out = self.swintr(x_in) 

        #block0
        x_bias_0_0_0, x = self.conv0(x_in)
        x_bias_0_1_0, x = self.conv1(x)

        renc0 = x

        x_bias_0_0_1 = self.down_0_0_1(x_bias_0_0_0)
        x_bias_0_0_2 = self.down_0_0_2(x_bias_0_0_1)
        x_bias_0_0_3 = self.down_0_0_3(x_bias_0_0_2)
        x_bias_0_0_4 = self.down_0_0_4(x_bias_0_0_3)

        x_bias_0_1_1 = self.down_0_1_1(x_bias_0_1_0)
        x_bias_0_1_2 = self.down_0_1_2(x_bias_0_1_1)
        x_bias_0_1_3 = self.down_0_1_3(x_bias_0_1_2)
        x_bias_0_1_4 = self.down_0_1_4(x_bias_0_1_3)

        #block1
        x1 = hidden_states_out[0]
        enc1 = self.encoder1(self.proj_feat(x1))
        
        x_bias_1_0_1, x = self.conv2(cat([enc1, x_bias_0_0_1], dim=1))
        x_bias_1_1_1, x = self.conv3(cat([x, x_bias_0_0_1, x_bias_0_1_1], dim=1))

        renc1 = x
        x_bias_1_0_0 = self.up_1_0_0(x_bias_1_0_1)
        x_bias_1_0_2 = self.down_1_0_2(x_bias_1_0_1)
        x_bias_1_0_3 = self.down_1_0_3(x_bias_1_0_2)
        x_bias_1_0_4 = self.down_1_0_4(x_bias_1_0_3)

        x_bias_1_1_0 = self.up_1_1_0(x_bias_1_1_1)
        x_bias_1_1_2 = self.down_1_1_2(x_bias_1_1_1)
        x_bias_1_1_3 = self.down_1_1_3(x_bias_1_1_2)
        x_bias_1_1_4 = self.down_1_1_4(x_bias_1_1_3)

        #block2
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))

        x_bias_2_0_2, x = self.conv4(cat([enc2, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2], dim=1))
        x_bias_2_1_2, x = self.conv5(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2], dim=1))

        renc2 = x
        x_bias_2_0_1 = self.up_2_0_1(x_bias_2_0_2)
        x_bias_2_0_0 = self.up_2_0_0(x_bias_2_0_1)
        x_bias_2_0_3 = self.down_2_0_3(x_bias_2_0_2)
        x_bias_2_0_4 = self.down_2_0_4(x_bias_2_0_3)

        x_bias_2_1_1 = self.up_2_1_1(x_bias_2_1_2)
        x_bias_2_1_0 = self.up_2_1_0(x_bias_2_1_1)
        x_bias_2_1_3 = self.down_2_1_3(x_bias_2_1_2)
        x_bias_2_1_4 = self.down_2_1_4(x_bias_2_1_3)

        #block3
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))

        x_bias_3_0_3, x = self.conv6(
            cat([enc3, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3], dim=1))
        x_bias_3_1_3, x = self.conv7(cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3,
                                          x_bias_2_1_3], dim=1))

        renc3 = x

        x_bias_3_0_2 = self.up_3_0_2(x_bias_3_0_3)
        x_bias_3_0_1 = self.up_3_0_1(x_bias_3_0_2)
        x_bias_3_0_0 = self.up_3_0_0(x_bias_3_0_1)
        x_bias_3_0_4 = self.down_3_0_4(x_bias_3_0_3)

        x_bias_3_1_2 = self.up_3_1_2(x_bias_3_1_3)
        x_bias_3_1_1 = self.up_3_1_1(x_bias_3_1_2)
        x_bias_3_1_0 = self.up_3_1_0(x_bias_3_1_1)
        x_bias_3_1_4 = self.down_3_1_4(x_bias_3_1_3)


        #block4
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))

        x_bias_4_0_4, x = self.conv8(
            cat([enc4, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4, x_bias_2_1_4, x_bias_3_0_4],
                dim=1))
        x_bias_4_1_4, x = self.conv9(cat([x, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4,
                                          x_bias_2_1_4, x_bias_3_0_4, x_bias_3_1_4], dim=1))

        renc4 = x

        x_bias_4_0_3 = self.up_4_0_3(x_bias_4_0_4)
        x_bias_4_0_2 = self.up_4_0_2(x_bias_4_0_3)
        x_bias_4_0_1 = self.up_4_0_1(x_bias_4_0_2)
        x_bias_4_0_0 = self.up_4_0_0(x_bias_4_0_1)
        x_bias_4_1_3 = self.up_4_1_3(x_bias_4_1_4)
        x_bias_4_1_2 = self.up_4_1_2(x_bias_4_1_3)
        x_bias_4_1_1 = self.up_4_1_1(x_bias_4_1_2)
        x_bias_4_1_0 = self.up_4_1_0(x_bias_4_1_1)

        x5 = xall
        enc5 = self.encoder5(self.proj_feat(x5))


        if xmim_in is not None:
            xmim_all, hidden_states_out_mim = self.swintr_mim(xmim_in)

            #block0
            xmim_bias_0_0_0, xmim = self.mim_conv0(xmim_in)
            xmim_bias_0_1_0, xmim = self.mim_conv1(xmim)

            xmim_bias_0_0_1 = self.mim_down_0_0_1(xmim_bias_0_0_0)
            xmim_bias_0_0_2 = self.mim_down_0_0_2(xmim_bias_0_0_1)
            xmim_bias_0_0_3 = self.mim_down_0_0_3(xmim_bias_0_0_2)
            xmim_bias_0_0_4 = self.mim_down_0_0_4(xmim_bias_0_0_3)

            xmim_bias_0_1_1 = self.mim_down_0_1_1(xmim_bias_0_1_0)
            xmim_bias_0_1_2 = self.mim_down_0_1_2(xmim_bias_0_1_1)
            xmim_bias_0_1_3 = self.mim_down_0_1_3(xmim_bias_0_1_2)
            xmim_bias_0_1_4 = self.mim_down_0_1_4(xmim_bias_0_1_3)

            #block1
            xmim1 = hidden_states_out_mim[0]
            xmim_enc1 = self.mim_encoder1(self.proj_feat(xmim1))


            xmim_bias_1_0_1, xmim = self.mim_conv2(cat([xmim_enc1, xmim_bias_0_0_1], dim=1))
            xmim_bias_1_1_1, xmim = self.mim_conv3(cat([xmim, xmim_bias_0_0_1, xmim_bias_0_1_1], dim=1))

            xmim_bias_1_0_0 = self.mim_up_1_0_0(xmim_bias_1_0_1)
            xmim_bias_1_0_2 = self.mim_down_1_0_2(xmim_bias_1_0_1)
            xmim_bias_1_0_3 = self.mim_down_1_0_3(xmim_bias_1_0_2)
            xmim_bias_1_0_4 = self.mim_down_1_0_4(xmim_bias_1_0_3)

            xmim_bias_1_1_0 = self.mim_up_1_1_0(xmim_bias_1_1_1)
            xmim_bias_1_1_2 = self.mim_down_1_1_2(xmim_bias_1_1_1)
            xmim_bias_1_1_3 = self.mim_down_1_1_3(xmim_bias_1_1_2)
            xmim_bias_1_1_4 = self.mim_down_1_1_4(xmim_bias_1_1_3)

            #block2
            xmim2 = hidden_states_out_mim[3]
            xmim_enc2 = self.mim_encoder2(self.proj_feat(xmim2))

            xmim_bias_2_0_2, xmim = self.mim_conv4(cat([xmim_enc2, xmim_bias_0_0_2, xmim_bias_0_1_2, xmim_bias_1_0_2], dim=1))
            xmim_bias_2_1_2, xmim = self.mim_conv5(cat([xmim, xmim_bias_0_0_2, xmim_bias_0_1_2, xmim_bias_1_0_2, xmim_bias_1_1_2], dim=1))

            xmim_bias_2_0_1 = self.mim_up_2_0_1(xmim_bias_2_0_2)
            xmim_bias_2_0_0 = self.mim_up_2_0_0(xmim_bias_2_0_1)
            xmim_bias_2_0_3 = self.mim_down_2_0_3(xmim_bias_2_0_2)
            xmim_bias_2_0_4 = self.mim_down_2_0_4(xmim_bias_2_0_3)

            xmim_bias_2_1_1 = self.mim_up_2_1_1(xmim_bias_2_1_2)
            xmim_bias_2_1_0 = self.mim_up_2_1_0(xmim_bias_2_1_1)
            xmim_bias_2_1_3 = self.mim_down_2_1_3(xmim_bias_2_1_2)
            xmim_bias_2_1_4 = self.mim_down_2_1_4(xmim_bias_2_1_3)

            #block3
            xmim3 = hidden_states_out_mim[6]
            xmim_enc3 = self.mim_encoder3(self.proj_feat(xmim3))

            xmim_bias_3_0_3, xmim = self.mim_conv6(cat([xmim_enc3, xmim_bias_0_0_3, xmim_bias_0_1_3, xmim_bias_1_0_3, xmim_bias_1_1_3, xmim_bias_2_0_3], dim=1))
            xmim_bias_3_1_3, xmim = self.mim_conv7(cat([xmim, xmim_bias_0_0_3, xmim_bias_0_1_3, xmim_bias_1_0_3, xmim_bias_1_1_3, xmim_bias_2_0_3, xmim_bias_2_1_3], dim=1))

            xmim_bias_3_0_2 = self.mim_up_3_0_2(xmim_bias_3_0_3)
            xmim_bias_3_0_1 = self.mim_up_3_0_1(xmim_bias_3_0_2)
            xmim_bias_3_0_0 = self.mim_up_3_0_0(xmim_bias_3_0_1)
            xmim_bias_3_0_4 = self.mim_down_3_0_4(xmim_bias_3_0_3)

            xmim_bias_3_1_2 = self.mim_up_3_1_2(xmim_bias_3_1_3)
            xmim_bias_3_1_1 = self.mim_up_3_1_1(xmim_bias_3_1_2)
            xmim_bias_3_1_0 = self.mim_up_3_1_0(xmim_bias_3_1_1)
            xmim_bias_3_1_4 = self.mim_down_3_1_4(xmim_bias_3_1_3)

            #block4
            xmim4 = hidden_states_out_mim[9]
            xmim_enc4 = self.mim_encoder4(self.proj_feat(xmim4))

            xmim_bias_4_0_4, xmim = self.mim_conv8(cat([xmim_enc4, xmim_bias_0_0_4, xmim_bias_0_1_4, xmim_bias_1_0_4, xmim_bias_1_1_4, xmim_bias_2_0_4, xmim_bias_2_1_4, xmim_bias_3_0_4], dim=1))
            xmim_bias_4_1_4, xmim = self.mim_conv9(cat([xmim, xmim_bias_0_0_4, xmim_bias_0_1_4, xmim_bias_1_0_4, xmim_bias_1_1_4, xmim_bias_2_0_4, xmim_bias_2_1_4, xmim_bias_3_0_4, xmim_bias_3_1_4], dim=1))

            xmim_bias_4_0_3 = self.mim_up_4_0_3(xmim_bias_4_0_4)
            xmim_bias_4_0_2 = self.mim_up_4_0_2(xmim_bias_4_0_3)
            xmim_bias_4_0_1 = self.mim_up_4_0_1(xmim_bias_4_0_2)
            xmim_bias_4_0_0 = self.mim_up_4_0_0(xmim_bias_4_0_1)
            xmim_bias_4_1_3 = self.mim_up_4_1_3(xmim_bias_4_1_4)
            xmim_bias_4_1_2 = self.mim_up_4_1_2(xmim_bias_4_1_3)
            xmim_bias_4_1_1 = self.mim_up_4_1_1(xmim_bias_4_1_2)
            xmim_bias_4_1_0 = self.mim_up_4_1_0(xmim_bias_4_1_1)

            xmim5 = xmim_all
            xmim_enc5 = self.mim_encoder5(self.proj_feat(xmim5))
            
            x_attended = self.linear_attention(enc5, xmim_enc5)
        else:
            x_attended = self.linear_attention(enc5, enc5)


        # block5
        x = self.up(x_attended)
        x_bias_3_2_3, x = self.conv10(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3], dim=1))
        x_bias_3_3_3, x = self.conv11(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3, x_bias_4_1_3], dim=1))

        rdnc4 = x
        x_bias_3_2_2 = self.up_3_2_2(x_bias_3_2_3)
        x_bias_3_2_1 = self.up_3_2_1(x_bias_3_2_2)
        x_bias_3_2_0 = self.up_3_2_0(x_bias_3_2_1)
        x_bias_3_3_2 = self.up_3_3_2(x_bias_3_3_3)
        x_bias_3_3_1 = self.up_3_3_1(x_bias_3_3_2)
        x_bias_3_3_0 = self.up_3_3_0(x_bias_3_3_1)

        # block6
        x = self.up(x)
        x_bias_2_2_2, x = self.conv12(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2], dim=1))
        x_bias_2_3_2, x = self.conv13(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2, x_bias_3_3_2], dim=1))

        rdnc3 = x
        x_bias_2_2_1 = self.up_2_2_1(x_bias_2_2_2)
        x_bias_2_2_0 = self.up_2_2_0(x_bias_2_2_1)
        x_bias_2_3_1 = self.up_2_3_1(x_bias_2_3_2)
        x_bias_2_3_0 = self.up_2_3_0(x_bias_2_3_1)

        # block7
        x = self.up(x)
        x_bias_1_2_1, x = self.conv14(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1], dim=1))
        x_bias_1_3_1, x = self.conv15(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1, x_bias_2_3_1], dim=1))

        rdnc2 = x
        x_bias_1_2_0 = self.up_1_2_0(x_bias_1_2_1)
        x_bias_1_3_0 = self.up_1_3_0(x_bias_1_3_1)

        # block8
        x = self.up(x)
        x_bias_0_2_0, x = self.conv16(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0],
                                          dim=1))

        x_bias_0_3_0, x = self.conv17(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0,
                                           x_bias_1_3_0], dim=1))
        rdnc1 =x 


        if xmim_in is not None:
            xmim = self.up(xmim_enc5)
            xmim_bias_3_2_3, xmim = self.mim_conv10(
                cat([xmim, xmim_bias_0_0_3, xmim_bias_0_1_3, xmim_bias_1_0_3, xmim_bias_1_1_3, xmim_bias_2_0_3, xmim_bias_2_1_3, xmim_bias_3_0_3,   
            xmim_bias_3_1_3, xmim_bias_4_0_3], dim=1))
            xmim_bias_3_3_3, xmim = self.mim_conv11(
                cat([xmim, xmim_bias_0_0_3, xmim_bias_0_1_3, xmim_bias_1_0_3, xmim_bias_1_1_3, xmim_bias_2_0_3, xmim_bias_2_1_3, xmim_bias_3_0_3,   
                xmim_bias_3_1_3, xmim_bias_4_0_3, xmim_bias_4_1_3], dim=1))

            rdncm4 = xmim
            xmim_bias_3_2_2 = self.mim_up_3_2_2(xmim_bias_3_2_3)
            xmim_bias_3_2_1 = self.mim_up_3_2_1(xmim_bias_3_2_2)
            xmim_bias_3_2_0 = self.mim_up_3_2_0(xmim_bias_3_2_1)
            xmim_bias_3_3_2 = self.mim_up_3_3_2(xmim_bias_3_3_3)
            xmim_bias_3_3_1 = self.mim_up_3_3_1(xmim_bias_3_3_2)
            xmim_bias_3_3_0 = self.mim_up_3_3_0(xmim_bias_3_3_1)

            # block6
            xmim = self.up(xmim)
            xmim_bias_2_2_2, xmim = self.mim_conv12(cat([xmim, xmim_bias_0_0_2, xmim_bias_0_1_2, xmim_bias_1_0_2, xmim_bias_1_1_2, xmim_bias_2_0_2,
                                          xmim_bias_2_1_2, xmim_bias_3_0_2, xmim_bias_3_1_2, xmim_bias_4_0_2, xmim_bias_4_1_2,
                                          xmim_bias_3_2_2], dim=1))
            xmim_bias_2_3_2, xmim = self.mim_conv13(cat([xmim, xmim_bias_0_0_2, xmim_bias_0_1_2, xmim_bias_1_0_2, xmim_bias_1_1_2, xmim_bias_2_0_2,
                                          xmim_bias_2_1_2, xmim_bias_3_0_2, xmim_bias_3_1_2, xmim_bias_4_0_2, xmim_bias_4_1_2,
                                          xmim_bias_3_2_2, xmim_bias_3_3_2], dim=1))

            rdncm3 = xmim
            xmim_bias_2_2_1 = self.mim_up_2_2_1(xmim_bias_2_2_2)
            xmim_bias_2_2_0 = self.mim_up_2_2_0(xmim_bias_2_2_1)
            xmim_bias_2_3_1 = self.mim_up_2_3_1(xmim_bias_2_3_2)
            xmim_bias_2_3_0 = self.mim_up_2_3_0(xmim_bias_2_3_1)

            # block7
            xmim = self.up(xmim)

            xmim_bias_1_2_1, xmim = self.mim_conv14(cat([xmim, xmim_bias_0_0_1, xmim_bias_0_1_1, xmim_bias_1_0_1, xmim_bias_1_1_1, xmim_bias_2_0_1,
                                          xmim_bias_2_1_1, xmim_bias_3_0_1, xmim_bias_3_1_1, xmim_bias_4_0_1, xmim_bias_4_1_1,
                                          xmim_bias_3_2_1, xmim_bias_3_3_1,xmim_bias_2_2_1], dim=1))
            xmim_bias_1_3_1, xmim = self.mim_conv15(cat([xmim, xmim_bias_0_0_1, xmim_bias_0_1_1, xmim_bias_1_0_1, xmim_bias_1_1_1, xmim_bias_2_0_1,
                                          xmim_bias_2_1_1, xmim_bias_3_0_1, xmim_bias_3_1_1, xmim_bias_4_0_1, xmim_bias_4_1_1,
                                          xmim_bias_3_2_1, xmim_bias_3_3_1, xmim_bias_2_2_1,xmim_bias_2_3_1], dim=1))

            rdncm2 = xmim
            xmim_bias_1_2_0 = self.mim_up_1_2_0(xmim_bias_1_2_1)
            xmim_bias_1_3_0 = self.mim_up_1_3_0(xmim_bias_1_3_1)

            # block8
            xmim = self.up(xmim)

            
            xmim_bias_0_2_0, xmim = self.mim_conv16(cat([xmim, xmim_bias_0_0_0, xmim_bias_0_1_0, xmim_bias_1_0_0, xmim_bias_1_1_0, xmim_bias_2_0_0,
                                          xmim_bias_2_1_0, xmim_bias_3_0_0, xmim_bias_3_1_0, xmim_bias_4_0_0, xmim_bias_4_1_0,
                                          xmim_bias_3_2_0, xmim_bias_3_3_0, xmim_bias_2_2_0, xmim_bias_2_3_0, xmim_bias_1_2_0], dim=1))
            xmim_bias_0_3_0, xmim = self.mim_conv17(cat([xmim, xmim_bias_0_0_0, xmim_bias_0_1_0, xmim_bias_1_0_0, xmim_bias_1_1_0, xmim_bias_2_0_0,
                                          xmim_bias_2_1_0, xmim_bias_3_0_0, xmim_bias_3_1_0, xmim_bias_4_0_0, xmim_bias_4_1_0,
                                          xmim_bias_3_2_0, xmim_bias_3_3_0, xmim_bias_2_2_0, xmim_bias_2_3_0, xmim_bias_1_2_0, xmim_bias_1_3_0], dim=1))

            rdncm1 = xmim

        if xmim_in is not None:

            dl_loss = ((enc5-xmim_enc5)**2).mean()
            x = self.decoder_segmentation(x)
            xmim = self.decoder_mim(xmim)
            return renc0,rdnc1,renc1,rdnc2,renc2,rdnc3,renc3,rdnc4,x,xmim, dl_loss
        else:
            x = self.decoder_segmentation(x)
            return renc0,rdnc1,renc1,rdnc2,renc2,rdnc3,renc3,rdnc4,x,x,x

        
class SegXTR(nn.Module):
    def __init__(self, in_channels, out_channels, crop_size=(128, 128, 128),depth=(16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16), bias=4):
        super(SegXTR, self).__init__()
        self.FDTR_base = FDTR_base(in_channels,out_channels=out_channels,img_size=crop_size,depth=depth, bias=bias)

    def forward(self, x, xmim_in=None):
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (16 - x.size()[2] % 16) % 16
        diffY = (16 - x.size()[3] % 16) % 16
        diffX = (16 - x.size()[4] % 16) % 16

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        renc0,rdnc1,renc1,rdnc2,renc2,rdnc3,renc3,rdnc4,x,xmim,dl_loss = self.FDTR_base(x, xmim_in)

        return renc0,rdnc1,renc1,rdnc2,renc2,rdnc3,renc3,rdnc4,x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2], xmim[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2],dl_loss
