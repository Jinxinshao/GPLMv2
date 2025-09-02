
import torch.nn.functional as F
import math
import torch
from torch import nn
from inspect import isfunction

import torch
import torch.nn as nn

import torchvision.models as models

def gamma_embedding(gammas_list, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings for multiple gamma sequences.
    :param gammas_list: a list of 1-D Tensors, each containing N indices for a batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: a list of [N x dim] Tensors of positional embeddings.
    """
    embeddings_list = []
    for gammas in gammas_list:
        gammas = gammas.unsqueeze(0) # or gammas = gammas.view(1)

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=gammas.device)
        args = gammas[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        embeddings_list.append(embedding)
    return embeddings_list


class ChannelsAttention(nn.Module):
    def __init__(self,in_channels,scale=16):
        super(ChannelsAttention,self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
        				nn.Conv2d(in_channels,in_channels//16,1),
            			nn.ReLU(),
            			nn.Conv2d(in_channels//16,in_channels,1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.gap(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x 
    
class SpatialAttention(nn.Module):
    def __init__(self,k=7):
        super(SpatialAttention,self).__init__()
        padding = k//2
        self.conv = nn.Conv2d(2,1,kernel_size=k,stride=1,padding=padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        avg = torch.mean(x,dim=1,keepdim=True)
        max,_ = torch.max(x,dim=1,keepdim=True)
        out = self.conv(torch.cat([avg,max],dim=1))
        out = self.sigmoid(out)
        return out 

class DynamicAdjustmentLayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1):
        super(DynamicAdjustmentLayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

    
class CSAM(nn.Module):
    def __init__(self, in_channels):
        super(CSAM, self).__init__()
        
        self.channels_att = ChannelsAttention(in_channels)
        self.spatial_att = SpatialAttention()  # 确保空间注意力被正确初始化
        self.adjust = DynamicAdjustmentLayer(in_channels)  # 初始化动态调整层
        #self.adjust_fh = DynamicAdjustmentLayer(in_channels)

    def forward(self, fl, fh):
        f = fl + fh
        c_a = self.channels_att(f)
        s_a = self.spatial_att(f)  # 正确使用空间注意力
        att = c_a * s_a
        weights = self.adjust(f)
        #weights_fh = self.adjust_fh(f)
        #weights_fl, weights_fh = torch.split(self.adjust(f), 1, dim=1)  # 假设权重适用于fl和fh的不同融合方式
        _,  channels, height, width = f.size()
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # 增加三个维度，使其变为 [batch_size, 1, 1, 1]
        #weights_fh = weights_fh.unsqueeze(-1).unsqueeze(-1)  # 同上

        # 使用 expand 方法将权重扩展到与 fl 和 fh 相同的尺寸
        # 假设 fl 和 fh 的尺寸为 [batch_size, channels, height, width]
        # 这里 -1 表示该维度不扩展，沿用原尺寸
        weights = weights.expand(-1, channels, height, width)
        #weights_fh = weights_fh.expand(-1, channels, height, width)
        # print(fl.size())
        # print(fh.size())
        # print(weights_fh.size())
        # print(weights_fl.size())
        f_new = weights * fl + (1-weights) * fh  # 动态加权融合fl和fh
        
        out = f_new * att  # 应用注意力机制调整后的特征图
        #print(att.size())
        return out

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, relu):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ZPool(nn.Module):
    '''GAP + GMP'''
    def forward(self, x):
        # Global Max Pooling
        gap, _ = torch.max(x, 1)
        gap = gap.unsqueeze(1)
        
        # Global Average Pooling
        gmp = torch.mean(x, 1)
        gmp = gmp.unsqueeze(1)
        
        # Concatenate
        return torch.cat([gap, gmp], dim=1)

class AttentionGate(nn.Module):
    '''空间注意力在进行GAP+GMP，合并通道之后用到'''
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale  # 自动进行广播

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        # b,c,h,w -> b,h,c,w
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)  # 对H进行处理，进行空间注意力操作
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        # b,c,h,w -> b,w,h,c
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)

        return x_out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, device='cuda'):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)#.to(device)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)#.to(device)
        #self.bn = nn.GroupNorm(out_channels, out_channels).to(device)
        #self.eca = ECALayer(out_channels).to(device)
        self.TA = TripletAttention()#.to(device)
        # Move entire module to device after initialization     
        if device != 'cpu':
            self.to(device)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        #x = self.bn(x)
        x = self.TA(x)
        return x

class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# class FPN(nn.Module):
#     def __init__(self, device='cpu'):
#         super(FPN, self).__init__()
#         self.device = device
#         # Define the backbone layers
#         self.conv1 = DepthwiseSeparableConv(3, 24, 3, padding=1, device=self.device)
#         self.conv2 = DepthwiseSeparableConv(24, 48, 3, stride=2, padding=1, device=self.device)
#         self.conv3 = DepthwiseSeparableConv(48, 96, 3, stride=2, padding=1, device=self.device)
#         # Define the lateral layers with depthwise separable convolutions
#         self.lateral1 = DepthwiseSeparableConv(96, 96, 1, device=self.device)
#         self.lateral2 = DepthwiseSeparableConv(48, 96, 1, device=self.device)
#         self.lateral3 = DepthwiseSeparableConv(24, 96, 1, device=self.device)
#         # Define the smoothing layers with dilated convolutions for higher resolution
#         #self.smooth1 = DepthwiseSeparableConv(96, 96, 3, padding=2, dilation=2, device=self.device)
#         #self.smooth2 = DepthwiseSeparableConv(96, 96, 3, padding=2, dilation=2, device=self.device)
#         self.smooth3 = DepthwiseSeparableConv(96, 24, 3, padding=1, dilation=1, device=self.device)
#         self.CSAM_2 = CSAM(96).to(device)
#         self.CSAM_1 = CSAM(96).to(device)

#     def forward(self, x):
#         c1 = F.relu(self.conv1(x))
#         c2 = F.relu(self.conv2(c1))
#         c3 = F.relu(self.conv3(c2))

#         p3 = self.lateral1(c3)
#         p2 = self.CSAM_2(F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True) , self.lateral2(c2))
#         p1 = self.CSAM_1(F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=True) , self.lateral3(c1))

#         p1 = self.smooth3(p1)
#         p2 = DepthwiseSeparableConv(96, 48, 3, padding=1, device=self.device)(p2)
#         p3 = DepthwiseSeparableConv(96, 96, 3, padding=1, device=self.device)(p3)
#         return p1, p2, p3

class FPN(nn.Module):
    def __init__(self, device='cpu'):
        super(FPN, self).__init__()
        self.device = device
        # --- All layers are now defined in __init__ ---
        self.conv1 = DepthwiseSeparableConv(3, 24, 3, padding=1, device=self.device)
        self.conv2 = DepthwiseSeparableConv(24, 48, 3, stride=2, padding=1, device=self.device)
        self.conv3 = DepthwiseSeparableConv(48, 96, 3, stride=2, padding=1, device=self.device)
        
        self.lateral1 = DepthwiseSeparableConv(96, 96, 1, device=self.device)
        self.lateral2 = DepthwiseSeparableConv(48, 96, 1, device=self.device)
        self.lateral3 = DepthwiseSeparableConv(24, 96, 1, device=self.device)
        
        self.smooth3 = DepthwiseSeparableConv(96, 24, 3, padding=1, dilation=1, device=self.device)
        # --- Define the previously dynamic layers here ---
        self.smooth2_dynamic = DepthwiseSeparableConv(96, 48, 3, padding=1, device=self.device)
        self.smooth1_dynamic = DepthwiseSeparableConv(96, 96, 3, padding=1, device=self.device)
        
        self.CSAM_2 = CSAM(96).to(device)
        self.CSAM_1 = CSAM(96).to(device)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))

        p3 = self.lateral1(c3)
        p2_intermediate = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)
        p2 = self.CSAM_2(p2_intermediate, self.lateral2(c2))
        
        p1_intermediate = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=True)
        p1 = self.CSAM_1(p1_intermediate, self.lateral3(c1))

        # Apply the smoothing layers that were previously created dynamically
        p1 = self.smooth3(p1)
        p2 = self.smooth2_dynamic(p2)
        p3 = self.smooth1_dynamic(p3)
        
        return p1, p2, p3


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        if torch.is_tensor(noise_level): # check if noise_level is a tensor
            noise_level = noise_level.view(-1) # convert it to a vector
        #    encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))

        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        #直接使用不同的噪声水平来生成不同的gamma和beta
        #没有编码是否可以
        self.gamma_func = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.beta_func = nn.Conv2d(in_channels, out_channels, kernel_size=1) if use_affine_level else None

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma = self.gamma_func(noise_embed)
            beta = self.beta_func(noise_embed)
            x = (1 + gamma) * x + beta
        else:
            y = self.gamma_func(noise_embed)
            x = x + y
        return x



'''
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        self.conv1 = nn.Conv2d(in_channels,  out_channels*(1+self.use_affine_level), kernel_size=1)
        self.conv_zh = nn.Conv2d(in_channels=1280, out_channels=out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)



        #in_features

        #out_features
    def forward(self, x, noise_embed):
        batch, C, H ,W = x.shape
        n, c ,h, w = noise_embed.shape
        self.conv_zh.in_channels = c
        self.conv_zh.out_channels = C
        self.conv2.in_channels = C
        self.conv2.out_channels = C
        noise_embed = self.conv_zh(noise_embed)
        noise_embed = self.conv2(noise_embed)



        if self.use_affine_level:
            #all = features.shape[0] * features.shape[1] * features.shape[2]

            gamma, beta = self.conv1(noise_embed).chunk(2, dim=1)
            gamma = torch.nn.functional.interpolate(gamma, size=(H,W), mode='bilinear', align_corners=False)
            beta = torch.nn.functional.interpolate(beta, size=(H,W), mode='bilinear', align_corners=False)
            x = (1 + gamma) * x + beta
        else:
            y = self.conv1(noise_embed)
            y = torch.nn.functional.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
            x = x + y
        return x
'''


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)









class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.noise_func = FeatureWiseAffine(dim, dim, use_affine_level=True)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')
  # 加载预训练的 VGG16 模型
        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.dim = dim
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )





    def forward(self, x, time_emb):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x
        #N, C, H, W = time_emb.shape
        # 假设输入图像为大小为(3, 32, 32)，即3通道，32x32的图像


        # 构建模型
        #output_size = 1  # 假设有10个类别
        #num_filters = 32
        #kernel_sizes = [3, 5, 7]  # 不同大小的卷积核

 # 将模型移动到 GPU 上（如果有的话）
        #output = features(input)  # 对输入图像进行特征提取

        #model_Conv1DNet = Conv1DNet(C, output_size, num_filters, kernel_sizes)

        #dtype = torch.half
        #model_Conv1DNet#.type(dtype)
        #model_Conv1DNet.apply(lambda x: x.half()) # 将模型中的所有参数转换为半精度
        
        #model = BrainModel()
         # move model to device and convert to half-precision

        # 创建随机输入数据（
        #batch_size = 8
        #input_data = torch.randn(batch_size, input_channels, image_height, image_width)

        # 将输入数据调整为3D形状
        #t#ime_emb = time_emb.view(N, C, -1)
        #print(N)
        # 前向传播
        
        #print(output)
        #print(time_emb)
        #emb = self.cond_embed(output)
        x = self.noise_func(x,time_emb)

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [MixStructureBlock(dim=dim) for i in range(depth)])


    def forward(self, x,TM):
        
        for blk in self.blocks:
            x = blk(x,TM)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class MixDehazeNet(nn.Module):
    def __init__(self, in_chans=6, out_chans=6,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(MixDehazeNet, self).__init__()

        # setting
        self.patch_size = 4
        # 需要改输入通道的， self.patch_embed ，self.patch_merge1 ， self.patch_merge2 
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)



        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
        
        #self.mobilenet_v2 = models.mobilenet_v2(pretrained=False)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #vgg16.eval()  # 设置为评估模式，不更新梯度
        #self.features = self.mobilenet_v2.features  # 获取特征提取部分
        #self.features.to(device)
        self.fpn = FPN(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # self.fpn = FPN(device='cuda').cuda()
        #self.fpn.to(device)
    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x,TM):
        x = self.patch_embed(torch.cat([x, x], dim=1))
        #x = torch.randn(4, 3, 256, 256)  # batch size: 4; image size: 256x256; channels: 3
        p1, p2, p3 = self.fpn(TM)
        #TM = self.features(TM)
        x = self.layer1(x,p1)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x,p2)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x,p3)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x,p2)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x,p1)
        x = self.patch_unembed(x)
        return x

    def forward(self, x, TM):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        TM = self.check_image_size(TM)
        feat = self.forward_features(x,TM)
        # 2022/11/26
        #K, B = torch.split(feat, (1, 3), dim=1)
        K, B = torch.split(feat, [3, feat.size(1) - 3], dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def MixDehazeNet_t():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])

def MixDehazeNet_s():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[2, 2, 4, 2, 2])

def MixDehazeNet_b():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[4, 4, 8, 4, 4])

def MixDehazeNet_l():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[8, 8, 16, 8, 8])
    

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
        return self.model(x1, x2)

if __name__=='__main__':
    import os
    from torchinfo import summary

    gpu_ids = [0]
    gpu_str = ','.join(str(x) for x in gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Model ###')
    x = torch.rand(8, 6, 256, 256).to(device)
    b, c, h, w = 8, 6, 64, 64
    timsteps = 100
    model = MixDehazeNet_s(

    )
    x1 = torch.rand(8, 3, 256, 256).to(device)
    x2 = torch.rand(8, 3, 256, 256).to(device)
    #x_all = torch.cat([x1, x2], dim=1)
    model = model.to(device)
    emb = torch.ones((b, ))
    print("noise_level is:", emb.cpu().numpy())
    #for i in range(8):
        #print("这是第" + str(i + 1) + "次执行循环")
    x = model(x1, x2)
    #x = model(x)
    print(x.size())
    model = MixDehazeNet_s().to(device)
    model_wrapper = ModelWrapper(model)

    summary(model_wrapper, input_size=(1, 6, 224, 224))

