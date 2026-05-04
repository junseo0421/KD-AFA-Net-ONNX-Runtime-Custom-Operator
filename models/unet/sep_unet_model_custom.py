""" Full assembly of the parts to form the complete network """
import torch.nn
import torch
import torch.nn.functional as F

from models.unet.sep_unet_parts_onnx import *


def afa_forward_impl(x, low_attention_weight, high_attention_weight):
    cuton = 0.1

    x_ft = torch.fft.fft2(x, norm="ortho")
    x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))

    magnitude = torch.abs(x_shift)
    magnitude = magnitude.clamp(min=1e-6, max=1e6)
    phase = torch.angle(x_shift)

    h, w = x_shift.shape[2:4]
    cy, cx = int(h / 2), int(w / 2)
    rh, rw = int(cuton * cy), int(cuton * cx)

    low_pass = torch.zeros_like(magnitude)
    low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = \
        magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

    high_pass = magnitude - low_pass

    low_attn_map = F.conv2d(low_pass, low_attention_weight, padding=0)
    high_attn_map = F.conv2d(high_pass, high_attention_weight, padding=0)

    low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
    high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

    low_attn_map = F.softmax(low_attn_map, dim=1)
    high_attn_map = F.softmax(high_attn_map, dim=1)

    low_pass_att = low_attn_map * low_pass
    high_pass_att = high_attn_map * high_pass

    mag_out = low_pass_att + high_pass_att
    mag_out = mag_out.clamp(min=1e-6, max=1e6)

    real = mag_out * torch.cos(phase)
    imag = mag_out * torch.sin(phase)

    fre_out = torch.complex(real, imag)
    x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

    out = torch.fft.ifft2(
        x_fft,
        s=(x.size(-2), x.size(-1)),
        norm="ortho"
    ).real

    return out


class AFAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, low_attention_weight, high_attention_weight):
        return afa_forward_impl(
            x,
            low_attention_weight,
            high_attention_weight
        )

    @staticmethod
    def symbolic(g, x, low_attention_weight, high_attention_weight):
        out = g.op(
            "kdafa::AFA",
            x,
            low_attention_weight,
            high_attention_weight,
            cuton_f=0.1,
            norm_s="ortho"
        )

        # AFA는 입력 x와 동일한 shape을 반환함: [N, C, H, W]
        # 따라서 출력 type을 입력 x와 동일하게 지정
        out.setType(x.type())

        return out


class V_Thin_Sep_UNet_4_custom_op(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(V_Thin_Sep_UNet_4_custom_op, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 4))
        self.down1 = (SepDown(4, 8))
        self.down2 = (SepDown(8, 16))
        self.down3 = (SepDown(16, 32))
        factor = 2 if bilinear else 1

        # decoder
        self.up2 = (SepUp(32, 16 // factor, bilinear))
        self.up3 = (SepUp(16, 8 // factor, bilinear))
        self.up4 = (SepUp(8, 4, bilinear))

        # Output layer
        self.outc = (OutConv(4, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.empty(4, 4, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.empty(4, 4, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.empty(8, 8, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.empty(8, 8, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.empty(16, 16, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.empty(16, 16, 1, 1))

        nn.init.kaiming_uniform_(self.low_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')

        self.concat_conv1 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(4, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-4)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-4)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")
        x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))

        magnitude = torch.abs(x_shift)
        magnitude = magnitude.clamp(min=1e-6, max=1e6)

        phase = torch.angle(x_shift)

        h, w = x_shift.shape[2:4]
        cy, cx = int(h / 2), int(w / 2)
        rh, rw = int(cuton * cy), int(cuton * cx)

        low_pass = torch.zeros_like(magnitude)
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight,
                                                  padding=0)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
        high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att
        mag_out = mag_out.clamp(min=1e-6, max=1e6)

        real = mag_out * torch.cos(phase)
        imag = mag_out * torch.sin(phase)

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real

        return out

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x3_out = AFAFunction.apply(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)  # Concatenate along the channel dimension
        x3_output = self.concat_conv3(x3_concat)  # Reduce back to original channels
        x3_output = self.bn3(x3_output)  # BatchNormalization
        x3_output = self.relu(x3_output)  # ReLU

        x2_out = AFAFunction.apply(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)  # Concatenate along the channel dimension
        x2_output = self.concat_conv2(x2_concat)  # Reduce back to original channels
        x2_output = self.bn2(x2_output)  # BatchNormalization
        x2_output = self.relu(x2_output)  # ReLU

        x1_out = AFAFunction.apply(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)  # Concatenate along the channel dimension
        x1_output = self.concat_conv1(x1_concat)  # Reduce back to original channels
        x1_output = self.bn1(x1_output)  # BatchNormalization
        x1_output = self.relu(x1_output)  # ReLU

        # decoder
        x = self.up2(x4, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4}

