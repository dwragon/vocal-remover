import torch
from torch import nn
import torch.nn.functional as F

from lib import layers


class BaseASPPNet(nn.Module):

    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):

    def __init__(self, ch, n_fft, mean=None, std=None):
        super(CascadedASPPNet, self).__init__()
        self.low_band_net = BaseASPPNet(2, 16)
        self.high_band_net = BaseASPPNet(2, 16)

        self.bridge = layers.Conv2DBNActiv(18, ch, 1, 1, 0)
        self.full_band_net = BaseASPPNet(ch, ch * 2)

        self.out = nn.Conv2d(ch * 2, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(16, 2, 1, bias=False)

        self.phase_bridge = layers.Conv2DBNActiv(18, ch, 1, 1, 0)
        self.phase_net = BaseASPPNet(ch, ch * 2)

        self.phase_left_out = nn.Conv2d(ch * 2, 16, 1, bias=False)
        self.phase_right_out = nn.Conv2d(ch * 2, 16, 1, bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        if mean is None:
            mean = torch.zeros(self.output_bin, 1)
        else:
            mean = torch.from_numpy(mean[:, None])

        if std is None:
            std = torch.ones(self.output_bin, 1)
        else:
            std = torch.from_numpy(std[:, None])

        self.mean = nn.Parameter(mean)
        self.std = nn.Parameter(std)

        self.offset = 128

    def forward(self, x_mag, x_phase):
        mix = x_mag.detach()
        x_mag = x_mag.clone()

        x_mag -= self.mean
        x_mag /= self.std

        x_mag = x_mag[:, :, :self.max_bin]

        bandw = x_mag.size()[2] // 2
        aux = torch.cat([
            self.low_band_net(x_mag[:, :, :bandw]),
            self.high_band_net(x_mag[:, :, bandw:])
        ], dim=2)

        h_mag = torch.cat([x_mag, aux], dim=1)
        h_mag = self.full_band_net(self.bridge(h_mag))
        h_mag = F.relu(self.out(h_mag))
        h_mag = F.pad(
            input=h_mag,
            pad=(0, 0, 0, self.output_bin - h_mag.size()[2]),
            mode='replicate')

        if x_phase is None:
            h_phase_left = None
            h_phase_right = None
        else:
            x_phase = x_phase[:, :, :self.max_bin]
            h_phase = torch.cat([x_phase, aux], dim=1)
            h_phase = self.phase_net(self.phase_bridge(h_phase))
            h_phase_left = self.phase_left_out(h_phase)
            h_phase_right = self.phase_right_out(h_phase)
            h_phase_left = F.pad(
                input=h_phase_left,
                pad=(0, 0, 0, self.output_bin - h_phase_left.size()[2]),
                mode='replicate')
            h_phase_right = F.pad(
                input=h_phase_right,
                pad=(0, 0, 0, self.output_bin - h_phase_right.size()[2]),
                mode='replicate')

        if self.training:
            aux = F.relu(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode='replicate')
            return h_mag * mix, h_phase_left, h_phase_right, aux * mix
        else:
            return h_mag * mix, h_phase_left, h_phase_right

    def predict(self, x_mag, x_phase):
        h_mag, h_phase_left, h_phase_right = self.forward(x_mag, x_phase)

        if self.offset > 0:
            h_mag = h_mag[:, :, :, self.offset:-self.offset]
            assert h_mag.size()[3] > 0

            if h_phase_left is not None and h_phase_right is not None:
                h_phase_left = h_phase_left[:, :, :, self.offset:-self.offset]
                h_phase_right = h_phase_right[:, :, :, self.offset:-self.offset]
                assert h_phase_left.size()[3] > 0 and h_phase_right.size()[3] > 0

        return h_mag, h_phase_left, h_phase_right
