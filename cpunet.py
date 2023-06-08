# ADAPTED FROM CELLPOSE
# https://github.com/MouseLand/cellpose

# Copyright © 2020 Howard Hughes Medical Institute
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )


def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x, t=None):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        if t is not None:  # add in time embedding
            if t.shape[1] >= x.shape[1]:
                x = x + t[:, :x.shape[1], None, None]
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x, t=None):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.add_module('res_down_%d' % n, resdown(nbase[n], nbase[n + 1], sz))
            else:
                raise AssertionError
                self.down.add_module('conv_down_%d' % n, convdown(nbase[n], nbase[n + 1], sz))

    def forward(self, x, t=None):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y, t))
        return xd


class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        if concatenation:
            self.conv = batchconv(in_channels * 2, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.conv = batchconv(in_channels, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
                                                      concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y)
        x = x + self.conv[3](style, self.conv[2](style, x))
        return x


class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
                                                      concatenation=concatenation))

    def forward(self, x, y, style):
        x = self.conv[1](style, self.conv[0](x), y=y)
        return x


class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5

        return style


class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d' % (n - 1),
                                   resup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))
            else:
                self.up.add_module('conv_up_%d' % (n - 1),
                                   convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))

    def forward(self, style, xd):
        x = self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up) - 2, -1, -1):
            x = self.upsampling(x)
            x = self.up[n](x, xd[n], style)
        return x


class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz,
                 residual_on=True, concatenation=False):
        super(CPnet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.concatenation = concatenation
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.time_embedder = SinusoidalPositionEmbeddings(128)

    def forward(self, data, t=None):
        if t is not None:
            t = self.time_embedder(t)
        T0 = self.downsample(data, t)
        style = 0 * self.make_style(T0[-1])
        T0 = self.upsample(style, T0)
        T0 = self.output(T0)
        return T0


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def create_net(nchan=2, nclasses=1, nbase=(32, 64, 128, 256)):
    nbase = [nchan] + list(nbase)
    net = CPnet(nbase,
                nclasses,
                sz=3,
                residual_on=True,
                concatenation=False)
    return net


if __name__ == '__main__':
    a = torch.randn(12, 2, 16 * 7, 16 * 7)
    net = create_net(2, 5)
    b = net(a)
