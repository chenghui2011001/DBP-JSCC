import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm, remove_weight_norm


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Mirror HiFi-GAN style padding helper.

    This matches `academicodec.utils.get_padding` so we can port the
    discriminators without changing their behavior.
    """

    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize Conv layers with a small normal distribution.

    This is the same scheme used in HiFi-GAN / HiFiCodec.
    """

    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)


class DiscriminatorP(nn.Module):
    """Period-based 2D discriminator (HiFi-GAN style)."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3,
                 use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(
                    1,
                    32,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )
            ),
            norm_f(
                Conv2d(
                    32,
                    128,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )
            ),
            norm_f(
                Conv2d(
                    128,
                    512,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )
            ),
            norm_f(
                Conv2d(
                    512,
                    1024,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )
            ),
            norm_f(
                Conv2d(
                    1024,
                    1024,
                    (kernel_size, 1),
                    1,
                    padding=(2, 0),
                )
            ),
        ])

        self.conv_post = norm_f(
            Conv2d(
                1024,
                1,
                (3, 1),
                1,
                padding=(1, 0),
            )
        )

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        fmap = []

        # [B, C, T] -> periodic 2D [B, C, T//period, period]
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def remove_weight_norm(self) -> None:
        for l in self.convs:
            remove_weight_norm(l)
        remove_weight_norm(self.conv_post)


class MultiPeriodDiscriminator(nn.Module):
    """HiFi-GAN Multi-Period Discriminator.

    Takes real/fake waveforms with shape [B, 1, T] and returns per-period
    discriminator outputs and intermediate feature maps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):  # type: ignore[override]
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def remove_weight_norm(self) -> None:
        for d in self.discriminators:
            d.remove_weight_norm()


class DiscriminatorS(nn.Module):
    """Scale-based 1D discriminator (HiFi-GAN style)."""

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def remove_weight_norm(self) -> None:
        for l in self.convs:
            remove_weight_norm(l)
        remove_weight_norm(self.conv_post)


class MultiScaleDiscriminator(nn.Module):
    """HiFi-GAN Multi-Scale Discriminator over raw waveform."""

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):  # type: ignore[override]
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def remove_weight_norm(self) -> None:
        for d in self.discriminators:
            d.remove_weight_norm()


class BarkHFDiscriminator(nn.Module):
    """高频 Bark/BFCC 纹理判别器（Aether 专用）。

    设计目标：在 STFT 主约束之外，对 Bark 高频带上的竖纹
    结构施加对抗约束，使生成的高频纹理更加接近真实语音。

    输入应为 [B,1,T,F_hf]，其中 F_hf 对应若干高频 Bark 带。
    本模块自身不做 BFCC 提取，由训练脚本负责调用
    WaveToBFCC 并裁剪到高频带。
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        norm_f = weight_norm

        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(in_channels, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                norm_f(Conv2d(32, 128, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                norm_f(Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))),
                norm_f(Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(
            Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        fmap = []
        h = x
        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, LRELU_SLOPE)
            fmap.append(h)
        h = self.conv_post(h)
        fmap.append(h)
        h = torch.flatten(h, 1, -1)
        return h, fmap

    def remove_weight_norm(self) -> None:
        for l in self.convs:
            remove_weight_norm(l)
        remove_weight_norm(self.conv_post)


class F0PeriodDiscriminator(nn.Module):
    """基于 F0 周期的 2D 判别器（Aether 专用）。

    通过将波形按预测周期重排为 [B,1,N_periods,period] 的 2D
    patch，卷积网络可以直接感知沿周期方向的结构，从而对
    周期性与谐波纹理施加对抗约束。这里提供一个骨架实现：
    具体的周期重排逻辑由训练脚本负责预处理生成输入张量。
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        norm_f = weight_norm

        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))),
                norm_f(Conv2d(32, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))),
                norm_f(Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))),
                norm_f(Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(
            Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        fmap = []
        h = x
        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, LRELU_SLOPE)
            fmap.append(h)
        h = self.conv_post(h)
        fmap.append(h)
        h = torch.flatten(h, 1, -1)
        return h, fmap

    def remove_weight_norm(self) -> None:
        for l in self.convs:
            remove_weight_norm(l)
        remove_weight_norm(self.conv_post)


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    """HiFi-GAN feature matching loss.

    Expects lists of feature maps per discriminator; each entry is a list
    over layers, as returned by MPD/MSD.
    """

    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))

    return loss * 2.0


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """LSGAN-style discriminator loss used by HiFi-GAN."""

    loss: torch.Tensor = torch.tensor(0.0, device=disc_real_outputs[0].device)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """LSGAN-style generator loss used by HiFi-GAN."""

    loss: torch.Tensor = torch.tensor(0.0, device=disc_outputs[0].device)
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l

    return loss, gen_losses
