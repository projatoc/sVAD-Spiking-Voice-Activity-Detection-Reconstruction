"""Research reconstruction of Yang et al., ICASSP 2024 sVAD.
Not an official author implementation. Underspecified choices are configurable.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    sample_rate: int = 16000    # assumption; paper does not give this in Sec. 3.1
    frame_ms: float = 30.0
    hop_ms: float = 15.0
    features: int = 20
    sinc_kernel: int = 101    # assumption
    conv_kernel: int = 3      # assumption
    conv_bias: bool = True
    attention_width1: int = 31  # sweep candidate, NOT known from paper
    attention_width2: int = 31
    attention_bias: bool = True
    recurrent_hidden: int = 32
    recurrent_type: str = 'diagonal' # also 'dense'
    recurrent_bias: bool = True
    readout_bias: bool = True
    alpha: float = 0.5
    threshold: float = 0.3
    surrogate_a: float = 4.0
    lambda_mse: float = 1.0  # assumption

    @property
    def frame_samples(self):
        return round(self.sample_rate * self.frame_ms / 1000)

    @property
    def hop_samples(self):
        return round(self.sample_rate * self.hop_ms / 1000)


class BoxcarSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        surrogate = (x.abs() < ctx.width / 2).to(x.dtype) / ctx.width
        return grad_output * surrogate, None


def spike(x, threshold, width):
    return BoxcarSpike.apply(x - threshold, width)


class SincConv1d(nn.Module):
    """Trainable normalized bandpass kernels with 2 parameters per filter.
    Not guaranteed to match authors' exact SincNet parametrization.
    """
    def __init__(self, filters=20, kernel_size=101, sample_rate=16000):
        super().__init__()
        assert kernel_size % 2 == 1
        self.filters = filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        low = torch.linspace(80.0, sample_rate / 2 - 900.0, filters)
        high = torch.clamp(low + 600.0, max=sample_rate / 2 - 50)
        self.low_hz_ = nn.Parameter(low / (sample_rate / 2))
        self.band_hz_ = nn.Parameter((high - low) / (sample_rate / 2))
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer('n', n)
        self.register_buffer('window', torch.hamming_window(kernel_size, periodic=False))

    def forward(self, x):
        nyquist = self.sample_rate / 2
        low = torch.clamp(self.low_hz_.abs() * nyquist, min=40., max=nyquist - 100.)
        high = torch.minimum(low + 40. + self.band_hz_.abs() * nyquist,
                             torch.full_like(low, nyquist - 1.))
        # ideal band-pass impulse response in cycles/sample
        lf = low[:, None] / self.sample_rate
        hf = high[:, None] / self.sample_rate
        n = self.n[None, :]
        kernels = (2 * hf * torch.sinc(2 * hf * n) - 2 * lf * torch.sinc(2 * lf * n))
        kernels = kernels * self.window[None, :]
        return F.conv1d(x, kernels[:, None, :], padding=self.kernel_size // 2)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sinc = SincConv1d(cfg.features, cfg.sinc_kernel, cfg.sample_rate)
        self.conv = nn.Conv1d(cfg.features, cfg.features,
                              cfg.conv_kernel, padding=cfg.conv_kernel // 2,
                              bias=cfg.conv_bias)

    def forward(self, frames):
        # frames: (batch, time_frames, frame_samples)
        b, t, n = frames.shape
        z = frames.reshape(b * t, 1, n)
        z = self.sinc(z).abs()  # rectified amplitude envelope; research assumption
        z = self.conv(z)
        z = F.adaptive_avg_pool1d(z, 1).squeeze(-1)
        z = F.relu(F.layer_norm(z, (self.cfg.features,)))
        return z.reshape(b, t, -1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(cfg.features, cfg.attention_width1, bias=cfg.attention_bias),
            nn.Linear(cfg.attention_width1, cfg.attention_width2, bias=cfg.attention_bias),
            nn.Linear(cfg.attention_width2, cfg.features, bias=cfg.attention_bias)
        ])
        self.cfg = cfg

    def forward(self, features):
        b, t, _ = features.shape
        states = [None] * len(self.layers)
        outs = []
        for i in range(t):
            a = features[:, i]
            for j, layer in enumerate(self.layers):
                cur = layer(a)
                u = cur if states[j] is None else self.cfg.alpha * states[j][0] + cur - self.cfg.threshold * states[j][1]
                a = spike(u, self.cfg.threshold, self.cfg.surrogate_a)
                states[j] = (u, a)
            outs.append(a)
        return torch.stack(outs, 1)


class SpikingRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.features, cfg.recurrent_hidden, bias=cfg.recurrent_bias)
        if cfg.recurrent_type == 'diagonal':
            self.recurrent_weight = nn.Parameter(torch.randn(cfg.recurrent_hidden) * 0.05)
        elif cfg.recurrent_type == 'dense':
            self.recurrent_weight = nn.Parameter(torch.randn(cfg.recurrent_hidden, cfg.recurrent_hidden) * 0.05)
        else:
            raise ValueError('recurrent_type must be diagonal or dense')
        self.readout = nn.Linear(cfg.recurrent_hidden, 2, bias=cfg.readout_bias)

    def forward(self, x):
        b, t, _ = x.shape
        u = x.new_zeros(b, self.cfg.recurrent_hidden)
        previous = torch.zeros_like(u)
        output = []
        for i in range(t):
            if self.cfg.recurrent_type == 'diagonal':
                rec = previous * self.recurrent_weight
            else:
                rec = F.linear(previous, self.recurrent_weight)
            current = self.input(x[:, i]) + rec
            u = self.cfg.alpha * u + current - self.cfg.threshold * previous
            previous = spike(u, self.cfg.threshold, self.cfg.surrogate_a)
            output.append(self.readout(previous))
        return torch.stack(output, 1)


class SVAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.encoder_lif_threshold = cfg.threshold
        self.attention = Attention(cfg)
        self.classifier = SpikingRNN(cfg)

    def encode(self, frames):
        analog = self.encoder(frames)
        u = torch.zeros_like(analog[:, 0])
        prev = torch.zeros_like(u)
        spikes = []
        for t in range(analog.shape[1]):
            u = self.cfg.alpha * u + analog[:, t] - self.cfg.threshold * prev
            prev = spike(u, self.cfg.threshold, self.cfg.surrogate_a)
            spikes.append(prev)
        return torch.stack(spikes, 1)

    def forward(self, noisy_frames, clean_frames=None):
        noisy = self.encode(noisy_frames)
        mask = self.attention(noisy)
        attended = noisy * mask
        logits = self.classifier(attended)
        clean = self.encode(clean_frames) if clean_frames is not None else None
        return {'logits': logits, 'features': noisy, 'mask': mask,
                'attended': attended, 'clean_features': clean}


def combined_loss(output, frame_labels, lambda_mse=1.0):
    ce = F.cross_entropy(output['logits'].reshape(-1, 2), frame_labels.reshape(-1))
    if output['clean_features'] is None:
        return ce, {'ce': float(ce.detach()), 'mse': None}
    mse = F.mse_loss(output['attended'], output['clean_features'])
    return ce + lambda_mse * mse, {'ce': float(ce.detach()), 'mse': float(mse.detach())}


def frame_audio(waveforms, cfg):
    """(batch, samples) -> (batch, frames, frame_samples)."""
    if waveforms.shape[-1] < cfg.frame_samples:
        raise ValueError('audio shorter than one analysis frame')
    return waveforms.unfold(-1, cfg.frame_samples, cfg.hop_samples)


def parameter_breakdown(model):
    return {name: sum(p.numel() for p in sub.parameters())
            for name, sub in [('encoder', model.encoder),
                              ('attention', model.attention),
                              ('classifier', model.classifier)]}
