"""Per-token RealNVP: h <-> (z, r), without pooling or a separate decoder."""
import torch
from torch import nn


class Coupling(nn.Module):
    def __init__(self, channels, hidden, flip, scale_bound=1.5):
        super().__init__()
        self.flip = flip
        self.bound = scale_bound
        self.net = nn.Sequential(nn.Linear(channels // 2, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, inverse=False):
        a, b = x.chunk(2, dim=-1)
        if self.flip:
            a, b = b, a
        scale, shift = self.net(a).chunk(2, dim=-1)
        scale = self.bound * torch.tanh(scale / self.bound)
        b = (b - shift) * torch.exp(-scale) if inverse else b * torch.exp(scale) + shift
        return torch.cat((b, a) if self.flip else (a, b), dim=-1)


class DinoAdapter(nn.Module):
    def __init__(self, channels=3072, z_dim=1024, hidden=256, layers=4):
        super().__init__()
        if channels % 2 or not 0 < z_dim < channels or layers < 2:
            raise ValueError('Need even channels, 0 < z_dim < channels and at least two couplings')
        self.spec = dict(channels=channels, z_dim=z_dim, hidden=hidden, layers=layers)
        self.z_dim = z_dim
        self.register_buffer('center', torch.zeros(channels))
        self.register_buffer('spread', torch.ones(channels))
        self.blocks = nn.ModuleList(Coupling(channels, hidden, bool(i % 2)) for i in range(layers))

    def encode(self, h):
        if h.shape[-1] != self.spec['channels']:
            raise ValueError('Activation channel count does not match adapter')
        x = (h.float() - self.center) / self.spread
        for block in self.blocks:
            x = block(x)
        return x[..., :self.z_dim], x[..., self.z_dim:]

    def decode(self, z, r):
        x = torch.cat((z, r), dim=-1)
        for block in reversed(self.blocks):
            x = block(x, inverse=True)
        return x * self.spread + self.center

    def forward(self, h):
        return self.encode(h)[0]

    def edit(self, h, direction, alpha):
        """Positive alpha removes the positive-minus-negative concept direction."""
        z, r = self.encode(h)
        return self.decode(z - alpha * direction, r)
