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
    """Invertible per-token map. z is the first z_dim output coordinates.

    scale_bound and permute expose the two knobs that matter for how much of the
    DINO target this stack can represent. Fixed channel permutations between
    couplings (as in RealNVP/Glow) are exactly invertible and add no parameters,
    but they are not free: out/capacity_probe.json shows they help only when
    scale_bound is also raised, and hurt at the default bound. Defaults therefore
    stay at the originally committed architecture and the choice is made from the
    measured per-block ablation in out/arch_ablation.json.
    """

    def __init__(self, channels=3072, z_dim=1024, hidden=256, layers=4,
                 scale_bound=1.5, permute=False, perm_seed=0):
        super().__init__()
        if channels % 2 or not 0 < z_dim < channels or layers < 2:
            raise ValueError('Need even channels, 0 < z_dim < channels and at least two couplings')
        if scale_bound <= 0:
            raise ValueError('scale_bound must be positive')
        self.spec = dict(channels=channels, z_dim=z_dim, hidden=hidden, layers=layers,
                         scale_bound=scale_bound, permute=permute, perm_seed=perm_seed)
        self.z_dim = z_dim
        self.permute = permute
        self.register_buffer('center', torch.zeros(channels))
        self.register_buffer('spread', torch.ones(channels))
        self.blocks = nn.ModuleList(
            Coupling(channels, hidden, bool(i % 2), scale_bound) for i in range(layers))
        generator = torch.Generator().manual_seed(perm_seed)
        for i in range(layers):
            order = (torch.randperm(channels, generator=generator) if permute
                     else torch.arange(channels))
            self.register_buffer(f'perm_{i}', order)
            self.register_buffer(f'unperm_{i}', torch.argsort(order))

    def encode(self, h):
        if h.shape[-1] != self.spec['channels']:
            raise ValueError('Activation channel count does not match adapter')
        x = (h.float() - self.center) / self.spread
        for i, block in enumerate(self.blocks):
            x = block(x[..., getattr(self, f'perm_{i}')])
        return x[..., :self.z_dim], x[..., self.z_dim:]

    def decode(self, z, r):
        if z.shape[-1] != self.z_dim or r.shape[-1] != self.spec['channels'] - self.z_dim:
            raise ValueError('z/r split does not match adapter channels')
        x = torch.cat((z, r), dim=-1)
        for i in reversed(range(len(self.blocks))):
            x = self.blocks[i](x, inverse=True)[..., getattr(self, f'unperm_{i}')]
        return x * self.spread + self.center

    def forward(self, h):
        return self.encode(h)[0]

    def edit(self, h, direction, alpha):
        """Positive alpha removes the positive-minus-negative concept direction."""
        z, r = self.encode(h)
        if direction.shape[-1] != self.z_dim:
            raise ValueError('Direction dimension does not match z')
        if direction.ndim > 1 and direction.shape[-2] != z.shape[-2]:
            raise ValueError('Direction token count does not match the activation grid')
        return self.decode(z - alpha * direction.to(z.device), r)
