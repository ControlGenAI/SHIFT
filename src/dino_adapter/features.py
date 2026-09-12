"""Full-frame DINO features; both patch grids cover the same image extent."""
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor


def align_patches(patches, source_grid, target_grid):
    b, n, d = patches.shape
    if n != source_grid[0] * source_grid[1]:
        raise ValueError('Invalid source patch grid')
    grid = patches.transpose(1, 2).reshape(b, d, *source_grid)
    if tuple(source_grid) != tuple(target_grid):
        grid = F.interpolate(grid, size=target_grid, mode='bilinear', align_corners=False)
    return F.normalize(grid.flatten(2).transpose(1, 2), dim=-1)


class DinoFeatures:
    def __init__(self, model_name, size, device, revision=None):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32).to(
            device=device, dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.size, self.device = size, device
        if size % self.model.config.patch_size:
            raise ValueError('DINO size must be divisible by patch size')
        if getattr(self.model.config, 'num_register_tokens', 0):
            raise ValueError('Use a DINO checkpoint without register tokens')

    def tensor_features(self, rgb, target_grid=None):
        """Differentiable BCHW RGB [0,1] -> patches, CLS; no PIL/detach/clamp."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('Expected BCHW RGB')
        # A surrounding FLUX autocast must not silently change the DINO feature
        # space. Casting the final CLS back to float32 cannot undo that change.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            return self._tensor_features_fp32(rgb, target_grid)

    def _tensor_features_fp32(self, rgb, target_grid):
        rgb = F.interpolate(rgb.float(), (self.size, self.size), mode='bicubic',
                            align_corners=False, antialias=True)
        mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
        std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
        seq = self.model(pixel_values=(rgb - mean) / std).last_hidden_state.float()
        side = self.size // self.model.config.patch_size
        patches = align_patches(seq[:, 1:], (side, side), target_grid or (side, side))
        return patches, F.normalize(seq[:, 0], dim=-1)

    def cls_from_rgb(self, rgb):
        return self.tensor_features(rgb)[1]

    @torch.no_grad()
    def __call__(self, image, target_grid):
        rgb = pil_to_tensor(image.convert('RGB')).unsqueeze(0).to(self.device).float() / 255
        patches, cls = self.tensor_features(rgb, target_grid)
        return patches.cpu(), cls.cpu()
