"""One dual-stream post-block image hook, gated by diffusion step.

Compatible with SHIFT's FluxPipeline callback_on_step_end. No text steering.
The block returns (encoder_hidden_states, hidden_states); only the second one,
the image tokens, is ever read or replaced.
"""
import torch


class ImageBlockHook:
    def __init__(self, transformer, block, step, edit=None, capture=True, expected_tokens=None):
        if block < 0 or block >= len(transformer.transformer_blocks) or step < 0:
            raise ValueError('Invalid dual-stream block or step')
        self.block = transformer.transformer_blocks[block]
        self.block_index = block
        self.target_step = step
        self.edit = edit
        self.capture = capture
        self.expected_tokens = expected_tokens
        self.step = 0
        self.hits = 0
        self.captured = None
        self.handle = None

    def __enter__(self):
        self.step, self.hits, self.captured = 0, 0, None
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        if self.step != self.target_step:
            return output
        if self.hits:
            raise RuntimeError('Multiple block calls at selected step; true CFG / repeated forwards unsupported')
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Expected dual-stream block output (txt_hidden, img_hidden)')
        txt, img = output
        if img.ndim != 3:
            raise ValueError('Expected all image tokens [B,N,C]')
        if txt.ndim != 3:
            raise ValueError('Expected a text stream [B,S,C] alongside the image tokens')
        # The block returns (text, image); a swap would silently steer the prompt.
        if self.expected_tokens is not None and img.shape[1] != self.expected_tokens:
            raise ValueError(f'Expected {self.expected_tokens} image tokens, got {img.shape[1]}; '
                             'dual-stream output order or spatial grid is not what the config assumes')
        self.hits += 1
        if self.capture:
            self.captured = img.detach().to('cpu', copy=True)
        if self.edit is None:
            return output
        changed = self.edit(img)
        if changed.shape != img.shape or not torch.isfinite(changed).all():
            raise ValueError('Adapter returned invalid image activations')
        return txt, changed.to(device=img.device, dtype=img.dtype)

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        self.step = step + 1
        return callback_kwargs

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if exc_type is None and self.hits != 1:
            raise RuntimeError(f'Expected one intervention/capture, observed {self.hits}')
