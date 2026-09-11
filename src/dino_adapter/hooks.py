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


class MultiImageBlockHook:
    """Same step-0 image edit on several dual-stream blocks in one generation.

    Each block keeps its own edit; the diffusion step counter is shared so every
    hooked block fires once at the selected step and nowhere else.
    """

    def __init__(self, transformer, edits, step, expected_tokens=None):
        if not edits:
            raise ValueError('Need at least one (block, edit) pair')
        indices = [block for block, _ in edits]
        if len(set(indices)) != len(indices):
            raise ValueError('Duplicate blocks in simultaneous intervention')
        self.target_step = step
        self.expected_tokens = expected_tokens
        self.step = 0
        self.hooks = []
        for block, edit in edits:
            if block < 0 or block >= len(transformer.transformer_blocks) or step < 0:
                raise ValueError('Invalid dual-stream block or step')
            self.hooks.append(dict(
                module=transformer.transformer_blocks[block],
                block_index=block,
                edit=edit,
                hits=0,
                handle=None,
            ))

    def __enter__(self):
        self.step = 0
        for item in self.hooks:
            item['hits'] = 0
            item['handle'] = item['module'].register_forward_hook(self._make_hook(item))
        return self

    def _make_hook(self, item):
        def _hook(module, inputs, output):
            if self.step != self.target_step:
                return output
            if item['hits']:
                raise RuntimeError('Multiple block calls at selected step; true CFG / repeated forwards unsupported')
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError('Expected dual-stream block output (txt_hidden, img_hidden)')
            txt, img = output
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError('Expected dual streams [B,S,C] and [B,N,C]')
            if self.expected_tokens is not None and img.shape[1] != self.expected_tokens:
                raise ValueError(f'Expected {self.expected_tokens} image tokens, got {img.shape[1]}')
            item['hits'] += 1
            if item['edit'] is None:
                return output
            changed = item['edit'](img)
            if changed.shape != img.shape or not torch.isfinite(changed).all():
                raise ValueError('Adapter returned invalid image activations')
            return txt, changed.to(device=img.device, dtype=img.dtype)
        return _hook

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        self.step = step + 1
        return callback_kwargs

    def __exit__(self, exc_type, exc, tb):
        for item in self.hooks:
            if item['handle'] is not None:
                item['handle'].remove()
                item['handle'] = None
        if exc_type is None:
            bad = [item['block_index'] for item in self.hooks if item['hits'] != 1]
            if bad:
                raise RuntimeError(f'Expected one hit per block at step {self.target_step}; bad blocks={bad}')
