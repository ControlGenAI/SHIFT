"""Invertible, spatially aligned DINO bridge for SHIFT image-block outputs."""
from .adapter import DinoAdapter
from .hooks import ImageBlockHook, MultiImageBlockHook

__all__ = ['DinoAdapter', 'ImageBlockHook', 'MultiImageBlockHook']
