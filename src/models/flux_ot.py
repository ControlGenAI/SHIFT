"""
OT-focused FLUX pipeline wrapper.

This file keeps the full generation logic from src.models.flux.FluxPipeline,
but provides a dedicated pipeline class that is explicitly intended for
text pooled low-space OT steering.
"""

from typing import Any, Dict

from src.models.flux import FluxPipeline as _BaseFluxPipeline
from src.utils.utils import pooled_steering_mode


class FluxOTPipeline(_BaseFluxPipeline):
    """
    Drop-in replacement for FluxPipeline with stricter intent:
    txt_steering should use a pooled advanced vector (subspace_mean / monge).
    """

    def __call__(self, *args, **kwargs):
        txt_steering: Dict[str, Any] = kwargs.get("txt_steering", {"vector": None})
        vector = txt_steering.get("vector") if isinstance(txt_steering, dict) else None
        if vector is not None and pooled_steering_mode(vector) == "raw":
            raise ValueError(
                "FluxOTPipeline expects advanced pooled txt steering vector "
                "with pooled_steering_mode in {'subspace_mean', 'monge'}. "
                "Use src.models.flux.FluxPipeline for legacy text steering."
            )
        return super().__call__(*args, **kwargs)


# Keep import style consistent with existing code.
FluxPipeline = FluxOTPipeline

