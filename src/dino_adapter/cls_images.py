"""Image conversion shared by CLS objectives and the final SHIFT decode."""


def one_step_latents(latents, velocity, sigma):
    """FlowMatchEuler's update to sigma=0, including its dtype conversions.

    Preserve the native dtype of `sigma * velocity` and cast the updated sample
    BEFORE VAE unpack/scale/shift. Moving either cast changes BF16 forward values.
    """
    return (latents.float() + (-sigma) * velocity).to(velocity.dtype)


def decode_latents(pipe, latents, resolution):
    height, width = resolution
    unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    return pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]


def pipeline_rgb(pipe, decoded):
    processor = getattr(pipe, 'image_processor', None)
    if processor is None:
        # Analytic pipeline fixtures use the default Diffusers conversion.
        from diffusers.image_processor import VaeImageProcessor
        return VaeImageProcessor.denormalize(decoded).float()
    return processor.postprocess(decoded, output_type='pt').float()


def decode_final(pipe, latents, resolution):
    decoded = decode_latents(pipe, latents, resolution)
    return decoded, pipeline_rgb(pipe, decoded)
