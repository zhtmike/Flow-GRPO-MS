from mindone.diffusers import (
    AutoencoderKL,
    AutoencoderKLWan,
    SD3Transformer2DModel,
    WanTransformer3DModel,
)
from mindone.transformers import (
    CLIPTextModelWithProjection,
    T5EncoderModel,
    UMT5EncoderModel,
)
from transformers import CLIPTextConfig, T5Config, UMT5Config

from .trainer import StableDiffusion3PipelineWithSDELogProb, WanPipelineWithSDELogProb


def init_sd3_debug_pipeline(model_path: str) -> StableDiffusion3PipelineWithSDELogProb:
    """Init the pipeline with models containing only 1 layers for easier debugging & faster creating"""
    vae_config = AutoencoderKL.load_config(model_path, subfolder="vae")
    vae_config["layers_per_block"] = 1
    vae = AutoencoderKL.from_config(vae_config)

    transformer_config = SD3Transformer2DModel.load_config(
        model_path, subfolder="transformer"
    )
    transformer_config["num_layers"] = 1
    transformer = SD3Transformer2DModel.from_config(transformer_config)

    text_encoder_config = CLIPTextConfig.from_pretrained(
        model_path, subfolder="text_encoder"
    )
    text_encoder_config.num_hidden_layers = 1
    text_encoder = CLIPTextModelWithProjection(text_encoder_config)

    text_encoder_2_config = CLIPTextConfig.from_pretrained(
        model_path, subfolder="text_encoder_2"
    )
    text_encoder_2_config.num_hidden_layers = 1
    text_encoder_2 = CLIPTextModelWithProjection(text_encoder_2_config)

    text_encoder_3_config = T5Config.from_pretrained(
        model_path, subfolder="text_encoder_3"
    )
    text_encoder_3_config.num_layers = 1
    text_encoder_3 = T5EncoderModel(text_encoder_3_config)

    pipeline = StableDiffusion3PipelineWithSDELogProb.from_pretrained(
        model_path,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
    )
    return pipeline


def init_wan21_debug_pipeline(model_path: str) -> WanPipelineWithSDELogProb:
    """Init the pipeline with models containing only 1 layers for easier debugging & faster creating"""
    vae_config = AutoencoderKLWan.load_config(model_path, subfolder="vae")
    vae_config["layers_per_block"] = 1
    vae = AutoencoderKLWan.from_config(vae_config)

    transformer_config = WanTransformer3DModel.load_config(
        model_path, subfolder="transformer"
    )
    transformer_config["num_layers"] = 1
    transformer = WanTransformer3DModel.from_config(transformer_config)

    text_encoder_config = UMT5Config.from_pretrained(
        model_path, subfolder="text_encoder"
    )
    text_encoder_config.num_layers = 1
    text_encoder = UMT5EncoderModel(text_encoder_config)

    pipeline = WanPipelineWithSDELogProb.from_pretrained(
        model_path,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
    )
    return pipeline
