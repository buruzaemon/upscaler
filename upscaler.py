# -*- coding: utf-8 -*-
"""
upscaler.py

Condensed script version of the Stable Diffusion Upscaler Demo Colab notebook
(https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4)
"""
import argparse
import hashlib
import logging
import numpy as np
import os
import re
import requests
import sys
import time

from pathlib import Path

from diffusers import AutoencoderKL, AutoPipelineForText2Image

import torch
import torch.nn.functional as F
from torch import nn

import k_diffusion as K
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)


# URLs for configuration and model of the Upscaler in the Colab notebook
UPSCALER_CONFIG_PATH = 'https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json'
UPSCALER_MODEL_PATH = 'https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'

# Model name on HF for the CLIP tokenizer and embedder used in the Upscaler
# per the Colab notebook
CLIP_MODEL_NAME = 'openai/clip-vit-large-patch14'

# Model name on HF for Stable Diffusion v1.5
# Used in both the variational autoencoder in the Upscaler,
# as well as the HF pipeline for generating the initial image
SD_MODEL_NAME = 'runwayml/stable-diffusion-v1-5'


RETURN_OK = 0
RETURN_NG = 1


##
# from 2c. Fetch models
#
def fetch(url_or_path):
    if url_or_path.startswith('http:') or url_or_path.startswith('https:'):
        _, ext = os.path.splitext(os.path.basename(url_or_path))
        cachekey = hashlib.md5(url_or_path.encode('utf-8')).hexdigest()
        cachename = f'{cachekey}{ext}'
        if not os.path.exists(f'cache/{cachename}'):
            os.makedirs('tmp', exist_ok=True)
            os.makedirs('cache', exist_ok=True)
            res = requests.get(url_or_path)
            res.raise_for_status()
            with open(f'tmp/{cachename}', mode='wb') as fout:
                fout.write(res.content)
            os.rename(f'tmp/{cachename}', f'cache/{cachename}')
        return f'cache/{cachename}'
    return url_or_path


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1., embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode='nearest') * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            input,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs
        )


def make_upscaler_model(config_path, model_path, pooler_dim=768, train=False, device='cpu'):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config['model']['sigma_data'],
        embed_dim=config['model']['mapping_cond_dim'] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_ema'])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


##
# from Set up some functions and load the text encoder
#
class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
          # Shortcut for when we don't need to run both.
          if self.cond_scale == 0.0:
            c_in = self.uc
          elif self.cond_scale == 1.0:
            c_in = c
          return self.inner_model(x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in)

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_length=True, return_overflowing_tokens=False,
                                 padding='max_length', return_tensors='pt')
        input_ids = tok_out['input_ids'][indexer]
        attention_mask = 1 - tok_out['attention_mask'][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPTextModel, logging
        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = self.transformer.eval().requires_grad_(False).to(device)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(input_ids=input_ids.to(self.device), output_hidden_states=True)
        return clip_out.hidden_states[-1], cross_cond_padding.to(self.device), clip_out.pooler_output


def main(args):

    # Model configuration values
    #SD_C = 4 # Latent dimension
    #SD_F = 8 # Latent patch size (pixels per latent)
    SD_Q = 0.18215 # sd_model.scale_factor; scaling for latents in first stage models

    # parameters
    #num_samples = 1 #@param {type: 'integer'}
    batch_size = 1 #@param {type: 'integer'}
    guidance_scale = 1 #@param {type: 'slider', min: 0.0, max: 10.0, step:0.5}
    noise_aug_level = 0 #@param {type: 'slider', min: 0.0, max: 0.6, step:0.025}
    noise_aug_type = 'gaussian' #@param ["gaussian", "fake"]
    sampler = 'k_dpm_adaptive' #@param ["k_euler", "k_euler_ancestral", "k_dpm_2_ancestral", "k_dpm_fast", "k_dpm_adaptive"]
    steps = 50 #@param {type: 'integer'}
    tol_scale = 0.25 #@param {type: 'number'}
    eta = 1.0 #@param {type: 'number'}

    device = torch.device('cuda')

    try:
        # 2c. Fetch models
        logger.info('Creating models for Upscaler (including VAE part)')
        model_up = make_upscaler_model(
            fetch(UPSCALER_CONFIG_PATH),
            fetch(UPSCALER_MODEL_PATH)
        )

        # Stable Diffusion vae used in upscaling process
        sd_vae = AutoencoderKL.from_pretrained(
            SD_MODEL_NAME,
            subfolder='vae'
        )

        # CLIP tokenizer transform & embedder used in
        tok_up = CLIPTokenizerTransform()
        text_encoder_up = CLIPEmbedder(device=device)
        logger.info("tokenizers set up")

        # Stable Diffusion HF pipeline for generating the initial image
        logger.info('Creating Stable Diffusion HF pipeline for initial image generation')
        sd_pipeline = AutoPipelineForText2Image.from_pretrained(
            SD_MODEL_NAME,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        # Load models on GPU
        logger.info(f'Moving models to {device} device')
        model_up = model_up.to(device)
        sd_vae = sd_vae.to(device)
        sd_pipeline = sd_pipeline.to(device)


        # 3c. Run the model
        @torch.no_grad()
        def condition_up(prompts):
            return text_encoder_up(tok_up(prompts))

        @torch.no_grad()
        def upscale_image(prompt, seed=0):
            timestamp = int(time.time())
            if not seed:
                logger.info('No seed was provided, using the current time.')
                seed = timestamp
            logger.info(f'Generating with seed={seed}')
            seed_everything(seed)

            # create initial image with stable diffusion
            input_image = sd_pipeline(prompt, num_inference_steps=25).images[0]
            image = input_image
            image = TF.to_tensor(image).to(device) * 2 - 1

            uc = condition_up([""])
            c = condition_up([prompt])

            # >>>
            low_res_latent = sd_vae.encode(image.unsqueeze(0)).latent_dist.sample() * SD_Q

            [_, C, H, W] = low_res_latent.shape

            # Noise levels from stable diffusion.
            sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

            model_wrap = CFGUpscaler(model_up, uc, cond_scale=guidance_scale)
            low_res_sigma = torch.full([1], noise_aug_level, device=device)
            x_shape = [batch_size, C, 2*H, 2*W]

            def do_sample(noise, extra_args):
                # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
                sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp().to(device)
                if sampler == 'k_euler':
                    return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
                elif sampler == 'k_euler_ancestral':
                    return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
                elif sampler == 'k_dpm_2_ancestral':
                    return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
                elif sampler == 'k_dpm_fast':
                    return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, steps, extra_args=extra_args, eta=eta)
                elif sampler == 'k_dpm_adaptive':
                    sampler_opts = dict(s_noise=1., rtol=tol_scale * 0.05, atol=tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0)
                    return K.sampling.sample_dpm_adaptive(model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=eta, **sampler_opts)

            if noise_aug_type == 'gaussian':
                latent_noised = low_res_latent + noise_aug_level * torch.randn_like(low_res_latent)
            elif noise_aug_type == 'fake':
                latent_noised = low_res_latent * (noise_aug_level ** 2 + 1)**0.5
            extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}
            noise = torch.randn(x_shape, device=device)
            up_latents = do_sample(noise, extra_args)

            pixels = sd_vae.decode(up_latents/SD_Q)
            pixels = pixels.sample.add(1).div(2).clamp(0,1).squeeze()

            return TF.to_pil_image(pixels)

        logger.info('Starting process...')
        logger.info(f'Prompt:\n  {args.prompt}')
        logger.info(f'Seed: {args.seed}')
        logger.info(f'Output file format: {args.format}')

        output_image = upscale_image(args.prompt, args.seed)
        output_image.save(args.output_file, format=args.format)

        logger.info(f'Created upscaled image file: {args.output_file}')

        retval = RETURN_OK
    except:
        logger.exception('Unexpected error in script execution')
        retval = RETURN_NG
    finally:
        return retval

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='upscaler.py: Stable Diffusion upscaler')
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default=None,
        required=True,
        help='Text prompt for generating an image.'
    )
    parser.add_argument(
        '--output-file', '-o',
        type=str,
        default=None,
        required=True,
        help='Path for output image file, JPEG or PNG format',
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=0,
        required=False,
        help='seed value',
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if re.match('.+\.(jpg|jpeg|png)', args.output_file, re.IGNORECASE):
        fpath = Path(args.output_file)
        fext = fpath.name.split('.')[-1].lower()
        if fext == 'png':
            image_format = 'PNG'
        else:
            image_format = 'JPEG'
        setattr(args, 'format', image_format)
    else:
        raise ValueError('Only JPEG and PNG formats are supported!')

    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s'
    )
    args = parse_args()
    stime = time.time()
    retval = main(args)
    etime = time.time()
    logger.info(f'Script completed in {etime-stime:0.1f} s')
    sys.exit(retval)

