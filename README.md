# upscaler

A stand-alone Python script that implements the [Colab notebook "Stable Diffusion Upscaler Demo"](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4).

This script uses Stable Diffusion v1-5 internally (see [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)), along with the Upscaler model in the aforementioned Colab notebook.

## Installation
### 
1. On a GPU-equipped environment, create a virtual Python enviroment for Python 3.10
1. Activate your virtual Python environment.
1. Use `environment.yml` (`conda env create -f environment.yml`) or `requirements.txt` (`pip install -r requirements.txt`) to install the dependencies.

## Usage
#### Script arguments
* `--prompt` | `-p`: text prompt for guiding the image generation, required. NOTE that the prompt may be truncated as CLIP can only handle sequences up to 77 tokens.
* `--output-file` | `-o`: path for ouputting the generated image file, required. NOTE that this script only supports image formats of JPEG and PNG.
* `--seed` | `-s`: seed value for global random state, optional; defaults to `0`, which will use the current time as the seed

### Execution
Run this script like so:

    python upscaler.py --prompt [prompt for image generation] --output-file [path to output image file, either JPEG or PNG] --seed [some integer seed value]


## Example

Here is an example of a 1024 x 1024 pixel upscaled image generated with Stable Diffusion v1-5 along with the Upscaler in the ["Stable Diffusion Upscaler Demo"](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4) Colab notebook. 

The prompt comes directly from the [Stable Diffusion prompt: a definitive guide](https://stable-diffusion-art.com/prompt-guide/).

    python upscaler.py --prompt "dog, autumn in paris, ornate, beautiful, atmosphere, vibe, mist, smoke, fire, chimney, rain, wet, pristine, puddles, melting, dripping, snow, creek, lush, ice, bridge, forest, roses, flowers, by stanley artgerm lau, greg rutkowski, thomas kindkade, alphonse mucha, loish, norman rockwell" --seed 8675309 --output-file "example.png"

![alt example generated image](https://github.com/buruzaemon/upscaler/blob/main/images/example.png)
