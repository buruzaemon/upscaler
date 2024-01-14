# upscaler
A stand-alone Python script that implements the [Colab notebook "Stable Diffusion Upscaler Demo"](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4).

This script uses Stable Diffusion v1-5 internally (see [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)), along with the Upscaler model in the aforementioned Colab notebook.

### Installation
1. Create a Python virtual environment for Python 3.10.
1. Use `environment.yml` (`conda`) or `requirements.txt` (`pip`) to install the dependencies.

### Usage
#### Script arguments
* `--prompt` | `-p`: text prompt for guiding the image generation, required. NOTE that the prompt may be truncated as CLIP can only handle sequences up to 77 tokens.
* `--output-file` | `-o`: path for ouputting the generated image file, required. NOTE that this script only supports image formats of JPEG and PNG.
* `--seed` | `-s`: seed value for global random state, optional; defaults to `0`, which will use the current time as the seed
