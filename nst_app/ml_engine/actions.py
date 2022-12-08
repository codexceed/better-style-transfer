import os
import time
from pathlib import Path

import torch

from nst_app.ml_engine.nst import neural_style_transfer

MODEL_FILE = "mnist_cnn.pt"
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _get_device():
    use_cuda = os.getenv("CUDA") == "True"

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_stylized_image(
    content_image: str,
    style_image: str,
    content_weight: float,
    style_weight: float,
    img_id: str,
) -> Path:
    """
    Train the neural net on MNIST.
    Args:
        content_image: Path to content image
        style_image: Path to style image
        content_weight: Weight (priority) for content
        style_weight: Weight (priority) for style
        img_id: Unique identifier for the output

    Returns:
        Path to stylized image
    """
    optimization_config = {
        "content_img_name": content_image,
        "style_img_name": style_image,
        "height": 400,
        "content_weight": content_weight,
        "style_weight": style_weight,
        "tv_weight": 1.0,
        "optimizer": "lbfgs",
        "model": "vgg19",
        "init_method": "content",
        "saving_freq": -1,
        "content_layer": -1,
        "style_layers": -1,
        "content_images_dir": str(DATA_DIR / "content-images"),
        "style_images_dir": str(DATA_DIR / "style-images"),
        "output_img_dir": str(DATA_DIR / img_id),
        "img_format": (4, ".jpg"),
    }

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    startTime = time.time()
    results_dir = neural_style_transfer(optimization_config)
    result_img = next(results_dir.glob("*.jpg"))
    endTime = time.time()
    print(f"Time: {endTime - startTime}")

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)

    return result_img
