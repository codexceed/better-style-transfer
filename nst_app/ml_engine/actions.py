import os
from pathlib import Path

import torch
import time

from nst_app.ml_engine.nst import neural_style_transfer

MODEL_FILE = "mnist_cnn.pt"


def _get_device():
    use_cuda = os.getenv("CUDA") == "True"

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_stylized_image(content_image: str, style_image: str) -> Path:
    """Train the neural net on MNIST."""
    optimization_config = {
        "content_img_name": content_image,
        "style_img_name": style_image,
        "height": 400,
        "content_weight": 100000.0,
        "style_weight": 30000.0,
        "tv_weight": 1.0,
        "optimizer": "lbfgs",
        "model": "vgg19",
        "init_method": "content",
        "saving_freq": -1,
        "content_layer": -1,
        "style_layers": -1,
        "content_images_dir": "/Users/sarthak/Repos/better-style-transfer/data/content-images",
        "style_images_dir": "/Users/sarthak/Repos/better-style-transfer/data/style-images",
        "output_img_dir": "/Users/sarthak/Repos/better-style-transfer/data/output-images",
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
