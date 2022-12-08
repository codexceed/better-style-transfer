import os
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import LBFGS, Adam

from nst_app.ml_engine.utils import utils


def build_loss(
    neural_net,
    optimizing_img,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config,
):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[
        content_feature_maps_index
    ].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction="mean")(
        target_content_representation, current_content_representation
    )

    style_loss = 0.0
    current_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(current_set_of_feature_maps)
        if cnt in style_feature_maps_indices
    ]
    for gram_gt, gram_hat in zip(
        target_style_representation, current_style_representation
    ):
        style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = (
        config["content_weight"] * content_loss
        + config["style_weight"] * style_loss
        + config["tv_weight"] * tv_loss
    )

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(
    neural_net,
    optimizer,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config,
):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(
            neural_net,
            optimizing_img,
            target_representations,
            content_feature_maps_index,
            style_feature_maps_indices,
            config,
        )
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def neural_style_transfer(config):
    print(f"Content Image: {config['content_img_name']}")
    print(f"Style Image: {config['style_img_name']}")
    print(f"Content Weight: {config['content_weight']}")
    print(f"Style Weight: {config['style_weight']}")
    print(f"TV Weight: {config['tv_weight']}")
    print(f"Init Method: {config['init_method']}")
    print(f"Content Layer: {config['content_layer']}")
    print(f"Max Style Layer: {config['style_layers']}")

    content_img_path = os.path.join(
        config["content_images_dir"], config["content_img_name"]
    )
    style_img_path = os.path.join(config["style_images_dir"], config["style_img_name"])

    out_dir_name = (
        "combined_"
        + os.path.split(content_img_path)[1].split(".")[0]
        + "_"
        + os.path.split(style_img_path)[1].split(".")[0]
    )
    dump_path = os.path.join(config["output_img_dir"], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config["height"], device)
    style_img = utils.prepare_img(style_img_path, config["height"], device)

    # init image has same dimension as content image - this is a hard constraint
    if config["init_method"] == "uniform":
        uniform_noise_img = np.random.uniform(-90.0, 90.0, content_img.shape).astype(
            np.float32
        )
        init_img = torch.from_numpy(uniform_noise_img).float().to(device)
    elif config["init_method"] == "gaussian":
        gaussian_noise_img = np.random.normal(
            loc=0, scale=90.0, size=content_img.shape
        ).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config["init_method"] == "content":
        init_img = content_img
    elif config["init_method"] == "style":
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(
            style_img_path, np.asarray(content_img.shape[2:]), device
        )
        init_img = style_img_resized
    else:
        print("Error: Incorrect usage of init_method argument.")

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    (
        neural_net,
        content_feature_maps_index_name,
        style_feature_maps_indices_names,
    ) = utils.prepare_model(
        config["model"], config["content_layer"], config["style_layers"], device
    )
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[
        content_feature_maps_index_name[0]
    ].squeeze(axis=0)
    target_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(style_img_set_of_feature_maps)
        if cnt in style_feature_maps_indices_names[0]
    ]
    target_representations = [
        target_content_representation,
        target_style_representation,
    ]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    #
    # Start of optimization procedure
    #
    if config["optimizer"] == "adam":
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(
            neural_net,
            optimizer,
            target_representations,
            content_feature_maps_index_name[0],
            style_feature_maps_indices_names[0],
            config,
        )
        for cnt in range(num_of_iterations[config["optimizer"]]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(
                    f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                )
                utils.save_and_maybe_display(
                    optimizing_img,
                    dump_path,
                    config,
                    cnt,
                    num_of_iterations[config["optimizer"]],
                    should_display=False,
                )
    elif config["optimizer"] == "lbfgs":
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS(
            (optimizing_img,),
            max_iter=num_of_iterations["lbfgs"],
            line_search_fn="strong_wolfe",
        )
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(
                neural_net,
                optimizing_img,
                target_representations,
                content_feature_maps_index_name[0],
                style_feature_maps_indices_names[0],
                config,
            )
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(
                    f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                )
                utils.save_and_maybe_display(
                    optimizing_img,
                    dump_path,
                    config,
                    cnt,
                    num_of_iterations[config["optimizer"]],
                    should_display=False,
                )

            cnt += 1
            return total_loss

        optimizer.step(closure)
        utils.save_and_maybe_display(
            optimizing_img,
            dump_path,
            config,
            cnt,
            num_of_iterations[config["optimizer"]],
            should_display=False,
        )

    return Path(dump_path)
