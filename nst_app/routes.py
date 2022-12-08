"""Defines API endpoints."""
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import make_response, request, send_file

from nst_app import app

from .ml_engine.actions import DATA_DIR, get_stylized_image

IMG_MAP = "img_map.json"


@app.route("/status", methods=["GET"])
def status() -> Any:
    """Get service status"""
    response = make_response("ML Service up", 200)
    return response


@app.route("/stylize", methods=["POST"])
def stylize() -> Any:
    """Generate and return stylized image."""
    content_image, style_image = request.files.get("content_file"), request.files.get(
        "style_file"
    )
    content_weight, style_weight = request.values.get(
        "content_weight", 100000.0
    ), request.values.get("style_weight", 30000.0)

    if content_image:
        content_image.save(DATA_DIR / "content-images" / content_image.filename)
        content = content_image.filename
    else:
        content = request.values.get("content")

    if style_image:
        style_image.save(DATA_DIR / "style-images" / style_image.filename)
        style = style_image.filename
    else:
        style = request.values.get("style")

    if content is None or style is None:
        response = make_response(
            "Invalid request. Please ensure that you specify both content and style",
            400,
        )
        return response

    img_id = str(uuid.uuid4())
    out_dir_name = f"combined_{content.split('.')[0]}_{style.split('.')[0]}"
    style_thread = threading.Thread(
        target=get_stylized_image,
        args=(content, style, content_weight, style_weight, img_id),
    )
    style_thread.start()

    try:
        with open(IMG_MAP, "r") as f:
            img_map = json.load(f)
    except FileNotFoundError:
        img_map = {}

    img_map[img_id] = str(DATA_DIR / out_dir_name / img_id)

    with open(IMG_MAP, "w") as f:
        json.dump(img_map, f)

    return make_response(img_id, 202)


@app.route("/get-style", methods=["POST"])
def get_style_image() -> Any:
    """Get service status"""
    img_id = request.values.get("img_id")
    if not img_id:
        response = make_response(
            "Invalid request. Please specify 'img_id'",
            400,
        )
        return response

    with open(IMG_MAP, "r") as f:
        img_dir = Path(json.load(f)[img_id])

    img_file = next(img_dir.glob("*.jpg"))
    response = send_file(img_file, mimetype="image/gif")
    return response
