"""Defines API endpoints."""
from typing import Any

from flask import make_response, request, send_file

from nst_app import app

from .ml_engine.actions import DATA_DIR, get_stylized_image


@app.route("/status", methods=["GET"])
def status() -> Any:
    """Get service status"""
    response = make_response("ML Service up", 201)
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

    result_img = get_stylized_image(content, style, content_weight, style_weight)
    response = send_file(result_img, mimetype="image/gif")
    return response
