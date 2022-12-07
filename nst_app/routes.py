"""Defines API endpoints."""
from typing import Any

from flask import make_response, request, send_file

from nst_app import app

from .ml_engine.actions import get_stylized_image


@app.route("/status", methods=["GET"])
def status() -> Any:
    """Get service status"""
    response = make_response("ML Service up", 201)
    return response


@app.route("/stylize", methods=["POST"])
def stylize() -> Any:
    """Generate and return stylized image."""
    content, style = request.values.get("content"), request.values.get("style")
    result_img = get_stylized_image(content, style)
    response = send_file(result_img, mimetype="image/gif")
    return response
