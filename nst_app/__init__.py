"""Initializes the web app."""
from flask import Flask

app = Flask(__name__)

from nst_app import routes  # NOQA
