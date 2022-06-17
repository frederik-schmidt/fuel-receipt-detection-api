import os
from functools import wraps
from typing import Callable, Union

import requests
from flask import Flask, request, redirect, url_for, render_template, make_response
from werkzeug.security import check_password_hash as check_hash
from werkzeug.wrappers import Response

from scan_receipt import scan_receipt_main
from settings import API_USERNAME_HASH, API_PASSWORD_HASH, UPLOAD_FOLDER
from utils import allowed_file, format_image_path, convert_response_code_to_text

app = Flask(__name__)

app.secret_key = "secret key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def auth_required(func: Callable) -> Union[Callable, Response]:
    """Decorator which hashes the provided username and password and compares with the
    actual hashed username and password. If both are present and match, the decorated
    function gets executed, otherwise an error 401 gets returned."""
    @wraps(func)
    def decorator(*args, **kwargs):
        auth = request.authorization
        if auth and auth.username and auth.password:
            username_valid = check_hash(API_USERNAME_HASH, auth.username)
            password_valid = check_hash(API_PASSWORD_HASH, auth.password)
            if username_valid and password_valid:
                return func(*args, **kwargs)
        return make_response(
            convert_response_code_to_text(401),
            401,
            {"WWW-Authenticate": "Basic realm='Login Required'"})
    return decorator


def request_image_scan() -> Union[int, tuple]:
    """Handles an HTTP request for an image scan and returns the appropriate result."""
    if request.method == "POST":
        if "file" not in request.files:
            return 422
        file = request.files["file"]
        # If no file is selected, the browser submits an empty file without a filename
        if not file or file.filename == "":
            return 422
        if not allowed_file(file.filename):
            return 415
        else:
            try:
                img_path = format_image_path(file.filename)
                file.save(img_path)  # Save image to UPLOAD_FOLDER
                scan_result, filepath_scanned = scan_receipt_main(img_path=img_path)
                return 200, scan_result, filepath_scanned
            except requests.exceptions.ConnectionError:
                return 400


@app.route("/api/v1/scan", methods=["POST"])
@auth_required
def request_scan() -> Response:
    """Handles an HTTP request to the scan endpoint."""
    response = request_image_scan()
    if isinstance(response, tuple):
        response_code, response_text = response[0], response[1]
    else:
        response_code = response
        response_text = convert_response_code_to_text(response_code)
    return make_response(response_text, response_code)


@app.route("/", methods=["GET", "POST"])
@auth_required
def request_index():
    """Handles an HTTP request to the index page."""
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        response = request_image_scan()
        if isinstance(response, tuple):
            response_text, filepath_scanned = response[1], response[2]
            return render_template(
                "index.html",
                output=response_text,
                filename=os.path.split(filepath_scanned)[1]
            )
        else:
            response_text = convert_response_code_to_text(response)
            return render_template("index.html", output=response_text)


@app.route("/display/<filename>")
@auth_required
def display_image(filename):
    """Displays an image."""
    return redirect(url_for("static", filename="uploads/" + filename), code=301)
