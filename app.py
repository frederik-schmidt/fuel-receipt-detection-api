import os
from functools import wraps
from typing import Callable, Union

import requests
from flask import Flask, request, render_template, make_response
from werkzeug.security import check_password_hash as check_hash
from werkzeug.wrappers import Response

from scan_receipt import scan_receipt_main
from settings import API_USERNAME_HASH, API_PASSWORD_HASH, GCS_BUCKET_NAME, GCS_BUCKET_SUBFOLDER, UPLOAD_FOLDER
from utils import allowed_file, format_image_path, response_code_to_text, wrap_into_json, upload_to_gcs

app = Flask(__name__)

app.secret_key = "secret key"
app.config["JSON_SORT_KEYS"] = False
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
        response_code = 401
        response_text = response_code_to_text(response_code)
        response_text = wrap_into_json(response_text)
        callback = {"WWW-Authenticate": "Basic realm='Login Required'"}
        return make_response(response_text, response_code, callback)
    return decorator


def request_image_scan() -> Union[int, tuple]:
    """Handles an HTTP request for an image scan and returns the appropriate result."""
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
            # Save receipt to UPLOAD_FOLDER
            img_path = format_image_path(file.filename)
            file.save(img_path)

            # Scan receipt
            scan_result, img_path_scanned = scan_receipt_main(img_path=img_path)

            # Upload scanned receipt to GCS
            img_name_scanned = os.path.split(img_path_scanned)[1]
            remote_filename = f"{GCS_BUCKET_SUBFOLDER}/{img_name_scanned}"
            upload_to_gcs(
                bucket_name=GCS_BUCKET_NAME,
                local_filename=img_path_scanned,
                remote_filename=remote_filename,
            )
            gcs_base_uri = "https://storage.cloud.google.com/"
            gcs_uri = gcs_base_uri + GCS_BUCKET_NAME + "/" + remote_filename

            # Remove temp files
            for file in [img_path, img_path_scanned]:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass

            return 200, scan_result, gcs_uri
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
        response_text = response_code_to_text(response_code)
    return make_response(wrap_into_json(response_text), response_code)


@app.route("/", methods=["GET", "POST"])
@auth_required
def render_web_interface():
    """Handles an HTTP request to the web interface."""
    if request.method == "POST":
        response = request_image_scan()
        if isinstance(response, tuple):
            response_text, gcs_uri = response[1], response[2]
            return render_template("index.html", output=response_text, gcs_uri=gcs_uri)
        else:
            response_text = response_code_to_text(response)
            return render_template("index.html", output=response_text)
    else:
        return render_template("index.html")


@app.errorhandler(404)
def url_not_found(e):
    """Creates custom response for requests to unknown URLs."""
    response_code = 404
    response_text = response_code_to_text(response_code)
    response_text = wrap_into_json(response_text)
    return make_response(response_text, response_code)


@app.errorhandler(405)
def url_not_found(e):
    """Creates custom response for requests with not allowed method."""
    response_code = 405
    response_text = response_code_to_text(response_code)
    response_text = wrap_into_json(response_text)
    return make_response(response_text, response_code)


@app.errorhandler(500)
def internal_server_error(e):
    """Creates custom response for requests with internal server error."""
    response_code = 400
    response_text = response_code_to_text(response_code)
    response_text = wrap_into_json(response_text)
    return make_response(response_text, response_code)
