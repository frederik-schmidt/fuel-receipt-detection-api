import os
from functools import wraps
from typing import Callable, Union

from flask import Flask, request, redirect, url_for, render_template, make_response
from werkzeug.security import check_password_hash as check_hash

from scan_receipt import scan_receipt_main
from settings import ALLOWED_EXTENSIONS, USER_HASH, PASS_HASH, UPLOAD_FOLDER
from utils import allowed_file, format_image_path

app = Flask(__name__)

app.secret_key = "secret key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def auth_required(func: Callable) -> Callable:
    """Decorator which hashes the provided username and password and compares with the
    actual hashed username and password. If both match, the decorated function gets
    executed, otherwise an error 401 gets returned."""
    @wraps(func)
    def decorator(*args, **kwargs):
        auth = request.authorization
        if auth and check_hash(USER_HASH, auth.username) and check_hash(PASS_HASH, auth.password):
            return func(*args, **kwargs)
        else:
            return make_response(
                "Unauthorized Access",
                401,
                {"WWW-Authenticate": "Basic realm='Login Required'"})
    return decorator


def request_image_scan() -> Union[str, tuple]:
    """Handles an HTTP request for an image scan and returns the appropriate result."""
    if request.method == "POST":
        if "file" not in request.files:
            return "Request must contain a file"
        file = request.files["file"]
        # If no file is selected, the browser submits an empty file without a filename
        if file.filename == "":
            return "Request must contain a file"
        if not file or not allowed_file(file.filename):
            allowed_extensions = ", ".join(ALLOWED_EXTENSIONS)
            return f"File type must be one of the following: {allowed_extensions}"
        else:
            img_path = format_image_path(file.filename)
            file.save(img_path)  # Save image to UPLOAD_FOLDER
            scan_result, filepath_scanned = scan_receipt_main(img_path=img_path)
            return scan_result, filepath_scanned


@app.route("/api/v1/scan", methods=["POST"])
@auth_required
def request_api() -> dict:
    response = request_image_scan()
    if isinstance(response, tuple):
        scan_result, _ = response
    else:
        scan_result = response
    return scan_result


@app.route("/", methods=["GET", "POST"])
@auth_required
def index_page():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        response = request_image_scan()
        if isinstance(response, tuple):
            scan_result, filepath_scanned = response
            return render_template(
                "index.html",
                output=scan_result,
                filename=os.path.split(filepath_scanned)[1]
            )
        else:
            scan_result = response
            return render_template("index.html", output=scan_result)


@app.route("/display/<filename>")
@auth_required
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename), code=301)
