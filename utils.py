import json
import os

from werkzeug.utils import secure_filename

from settings import ALLOWED_EXTENSIONS, UPLOAD_FOLDER


def allowed_file(filepath: str) -> bool:
    """Returns True if the file name provided is valid and has an allowed extension."""
    return "." in filepath and filepath.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_from_json(filepath: str) -> list:
    """Returns the content of a json file as a list of dictionaries."""
    with open(filepath, "r") as f:
        data = json.load(f)
        f.close()
    return data


def write_to_json(filepath: str, data: dict) -> None:
    """Writes the content of a dictionary to a json file."""
    with open(filepath, "w") as f:
        json.dump(data, f)
        f.close()


def format_image_path(img_path, option=None):
    """Formats the image file name and changes the path to UPLOAD_FOLDER. Optionally, a
    json path or an image path with added '_processed' can be returned.

    >>> img_path_t = './data/IMG 1234.jpg'
    >>> format_image_path(img_path_t)
    './static/uploads/IMG_1234.jpg'
    >>> format_image_path(img_path_t, option="json")
    './static/uploads/IMG_1234.json'
    >>> format_image_path(img_path_t, option="scanned")
    './static/uploads/IMG_1234_scanned.jpg'
    """
    filename = os.path.split(img_path)[1]
    filename = secure_filename(filename)

    if option == "json":
        filename = filename.rsplit(".", 1)[0] + ".json"
    elif option == "scanned":
        img_name_bas, file_ext = filename.rsplit(".", 1)[0], filename.rsplit(".", 1)[1]
        filename = img_name_bas + "_scanned." + file_ext
    else:
        pass

    return os.path.join(UPLOAD_FOLDER, filename)
