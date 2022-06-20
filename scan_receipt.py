import io
import os
import re

import dateutil
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToDict

from settings import ALLOWED_EXTENSIONS, API_SECRET, PRODUCTION
from utils import allowed_file, read_from_json, write_to_json, format_image_path

matplotlib.use("Agg")


def detect_text(image_path: str) -> list:
    """Detects text in an image using the Google Cloud Vision API.
    If the json of the image is available, the API will not be called again.
    Docs: https://cloud.google.com/vision/docs/ocr#vision_text_detection-python"""

    if not allowed_file:
        raise ValueError(f"Supported file types are: {', '.join(ALLOWED_EXTENSIONS)}")

    json_path = format_image_path(image_path, option="json")

    if os.path.isfile(json_path) and not PRODUCTION:
        response_text = read_from_json(json_path)
        return response_text

    else:
        client = vision.ImageAnnotatorClient.from_service_account_info(info=API_SECRET)

        with io.open(image_path, "rb") as image_file:
            content = image_file.read()
        image = types.Image(content=content)
        response = client.text_detection(image=image)
        response_dict = MessageToDict(response._pb)
        response_text = response_dict["textAnnotations"]

        if not PRODUCTION:
            write_to_json(json_path, response_text)

        return response_text


def format_text(text_raw: list) -> str:
    """Formats the text detected in an image."""
    text = text_raw[0]["description"]
    text = text.replace("\n", " ")
    return text


def get_regex_patterns() -> dict:
    """Returns the patterns for each of the features to extract.
    Explanation:
        \d+     - one or more digits
        \d{2}   - exactly two digits
        \d{1,3} - one to three digits
        (?:.|,) - optional comma or dot
        [.|,]   - mandatory comma or dot
        \s?     - optional whitespace
    """
    date = r"(?:\s|\n)\d{1,2}[.-]\d{2}[.-]\d{2,4}(?:\s|\n)"
    time = r"\d{1,2}:\d{2}"
    fuel_type = r"(?:diesel|DIESEL|Diesel|super|SUPER|Super)"
    volume = r"(?:Liter|liter|l|1)"  # Sometimes l gets recognized as 1
    currency = r"(?:€|EUR|Eur|eur)"
    per = r"(?:/|PRO|pro|PER|per)"
    tax_rate = r"\d+[.|,]\d+\s?%"
    amount = r"\d{1,3}[.|,]\d{1,2}"
    unit_amount = r"\d{1}[.|,]\d{3}"

    regex_pattern = {
        "date": date,
        "time": time,
        "fuel_type": fuel_type,
        "tax_rate": tax_rate,
        "amount": f"(?:\s|\n){amount}\s{volume}",
        "price_per_unit": [
            f"{unit_amount}\s{currency}\s?{per}\s?{volume}",
            f"{currency}\s{unit_amount}\s?{per}\s?{volume}"
        ],
        "price_incl_tax": [
            f"{amount}\s{currency}(?:\s|\n)",
            f"{currency}\s{amount}(?:\s|\n)",
        ],
    }
    return regex_pattern


def standardize_date(date: str) -> str:
    """Standardizes a date string to YYYY-MM-DD."""
    return dateutil.parser.parse(date, dayfirst=True, fuzzy=True).strftime("%Y-%m-%d")


def preprocess_float_matches(matches: list) -> list:
    """Extracts the numbers from a list of strings and converts it into a list of floats.
     Each string must contain only one float."""
    # Replace dot with comma
    matches_form = [i.replace(",", ".") for i in matches]
    # Extract numbers and convert to float
    matches_form = [float(re.sub("[^0-9.]", "", i)) for i in matches_form]
    return matches_form


def extract_coordinates_in_image(text_raw: list, target: str) -> tuple:
    """Extracts coordinates of a target using the output of the Google Cloud Vision API
    call."""
    coordinates = None
    for elem in text_raw[1:]:  # The first element which contains all text is skipped
        if target in elem["description"]:
            poly = elem["boundingPoly"]

            x_coordinates = [coordinate["x"] for coordinate in poly["vertices"]]
            y_coordinates = [coordinate["y"] for coordinate in poly["vertices"]]

            x_1, x_2 = min(x_coordinates), max(x_coordinates)
            y_1, y_2 = min(y_coordinates), max(y_coordinates)

            coordinates = (x_1, x_2, y_1, y_2)
            break  # Stop after first hit
    return coordinates


def extract_feature(text_raw: list, to_extract: str) -> tuple:
    """Extracts features, such as date, time, fuel amount, as well as their coordinates
    from the output of the Google Cloud Vision API call."""
    result, result_raw, coordinates = None, None, None

    # Prepare text and pattern to search for
    text = format_text(text_raw)
    patterns = get_regex_patterns()
    pattern = patterns[to_extract]

    # Find all strings in text that match the pattern
    if to_extract == "price_incl_tax":
        # We want to find the first matching price after the fuel amount, because this
        # is most likely the amount paid for the fuel amount
        _, amount_raw, _ = extract_feature(text_raw, "amount")
        text_after_amount = text.split(str(amount_raw))[1]
        matches = re.compile(f"({'|'.join(pattern)})").findall(text_after_amount)

    else:
        if isinstance(pattern, list):
            matches = re.compile(f"({'|'.join(pattern)})").findall(text)
        else:
            matches = re.findall(pattern, text)

    # Format the strings, select the appropriate one, and extract coordinates
    if len(matches) > 0:
        if to_extract == "date":
            matches_form = [standardize_date(date) for date in matches]
            result = sorted(matches_form)[-1]  # Select last date

            # Find the non-formatted string to be able to find its coordinates in text_raw
            max_idx = matches_form.index(max(matches_form))
            result_raw = re.sub("[^0-9.,-/]", "", matches[max_idx])

        elif to_extract == "time":
            result_raw = sorted(matches)[-1]  # Select last time
            result = result_raw

        elif to_extract == "fuel_type":
            result_raw = matches[0]  # Select first match
            result = result_raw.lower()

        elif to_extract in ["tax_rate", "amount", "price_per_unit", "price_incl_tax"]:
            # Drop matches without a digit
            matches = [match for match in matches if any(c.isdigit() for c in match)]
            result_raw = re.sub("[^0-9.,]", "", matches[0])
            matches_form = preprocess_float_matches(matches)

            if to_extract == "tax_rate":
                result = matches_form[0] / 100  # Take first tax amount divided by 100

            elif to_extract == "amount":
                result = round(matches_form[0], 2)  # Select first matching amount
                if " " in result_raw.strip():
                    result_raw = result_raw.strip().split(" ")[0]

            elif to_extract == "price_per_unit":
                result = round(matches_form[0], 3)  # Select first matching price

            elif to_extract == "price_incl_tax":
                result = round(matches_form[0], 2)  # Select first price after amount

        else:
            raise ValueError(f"Feature '{to_extract}' not supported.")
        coordinates = extract_coordinates_in_image(text_raw, result_raw)
    return result, result_raw, coordinates


def save_raw_image_with_original_orientation(img_path: str) -> str:
    """
    Checks whether an image got rotated and fixes the orientation, if necessary.
    Images taken by mobile devices may be saved as Landscape Left, even if the image
    was taken in portrait mode. This is inconvenient in the web interface.
    Source: https://stackoverflow.com/questions/13872331/rotating-an-image-with-orientation-specified-in-exif-using-python-without-pil-in
    """
    image = Image.open(img_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    img_path_new = format_image_path(img_path)
    image.save(img_path_new)
    image.close()
    return img_path_new


def save_scanned_image(
        img_path: str, coordinates: dict = None, display_img: bool = False
) -> str:
    """Displays an image. If coordinates are provided, a red box is drawn around.
    Coordinates must be a dict with tuples (x_1, x_2, y_1, y_2) as values."""
    # Prepare image
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Add red boxes around features
    if coordinates:
        for coordinate in coordinates.values():
            if coordinate:  # Some features may have no coordinates
                (x_1, x_2, y_1, y_2) = coordinate
                rect = patches.Rectangle(
                    (x_1, y_1),
                    x_2 - x_1,  # Width
                    y_2 - y_1,  # Height
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

    # Display image
    if display_img:
        plt.show()

    # Save processed image
    img_path_scanned = format_image_path(img_path, option="scanned")
    plt.savefig(img_path_scanned, bbox_inches="tight")
    plt.close()
    return img_path_scanned


def scan_receipt_main(img_path: str, display_img: bool = False) -> tuple:
    """Runs the entire process for an image: detect text in image, extract features
    from text, and save scanned image with boxes around features."""
    # Save image to UPLOAD_FOLDER
    img_path = save_raw_image_with_original_orientation(img_path)

    # Detect text in image
    text_raw = detect_text(img_path)

    # Extract features from text
    result, coordinates = {}, {}
    for feat in [
        "date",
        "time",
        "fuel_type",
        "tax_rate",
        "amount",
        "price_per_unit"
    ]:
        result[feat], _, coordinates[feat] = extract_feature(text_raw, to_extract=feat)

    # If a price per unit was found, calculate price incl. tax
    if result["price_per_unit"]:
        if result["amount"]:
            result["price_incl_tax"] = round(result["price_per_unit"] * result["amount"], 2)
        else:
            result["price_incl_tax"] = None
    # If no price per unit was found, extract the price incl. tax and calculate price per unit
    else:
        feat = "price_incl_tax"
        result[feat], _, coordinates[feat] = extract_feature(text_raw, to_extract=feat)
        if result["price_incl_tax"] and result["amount"]:
            result["price_per_unit"] = round(result["price_incl_tax"] / result["amount"], 3)

    # Save scanned image with boxes around features to UPLOAD_FOLDER
    img_path_scanned = save_scanned_image(
        img_path=img_path, coordinates=coordinates, display_img=display_img
    )

    return result, img_path_scanned
