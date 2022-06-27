import io
import os
import re
from datetime import datetime
from typing import Union

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import ExifTags, Image
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToDict

from settings import ALLOWED_EXTENSIONS, API_SECRET, PRODUCTION
from utils import (
    allowed_file,
    format_image_path,
    read_from_json,
    standardize_date,
    write_to_json,
)

matplotlib.use("Agg")


def save_raw_image_with_original_orientation(img_path: str) -> str:
    """ Checks whether an image got rotated and fixes the orientation, if necessary.
    Images taken by mobile devices may be saved as Landscape Left, even if the image was
    taken in portrait mode. This is inconvenient in the web interface."""
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
    img_path = format_image_path(img_path)  # make path to UPLOAD_FOLDER
    image.save(img_path)
    image.close()
    return img_path


def detect_text(img_path: str) -> list:
    """Detects text in an image using the Google Cloud Vision API.
    If the json of the image is available, the API will not be called again.
    Docs: https://cloud.google.com/vision/docs/ocr#vision_text_detection-python"""
    if not allowed_file:
        raise ValueError(f"Supported file types are: {', '.join(ALLOWED_EXTENSIONS)}")
    json_path = format_image_path(img_path, option="json")

    if os.path.isfile(json_path) and not PRODUCTION:
        response_text = read_from_json(json_path)
        return response_text

    else:
        client = vision.ImageAnnotatorClient.from_service_account_info(info=API_SECRET)
        with io.open(img_path, "rb") as image_file:
            content = image_file.read()
        image = types.Image(content=content)
        response = client.text_detection(image=image)
        response_dict = MessageToDict(response._pb)
        response_text = response_dict["textAnnotations"]

        if not PRODUCTION:
            write_to_json(json_path, response_text)

        return response_text


def get_elements_and_coordinates(text_raw: list) -> list:
    """Extracts coordinates of each word detected by the Google Cloud Vision API call.
    Coordinates are stored in a dictionary with 'description' and 'coordinates' keys."""
    coordinates = []
    for elem in text_raw[1:]:  # skip first element which contains the entire text
        target = elem["description"]
        poly = elem["boundingPoly"]

        x_coordinates = [coordinate["x"] for coordinate in poly["vertices"]]
        y_coordinates = [coordinate["y"] for coordinate in poly["vertices"]]
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)

        coordinates.append(
            {"description": target, "coordinates": (x_min, x_max, y_min, y_max)}
        )
    return coordinates


def sort_elements_by_coordinates(coords: list) -> list:
    """Sorts the elements of the receipt by its coordinates.
    This iterative approach compares a word's midpoint on the y-axis with the mean
    midpoint of the words in the current row. If the word's  starts below the mean
    midpoint on the y-axis, it becomes part of a new row."""
    # sort coordinates by y-axis position
    coords = sorted(coords, key=lambda d: d["coordinates"][2])

    # extract the row of each word
    row, midpoint_mean, midpoints = 1, 9999, []
    for c in coords:
        y_min, y_max = c["coordinates"][2], c["coordinates"][3]
        y_midpoint = y_min + 0.5 * (y_max - y_min)

        if y_min > midpoint_mean:
            row += 1  # increase the row
            midpoints = [y_midpoint]  # reset the midpoints
        else:
            midpoints.append(y_midpoint)

        midpoint_mean = sum(midpoints) / len(midpoints)
        c["row"] = row

    # sort coordinates by row and x-axis position
    coords = sorted(coords, key=lambda d: (d["row"], d["coordinates"][0]))

    return coords


def get_regex_patterns() -> dict:
    """Returns the patterns for each of the features to extract.
    Explanation:
        \d+     - one or more digits
        \d{2}   - exactly two digits
        \d{1,3} - one to three digits
        (?:.|,) - optional comma or dot
        [.|,]   - mandatory comma or dot
        \s?     - optional whitespace
        \b      - word boundary
    """
    date = r"(?:\s|\n)\d{1,2}[.-]\d{2}[.-]\d{2,4}(?:\s|\n)"
    time = r"\d{1,2}:\d{2}"
    fuel_type = r"(?:diesel|DIESEL|Diesel|super|SUPER|Super)"
    volume = r"(?:Liter|liter|l|1\s)"  # sometimes l gets recognized as 1
    currency = r"(?:€|EUR|Eur|eur)"
    per = r"(?:/|PRO|pro|PER|per)"
    tax_rate = r"\d+[.|,]\d+\s?%"
    amount = r"\d{1,3}[.|,]\d{1,2}"
    unit_amount = r"\d{1}[.|,]\d{3}"

    regex_pattern = {
        "date": [date],
        "time": [time],
        "fuel_type": [fuel_type],
        "tax_rate": [tax_rate],
        "amount": [rf"(?:\s|\n){amount}\s{volume}"],
        "unit_price": [
            rf"{unit_amount}\s{currency}\s?{per}\s?{volume}",
            rf"{currency}\s{unit_amount}\s?{per}\s?{volume}",
        ],
        "price": [
            rf"{amount}\s{currency}(?:\s|\n)",
            rf"{currency}\s{amount}(?:\s|\n)",
        ],
    }
    return regex_pattern


def format_float_matches(matches: list) -> tuple:
    """Extracts numbers from a list of strings and converts it into a list of floats.
     Each string must contain only one float.

     >>> m = [' 12,34 EUR ', ' EUR', '23.45']
     >>> format_float_matches(m)
     (['12,34', '23.45'], [12.34, 23.45])
     """
    # drop non-numeric matches
    matches = [match for match in matches if any(c.isdigit() for c in match)]

    matches_f, matches_float = [], []
    for match in matches:
        # remove letters and special characters
        match_form = re.sub(r"[^\d\s.,]", "", match)
        # remove leading or lagging whitespace
        match_form = match_form.strip()
        # select all text before a whitespace, if necessary
        if " " in match_form:
            match_form = match_form.split(" ")[0]
        # convert to float
        match_float = match_form.replace(",", ".")
        match_float = float(re.sub(r"[^\d.]", "", match_float))
        matches_f.append(match_form)
        matches_float.append(match_float)

    return matches_f, matches_float


def get_float_match(matches: list, matches_f: list, first_match: bool = True) -> tuple:
    """Extracts the matching float number from a list of raw and formatted floats.
    There is an option to select the first match, which is used e.g. for extracting the
    tax rate from the receipt, and an option to select the maximum match, which is used
    for the amount.

    >>> get_float_match(['12,34', '23.45'], [12.34, 23.45])
    (12.34, '12,34')
    """
    if first_match:
        result = matches_f[0]
        idx = 0
    else:
        result = max(matches_f)
        idx = matches_f.index(max(matches_f))

    result_raw = matches[idx]
    return result_raw, result


def find_feature_coordinates(coords: list, feature: str) -> Union[str, None]:
    """Extracts coordinates of a feature like date, time, etc., except for the vendor.
    This is done by iterating over all text elements until the target element is found.
    Then, its coordinates are returned."""
    for c in coords:
        if feature in c["description"]:
            return c["coordinates"]
    return None


def find_vendor_name(coords: list) -> Union[str, None]:
    """Extracts the vendor name from all coordinates by selecting and joining all
    elements until the row which contains the zip code. This is a special case since it
    is stored in multiple rows."""
    zipcode_row = None
    for c in coords:
        if bool(re.search(r"\b\d{5}\b", c["description"])):
            zipcode_row = c["row"]
            break
    if not zipcode_row or zipcode_row > 6:
        return None
    vendor = [c["description"] for c in coords if c["row"] <= zipcode_row]
    return " ".join(vendor)


def find_vendor_coordinates(coords: list, vendor: str) -> tuple:
    """Extracts the vendor coordinates.
    This is done by iterating over all text elements and updating the current
    coordinates, if necessary. The function stops updating the coordinates when a text
    element is not in the vendor name anymore."""
    x_min_glob, x_max_glob, y_min_glob, y_max_glob = 99999, 0, 99999, 0

    for c in coords:
        if c["description"] not in vendor:
            break
        # update the global coordinates if the individual coordinates are exceeding
        x_min, x_max, y_min, y_max = c["coordinates"]
        x_min_glob = x_min if x_min < x_min_glob else x_min_glob
        x_max_glob = x_max if x_min > x_max_glob else x_max_glob
        y_min_glob = y_min if y_min < y_min_glob else y_min_glob
        y_max_glob = y_max if y_max > y_max_glob else y_max_glob

    return x_min_glob, x_max_glob, y_min_glob, y_max_glob


def extract_vendor(coords: list) -> tuple:
    """Extracts the unit price and its coordinates from the output of the Google Cloud
    Vision API. This is done by selecting and joining all elements until the row which
    contains the zip code."""
    result_raw = find_vendor_name(coords)
    result = result_raw.replace(" .", ".")
    coords = find_vendor_coordinates(coords, result)

    return result, result_raw, coords


def extract_date(text: str, coords: list, regex_pattern: str) -> tuple:
    """Extracts the date and its coordinates from the output of the Google Cloud Vision
    API. All date-like strings are extracted using regular expressions and the maximum
    date gets returned.

    >>> t = 'text with one 01.01.2020 date and another 01.07.2020 date'
    >>> r = get_regex_patterns()['date']
    >>> c = [
    ... {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ... {'description': '01.07.2020', 'coordinates': (559, 939, 46, 131)},
    ... {'description': '01.01.2020', 'coordinates': (291, 368, 100, 135)}
    ... ]
    >>> extract_date(text=t, coords=c, regex_pattern=r)
    ('2020-07-01', '01.07.2020', (559, 939, 46, 131))
     """
    result, result_raw, feature_coords = None, None, None
    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches

    if len(matches) > 0:
        matches_f = [standardize_date(date) for date in matches]  # format matches
        result = sorted(matches_f)[-1]  # find last date
        max_idx = matches_f.index(max(matches_f))  # find index of raw last date
        result_raw = re.sub(r"[^\d.,-/]", "", matches[max_idx])  # find raw last date
        feature_coords = find_feature_coordinates(coords, result_raw)

    return result, result_raw, feature_coords


def extract_time(text: str, coords: list, regex_pattern: str) -> tuple:
    """Extracts the time and its coordinates from the output of the Google Cloud Vision
    API. All time-like strings are extracted using regular expressions and the maximum
    time gets returned.

    >>> t = 'text with one 9:41 time and another 9:45:23 time'
    >>> r = get_regex_patterns()['time']
    >>> c = [
    ... {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ... {'description': '9:45:23', 'coordinates': (559, 939, 46, 131)},
    ... {'description': '9:41', 'coordinates': (291, 368, 100, 135)}
    ... ]
    >>> extract_time(text=t, coords=c, regex_pattern=r)
    ('9:45', '9:45', (559, 939, 46, 131))
    """
    result, result_raw, feature_coords = None, None, None
    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches

    if len(matches) > 0:
        result = sorted(matches)[-1]  # find last time
        result_raw = result  # no need to format the time
        feature_coords = find_feature_coordinates(coords, result_raw)

    return result, result_raw, feature_coords


def extract_fuel_type(text: str, coords: list, regex_pattern: str) -> tuple:
    """Extracts the fuel type and its coordinates from the output of the Google Cloud
    Vision API. All matching elements are extracted using regular expressions and the
    first fuel type gets returned.

    >>> t = 'text with one Diesel E10 fuel type'
    >>> r = get_regex_patterns()['fuel_type']
    >>> c = [
    ...    {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ...    {'description': 'Diesel', 'coordinates': (559, 939, 46, 131)}
    ... ]
    >>> extract_fuel_type(text=t, coords=c, regex_pattern=r)
    ('diesel', 'Diesel', (559, 939, 46, 131))
    """
    result, result_raw, feature_coords = None, None, None
    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches

    if len(matches) > 0:
        result_raw = matches[0]  # select first fuel type match
        result = result_raw.lower()  # format the fuel type
        feature_coords = find_feature_coordinates(coords, result_raw)

    return result, result_raw, feature_coords


def extract_tax_rate(text: str, coords: list, regex_pattern: str) -> tuple:
    """Extracts the tax rate and its coordinates from the output of the Google Cloud
    Vision API. All matching elements are extracted using regular expressions and the
    first tax rate gets returned.

    >>> t = 'text with one 19,00 % percentage and another 5,1 % percentage'
    >>> r = get_regex_patterns()['tax_rate']
    >>> c = [
    ... {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ... {'description': '19,00', 'coordinates': (559, 939, 46, 131)},
    ... {'description': '5,1', 'coordinates': (291, 368, 100, 135)}
    ... ]
    >>> extract_tax_rate(text=t, coords=c, regex_pattern=r)
    (0.19, '19,00', (559, 939, 46, 131))
    """
    result, result_raw, feature_coords = None, None, None

    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches
    if len(matches) > 0:
        matches, matches_f = format_float_matches(matches)
        result_raw, result = get_float_match(matches, matches_f, first_match=True)
        result = result / 100  # first tax amount divided by 100
        feature_coords = find_feature_coordinates(coords, result_raw)

    return result, result_raw, feature_coords


def extract_amount(text: str, coords: list, regex_pattern: str) -> tuple:
    """Extracts the amount and its coordinates from the output of the Google Cloud
    Vision API. This is tricky, since there may be other amounts on the receipt than the
    fuel amount, e.g. 0.5 l of engine oil. The fuel amount is always close to the fuel
    type on the receipt. Thus, the amount before and after the fuel type gets extracted
    and the higher of the two get returned.

    >>> t = 'text 30,32 l amount, 1,649 € / l unit price, 0,5 l amount, 50,00 € price'
    >>> r = get_regex_patterns()['amount']
    >>> c = [
    ...    {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ...    {'description': '30,32', 'coordinates': (559, 939, 46, 131)},
    ...    {'description': '0,5', 'coordinates': (291, 368, 100, 135)}
    ... ]
    >>> extract_amount(text=t, coords=c, regex_pattern=r)
    (30.32, '30,32', (559, 939, 46, 131))
    """
    result, result_raw, feature_coords = None, None, None

    # select the maximum amount-like string before and after the fuel type
    fuel_type_pat = get_regex_patterns()["fuel_type"]
    _, fuel_type, _ = extract_fuel_type(text, coords, fuel_type_pat)
    text_before, text_after = text.split(fuel_type, 1)
    matches_before = re.compile(f"({'|'.join(regex_pattern)})").findall(text_before)
    matches_after = re.compile(f"({'|'.join(regex_pattern)})").findall(text_after)
    try:
        last_match_before = [matches_before[-1]]
    except IndexError:
        last_match_before = []
    try:
        first_match_after = [matches_after[0]]
    except IndexError:
        first_match_after = []
    matches = last_match_before + first_match_after

    if len(matches) > 0:
        matches, matches_f = format_float_matches(matches)
        result_raw, result = get_float_match(matches, matches_f, first_match=False)
        result = round(result, 2)
        feature_coords = find_feature_coordinates(coords, result_raw)

    return result, result_raw, feature_coords


def extract_unit_price(text: str, coords: list, regex_pattern: str) -> tuple:
    """Extracts the unit price and its coordinates from the output of the Google Cloud
    Vision API. All matching elements are extracted using regular expressions and the
    first unit price gets returned.

    >>> t = 'text 30,32 l amount, 1,649 € / l unit price, 0,5 l amount, 50,00 € price'
    >>> r = get_regex_patterns()['unit_price']
    >>> c = [
    ...    {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ...    {'description': '1,649', 'coordinates': (559, 939, 46, 131)},
    ...    {'description': '0,5', 'coordinates': (291, 368, 100, 135)}
    ... ]
    >>> extract_unit_price(text=t, coords=c, regex_pattern=r)
    (1.649, '1,649', (559, 939, 46, 131))
    """
    result, result_raw, feature_coords = None, None, None

    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches
    if len(matches) > 0:
        matches, matches_f = format_float_matches(matches)
        result_raw, result = get_float_match(matches, matches_f, first_match=True)
        result = round(result, 3)
        feature_coords = find_feature_coordinates(coords, result_raw)

    return result, result_raw, feature_coords


def extract_price(text: str, coords: list, regex_pattern: str) -> tuple:
    """Extracts the price and its coordinates from the output of the Google Cloud Vision
    API. All matching elements are extracted using regular expressions and the first
     price gets returned.

    >>> t = 'text 30,32 l amount, 1,649 € / l unit price, 0,5 l amount, 50,00 € price'
    >>> r = get_regex_patterns()['price']
    >>> c = [
    ...    {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ...    {'description': '50,00', 'coordinates': (559, 939, 46, 131)},
    ...    {'description': '0,5', 'coordinates': (291, 368, 100, 135)}
    ... ]
    >>> extract_price(text=t, coords=c, regex_pattern=r)
    (50.0, '50,00', (559, 939, 46, 131))
    """
    result, result_raw, feature_coords = None, None, None

    # find first price after fuel amount, because this most likely the fuel price
    amount_pattern = get_regex_patterns()["amount"]
    _, amount_raw, _ = extract_amount(text, coords, amount_pattern)
    text = text.split(str(amount_raw))[1]  # text after amount

    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches
    if len(matches) > 0:
        matches, matches_f = format_float_matches(matches)
        result_raw, result = get_float_match(matches, matches_f, first_match=True)
        result = round(result, 2)
        feature_coords = find_feature_coordinates(coords, result_raw)

    return result, result_raw, feature_coords


def extract_features(img_path: str, text: str, coords: list) -> tuple:
    """Extracts all results by the individual function for each of the features."""
    regex_patterns = get_regex_patterns()
    img_name = os.path.split(img_path)[1].split(".")[0]
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S:%fZ")
    result, feature_coords = {"timestamp": timestamp, "id": img_name}, {}

    result_ind = extract_vendor(coords)
    result["vendor"], _, feature_coords["vendor"] = result_ind  # unpack result

    result_ind = extract_date(text, coords, regex_patterns["date"])
    result["date"], _, feature_coords["date"] = result_ind  # unpack result

    result_ind = extract_time(text, coords, regex_patterns["time"])
    result["time"], _, feature_coords["time"] = result_ind  # unpack result

    result_ind = extract_fuel_type(text, coords, regex_patterns["fuel_type"])
    result["fuel_type"], _, feature_coords["fuel_type"] = result_ind  # unpack result

    result_ind = extract_tax_rate(text, coords, regex_patterns["tax_rate"])
    result["tax_rate"], _, feature_coords["tax_rate"] = result_ind  # unpack result

    result_ind = extract_amount(text, coords, regex_patterns["amount"])
    result["amount"], _, feature_coords["amount"] = result_ind  # unpack result

    result_ind = extract_unit_price(text, coords, regex_patterns["unit_price"])
    result["unit_price"], _, feature_coords["unit_price"] = result_ind  # unpack result

    # if unit_price was found, calculate the price
    if result["unit_price"]:
        if result["amount"]:
            result["price"] = round(result["unit_price"] * result["amount"], 2)
        else:
            result["price"] = None
    # if no unit_price was found, extract the price and calculate the unit_price
    else:
        result_ind = extract_price(text, coords, regex_patterns["price"])
        result["price"], _, feature_coords["price"] = result_ind
        if result["price"] and result["amount"]:
            result["unit_price"] = round(result["price"] / result["amount"], 3)

    return result, feature_coords


def save_scanned_image(
    img_path: str, feature_coords: dict = None, display_img: bool = False
) -> None:
    """Displays an image. If coordinates are provided, a red box is drawn around.
    Coordinates must be a dict with tuples (x_1, x_2, y_1, y_2) as values."""
    # prepare image
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # draw red boxes around features
    if feature_coords:
        for c in feature_coords.values():
            if c:  # some features may have no coordinates
                (x_1, x_2, y_1, y_2) = c
                rect = patches.Rectangle(
                    (x_1, y_1),
                    x_2 - x_1,  # Width
                    y_2 - y_1,  # Height
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

    # display image
    if display_img:
        plt.show()

    # save processed image
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()


def scan_receipt_main(img_path: str, display_img: bool = False) -> dict:
    """Runs the entire process for an image: detect text in image, extract features
    from text, and save scanned image with boxes around features."""
    # save originally oriented image
    img_path = save_raw_image_with_original_orientation(img_path)

    # detect and format text in image
    text_raw = detect_text(img_path)
    coords = get_elements_and_coordinates(text_raw)
    coords = sort_elements_by_coordinates(coords)
    text = " ".join([c["description"] for c in coords])

    # extract features from text
    result, feature_coords = extract_features(img_path, text, coords)

    # save scanned image with boxes around features to UPLOAD_FOLDER
    save_scanned_image(img_path, feature_coords, display_img)

    return result
