import io
import os
import re
from typing import Union

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
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
    """
    Checks whether an image got rotated and fixes the orientation, if necessary. Images
    taken by mobile devices may be saved as Landscape Left, even if the image was taken
    in portrait mode. This is inconvenient in the web interface.
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
    img_path = format_image_path(img_path)  # path to UPLOAD_FOLDER
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
    coordinates_all = []
    for elem in text_raw[1:]:  # skip first element which contains the entire text
        target = elem["description"]
        poly = elem["boundingPoly"]

        x_coordinates = [coordinate["x"] for coordinate in poly["vertices"]]
        y_coordinates = [coordinate["y"] for coordinate in poly["vertices"]]
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)

        coordinates = (x_min, x_max, y_min, y_max)
        coordinates_all.append({"description": target, "coordinates": coordinates})
    return coordinates_all


def sort_elements_by_coordinates(coordinates_all: list) -> list:
    """Sorts the text by locating its position on the image from the coordinates.
    We cannot order by y-axis location first, and then by x-axis location, since an
    image might be slightly shifted, which would cause an incorrect order, if the most
    right word is higher than the most left word. Therefore, we calculate the midpoint
    of a word on the y-axis and identify whether the next word is in a new row."""

    def identify_new_row(df) -> int:
        """Identifies whether the coordinates are in a new row. Checks whether the
        midpoint of a word on the x-axis is within the x-axis range of the next word.
        If yes, the next word is in the same row, if not it is in a new row."""
        same_range = (df["ymid_next"] > df["ymin"]) and (df["ymid_next"] < df["ymax"])
        return 0 if same_range else 1

    df = pd.DataFrame(coordinates_all)
    # split coordinates into separate columns
    df_coordinates = pd.DataFrame(df["coordinates"].tolist(), index=df.index)
    df[["xmin", "xmax", "ymin", "ymax"]] = df_coordinates
    # sort by most top words - sometimes the most left rows are on top by default
    df = df.sort_values(by="ymin")
    # calculate the midpoint of the word on the y-axis
    df["ymid"] = df["ymax"] - ((df["ymax"] - df["ymin"]) / 2)
    # calculate the midpoint of the next word on the y-axis
    df["ymid_next"] = df["ymid"].shift(1)
    # identify whether next word is a new row depending on the overlap
    df["new_row"] = df.apply(identify_new_row, axis=1)
    # count the rows
    df["row"] = df["new_row"].cumsum()
    # sort by rows which refers to y-axis and by x-axis
    df = df.sort_values(by=["row", "xmin"])
    # convert back to dict
    coordinates_all = df[["description", "coordinates", "row"]].to_dict("records")
    return coordinates_all


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
    volume = r"(?:Liter|liter|l|1)"  # sometimes l gets recognized as 1
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


def preprocess_float_matches(matches: list) -> tuple:
    """Extracts numbers from a list of strings and converts it into a list of floats.
     Each string must contain only one float.

     >>> m = [' 12,34 EUR ', ' EUR', '23.45']
     >>> preprocess_float_matches(m)
     ([12.34, 23.45], '12,34')
     """
    # drop non-numeric matches
    matches = [match for match in matches if any(c.isdigit() for c in match)]
    # select the first match
    result_raw = matches[0].strip()
    # select the first section if the match contains a whitespace
    if " " in result_raw:
        result_raw = result_raw.strip().split(" ")[0]
    result_raw = re.sub(r"[^\d.,]", "", result_raw)
    # replace dot with comma
    matches_form = [i.replace(",", ".") for i in matches]
    # extract numbers and convert to float
    matches_form = [float(re.sub(r"[^\d.]", "", i)) for i in matches_form]
    return matches_form, result_raw


def find_element_coordinates(coordinates_all: list, elem: str) -> Union[str, None]:
    """Extracts coordinates of an element - except for the vendor.
    This is done by iterating over all text elements until the target element is found.
    Then, its coordinates are returned."""
    for i in coordinates_all:
        if elem in i["description"]:
            return i["coordinates"]
    return None


def find_vendor_name(coordinates_all: list) -> Union[str, None]:
    """Extracts the vendor name from all coordinates by selecting and joining all
    elements until the row which contains the zip code. This is a special case since it
    is stored in multiple rows."""
    zipcode_row = None
    for c in coordinates_all:
        if bool(re.search(r"\b\d{5}\b", c["description"])):
            zipcode_row = c["row"]
            break
    if not zipcode_row or zipcode_row > 6:
        return None
    vendor = [c["description"] for c in coordinates_all if c["row"] <= zipcode_row]
    return " ".join(vendor)


def find_vendor_coordinates(coordinates_all: list, vendor: str) -> tuple:
    """Extracts the vendor coordinates.
    This is done by iterating over all text elements and updating the current
    coordinates if necessary. Then function updating the coordinates when a text element
    is not in the vendor name anymore."""
    x_min_glob, x_max_glob, y_min_glob, y_max_glob = 99999, 0, 99999, 0

    for c in coordinates_all:
        if c["description"] not in vendor:
            break
        # update the global coordinates if the individual coordinates are exceeding
        x_min, x_max, y_min, y_max = c["coordinates"]
        x_min_glob = x_min if x_min < x_min_glob else x_min_glob
        x_max_glob = x_max if x_min > x_max_glob else x_max_glob
        y_min_glob = y_min if y_min < y_min_glob else y_min_glob
        y_max_glob = y_max if y_max > y_max_glob else y_max_glob

    return x_min_glob, x_max_glob, y_min_glob, y_max_glob


def extract_vendor(coordinates_all: list) -> tuple:
    """Extracts the unit price and its coordinates from the output of the Google Cloud
    Vision API. This is done by selecting and joining all elements until the row which
    contains the zip code."""
    result = find_vendor_name(coordinates_all)
    coordinates = find_vendor_coordinates(coordinates_all, result)

    return result, result, coordinates


def extract_date(text: str, coordinates_all: list, regex_pattern: str) -> tuple:
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
    >>> extract_date(text=t, coordinates_all=c, regex_pattern=r)
    ('2020-07-01', '01.07.2020', (559, 939, 46, 131))
     """
    result, result_raw, coordinates = None, None, None  # initialize values
    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches

    if len(matches) > 0:
        matches_f = [standardize_date(date) for date in matches]  # format matches
        result = sorted(matches_f)[-1]  # find last date
        max_idx = matches_f.index(max(matches_f))  # find index of raw last date
        result_raw = re.sub(r"[^\d.,-/]", "", matches[max_idx])  # find raw last date
        coordinates = find_element_coordinates(coordinates_all, result_raw)

    return result, result_raw, coordinates


def extract_time(text: str, coordinates_all: list, regex_pattern: str) -> tuple:
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
    >>> extract_time(text=t, coordinates_all=c, regex_pattern=r)
    ('9:45', '9:45', (559, 939, 46, 131))
    """
    result, result_raw, coordinates = None, None, None  # initialize values
    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches

    if len(matches) > 0:
        result = sorted(matches)[-1]  # find last time
        result_raw = result  # no need to format the time
        coordinates = find_element_coordinates(coordinates_all, result_raw)

    return result, result_raw, coordinates


def extract_fuel_type(text: str, coordinates_all: list, regex_pattern: str) -> tuple:
    """Extracts the fuel type and its coordinates from the output of the Google Cloud
    Vision API. All matching elements are extracted using regular expressions and the
    first fuel type gets returned.

    >>> t = 'text with one Diesel E10 fuel type'
    >>> r = get_regex_patterns()['fuel_type']
    >>> c = [
    ...    {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ...    {'description': 'Diesel', 'coordinates': (559, 939, 46, 131)}
    ... ]
    >>> extract_fuel_type(text=t, coordinates_all=c, regex_pattern=r)
    ('diesel', 'Diesel', (559, 939, 46, 131))
    """
    result, result_raw, coordinates = None, None, None  # initialize values
    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches

    if len(matches) > 0:
        result_raw = matches[0]  # select first fuel type match
        result = result_raw.lower()  # format the fuel type
        coordinates = find_element_coordinates(coordinates_all, result_raw)

    return result, result_raw, coordinates


def extract_tax_rate(text: str, coordinates_all: list, regex_pattern: str) -> tuple:
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
    >>> extract_tax_rate(text=t, coordinates_all=c, regex_pattern=r)
    (0.19, '19,00', (559, 939, 46, 131))
    """
    result, result_raw, coordinates = None, None, None  # initialize values

    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches
    if len(matches) > 0:
        matches_form, result_raw = preprocess_float_matches(matches)
        result = matches_form[0] / 100  # first tax amount divided by 100
        coordinates = find_element_coordinates(coordinates_all, result_raw)

    return result, result_raw, coordinates


def extract_amount(text: str, coordinates_all: list, regex_pattern: str) -> tuple:
    """Extracts the amount and its coordinates from the output of the Google Cloud
    Vision API. All matching elements are extracted using regular expressions and the
    first amount gets returned.

    >>> t = 'text 30,32 l amount, 1,649 € / l unit price, 0,5 l amount, 50,00 € price'
    >>> r = get_regex_patterns()['amount']
    >>> c = [
    ...    {'description': 'string', 'coordinates': (409, 528, 39, 112)},
    ...    {'description': '30,32', 'coordinates': (559, 939, 46, 131)},
    ...    {'description': '0,5', 'coordinates': (291, 368, 100, 135)}
    ... ]
    >>> extract_amount(text=t, coordinates_all=c, regex_pattern=r)
    (30.32, '30,32', (559, 939, 46, 131))
    """
    result, result_raw, coordinates = None, None, None  # initialize values

    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches
    if len(matches) > 0:
        matches_form, result_raw = preprocess_float_matches(matches)
        result = round(matches_form[0], 2)  # select first matching amount
        coordinates = find_element_coordinates(coordinates_all, result_raw)

    return result, result_raw, coordinates


def extract_unit_price(text: str, coordinates_all: list, regex_pattern: str) -> tuple:
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
    >>> extract_unit_price(text=t, coordinates_all=c, regex_pattern=r)
    (1.649, '1,649', (559, 939, 46, 131))
    """
    result, result_raw, coordinates = None, None, None  # initialize values

    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches
    if len(matches) > 0:
        matches_form, result_raw = preprocess_float_matches(matches)
        result = round(matches_form[0], 3)  # select first matching unit price
        coordinates = find_element_coordinates(coordinates_all, result_raw)

    return result, result_raw, coordinates


def extract_price(text: str, coordinates_all: list, regex_pattern: str) -> tuple:
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
    >>> extract_price(text=t, coordinates_all=c, regex_pattern=r)
    (50.0, '50,00', (559, 939, 46, 131))
    """
    result, result_raw, coordinates = None, None, None  # initialize values

    # find first price after fuel amount, because this most likely the fuel price
    amount_pattern = get_regex_patterns()["amount"]
    _, amount_raw, _ = extract_amount(text, coordinates_all, amount_pattern)
    text = text.split(str(amount_raw))[1]  # text after amount

    matches = re.compile(f"({'|'.join(regex_pattern)})").findall(text)  # find matches
    if len(matches) > 0:
        matches_form, result_raw = preprocess_float_matches(matches)
        result = round(matches_form[0], 2)  # select first matching fuel price
        coordinates = find_element_coordinates(coordinates_all, result_raw)

    return result, result_raw, coordinates


def save_scanned_image(
    img_path: str, coordinates: dict = None, display_img: bool = False
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
    if coordinates:
        for coordinate in coordinates.values():
            if coordinate:  # some features may have no coordinates
                (x_1, x_2, y_1, y_2) = coordinate
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
    coordinates_all = get_elements_and_coordinates(text_raw)
    coordinates_all = sort_elements_by_coordinates(coordinates_all)
    text = " ".join([c["description"] for c in coordinates_all])

    # extract features from text
    regex_patterns = get_regex_patterns()
    img_name = os.path.split(img_path)[1].split(".")[0]
    result, coordinates = {"id": img_name}, {}

    result_ind = extract_vendor(coordinates_all)
    result["vendor"], _, coordinates["vendor"] = result_ind  # unpack result

    result_ind = extract_date(text, coordinates_all, regex_patterns["date"])
    result["date"], _, coordinates["date"] = result_ind  # unpack result

    result_ind = extract_time(text, coordinates_all, regex_patterns["time"])
    result["time"], _, coordinates["time"] = result_ind  # unpack result

    result_ind = extract_fuel_type(text, coordinates_all, regex_patterns["fuel_type"])
    result["fuel_type"], _, coordinates["fuel_type"] = result_ind  # unpack result

    result_ind = extract_tax_rate(text, coordinates_all, regex_patterns["tax_rate"])
    result["tax_rate"], _, coordinates["tax_rate"] = result_ind  # unpack result

    result_ind = extract_amount(text, coordinates_all, regex_patterns["amount"])
    result["amount"], _, coordinates["amount"] = result_ind  # unpack result

    result_ind = extract_unit_price(text, coordinates_all, regex_patterns["unit_price"])
    result["unit_price"], _, coordinates["unit_price"] = result_ind  # unpack result

    # if unit_price was found, calculate the price
    if result["unit_price"]:
        if result["amount"]:
            result["price"] = round(result["unit_price"] * result["amount"], 2)
        else:
            result["price"] = None
    # if no unit_price was found, extract the price and calculate the unit_price
    else:
        result_ind = extract_price(text, coordinates_all, regex_patterns["price"])
        result["price"], _, coordinates["price"] = result_ind
        if result["price"] and result["amount"]:
            result["unit_price"] = round(result["price"] / result["amount"], 3)

    # save scanned image with boxes around features to UPLOAD_FOLDER
    save_scanned_image(img_path, coordinates, display_img)

    return result
