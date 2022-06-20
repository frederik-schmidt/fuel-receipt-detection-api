import json
import os
from typing import Union

from flask import jsonify
from google.cloud import bigquery
from google.cloud import storage
from google.oauth2 import service_account
from werkzeug.utils import secure_filename

from settings import (
    ALLOWED_EXTENSIONS,
    API_SECRET,
    GCP_PROJECT_ID,
    BQ_TABLE_ID,
    UPLOAD_FOLDER
)


def allowed_file(filepath: str) -> bool:
    """Returns True if the file name provided is valid and has an allowed extension."""
    return "." in filepath and filepath.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def response_code_to_text(code) -> Union[str, None]:
    """Converts the response code of an HTTP request to an appropriate description."""
    mapping = {
        400: "400 Bad Request",
        401: "401 Unauthorized Access",
        404: "404 Not Found",
        405: "405 Method Not Allowed",
        415: "415 Unsupported Media Type",
        422: "422 Unprocessable Entity"
    }
    return mapping.get(code, None)


def wrap_into_json(response_text: Union[str, dict]):
    """Wraps a plain response text into json."""
    if isinstance(response_text, dict):
        return jsonify({"success": True, "data": response_text})
    else:
        return jsonify({"success": False, "message": response_text})


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


def load_file_into_gcs(
        bucket_name: str, local_filename: str, remote_filename: str
) -> None:
    """Uploads a file to a Google Cloud Storage bucket."""
    credentials = service_account.Credentials.from_service_account_info(API_SECRET)
    client = storage.Client(project=GCP_PROJECT_ID, credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_filename)
    blob.upload_from_filename(local_filename)


def load_json_into_bq(json_data: dict, table_id: str = BQ_TABLE_ID) -> None:
    """Uploads a json to a Google BigQuery table."""
    table_schema = [
        bigquery.SchemaField("id", "STRING", "REQUIRED"),
        bigquery.SchemaField("date", "DATE", "NULLABLE"),
        bigquery.SchemaField("time", "STRING", "NULLABLE"),
        bigquery.SchemaField("fuel_type", "STRING", "NULLABLE"),
        bigquery.SchemaField("tax_rate", "FLOAT", "NULLABLE"),
        bigquery.SchemaField("amount", "FLOAT", "NULLABLE"),
        bigquery.SchemaField("price_per_unit", "FLOAT", "NULLABLE"),
        bigquery.SchemaField("price_incl_tax", "FLOAT", "NULLABLE"),
    ]
    credentials = service_account.Credentials.from_service_account_info(API_SECRET)
    client = bigquery.Client(project=GCP_PROJECT_ID, credentials=credentials)
    job_config = bigquery.LoadJobConfig(
        schema=table_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        allow_quoted_newlines=True,
    )

    client.load_table_from_json(
        json_rows=[json_data], destination=table_id, job_config=job_config,
    )
