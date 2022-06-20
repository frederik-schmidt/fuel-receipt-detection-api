import json
import os

from dotenv import load_dotenv

load_dotenv()

PRODUCTION = os.environ.get("IS_HEROKU", False)

UPLOAD_FOLDER = "./static/uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tiff"}

API_USERNAME_HASH = os.environ.get("API_USERNAME_HASH")
API_PASSWORD_HASH = os.environ.get("API_PASSWORD_HASH")

API_SECRET = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
API_SECRET = json.loads(API_SECRET)
API_SECRET = {i: v.replace("\\n", "\n") for i, v in API_SECRET.items()}

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_BUCKET_SUBFOLDER = os.environ.get("GCS_BUCKET_SUBFOLDER")
