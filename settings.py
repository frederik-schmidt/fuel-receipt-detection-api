import json
import os

from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = "./static/uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tiff"}

API_SECRET = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
API_SECRET = json.loads(API_SECRET)
API_SECRET = {i: v.replace("\\n", "\n") for i, v in API_SECRET.items()}

API_USERNAME_HASH = os.environ.get("API_USERNAME_HASH")
API_PASSWORD_HASH = os.environ.get("API_PASSWORD_HASH")
