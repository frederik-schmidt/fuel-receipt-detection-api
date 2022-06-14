import os

from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = "./static/uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tiff"}
API_SECRET = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
USER_HASH = "pbkdf2:sha256:260000$G3aDYDGmEoTM5084$3e36ae16785da2e9341034a921c5754dcadca4b154ce9a38e7e931026ec5fe7d"
PASS_HASH = "pbkdf2:sha256:260000$mhMuC5KNbBtAEZGd$e4a0b02dcf8fc78cc07348f9322d5862d0a67c3f810cf3dd242fb8054a53f442"
