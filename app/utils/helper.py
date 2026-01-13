import shutil
import os
from app.config import CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, CLOUDINARY_CLOUD_NAME
import cloudinary
from cloudinary.uploader import upload


cloudinary.config(
    cloud_name = CLOUDINARY_CLOUD_NAME,
    api_key = CLOUDINARY_API_KEY,
    api_secret = CLOUDINARY_API_SECRET
)


def delete_file(file_path):
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        
        
        
def cloudinary_file_upload(file_path):
    
    if file_path.lower().endswith(".pdf"):
        try:
            result = upload(
                file = file_path,
                resource_type = "auto",
                folder = "pdfs"
            )
            return result["secure_url"]   
        except Exception as e:
            raise ValueError(str(e))
    else:
        try:
            result = upload(
                file = file_path,
                resource_type = "auto"
            )
            return result["secure_url"]   
        except Exception as e:
            raise ValueError(str(e))