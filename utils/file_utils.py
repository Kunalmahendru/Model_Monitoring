import os
from fastapi import UploadFile

UPLOAD_DIR = "uploaded_datasets"

async def save_uploaded_file(file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    return file_path
