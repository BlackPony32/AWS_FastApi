from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import pandas as pd
import shutil
import requests
import logging

log_file_path = "logging_file.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

streamlit_url = os.getenv('STREAMLIT_URL')
# FastAPI app initialization
app = FastAPI()
# Constants for upload directories
UPLOAD_DIR = "uploads"
UPLOAD_DIR_MANY = "uploads_many"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR_MANY, exist_ok=True)

# Initialize global variables
url = None
user_id = None
file_name = None


# Pydantic models for request validation
class ChatRequest(BaseModel):
    prompt: str

class DownloadRequest(BaseModel):
    url: str
    user_id: Optional[str] = None
    filename: Optional[str] = None

# Utility functions
def cleanup_uploads_folder(upload_dir: str):
    try:
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        logging.error(f"Error cleaning up uploads folder: {str(e)}")

@app.post("/link_file_and_name/")
async def link_file_and_name(request: DownloadRequest):
    cleanup_uploads_folder(UPLOAD_DIR)
    url = request.url
    filename = request.filename

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        return {"message": "File downloaded and converted successfully", "streamlit_url": streamlit_url}
    except requests.RequestException as e:
        logging.error(f"RequestException: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/link_file_and_name_for_ui/")
async def link_file_and_name(request: DownloadRequest):
    global url
    global file_name
    global user_id
    try:
        url = request.url
        user_id = request.user_id
        file_name = request.filename
        return JSONResponse(content={"url": url, "user_id": user_id, "file_name": file_name})
    except Exception as e:
        logger.info("Page was updated, link_file_and_name error")

@app.get("/get_file_info/")
async def get_file_info():
    global url
    global user_id
    global file_name
    try:
        if url and user_id and file_name:
            response = {"url": url, "user_id": user_id, "file_name": file_name}
            # Clear the variables
            url = None
            user_id = None
            file_name = None
            return response
        else:
            raise HTTPException(status_code=404, detail="No data available")
    except Exception as e:
        logger.info("Page was updated")





@app.on_event("shutdown")
def shutdown_event():
    cleanup_uploads_folder(UPLOAD_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)