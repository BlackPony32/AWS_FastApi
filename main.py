from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import pandas as pd
import shutil
import requests
import logging

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.agent_types import AgentType

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

def convert_excel_to_csv(excel_file_path):
    try:
        df = pd.read_excel(excel_file_path)
        csv_file_path = os.path.splitext(excel_file_path)[0] + ".csv"
        df.to_csv(csv_file_path, index=False)
        os.remove(excel_file_path)
        return csv_file_path
    except Exception as e:
        raise ValueError(f"Error converting Excel to CSV: {str(e)}")


@app.post("/link_file_and_name/")
async def link_file_and_name(request: DownloadRequest):
    cleanup_uploads_folder(UPLOAD_DIR)
    url = request.url
    filename = request.filename
    report_type_filenames = {
        'CUSTOMER_DETAILS': 'customer_details.xlsx',
        'TOP_CUSTOMERS': 'top_customers.xlsx',
        'ORDER_SALES_SUMMARY': 'order_sales_summary.xlsx',
        'THIRD_PARTY_SALES_SUMMARY': 'third_party_sales_summary.xlsx',
        'CURRENT_INVENTORY': 'current_inventory.xlsx',
        'LOW_STOCK_INVENTORY': 'low_stock_inventory.xlsx',
        'BEST_SELLERS': 'best_sellers.xlsx',
        'SKU_NOT_ORDERED': 'sku_not_ordered.xlsx',
        'REP_DETAILS': 'rep_details.xlsx',
        'REPS_SUMMARY': 'reps_summary.xlsx',
    }
    friendly_filename = report_type_filenames.get(filename, 'unknown.xlsx')
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        if response.headers.get('Content-Type') != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            raise HTTPException(status_code=400, detail="Unsupported file type")
        excel_file_path = os.path.join(UPLOAD_DIR, friendly_filename)
        with open(excel_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        global csv_file_path
        csv_file_path = convert_excel_to_csv(excel_file_path)

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
    url = request.url
    user_id = request.user_id
    file_name = request.filename
    return JSONResponse(content={"url": url, "user_id": user_id, "file_name": file_name})


@app.get("/get_file_info/")
async def get_file_info():
    global url
    global user_id
    global file_name
    if url and user_id and file_name:
        response = {"url": url, "user_id": user_id, "file_name": file_name}
        # Clear the variables
        url = None
        user_id = None
        file_name = None
        return response
    else:
        raise HTTPException(status_code=404, detail="No data available")


#_____START  BLOCK FOR CHAT TO CSV_____
def chat_with_file(prompt):
    filename = csv_file_path
    try:
        if filename is None or not os.path.exists(filename):
            raise HTTPException(status_code=400, detail=f"No file has been uploaded or downloaded yet {filename}")
            
        result = chat_with_agent(prompt, filename)
        
        return {"response": result}

    except ValueError as e:
        return {"error": f"ValueError: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

def chat_with_agent(input_string, file_path):
    try:
        # Assuming file_path is always CSV after conversion
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        result = agent.invoke(input_string)
        return result['output']
    except ImportError as e:
        raise ValueError("Missing optional dependency 'tabulate'. Use pip or conda to install tabulate.")
    except pd.errors.ParserError as e:
        raise ValueError("Parsing error occurred: " + str(e))
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")

@app.post("/chat_to_agent/")
async def chat_to_agent(request: ChatRequest):
    prompt = request.prompt
    response = chat_with_file(prompt)
    return {"response": response}

#_____END  BLOCK FOR CHAT TO CSV_____




@app.on_event("shutdown")
def shutdown_event():
    cleanup_uploads_folder(UPLOAD_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)