# File Analysis and LLM-Powered Insights

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## Overview
This application enables users to analyze files, interact with an AI-powered assistant (via LangChain), visualize data with Plotly, and extract meaningful insights. The app is built using FastAPI for backend processing and Streamlit for an interactive front-end experience.

## Features
- **File Upload & Analysis**: Supports CSV and Excel files.
- **AI-Powered Chat**: Communicate with the app using LLM (LangChain & OpenAI).
- **Data Visualizations**: Interactive Plotly charts for deeper insights.
- **Streamlit UI**: Web-based interface for easy access.
- **FastAPI Backend**: Handles file processing, API endpoints, and AI interactions.
- **Data Processing**: Utilizes Pandas for efficient data manipulation.

## Tech Stack
- **Backend**: FastAPI, LangChain, OpenAI API, Pydantic
- **Frontend**: Streamlit, Streamlit-Extras
- **Visualization**: Plotly, Kaleido
- **Data Processing**: Pandas, OpenPyXL, Statsmodels

## Installation

### Prerequisites
- Python 3.10+
- `pip` package manager
- OpenAI API key (if using LLM features)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/BlackPony32/AWS_Streamlit.git
   cd AWS_Streamlit
Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Set up environment variables:
Create a .env file in the root directory and add
OPENAI_API_KEY=your_openai_api_key

Usage
Running the Backend (FastAPI):
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Running the Frontend (Streamlit):
streamlit run app.py

Contributing
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit changes (git commit -m "Add new feature").
4. Push to the branch (git push origin feature-branch).
5. Create a Pull Request.
License
This project is licensed under the MIT License.
