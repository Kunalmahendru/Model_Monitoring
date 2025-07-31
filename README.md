# MLOps AutoML Project

An end-to-end AutoML platform that simplifies machine learning workflows from data upload to model deployment. Built with H2O AutoML, FastAPI, React, and MLflow for complete ML lifecycle management.

## What This Project Does

- Dataset upload and automatic analysis with summary statistics
- Smart target column selection and problem type detection (classification/regression)
- AutoML training with multiple algorithms using H2O's AutoML capabilities
- Interactive model leaderboard with performance metrics comparison
- One-click model deployment for live predictions
- Live prediction interface for new data
- Real-time training progress tracking

## Tech Stack

- **Backend**: FastAPI, H2O AutoML, MLflow
- **Frontend**: React, TypeScript, Vite
- **ML Framework**: H2O.ai AutoML
- **Database**: SQLite (MLflow tracking)

## Prerequisites

- Python 3.8+
- Node.js 16+
- Java 8+ (Required for H2O)

## Installation & Setup

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify Java installation (required for H2O)
java -version

# Frontend Setup
cd automl-frontend
npm install
npm run dev

# At last
uvicorn main:app --reload --host 0.0.0.0 --port 8000