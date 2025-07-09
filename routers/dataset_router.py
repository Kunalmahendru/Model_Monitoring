from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils.file_utils import save_uploaded_file
from services.automl_services import run_automl

import pandas as pd
from fastapi.responses import JSONResponse
from io import StringIO
import tempfile
import mlflow
import h2o
from h2o.automl import H2OAutoML
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Basic file size check (10MB limit)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Reset file pointer and validate CSV format
        await file.seek(0)
        try:
            df = pd.read_csv(StringIO(contents.decode()))
            if df.empty:
                raise HTTPException(status_code=400, detail="CSV file is empty")
            if len(df.columns) < 2:
                raise HTTPException(status_code=400, detail="CSV must have at least 2 columns (features + target)")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        # Save file
        file_path = await save_uploaded_file(file)

        # Log file as artifact with metadata
        with mlflow.start_run(run_name=f"Dataset_Upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.log_artifact(file_path, artifact_path="dataset")
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("file_size_mb", len(contents) / (1024*1024))
            mlflow.log_param("num_rows", len(df))
            mlflow.log_param("num_columns", len(df.columns))
            mlflow.log_param("target_column", df.columns[-1])

        logger.info(f"Dataset uploaded successfully: {file.filename}")
        return JSONResponse(status_code=200, content={
            "message": f"Dataset '{file.filename}' uploaded successfully!",
            "rows": len(df),
            "columns": len(df.columns),
            "target": df.columns[-1]
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

@router.post("/train-model/")
async def train_model(file: UploadFile = File(...)):
    try:
        logger.info("Starting AutoML training...")
        
        # Read and validate file
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Validate dataset
        df = pd.read_csv(tmp_path)
        if df.isnull().sum().sum() > 0:
            logger.warning("Dataset contains null values - H2O will handle this automatically")

        # Set up MLflow experiment
        experiment_name = "AutoML_Experiments"
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass  # Experiment already exists
        
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"AutoML_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log dataset info
            mlflow.log_artifact(tmp_path, artifact_path="dataset")
            mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("target_column", df.columns[-1])
            
            # Initialize H2O (handle if already running)
            try:
                h2o.init()
            except:
                h2o.init(force=True)
            
            # Load data and prepare for training
            data = h2o.import_file(tmp_path)
            
            # Get target and feature columns
            y = data.columns[-1]  # Assume last column is target
            x = data.columns[:-1]  # All other columns are features
            
            # Convert target to factor for classification
            data[y] = data[y].asfactor()
            
            # Split data for validation
            train, valid = data.split_frame(ratios=[0.8], seed=1)
            
            # Configure AutoML
            aml = H2OAutoML(
                max_models=10,  # Increased for better model variety
                seed=1,
                max_runtime_secs=300,  # 5 minutes max
                sort_metric="AUC"  # For classification
            )
            
            logger.info("Training AutoML models...")
            aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
            
            # Get leaderboard
            lb = aml.leaderboard.as_data_frame()
            
            # Log metrics for best model
            best_model = aml.leader
            mlflow.log_metric("best_auc", float(lb.iloc[0]['auc']))
            mlflow.log_metric("best_logloss", float(lb.iloc[0]['logloss']))
            mlflow.log_param("best_model_id", str(lb.iloc[0]['model_id']))
            mlflow.log_param("total_models_trained", len(lb))
            
            # Save best model
            model_path = h2o.save_model(best_model, path="/tmp", force=True)
            mlflow.log_artifact(model_path, artifact_path="best_model")
            
            logger.info(f"Training completed. Best model: {lb.iloc[0]['model_id']}")
            
            return JSONResponse(content={
                "message": f"Training completed! {len(lb)} models trained.",
                "leaderboard": lb.to_dict(orient="records"),
                "best_model": {
                    "id": str(lb.iloc[0]['model_id']),
                    "auc": float(lb.iloc[0]['auc']),
                    "logloss": float(lb.iloc[0]['logloss'])
                }
            })

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

@router.post("/preview")
async def preview_dataset(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        
        # Get basic statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        preview_data = {
            "columns": df.columns.tolist(),
            "rows": df.head(10).to_dict(orient='records'),  # Show more rows
            "shape": df.shape,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": df.isnull().sum().to_dict(),
            "target_column": df.columns[-1],
            "target_unique_values": df[df.columns[-1]].nunique() if df.columns[-1] in df.columns else 0
        }
        
        return JSONResponse(content=preview_data)
    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# from fastapi import APIRouter, UploadFile, File, HTTPException, Form
# from fastapi.responses import JSONResponse
# from utils.file_utils import save_uploaded_file
# import pandas as pd
# from io import StringIO
# import tempfile
# import mlflow
# import mlflow.h2o
# from mlflow.tracking import MlflowClient
# import h2o
# from h2o.automl import H2OAutoML
# import os
# import logging
# from datetime import datetime
# import json
# import pickle
# from typing import Dict, Any, List

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # MLflow Configuration
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# # Global variables for model storage
# trained_models: Dict[str, Any] = {}
# model_metadata: Dict[str, Dict] = {}

# router = APIRouter()

# def ensure_mlflow_experiment(experiment_name: str = "AutoML_Experiments"):
#     """Ensure MLflow experiment exists"""
#     try:
#         mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#         experiment = mlflow.get_experiment_by_name(experiment_name)
#         if experiment is None:
#             experiment_id = mlflow.create_experiment(experiment_name)
#             logger.info(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
#         else:
#             logger.info(f"Using existing experiment: {experiment_name}")
#         mlflow.set_experiment(experiment_name)
#         return True
#     except Exception as e:
#         logger.error(f"MLflow experiment setup error: {str(e)}")
#         return False

# def initialize_h2o():
#     """Initialize H2O cluster"""
#     try:
#         h2o.init()
#         logger.info("H2O initialized successfully")
#     except Exception as e:
#         try:
#             h2o.init(force=True)
#             logger.info("H2O initialized with force=True")
#         except Exception as e2:
#             logger.error(f"H2O initialization failed: {str(e2)}")
#             raise HTTPException(status_code=500, detail="Failed to initialize H2O")

# @router.post("/upload-dataset/")
# async def upload_dataset(file: UploadFile = File(...)):
#     """Upload and validate dataset"""
#     try:
#         # Validate file type
#         if not file.filename.endswith('.csv'):
#             raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
#         # Basic file size check (10MB limit)
#         contents = await file.read()
#         if len(contents) > 10 * 1024 * 1024:  # 10MB
#             raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
#         # Reset file pointer and validate CSV format
#         await file.seek(0)
#         try:
#             df = pd.read_csv(StringIO(contents.decode()))
#             if df.empty:
#                 raise HTTPException(status_code=400, detail="CSV file is empty")
#             if len(df.columns) < 2:
#                 raise HTTPException(status_code=400, detail="CSV must have at least 2 columns (features + target)")
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

#         # Save file
#         file_path = await save_uploaded_file(file)

#         # Ensure MLflow experiment exists
#         ensure_mlflow_experiment()

#         # Log file as artifact with metadata
#         with mlflow.start_run(run_name=f"Dataset_Upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
#             mlflow.log_artifact(file_path, artifact_path="dataset")
#             mlflow.log_param("filename", file.filename)
#             mlflow.log_param("file_size_mb", round(len(contents) / (1024*1024), 2))
#             mlflow.log_param("num_rows", len(df))
#             mlflow.log_param("num_columns", len(df.columns))
#             mlflow.log_param("target_column", df.columns[-1])
#             mlflow.log_param("upload_timestamp", datetime.now().isoformat())
            
#             # Log basic data statistics
#             mlflow.log_param("missing_values", df.isnull().sum().sum())
#             mlflow.log_param("numeric_columns", len(df.select_dtypes(include=['number']).columns))
#             mlflow.log_param("categorical_columns", len(df.select_dtypes(include=['object']).columns))

#         logger.info(f"Dataset uploaded successfully: {file.filename}")
#         return JSONResponse(status_code=200, content={
#             "message": f"Dataset '{file.filename}' uploaded successfully!",
#             "rows": len(df),
#             "columns": len(df.columns),
#             "target": df.columns[-1],
#             "run_id": run.info.run_id
#         })
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Upload error: {str(e)}")
#         return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

# @router.post("/train-model/")
# async def train_model(file: UploadFile = File(...)):
#     """Train AutoML models"""
#     try:
#         logger.info("Starting AutoML training...")
        
#         # Initialize H2O
#         initialize_h2o()
        
#         # Read and validate file
#         contents = await file.read()
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
#             tmp.write(contents)
#             tmp_path = tmp.name

#         # Validate dataset
#         df = pd.read_csv(tmp_path)
#         if df.isnull().sum().sum() > 0:
#             logger.warning("Dataset contains null values - H2O will handle this automatically")

#         # Ensure MLflow experiment exists
#         ensure_mlflow_experiment()

#         with mlflow.start_run(run_name=f"AutoML_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
#             run_id = run.info.run_id
            
#             # Log dataset info
#             mlflow.log_artifact(tmp_path, artifact_path="dataset")
#             mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
#             mlflow.log_param("target_column", df.columns[-1])
#             mlflow.log_param("training_timestamp", datetime.now().isoformat())
            
#             # Load data and prepare for training
#             data = h2o.import_file(tmp_path)
            
#             # Get target and feature columns
#             y = data.columns[-1]  # Assume last column is target
#             x = data.columns[:-1]  # All other columns are features
            
#             # Convert target to factor for classification
#             data[y] = data[y].asfactor()
            
#             # Split data for validation
#             train, valid = data.split_frame(ratios=[0.8], seed=1)
            
#             # Configure AutoML
#             aml = H2OAutoML(
#                 max_models=10,
#                 seed=1,
#                 max_runtime_secs=300,  # 5 minutes max
#                 sort_metric="AUC"
#             )
            
#             logger.info("Training AutoML models...")
#             aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
            
#             # Get leaderboard
#             lb = aml.leaderboard.as_data_frame()
            
#             # Log metrics for best model
#             best_model = aml.leader
#             mlflow.log_metric("best_auc", float(lb.iloc[0]['auc']))
#             mlflow.log_metric("best_logloss", float(lb.iloc[0]['logloss']))
#             mlflow.log_param("best_model_id", str(lb.iloc[0]['model_id']))
#             mlflow.log_param("total_models_trained", len(lb))
            
#             # Save best model to temporary location
#             model_path = h2o.save_model(best_model, path="/tmp", force=True)
#             mlflow.log_artifact(model_path, artifact_path="best_model")
            
#             # Store model in memory for predictions
#             best_model_id = str(lb.iloc[0]['model_id'])
#             trained_models[best_model_id] = best_model
#             model_metadata[best_model_id] = {
#                 "run_id": run_id,
#                 "model_path": model_path,
#                 "target_column": y,
#                 "feature_columns": x,
#                 "training_time": datetime.now().isoformat(),
#                 "metrics": {
#                     "auc": float(lb.iloc[0]['auc']),
#                     "logloss": float(lb.iloc[0]['logloss'])
#                 }
#             }
            
#             # Log model metadata
#             mlflow.log_dict(model_metadata[best_model_id], "model_metadata.json")
            
#             logger.info(f"Training completed. Best model: {best_model_id}")
            
#             return JSONResponse(content={
#                 "message": f"Training completed! {len(lb)} models trained.",
#                 "run_id": run_id,
#                 "leaderboard": lb.to_dict(orient="records"),
#                 "best_model": {
#                     "id": best_model_id,
#                     "auc": float(lb.iloc[0]['auc']),
#                     "logloss": float(lb.iloc[0]['logloss'])
#                 }
#             })

#     except Exception as e:
#         logger.error(f"Training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
#     finally:
#         # Clean up temp file
#         try:
#             os.unlink(tmp_path)
#         except:
#             pass

# @router.post("/predict/")
# async def predict(file: UploadFile = File(...), model_id: str = Form(...)):
#     """Make predictions using selected model"""
#     try:
#         logger.info(f"Making predictions with model: {model_id}")
        
#         # Check if model exists
#         if model_id not in trained_models:
#             raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
#         # Initialize H2O
#         initialize_h2o()
        
#         # Ensure MLflow experiment exists
#         ensure_mlflow_experiment()
        
#         with mlflow.start_run(run_name=f"Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
#             # Log prediction request
#             mlflow.log_param("model_id", model_id)
#             mlflow.log_param("prediction_file", file.filename)
#             mlflow.log_param("prediction_timestamp", datetime.now().isoformat())
            
#             # Read prediction data
#             contents = await file.read()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
#                 tmp.write(contents)
#                 tmp_path = tmp.name
            
#             # Load data for prediction
#             df = pd.read_csv(tmp_path)
#             data = h2o.import_file(tmp_path)
            
#             # Get the trained model
#             model = trained_models[model_id]
#             metadata = model_metadata[model_id]
            
#             # Make predictions
#             predictions = model.predict(data)
#             predictions_df = predictions.as_data_frame()
            
#             # Convert predictions to serializable format
#             predictions_list = []
#             for i in range(len(predictions_df)):
#                 pred_dict = {}
#                 for col in predictions_df.columns:
#                     value = predictions_df.iloc[i][col]
#                     if pd.isna(value):
#                         pred_dict[col] = None
#                     else:
#                         pred_dict[col] = float(value) if isinstance(value, (int, float)) else str(value)
#                 predictions_list.append(pred_dict)
            
#             # Log prediction results
#             mlflow.log_metric("num_predictions", len(predictions_list))
#             mlflow.log_param("prediction_columns", list(predictions_df.columns))
            
#             # Save predictions as artifact
#             pred_file = f"/tmp/predictions_{run.info.run_id}.csv"
#             predictions_df.to_csv(pred_file, index=False)
#             mlflow.log_artifact(pred_file, artifact_path="predictions")
            
#             # Clean up
#             os.unlink(tmp_path)
#             os.unlink(pred_file)
            
#             logger.info(f"Predictions completed: {len(predictions_list)} rows")
            
#             return JSONResponse(content={
#                 "message": f"Predictions completed for {len(predictions_list)} rows",
#                 "predictions": predictions_list,
#                 "model_id": model_id,
#                 "run_id": run.info.run_id
#             })
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# @router.get("/models")
# async def list_models():
#     """List all trained models"""
#     try:
#         client = MlflowClient()
        
#         # Get experiments
#         experiments = client.search_experiments()
        
#         models = []
#         for exp in experiments:
#             runs = client.search_runs(exp.experiment_id)
#             for run in runs:
#                 if run.info.status == "FINISHED" and "AutoML_Training" in run.info.run_name:
#                     model_info = {
#                         "run_id": run.info.run_id,
#                         "run_name": run.info.run_name,
#                         "experiment_id": exp.experiment_id,
#                         "experiment_name": exp.name,
#                         "status": run.info.status,
#                         "start_time": run.info.start_time,
#                         "end_time": run.info.end_time,
#                         "metrics": run.data.metrics,
#                         "params": run.data.params
#                     }
#                     models.append(model_info)
        
#         # Also add currently loaded models
#         loaded_models = []
#         for model_id, metadata in model_metadata.items():
#             loaded_models.append({
#                 "model_id": model_id,
#                 "metadata": metadata,
#                 "status": "loaded"
#             })
        
#         return JSONResponse(content={
#             "total_models": len(models),
#             "mlflow_models": models,
#             "loaded_models": loaded_models
#         })
        
#     except Exception as e:
#         logger.error(f"Model listing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

# @router.get("/model/{model_id}")
# async def get_model_details(model_id: str):
#     """Get details of a specific model"""
#     try:
#         if model_id not in model_metadata:
#             raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
#         metadata = model_metadata[model_id]
        
#         # Get MLflow run details
#         client = MlflowClient()
#         run = client.get_run(metadata["run_id"])
        
#         return JSONResponse(content={
#             "model_id": model_id,
#             "metadata": metadata,
#             "mlflow_run": {
#                 "run_id": run.info.run_id,
#                 "status": run.info.status,
#                 "start_time": run.info.start_time,
#                 "end_time": run.info.end_time,
#                 "metrics": run.data.metrics,
#                 "params": run.data.params
#             }
#         })
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Model details error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")

# @router.delete("/model/{model_id}")
# async def delete_model(model_id: str):
#     """Delete a model from memory"""
#     try:
#         if model_id not in trained_models:
#             raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
#         # Remove from memory
#         del trained_models[model_id]
#         del model_metadata[model_id]
        
#         logger.info(f"Model {model_id} deleted from memory")
        
#         return JSONResponse(content={
#             "message": f"Model {model_id} deleted successfully"
#         })
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Model deletion error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

# @router.post("/preview")
# async def preview_dataset(file: UploadFile = File(...)):
#     """Preview dataset before training"""
#     try:
#         # Validate file type
#         if not file.filename.endswith('.csv'):
#             raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
#         contents = await file.read()
#         df = pd.read_csv(StringIO(contents.decode()))
        
#         # Get basic statistics
#         numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#         categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
#         # Calculate basic statistics for numeric columns
#         numeric_stats = {}
#         for col in numeric_cols:
#             numeric_stats[col] = {
#                 "mean": float(df[col].mean()),
#                 "std": float(df[col].std()),
#                 "min": float(df[col].min()),
#                 "max": float(df[col].max()),
#                 "median": float(df[col].median())
#             }
        
#         # Calculate value counts for categorical columns (top 10)
#         categorical_stats = {}
#         for col in categorical_cols:
#             value_counts = df[col].value_counts().head(10)
#             categorical_stats[col] = {
#                 "unique_values": int(df[col].nunique()),
#                 "top_values": value_counts.to_dict()
#             }
        
#         preview_data = {
#             "filename": file.filename,
#             "columns": df.columns.tolist(),
#             "sample_rows": df.head(10).to_dict(orient='records'),
#             "shape": df.shape,
#             "numeric_columns": numeric_cols,
#             "categorical_columns": categorical_cols,
#             "numeric_stats": numeric_stats,
#             "categorical_stats": categorical_stats,
#             "missing_values": df.isnull().sum().to_dict(),
#             "target_column": df.columns[-1],
#             "target_unique_values": int(df[df.columns[-1]].nunique()) if df.columns[-1] in df.columns else 0,
#             "data_types": df.dtypes.astype(str).to_dict()
#         }
        
#         return JSONResponse(content=preview_data)
        
#     except Exception as e:
#         logger.error(f"Preview error: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

# @router.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     try:
#         # Check H2O status
#         h2o_status = "unknown"
#         try:
#             h2o.cluster().show_status()
#             h2o_status = "running"
#         except:
#             h2o_status = "stopped"
        
#         # Check MLflow connection
#         mlflow_status = "unknown"
#         try:
#             client = MlflowClient()
#             experiments = client.search_experiments()
#             mlflow_status = "connected"
#         except:
#             mlflow_status = "disconnected"
        
#         return JSONResponse(content={
#             "status": "healthy",
#             "timestamp": datetime.now().isoformat(),
#             "h2o_status": h2o_status,
#             "mlflow_status": mlflow_status,
#             "loaded_models": len(trained_models),
#             "mlflow_tracking_uri": MLFLOW_TRACKING_URI
#         })
        
#     except Exception as e:
#         logger.error(f"Health check error: {str(e)}")
#         return JSONResponse(status_code=500, content={
#             "status": "unhealthy",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         })

# @router.get("/experiments")
# async def list_experiments():
#     """List all MLflow experiments"""
#     try:
#         client = MlflowClient()
#         experiments = client.search_experiments()
        
#         experiment_list = []
#         for exp in experiments:
#             runs = client.search_runs(exp.experiment_id)
#             experiment_list.append({
#                 "experiment_id": exp.experiment_id,
#                 "name": exp.name,
#                 "lifecycle_stage": exp.lifecycle_stage,
#                 "total_runs": len(runs),
#                 "creation_time": exp.creation_time
#             })
        
#         return JSONResponse(content={
#             "experiments": experiment_list,
#             "total_experiments": len(experiment_list)
#         })
        
#     except Exception as e:
#         logger.error(f"Experiments listing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")

# @router.get("/runs/{experiment_id}")
# async def list_runs(experiment_id: str):
#     """List all runs in an experiment"""
#     try:
#         client = MlflowClient()
#         runs = client.search_runs(experiment_id)
        
#         run_list = []
#         for run in runs:
#             run_list.append({
#                 "run_id": run.info.run_id,
#                 "run_name": run.info.run_name,
#                 "status": run.info.status,
#                 "start_time": run.info.start_time,
#                 "end_time": run.info.end_time,
#                 "metrics": run.data.metrics,
#                 "params": run.data.params
#             })
        
#         return JSONResponse(content={
#             "experiment_id": experiment_id,
#             "runs": run_list,
#             "total_runs": len(run_list)
#         })
        
#     except Exception as e:
#         logger.error(f"Runs listing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")