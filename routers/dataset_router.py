from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from utils.file_utils import save_uploaded_file
from services.automl_services import run_automl

import pandas as pd
from fastapi.responses import JSONResponse
from io import StringIO
import tempfile
import mlflow
import mlflow.h2o
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import *
import os
import logging
from datetime import datetime
import json
import time
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

router = APIRouter()

# Model mapping for H2O - Only actual H2O algorithms
MODEL_MAPPING = {
    'xgboost': (['XGBoost'], 'XGBoost', '🚀'),
    'randomforest': (['DRF'], 'Random Forest (DRF)', '🌲'),
    'gbm': (['GBM'], 'Gradient Boosting Machine', '⚡'),
    'glm': (['GLM'], 'Generalized Linear Model', '📊'),
    'neural': (['DeepLearning'], 'Deep Learning', '🧠'),
    'ensemble': (['StackedEnsemble'], 'Stacked Ensemble', '🔗')
}

def ensure_mlflow_experiment(experiment_name: str = "AutoML_Experiments"):
    """Ensure MLflow experiment exists and is set"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        return True
    except Exception as e:
        logger.error(f"MLflow experiment setup error: {str(e)}")
        return False

def extract_model_parameters(model):
    """Extract detailed hyperparameters from H2O model"""
    try:
        params = {}
        
        # Get the model's actual parameters
        if hasattr(model, 'actual_params') and model.actual_params:
            for key, value in model.actual_params.items():
                # Skip internal H2O parameters
                if not key.startswith('_') and key not in ['training_frame', 'validation_frame', 'response_column']:
                    params[key] = value
        
        # Algorithm-specific parameter extraction
        algo = model.algo.upper()
        
        if algo == 'DRF':  # Random Forest
            params.update({
                'ntrees': getattr(model, 'ntrees', None),
                'max_depth': getattr(model, 'max_depth', None),
                'min_rows': getattr(model, 'min_rows', None),
                'mtries': getattr(model, 'mtries', None),
                'sample_rate': getattr(model, 'sample_rate', None),
                'col_sample_rate': getattr(model, 'col_sample_rate', None),
                'binomial_double_trees': getattr(model, 'binomial_double_trees', None),
            })
        
        elif algo == 'GBM':  # Gradient Boosting
            params.update({
                'ntrees': getattr(model, 'ntrees', None),
                'max_depth': getattr(model, 'max_depth', None),
                'min_rows': getattr(model, 'min_rows', None),
                'learn_rate': getattr(model, 'learn_rate', None),
                'sample_rate': getattr(model, 'sample_rate', None),
                'col_sample_rate': getattr(model, 'col_sample_rate', None),
                'col_sample_rate_per_tree': getattr(model, 'col_sample_rate_per_tree', None),
                'min_split_improvement': getattr(model, 'min_split_improvement', None),
                'histogram_type': getattr(model, 'histogram_type', None),
                'regularization_x': getattr(model, 'regularization_x', None),
                'regularization_y': getattr(model, 'regularization_y', None),
            })
        
        elif algo == 'XGBOOST':  # XGBoost
            params.update({
                'ntrees': getattr(model, 'ntrees', None),
                'max_depth': getattr(model, 'max_depth', None),
                'min_rows': getattr(model, 'min_rows', None),
                'learn_rate': getattr(model, 'learn_rate', None),
                'sample_rate': getattr(model, 'sample_rate', None),
                'col_sample_rate': getattr(model, 'col_sample_rate', None),
                'col_sample_rate_per_tree': getattr(model, 'col_sample_rate_per_tree', None),
                'reg_alpha': getattr(model, 'reg_alpha', None),
                'reg_lambda': getattr(model, 'reg_lambda', None),
                'booster': getattr(model, 'booster', None),
                'normalize_type': getattr(model, 'normalize_type', None),
                'dropout_rate': getattr(model, 'dropout_rate', None),
            })
        
        elif algo == 'GLM':  # Generalized Linear Model
            params.update({
                'family': getattr(model, 'family', None),
                'solver': getattr(model, 'solver', None),
                'alpha': getattr(model, 'alpha', None),
                'lambda_': getattr(model, 'lambda_', None),
                'standardize': getattr(model, 'standardize', None),
                'remove_collinear_columns': getattr(model, 'remove_collinear_columns', None),
                'compute_p_values': getattr(model, 'compute_p_values', None),
                'max_iterations': getattr(model, 'max_iterations', None),
                'link': getattr(model, 'link', None),
            })
        
        elif algo == 'DEEPLEARNING':  # Neural Network
            params.update({
                'hidden': getattr(model, 'hidden', None),
                'epochs': getattr(model, 'epochs', None),
                'activation': getattr(model, 'activation', None),
                'learning_rate': getattr(model, 'learning_rate', None),
                'momentum': getattr(model, 'momentum', None),
                'dropout': getattr(model, 'dropout', None),
                'l1': getattr(model, 'l1', None),
                'l2': getattr(model, 'l2', None),
                'input_dropout_ratio': getattr(model, 'input_dropout_ratio', None),
                'hidden_dropout_ratios': getattr(model, 'hidden_dropout_ratios', None),
                'adaptive_rate': getattr(model, 'adaptive_rate', None),
                'rho': getattr(model, 'rho', None),
                'epsilon': getattr(model, 'epsilon', None),
            })
        
        elif algo == 'STACKEDENSEMBLE':  # Ensemble
            params.update({
                'metalearner_algorithm': getattr(model, 'metalearner_algorithm', None),
                'metalearner_nfolds': getattr(model, 'metalearner_nfolds', None),
                'metalearner_fold_assignment': getattr(model, 'metalearner_fold_assignment', None),
                'base_models': str(getattr(model, 'base_models', None)),
            })
        
        # Remove None values and convert to string for MLflow compatibility
        filtered_params = {}
        for key, value in params.items():
            if value is not None:
                if isinstance(value, (list, tuple)):
                    filtered_params[key] = str(value)
                elif isinstance(value, dict):
                    filtered_params[key] = json.dumps(value)
                else:
                    filtered_params[key] = str(value)
        
        return filtered_params
        
    except Exception as e:
        logger.warning(f"Could not extract parameters: {str(e)}")
        return {}

def extract_cv_metrics(model):
    """Extract cross-validation metrics if available"""
    try:
        cv_metrics = {}
        
        # Check if model has cross-validation summary
        if hasattr(model, '_model_json') and model._model_json:
            model_json = model._model_json
            
            # Extract cross-validation metrics
            if 'output' in model_json and 'cross_validation_metrics_summary' in model_json['output']:
                cv_summary = model_json['output']['cross_validation_metrics_summary']
                
                # Common CV metrics
                for metric in ['auc', 'logloss', 'rmse', 'mse', 'mean_per_class_error']:
                    if metric in cv_summary:
                        # Get mean value if available
                        metric_data = cv_summary[metric]
                        if isinstance(metric_data, list) and len(metric_data) > 0:
                            if isinstance(metric_data[0], list) and len(metric_data[0]) > 0:
                                cv_metrics[f"{metric}_mean"] = float(metric_data[0][0])
                                if len(metric_data[0]) > 1:
                                    cv_metrics[f"{metric}_std"] = float(metric_data[0][1])
        
        return cv_metrics
        
    except Exception as e:
        logger.warning(f"Could not extract CV metrics: {str(e)}")
        return {}

def map_h2o_to_frontend(h2o_model_id: str, selected_models: List[str]) -> dict:
    """Map H2O model back to frontend model selection"""
    
    # Create a clean mapping from H2O algorithm names to frontend info
    h2o_id_lower = h2o_model_id.lower()
    
    # Direct mapping from H2O algorithm to frontend
    algorithm_mapping = {
        'xgboost': ('xgboost', 'XGBoost', '🚀'),
        'drf': ('randomforest', 'Random Forest (DRF)', '🌲'),
        'gbm': ('gbm', 'Gradient Boosting Machine', '⚡'),
        'glm': ('glm', 'Generalized Linear Model', '📊'),
        'deeplearning': ('neural', 'Deep Learning', '🧠'),
        'stackedensemble': ('ensemble', 'Stacked Ensemble', '🔗')
    }
    
    # Find the matching algorithm
    for h2o_algo, (frontend_id, display_name, icon) in algorithm_mapping.items():
        if h2o_algo in h2o_id_lower:
            # Only return if this model type was actually selected
            if frontend_id in selected_models:
                return {
                    'frontend_id': frontend_id,
                    'display_name': display_name,
                    'icon': icon
                }
    
    # Default fallback
    return {
        'frontend_id': 'unknown',
        'display_name': h2o_model_id,
        'icon': '🤖'
    }

def initialize_h2o():
    """Initialize H2O cluster with proper error handling"""
    try:
        # Check if H2O is already running
        try:
            h2o.cluster().show_status()
            logger.info("H2O cluster already running")
            return True
        except:
            pass
        
        # Initialize H2O
        h2o.init(max_mem_size="4G", nthreads=-1, port=54321)
        logger.info("H2O initialized successfully")
        return True
    except Exception as e:
        try:
            h2o.init(force=True, max_mem_size="4G", nthreads=-1)
            logger.info("H2O initialized with force=True")
            return True
        except Exception as e2:
            logger.error(f"H2O initialization failed: {str(e2)}")
            return False

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

        # Ensure MLflow experiment exists
        ensure_mlflow_experiment()

        # Log file as artifact with metadata
        with mlflow.start_run(run_name=f"Dataset_Upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.log_artifact(file_path, artifact_path="dataset")
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("file_size_mb", round(len(contents) / (1024*1024), 2))
            mlflow.log_param("num_rows", len(df))
            mlflow.log_param("num_columns", len(df.columns))
            mlflow.log_param("target_column", df.columns[-1])
            mlflow.log_param("upload_timestamp", datetime.now().isoformat())

        logger.info(f"Dataset uploaded successfully: {file.filename}")
        return JSONResponse(status_code=200, content={
            "message": f"Dataset '{file.filename}' uploaded successfully!",
            "rows": len(df),
            "columns": len(df.columns),
            "target": df.columns[-1],
            "run_id": run.info.run_id
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

@router.post("/train-model/")
async def train_model(file: UploadFile = File(...), models: str = Form(...)):
    """Train selected AutoML models with MLflow logging"""
    tmp_path = None
    try:
        logger.info("Starting AutoML training...")
        
        # Parse selected models from frontend
        try:
            selected_models = json.loads(models)
            logger.info(f"Selected models for training: {selected_models}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid models format")
        
        if not selected_models:
            raise HTTPException(status_code=400, detail="No models selected for training")
        
        # Initialize H2O
        if not initialize_h2o():
            raise HTTPException(status_code=500, detail="Failed to initialize H2O cluster")
        
        # Read and validate file
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Validate dataset
        df = pd.read_csv(tmp_path)
        if df.isnull().sum().sum() > 0:
            logger.warning("Dataset contains null values - H2O will handle this automatically")

        # Ensure MLflow experiment exists
        ensure_mlflow_experiment()

        with mlflow.start_run(run_name=f"AutoML_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started with ID: {run_id}")
            
            # Log dataset info and training parameters
            mlflow.log_artifact(tmp_path, artifact_path="dataset")
            mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("target_column", df.columns[-1])
            mlflow.log_param("training_timestamp", datetime.now().isoformat())
            mlflow.log_param("selected_models", selected_models)
            mlflow.log_param("num_selected_models", len(selected_models))
            
            # Load data into H2O
            data = h2o.import_file(tmp_path)
            logger.info(f"Data loaded into H2O: {data.shape}")
            
            # Get target and feature columns
            y = data.columns[-1]  # Assume last column is target
            x = data.columns[:-1]  # All other columns are features
            
            # Convert target to factor for classification
            data[y] = data[y].asfactor()
            mlflow.log_param("problem_type", "classification")
            mlflow.log_param("num_features", len(x))
            mlflow.log_param("num_classes", data[y].nlevels()[0])
            
            # Split data for validation
            train, valid = data.split_frame(ratios=[0.8], seed=1)
            mlflow.log_param("train_size", train.nrows)
            mlflow.log_param("valid_size", valid.nrows)
            
            # Map selected models to H2O algorithms
            h2o_algorithms = []
            for model_id in selected_models:
                if model_id in MODEL_MAPPING:
                    h2o_algorithms.extend(MODEL_MAPPING[model_id][0])  # Get algorithms list
                else:
                    logger.warning(f"Unknown model ID: {model_id}, skipping...")
            
            # Remove duplicates and log
            h2o_algorithms = list(set(h2o_algorithms))
            mlflow.log_param("h2o_algorithms", h2o_algorithms)
            logger.info(f"H2O algorithms to train: {h2o_algorithms}")
            
            # Configure AutoML with selected algorithms
            aml = H2OAutoML(
                max_models=20,  # Allow more models for variety
                seed=1,
                max_runtime_secs=600,  # 10 minutes max
                sort_metric="AUC",  # For classification
                include_algos=h2o_algorithms,  # Only train selected algorithms
                exclude_algos=None,  # Don't exclude any from the selected ones
                nfolds=5,  # Cross-validation folds
                balance_classes=True,  # Handle class imbalance
                stopping_metric="AUC",
                stopping_tolerance=0.001,
                stopping_rounds=3
            )
            
            logger.info("Starting AutoML training with selected models...")
            start_time = datetime.now()
            
            # Train the models
            aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
            
            training_duration = (datetime.now() - start_time).total_seconds()
            mlflow.log_metric("training_duration_seconds", training_duration)
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
            # Get leaderboard
            lb = aml.leaderboard.as_data_frame()
            logger.info(f"Training completed. {len(lb)} models trained.")
            
            # Log overall training metrics
            mlflow.log_param("total_models_trained", len(lb))
            mlflow.log_metric("best_auc", float(lb.iloc[0]['auc']))
            mlflow.log_metric("best_logloss", float(lb.iloc[0]['logloss']))
            mlflow.log_param("best_model_id", str(lb.iloc[0]['model_id']))
            
            # Log individual model metrics
            for idx, row in lb.iterrows():
                model_prefix = f"model_{idx}"
                mlflow.log_metric(f"{model_prefix}_auc", float(row['auc']))
                mlflow.log_metric(f"{model_prefix}_logloss", float(row['logloss']))
                mlflow.log_param(f"{model_prefix}_id", str(row['model_id']))
                
                # **ENHANCED: Log detailed parameters for top 3 models**
                if idx < 3:  # Top 3 models
                    try:
                        model = h2o.get_model(str(row['model_id']))
                        
                        # Extract all model parameters
                        model_params = extract_model_parameters(model)
                        
                        # Create nested run for detailed parameter logging
                        with mlflow.start_run(run_name=f"TOP_{idx+1}_{str(row['model_id'])[:20]}", nested=True):
                            # Basic model info
                            mlflow.log_param("rank", idx + 1)
                            mlflow.log_param("model_id", str(row['model_id']))
                            mlflow.log_param("algorithm", model.algo)
                            mlflow.log_param("is_top_performer", True)
                            
                            # Performance metrics
                            mlflow.log_metric("auc", float(row['auc']))
                            mlflow.log_metric("logloss", float(row['logloss']))
                            if 'rmse' in row:
                                mlflow.log_metric("rmse", float(row['rmse']))
                            if 'mse' in row:
                                mlflow.log_metric("mse", float(row['mse']))
                            if 'mean_per_class_error' in row:
                                mlflow.log_metric("mean_per_class_error", float(row['mean_per_class_error']))
                            
                            # **Log all hyperparameters**
                            for param_name, param_value in model_params.items():
                                mlflow.log_param(f"hp_{param_name}", param_value)
                            
                            # **Extract and log cross-validation metrics**
                            cv_metrics = extract_cv_metrics(model)
                            for cv_metric, cv_value in cv_metrics.items():
                                mlflow.log_metric(f"cv_{cv_metric}", cv_value)
                            
                            # **Variable importance for tree-based models**
                            try:
                                if hasattr(model, 'varimp') and model.algo.upper() in ['DRF', 'GBM', 'XGBOOST']:
                                    varimp = model.varimp(use_pandas=True)
                                    if varimp is not None and len(varimp) > 0:
                                        # Log top 15 important features
                                        for var_idx, var_row in varimp.head(15).iterrows():
                                            feature_name = var_row['variable']
                                            importance = var_row['relative_importance']
                                            mlflow.log_metric(f"importance_{feature_name}", importance)
                                        
                                        # Save variable importance as artifact
                                        varimp_file = f"/tmp/varimp_rank_{idx+1}_{str(row['model_id'])[:10]}.csv"
                                        varimp.to_csv(varimp_file, index=False)
                                        mlflow.log_artifact(varimp_file, artifact_path="variable_importance")
                                        os.remove(varimp_file)
                                        
                                        # Log summary stats
                                        mlflow.log_metric("num_features_used", len(varimp))
                                        mlflow.log_metric("top_feature_importance", varimp.iloc[0]['relative_importance'])
                            except Exception as varimp_e:
                                logger.warning(f"Could not extract variable importance for {row['model_id']}: {str(varimp_e)}")
                            
                            # **Training metrics and model summary**
                            try:
                                # Training time if available
                                if hasattr(model, '_model_json') and 'output' in model._model_json:
                                    output = model._model_json['output']
                                    if 'run_time' in output:
                                        mlflow.log_metric("training_time_ms", output['run_time'])
                                    if 'model_summary' in output:
                                        summary = output['model_summary']
                                        if isinstance(summary, list) and len(summary) > 1:
                                            # Log model summary info (varies by algorithm)
                                            summary_data = summary[1] if len(summary) > 1 else summary[0]
                                            if isinstance(summary_data, list):
                                                for i, val in enumerate(summary_data[:5]):  # First 5 summary values
                                                    try:
                                                        mlflow.log_metric(f"summary_metric_{i}", float(val))
                                                    except (ValueError, TypeError):
                                                        mlflow.log_param(f"summary_param_{i}", str(val))
                            except Exception as summary_e:
                                logger.warning(f"Could not extract training summary for {row['model_id']}: {str(summary_e)}")
                            
                            # **Save model artifact for top 3**
                            try:
                                model_path = h2o.save_model(model, path="/tmp", force=True)
                                mlflow.log_artifact(model_path, artifact_path=f"top_models/rank_{idx+1}")
                                
                                # Also save MOJO if supported
                                if hasattr(model, 'save_mojo'):
                                    mojo_path = model.save_mojo(path="/tmp", force=True)
                                    mlflow.log_artifact(mojo_path, artifact_path=f"mojo_models/rank_{idx+1}")
                                    os.remove(mojo_path)
                                
                                os.remove(model_path)
                                logger.info(f"Saved artifacts for top model #{idx+1}: {row['model_id']}")
                            except Exception as save_e:
                                logger.warning(f"Could not save model artifacts for {row['model_id']}: {str(save_e)}")
                            
                            logger.info(f"Detailed logging completed for TOP MODEL #{idx+1}: {row['model_id']}")
                    
                    except Exception as detail_e:
                        logger.error(f"Could not log detailed parameters for top model {row['model_id']}: {str(detail_e)}")
                
                # Log model type based on model_id
                model_type = "unknown"
                model_id_str = str(row['model_id']).lower()
                if "xgboost" in model_id_str:
                    model_type = "XGBoost"
                elif "drf" in model_id_str:
                    model_type = "Random Forest"
                elif "gbm" in model_id_str:
                    model_type = "GBM"
                elif "glm" in model_id_str:
                    model_type = "GLM"
                elif "deeplearning" in model_id_str:
                    model_type = "Neural Network"
                elif "stackedensemble" in model_id_str:
                    model_type = "Ensemble"
                
                mlflow.log_param(f"{model_prefix}_type", model_type)
            
            # Save and log the best model
            best_model = aml.leader
            try:
                model_path = h2o.save_model(best_model, path="/tmp", force=True)
                mlflow.log_artifact(model_path, artifact_path="best_model")
                logger.info(f"Best model saved: {model_path}")
            except Exception as e:
                logger.error(f"Failed to save best model: {str(e)}")
            
            # Create response with enhanced model information
            leaderboard_with_types = []
            for _, row in lb.iterrows():
                model_dict = row.to_dict()
                
                # Use the clean mapping function
                frontend_info = map_h2o_to_frontend(str(row['model_id']), selected_models)
                model_dict['model_type'] = frontend_info['display_name']
                model_dict['frontend_id'] = frontend_info['frontend_id']
                
                leaderboard_with_types.append(model_dict)
            
            logger.info(f"Training completed successfully. MLflow run ID: {run_id}")
            
            # **Log summary insights about top 3 models**
            try:
                top_3_summary = {
                    "top_3_algorithms": [str(lb.iloc[i]['model_id']).split('_')[0] for i in range(min(3, len(lb)))],
                    "performance_gap": float(lb.iloc[0]['auc'] - lb.iloc[min(2, len(lb)-1)]['auc']) if len(lb) >= 3 else 0,
                    "best_auc": float(lb.iloc[0]['auc']),
                    "best_logloss": float(lb.iloc[0]['logloss']),
                }
                
                for key, value in top_3_summary.items():
                    if isinstance(value, list):
                        mlflow.log_param(f"summary_{key}", str(value))
                    else:
                        mlflow.log_metric(f"summary_{key}", value)
                
                logger.info(f"Top 3 models summary: {top_3_summary}")
            except Exception as summary_e:
                logger.warning(f"Could not log top 3 summary: {str(summary_e)}")
            
            return JSONResponse(content={
                "message": f"Training completed! {len(lb)} models trained from your selection.",
                "leaderboard": leaderboard_with_types,
                "best_model": {
                    "id": str(lb.iloc[0]['model_id']),
                    "auc": float(lb.iloc[0]['auc']),
                    "logloss": float(lb.iloc[0]['logloss']),
                    "type": leaderboard_with_types[0]['model_type']
                },
                "training_info": {
                    "duration_seconds": training_duration,
                    "selected_models": selected_models,
                    "h2o_algorithms": h2o_algorithms,
                    "mlflow_run_id": run_id
                }
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    finally:
        # Clean up temp file
        if tmp_path:
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

@router.post("/predict/")
async def predict_single(request_data: dict):
    """Make predictions using a trained model"""
    try:
        model_id = request_data.get('model_id')
        inputs = request_data.get('inputs', {})
        
        if not model_id or not inputs:
            raise HTTPException(status_code=400, detail="model_id and inputs are required")
        
        # Initialize H2O
        if not initialize_h2o():
            raise HTTPException(status_code=500, detail="Failed to initialize H2O cluster")
        
        # This is a placeholder - in production you'd load the saved model
        # For now, return a mock prediction
        logger.info(f"Prediction request for model {model_id} with inputs: {inputs}")
        
        return JSONResponse(content={
            "prediction": 0.75,  # Mock prediction
            "confidence": 0.85,
            "model_used": model_id,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "h2o_status": "running" if h2o.cluster().is_running() else "stopped"
    }