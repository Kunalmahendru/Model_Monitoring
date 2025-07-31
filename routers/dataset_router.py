from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from utils.file_utils import save_uploaded_file
from services.automl_services import run_automl

import pandas as pd
import numpy as np
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
import re
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
    'xgboost': (['XGBoost'], 'XGBoost', ''),
    'randomforest': (['DRF'], 'Random Forest (DRF)', ''),
    'gbm': (['GBM'], 'Gradient Boosting Machine', ''),
    'glm': (['GLM'], 'Generalized Linear Model', ''),
    'neural': (['DeepLearning'], 'Deep Learning', ''),
    'ensemble': (['StackedEnsemble'], 'Stacked Ensemble', '')
}

def apply_feature_engineering(data, feature_config, target_column):
    """Apply feature engineering transformations to H2O frame"""
    logger.info("Applying feature engineering transformations...")
    
    # Create a copy to avoid modifying original data
    processed_data = data
    columns_to_remove = []
    
    for column, config in feature_config.items():
        if column == target_column:
            continue  # Skip target column
            
        if not config.get('include', True):
            # Mark column for removal
            columns_to_remove.append(column)
            logger.info(f"Column '{column}' marked for exclusion")
            continue
        
        if column not in processed_data.columns:
            logger.warning(f"Column '{column}' not found in data, skipping...")
            continue
            
        # Apply missing value imputation
        impute_method = config.get('impute', 'auto')
        if impute_method != 'auto':
            processed_data = apply_imputation(processed_data, column, impute_method)
        
        # Apply encoding for categorical variables
        encoding_method = config.get('encoding', 'auto')
        if encoding_method != 'auto':
            processed_data = apply_encoding(processed_data, column, encoding_method)
        
        # Apply transformations for numeric variables
        transform_method = config.get('transform', 'none')
        if transform_method != 'none':
            processed_data = apply_transformation(processed_data, column, transform_method)
    
    # Remove excluded columns
    if columns_to_remove:
        remaining_columns = [col for col in processed_data.columns if col not in columns_to_remove]
        processed_data = processed_data[remaining_columns]
        logger.info(f"Removed {len(columns_to_remove)} excluded columns")
    
    logger.info(f"Feature engineering completed. Final shape: {processed_data.shape}")
    return processed_data

def apply_imputation(data, column, method):
    """Apply missing value imputation"""
    try:
        if method == 'mean':
            mean_val = data[column].mean()
            data[column] = data[column].fillna(mean_val)
        elif method == 'median':
            median_val = data[column].median()
            data[column] = data[column].fillna(median_val)
        elif method == 'mode':
            mode_val = data[column].mode()[0]
            data[column] = data[column].fillna(mode_val)
        elif method == 'constant':
            data[column] = data[column].fillna(0)
        elif method == 'drop':
            # H2O will handle this during training
            pass
        
        logger.info(f"Applied {method} imputation to column '{column}'")
    except Exception as e:
        logger.warning(f"Failed to apply {method} imputation to column '{column}': {e}")
    
    return data

def apply_encoding(data, column, method):
    """Apply categorical encoding"""
    try:
        if method == 'onehot':
            # H2O automatically handles one-hot encoding for factors
            data[column] = data[column].asfactor()
        elif method == 'label':
            # Convert to factor (H2O's equivalent of label encoding)
            data[column] = data[column].asfactor()
        elif method == 'target':
            # H2O doesn't support target encoding directly, convert to factor
            data[column] = data[column].asfactor()
        elif method == 'none':
            # Keep as numeric
            pass
        
        logger.info(f"Applied {method} encoding to column '{column}'")
    except Exception as e:
        logger.warning(f"Failed to apply {method} encoding to column '{column}': {e}")
    
    return data

def apply_transformation(data, column, method):
    """Apply numeric transformations"""
    try:
        if method == 'log':
            # Add small constant to handle zeros/negatives
            data[column] = (data[column] + 1e-6).log()
        elif method == 'sqrt':
            # Handle negatives by taking absolute value
            data[column] = data[column].abs().sqrt()
        elif method == 'standard':
            # Standardize (mean=0, std=1)
            mean_val = data[column].mean()
            std_val = data[column].sd()
            data[column] = (data[column] - mean_val) / std_val
        elif method == 'minmax':
            # Min-Max scaling (0-1)
            min_val = data[column].min()
            max_val = data[column].max()
            data[column] = (data[column] - min_val) / (max_val - min_val)
        elif method == 'robust':
            # Robust scaling using median and IQR
            try:
                median_val = data[column].median()
                q75_result = data[column].quantile([0.75])
                q25_result = data[column].quantile([0.25])
                
                # Safe conversion for quantile results
                if hasattr(q75_result, 'as_data_frame'):
                    q75 = q75_result.as_data_frame().iloc[0, 0]
                else:
                    q75 = float(q75_result)
                    
                if hasattr(q25_result, 'as_data_frame'):
                    q25 = q25_result.as_data_frame().iloc[0, 0]
                else:
                    q25 = float(q25_result)
                    
                iqr = q75 - q25
                if iqr > 0:
                    data[column] = (data[column] - median_val) / iqr
                else:
                    logger.warning(f"IQR is zero for column '{column}', skipping robust scaling")
            except Exception as robust_e:
                logger.warning(f"Failed to apply robust scaling to column '{column}': {robust_e}")
        
        logger.info(f"Applied {method} transformation to column '{column}'")
    except Exception as e:
        logger.warning(f"Failed to apply {method} transformation to column '{column}': {e}")
    
    return data

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

def apply_feature_engineering(data, feature_config: dict, target_column: str):
    """Apply feature engineering transformations to H2O data frame"""
    try:
        logger.info("Starting feature engineering...")
        logger.info(f"Columns before feature engineering: {data.columns}")
        logger.info(f"Feature config received: {feature_config}")
        
        # Track transformations applied
        transformations_applied = []
        
        for column, config in feature_config.items():
            logger.info(f"Processing column '{column}' with config: {config}")
            
            if column == target_column:
                logger.info(f"Skipping target column: {column}")
                continue
                
            if not config.get('include', True):
                # Remove column if not included
                if column in data.columns:
                    logger.info(f"EXCLUDING column '{column}' as include=False")
                    data = data.drop(column)
                    transformations_applied.append(f"Removed column: {column}")
                else:
                    logger.warning(f"Column '{column}' not found in data, cannot exclude")
                continue
            
            if column not in data.columns:
                logger.warning(f"Column '{column}' not found in data, skipping")
                continue
                
            # Apply imputation
            impute_method = config.get('impute', 'auto')
            if impute_method != 'auto':
                if impute_method == 'mean' and data[column].type == 'real':
                    data[column] = data[column].impute("mean")
                    transformations_applied.append(f"Imputed {column} with mean")
                elif impute_method == 'median' and data[column].type == 'real':
                    data[column] = data[column].impute("median")
                    transformations_applied.append(f"Imputed {column} with median")
                elif impute_method == 'mode':
                    data[column] = data[column].impute("mode")
                    transformations_applied.append(f"Imputed {column} with mode")
            
            # Apply encoding (H2O handles most of this automatically)
            encoding_method = config.get('encoding', 'auto')
            if encoding_method == 'onehot':
                # Force one-hot encoding by converting to factor
                data[column] = data[column].asfactor()
                transformations_applied.append(f"One-hot encoded: {column}")
            
            # Apply transformations
            transform_method = config.get('transform', 'none')
            if transform_method != 'none' and data[column].type == 'real':
                if transform_method == 'log':
                    data[column] = data[column].log()
                    transformations_applied.append(f"Log transformed: {column}")
                elif transform_method == 'sqrt':
                    data[column] = data[column].sqrt()
                    transformations_applied.append(f"Sqrt transformed: {column}")
        
        logger.info(f"Feature engineering completed. Transformations: {transformations_applied}")
        logger.info(f"Columns after feature engineering: {data.columns}")
        return data
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        return data

def log_hyperparameters_to_mlflow(model, model_type: str):
    """Extract and log hyperparameters from H2O model to MLflow"""
    try:
        # Get the actual model parameters
        model_params = model.params
        
        # Log key hyperparameters based on model type
        common_params = ['training_frame', 'validation_frame', 'nfolds', 'seed']
        
        if model_type.lower() == 'gbm':
            gbm_params = ['ntrees', 'max_depth', 'learn_rate', 'sample_rate', 
                         'col_sample_rate', 'min_rows', 'nbins']
            for param in gbm_params:
                if param in model_params and model_params[param]['actual'] is not None:
                    mlflow.log_param(f"gbm_{param}", model_params[param]['actual'])
        
        elif model_type.lower() == 'drf':
            drf_params = ['ntrees', 'max_depth', 'sample_rate', 'col_sample_rate_per_tree', 
                         'min_rows', 'nbins']
            for param in drf_params:
                if param in model_params and model_params[param]['actual'] is not None:
                    mlflow.log_param(f"drf_{param}", model_params[param]['actual'])
        
        elif model_type.lower() == 'glm':
            glm_params = ['alpha', 'lambda', 'family', 'link', 'solver']
            for param in glm_params:
                if param in model_params and model_params[param]['actual'] is not None:
                    mlflow.log_param(f"glm_{param}", model_params[param]['actual'])
        
        elif model_type.lower() == 'deeplearning':
            dl_params = ['hidden', 'epochs', 'activation', 'input_dropout_ratio', 
                        'hidden_dropout_ratios', 'l1', 'l2']
            for param in dl_params:
                if param in model_params and model_params[param]['actual'] is not None:
                    mlflow.log_param(f"dl_{param}", model_params[param]['actual'])
        
        elif model_type.lower() == 'xgboost':
            xgb_params = ['ntrees', 'max_depth', 'learn_rate', 'sample_rate', 
                         'col_sample_rate', 'subsample', 'reg_alpha', 'reg_lambda']
            for param in xgb_params:
                if param in model_params and model_params[param]['actual'] is not None:
                    mlflow.log_param(f"xgb_{param}", model_params[param]['actual'])
        
        # Log common parameters
        for param in common_params:
            if param in model_params and model_params[param]['actual'] is not None:
                mlflow.log_param(param, model_params[param]['actual'])
                
        # Log model summary
        mlflow.log_param("model_algorithm", model_type)
        mlflow.log_param("model_id", model.model_id)
        
    except Exception as e:
        logger.warning(f"Failed to log hyperparameters for {model_type}: {str(e)}")

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

def extract_cv_metrics(model, problem_type: str = "classification"):
    """Extract cross-validation metrics if available"""
    try:
        cv_metrics = {}
        
        # Check if model has cross-validation summary
        if hasattr(model, '_model_json') and model._model_json:
            model_json = model._model_json
            
            # Extract cross-validation metrics
            if 'output' in model_json and 'cross_validation_metrics_summary' in model_json['output']:
                cv_summary = model_json['output']['cross_validation_metrics_summary']
                
                # Metrics based on problem type
                if problem_type == "classification":
                    cv_metric_names = ['auc', 'logloss', 'mean_per_class_error']
                else:  # regression
                    cv_metric_names = ['rmse', 'mse', 'mean_residual_deviance']
                
                for metric in cv_metric_names:
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
        'xgboost': ('xgboost', 'XGBoost', ''),
        'drf': ('randomforest', 'Random Forest (DRF)', ''),
        'gbm': ('gbm', 'Gradient Boosting Machine', ''),
        'glm': ('glm', 'Generalized Linear Model', ''),
        'deeplearning': ('neural', 'Deep Learning', ''),
        'stackedensemble': ('ensemble', 'Stacked Ensemble', '')
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
        'icon': ''
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
async def upload_dataset(file: UploadFile = File(...), target_column: str = Form(None)):
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

        # Determine target column - DON'T default, let user choose
        if target_column and target_column in df.columns:
            selected_target = target_column
        else:
            # Don't default to any column - let user choose in frontend
            selected_target = None
            
        # If target column is specified, validate it exists and analyze it
        if selected_target:
            if selected_target not in df.columns:
                raise HTTPException(status_code=400, detail=f"Target column '{selected_target}' not found in dataset")

        # If target column is specified, do basic validation only
        if selected_target:
            if selected_target not in df.columns:
                raise HTTPException(status_code=400, detail=f"Target column '{selected_target}' not found in dataset")

            # Basic target validation - just check if it has valid data
            target_data = df[selected_target].dropna()
            if len(target_data) == 0:
                raise HTTPException(status_code=400, detail=f"Target column '{selected_target}' is completely empty")
            if target_data.nunique() == 1:
                raise HTTPException(status_code=400, detail=f"Target column '{selected_target}' has only one unique value")

            # Calculate missing value summary (excluding target column)
            feature_columns = [col for col in df.columns if col != selected_target]
            missing_counts = df[feature_columns].isnull().sum()
            top_missing = missing_counts.sort_values(ascending=False).head(5)
            top_missing_summary = top_missing[top_missing > 0].to_dict()
            warnings = []
            for col, count in top_missing_summary.items():
                percent = (count / len(df)) * 100
                if percent > 20:
                    warnings.append(f"Column '{col}' has {count} missing values ({percent:.1f}%)")

            # Check target column missing values
            target_missing = df[selected_target].isnull().sum()
            if target_missing > 0:
                target_missing_percent = (target_missing / len(df)) * 100
                warnings.append(f"Target column '{selected_target}' has {target_missing} missing values ({target_missing_percent:.1f}%)")
                
            target_analysis = {
                "unique_values": int(target_data.nunique()),
                "sample_values": target_data.head(5).tolist(),
                "missing_count": int(target_missing)
            }
        else:
            # No target column selected yet
            warnings = []
            target_analysis = None
            top_missing_summary = {}
            
            # Calculate missing values for all columns since no target selected
            missing_counts = df.isnull().sum()
            top_missing = missing_counts.sort_values(ascending=False).head(5)
            top_missing_summary = top_missing[top_missing > 0].to_dict()
            for col, count in top_missing_summary.items():
                percent = (count / len(df)) * 100
                if percent > 20:
                    warnings.append(f"Column '{col}' has {count} missing values ({percent:.1f}%)")

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
            mlflow.log_param("target_column", selected_target if selected_target else "not_selected")
            mlflow.log_param("upload_timestamp", datetime.now().isoformat())
            
            # Only log target analysis if target column is selected
            if selected_target and target_analysis:
                mlflow.log_param("target_unique_values", target_analysis["unique_values"])
                mlflow.log_param("target_missing_count", target_analysis["missing_count"])

        target_msg = f", Target: {selected_target}" if selected_target else " (no target column selected yet)"
        logger.info(f"Dataset uploaded successfully: {file.filename}{target_msg}")
        
        return JSONResponse(status_code=200, content={
            "message": f"Dataset '{file.filename}' uploaded successfully!",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),  # Add column names for frontend dropdown
            "target": selected_target,
            "target_analysis": target_analysis,
            "top_missing": top_missing_summary,
            "warnings": warnings,
            "run_id": run.info.run_id
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

@router.post("/analyze-target/")
async def analyze_target_column(file: UploadFile = File(...), target_column: str = Form(...)):
    """Analyze the selected target column without determining problem type"""
    try:
        # Read file contents
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        
        # Validate target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")

        # Analyze target column
        target_data = df[target_column].dropna()
        unique_values = target_data.nunique()
        total_values = len(target_data)
        
        # Determine if target is numeric
        is_numeric = pd.api.types.is_numeric_dtype(target_data)

        # Calculate missing values for target
        target_missing = df[target_column].isnull().sum()
        
        # Create warnings
        warnings = []
        if target_missing > 0:
            target_missing_percent = (target_missing / len(df)) * 100
            warnings.append(f"Target column '{target_column}' has {target_missing} missing values ({target_missing_percent:.1f}%)")

        logger.info(f"Target column '{target_column}' analyzed - letting user choose problem type")
        
        return JSONResponse(content={
            "target_analysis": {
                "unique_values": int(unique_values),
                "is_numeric": bool(is_numeric),
                "sample_values": target_data.head(10).tolist(),
                "missing_count": int(target_missing),
                "total_count": len(df)
            },
            "warnings": warnings
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Target analysis error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Target analysis failed: {str(e)}"})

@router.post("/train-model/")
async def train_model(
    file: UploadFile = File(...), 
    models: str = Form(...), 
    target_column: str = Form(None), 
    problem_type: str = Form(...),  # User-selected problem type
    feature_config: str = Form("{}"),
    advanced_settings: str = Form("{}")
):
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
        logger.info(f"File contents size: {len(contents)} bytes")
        
        # First create temp file with original data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Validate dataset with cleaning
        df = pd.read_csv(tmp_path, on_bad_lines='skip')  # Skip malformed CSV lines
        logger.info(f"Pandas DataFrame loaded successfully:")
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Columns: {df.columns.tolist()}")
        logger.info(f"  - Column count: {len(df.columns)}")
        
        # Simple data cleaning - only fix what H2O can't handle
        original_shape = df.shape
        cleaned_rows_removed = 0
        rows_to_drop = []
        
        logger.info("Starting simple data cleaning...")
        
        # 1. Remove rows with obviously malformed structure (completely empty or too few values)
        expected_col_count = len(df.columns)
        
        for idx, row in df.iterrows():
            # Count actual non-null, non-empty values
            valid_values = 0
            for val in row:
                if pd.notna(val) and str(val).strip() != '':
                    valid_values += 1
            
            # If row has less than 30% of expected columns filled, it's malformed
            if valid_values < expected_col_count * 0.3:
                rows_to_drop.append(idx)
                logger.debug(f"Row {idx} marked for removal: only {valid_values}/{expected_col_count} valid values")
        
        # 2. Check first column for mixed alphanumeric patterns (like "1al" instead of "1")
        if len(df) > 0:
            first_col = df.columns[0]
            first_col_str = df[first_col].astype(str)
            
            # Look for patterns like "123abc" which indicate data corruption
            for idx, val in enumerate(first_col_str):
                if pd.notna(val) and val.strip():
                    # Check for mixed alphanumeric starting with digits (corruption pattern)
                    if re.match(r'^[0-9]+[a-zA-Z]+', val.strip()):
                        if idx not in rows_to_drop:
                            rows_to_drop.append(idx)
                            logger.debug(f"Row {idx} marked for removal: corrupted first column value '{val}'")
        
        # 3. Remove identified malformed rows
        if rows_to_drop:
            cleaned_rows_removed = len(rows_to_drop)
            df = df.drop(rows_to_drop).reset_index(drop=True)
            logger.warning(f"Removed {cleaned_rows_removed} malformed rows due to structural issues")
            logger.info(f"Dataset shape after cleaning: {df.shape} (was {original_shape})")
            
            # Save cleaned dataset to a new temp file
            os.unlink(tmp_path)  # Remove original temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as cleaned_tmp:
                df.to_csv(cleaned_tmp, index=False)
                tmp_path = cleaned_tmp.name
            logger.info(f"Saved cleaned dataset to new temp file: {tmp_path}")
        else:
            logger.info("No malformed rows detected, dataset is clean")
        
        if df.isnull().sum().sum() > 0:
            logger.warning("Dataset contains null values - H2O will handle this automatically")

        # DEBUG: Log received parameters
        logger.info(f"Received parameters:")
        logger.info(f"  - target_column: '{target_column}'")
        logger.info(f"  - problem_type: '{problem_type}'")
        logger.info(f"  - selected_models: {selected_models}")

        # CRITICAL: Ensure target column is provided by user
        if not target_column or target_column.strip() == "":
            raise HTTPException(
                status_code=400, 
                detail="Target column is required! Please:\n1. Upload your dataset\n2. Select a target column from the dropdown\n3. Then start training"
            )

        # CRITICAL: Ensure problem type is provided by user
        if not problem_type or problem_type.strip() == "":
            raise HTTPException(
                status_code=400, 
                detail="Problem type is required! Please select either 'classification' or 'regression'"
            )
        
        # Validate problem type
        if problem_type.lower() not in ['classification', 'regression']:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid problem type '{problem_type}'. Must be 'classification' or 'regression'"
            )
        
        problem_type = problem_type.lower()  # Normalize to lowercase
        logger.info(f"Problem type validation passed: '{problem_type}'")

        # Determine target column - MUST be user-selected
        if target_column in df.columns:
            y_column = target_column
            logger.info(f"Using user-selected target column: '{y_column}'")
        else:
            logger.error(f"User-selected target column '{target_column}' not found in dataset!")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}"
            )

        logger.info(f" Target column validation passed: '{y_column}'")

        # Ensure MLflow experiment exists
        ensure_mlflow_experiment()

        with mlflow.start_run(run_name=f"AutoML_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started with ID: {run_id}")
            
            # Log dataset info and training parameters
            mlflow.log_artifact(tmp_path, artifact_path="dataset")
            mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("target_column", y_column)
            mlflow.log_param("target_column_name", y_column)  # Ensure target is logged with clear name
            mlflow.log_param("problem_type", problem_type)  # Log user-selected problem type
            mlflow.log_param("training_timestamp", datetime.now().isoformat())
            mlflow.log_param("selected_models", selected_models)
            mlflow.log_param("num_selected_models", len(selected_models))
            
            # Load data into H2O
            logger.info(f"Loading data into H2O from temp file: {tmp_path}")
            data = h2o.import_file(tmp_path)
            logger.info(f"Data loaded into H2O successfully:")
            logger.info(f"  - H2O Shape: {data.shape}")
            logger.info(f"  - H2O Columns: {data.columns}")
            logger.info(f"  - H2O Column count: {len(data.columns)}")
            
            # CRITICAL DEBUG: Verify target column exists in H2O frame
            if y_column not in data.columns:
                logger.error(f"CRITICAL ERROR: Target column '{y_column}' not found in H2O frame!")
                logger.error(f"H2O frame columns: {data.columns}")
                logger.error(f"Pandas columns were: {df.columns.tolist()}")
                raise HTTPException(status_code=400, detail=f"Target column '{y_column}' not found in H2O frame. H2O columns: {data.columns}")
            
            logger.info(f"Confirmed target column '{y_column}' exists in H2O frame")
            
            # Parse feature engineering configuration
            try:
                feature_config_dict = json.loads(feature_config) if feature_config != "{}" else {}
                advanced_settings_dict = json.loads(advanced_settings) if advanced_settings != "{}" else {}
                logger.info(f"Feature engineering config: {feature_config_dict}")
                logger.info(f"Advanced settings: {advanced_settings_dict}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse feature config, using defaults: {e}")
                feature_config_dict = {}
                advanced_settings_dict = {}
            
            # Apply feature engineering
            if feature_config_dict:
                logger.info(f"Applying feature engineering to {len(feature_config_dict)} columns...")
                data_before_fe = data.shape
                data = apply_feature_engineering(data, feature_config_dict, y_column)
                logger.info(f"Feature engineering applied. Shape before: {data_before_fe}, after: {data.shape}")
                logger.info(f"Columns after feature engineering: {data.columns}")
            else:
                logger.info("No feature engineering configuration provided, using default H2O processing")
            
            # Apply target column treatment based on user-selected problem type
            try:
                target_unique_values = data[y_column].nlevels()  # H2O method to get unique count
                target_unique_values = int(target_unique_values[0]) if isinstance(target_unique_values, list) else int(target_unique_values)
            except (ValueError, TypeError, IndexError):
                target_unique_values = 0
            
            # Safe conversion for target sample
            try:
                target_sample = data[y_column].as_data_frame()[y_column].dropna()
                sample_values = target_sample.unique()[:10].tolist()
            except (AttributeError, TypeError, IndexError):
                # Fallback if as_data_frame fails
                try:
                    sample_values = data[y_column].head(10).as_data_frame().iloc[:, 0].dropna().unique().tolist()
                except:
                    sample_values = ["N/A"]
            
            logger.info(f"Target column '{y_column}' analysis:")
            logger.info(f"  - Unique values: {target_unique_values}")
            logger.info(f"  - Sample values: {sample_values}")
            logger.info(f"  - Data type: {data[y_column].dtype}")
            logger.info(f"  - User-selected problem type: {problem_type}")
            
            # Apply target column treatment based on user selection
            if problem_type == "classification":
                logger.info(f"Converting target '{y_column}' to factor for classification (user selected)")
                data[y_column] = data[y_column].asfactor()
            else:  # regression
                logger.info(f"Keeping target '{y_column}' as numeric for regression (user selected)")
                # Ensure target is numeric for regression
                if data[y_column].isfactor():
                    logger.info(f"Converting factor target '{y_column}' to numeric for regression")
                    data[y_column] = data[y_column].asnumeric()
            
            # Update advanced training settings
            train_split = advanced_settings_dict.get('trainSplit', 0.8)
            cv_folds = advanced_settings_dict.get('cvFolds', 5)
            max_runtime = advanced_settings_dict.get('maxRuntime', 300)
            
            # Get target and feature columns
            y = y_column
            x = [col for col in data.columns if col != y]  # All columns except target
            
            logger.info(f"Training configuration:")
            logger.info(f"  - Target column (y): '{y}'")
            logger.info(f"  - Problem type: {problem_type}")
            logger.info(f"  - Feature columns (x): {x}")
            logger.info(f"  - Number of features: {len(x)}")
            
            # Split data for validation using advanced settings
            train, valid = data.split_frame(ratios=[train_split], seed=1)
            mlflow.log_param("train_split_ratio", train_split)
            
            # Safe conversion for H2O frame properties that might return unexpected types
            try:
                train_size = train.nrows
                train_size = int(train_size[0]) if isinstance(train_size, list) else int(train_size)
            except (ValueError, TypeError, IndexError):
                train_size = 0
            mlflow.log_param("train_size", train_size)
            
            try:
                valid_size = valid.nrows
                valid_size = int(valid_size[0]) if isinstance(valid_size, list) else int(valid_size)
            except (ValueError, TypeError, IndexError):
                valid_size = 0
            mlflow.log_param("valid_size", valid_size)
            
            # Log target column statistics for better tracking (after data is split)
            try:
                unique_vals = train[y].nlevels()
                unique_vals = int(unique_vals[0]) if isinstance(unique_vals, list) else int(unique_vals)
            except (ValueError, TypeError, IndexError):
                unique_vals = 0
                
            try:
                missing_count_result = train[y].isna().sum()
                if hasattr(missing_count_result, 'as_data_frame'):
                    missing_count = missing_count_result.as_data_frame().iloc[0, 0]
                    missing_count = int(missing_count[0]) if isinstance(missing_count, list) else int(missing_count)
                else:
                    # If it's already a scalar (float/int), convert directly
                    missing_count = int(missing_count_result)
            except (ValueError, TypeError, IndexError, AttributeError):
                missing_count = 0
                
            try:
                total_rows = train.nrows
                total_rows = int(total_rows[0]) if isinstance(total_rows, list) else int(total_rows)
            except (ValueError, TypeError, IndexError):
                total_rows = 0
            
            target_stats = {
                "unique_values": unique_vals,  # H2O method for unique count (safely converted)
                "data_type": str(train[y].dtype),
                "missing_count": missing_count,  # H2O method for NA count (safely converted)
                "total_rows": total_rows
            }
            
            for stat_key, stat_value in target_stats.items():
                mlflow.log_param(f"target_{stat_key}", stat_value)
            
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
            
            # Configure AutoML with selected algorithms and advanced settings
            aml_config = {
                "max_models": 20,  # Allow more models for variety
                "seed": 1,
                "max_runtime_secs": max_runtime,  # Use advanced setting
                "include_algos": h2o_algorithms,  # Only train selected algorithms
                "exclude_algos": None,  # Don't exclude any from the selected ones
                "nfolds": cv_folds,  # Use advanced setting
                "stopping_tolerance": 0.001,
                "stopping_rounds": 3
            }
            
            aml = H2OAutoML(**aml_config)
            
            logger.info("Starting AutoML training with selected models...")
            start_time = datetime.now()
            
            # Train the models
            aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
            
            training_duration = (datetime.now() - start_time).total_seconds()
            mlflow.log_metric("training_duration_seconds", training_duration)
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
            # Get leaderboard - using user-selected problem type
            lb = aml.leaderboard.as_data_frame()
            logger.info(f"Training completed. {len(lb)} models trained.")
            logger.info(f"Leaderboard columns: {lb.columns.tolist()}")  # Debug: log actual columns
            
            # Use the user-selected problem type (no automatic detection)
            logger.info(f"Using user-selected problem type: {problem_type}")
            
            # Log the user-selected problem type and target column info
            mlflow.log_param("detected_problem_type", problem_type)  # Keep same param name for consistency
            mlflow.log_param("target_column", y_column)
            mlflow.log_param("target_column_final", y_column)  # Final confirmation of target used
            mlflow.log_param("num_features", len(x))
            mlflow.log_param("feature_columns", x)
            
            # Log overall training metrics
            mlflow.log_param("total_models_trained", len(lb))
            
            # Log metrics based on problem type with NaN handling
            if problem_type == "classification":
                if 'auc' in lb.columns and not np.isnan(lb.iloc[0]['auc']):
                    mlflow.log_metric("best_auc", float(lb.iloc[0]['auc']))
                if 'logloss' in lb.columns and not np.isnan(lb.iloc[0]['logloss']):
                    mlflow.log_metric("best_logloss", float(lb.iloc[0]['logloss']))
            elif problem_type == "regression":
                if 'rmse' in lb.columns and not np.isnan(lb.iloc[0]['rmse']):
                    mlflow.log_metric("best_rmse", float(lb.iloc[0]['rmse']))
                if 'mse' in lb.columns and not np.isnan(lb.iloc[0]['mse']):
                    mlflow.log_metric("best_mse", float(lb.iloc[0]['mse']))
                    
            mlflow.log_param("best_model_id", str(lb.iloc[0]['model_id']))
            
            # Log individual model metrics
            for idx, row in lb.iterrows():
                model_prefix = f"model_{idx}"
                
                # Log metrics based on problem type with NaN handling
                if problem_type == "classification":
                    if 'auc' in row and not np.isnan(row['auc']):
                        mlflow.log_metric(f"{model_prefix}_auc", float(row['auc']))
                    if 'logloss' in row and not np.isnan(row['logloss']):
                        mlflow.log_metric(f"{model_prefix}_logloss", float(row['logloss']))
                elif problem_type == "regression":
                    if 'rmse' in row and not np.isnan(row['rmse']):
                        mlflow.log_metric(f"{model_prefix}_rmse", float(row['rmse']))
                    if 'mse' in row and not np.isnan(row['mse']):
                        mlflow.log_metric(f"{model_prefix}_mse", float(row['mse']))
                        
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
                            
                            # Log target column info for this model
                            mlflow.log_param("target_column", y_column)
                            mlflow.log_param("target_column_name", y_column)
                            mlflow.log_param("problem_type", problem_type)
                            
                            # Performance metrics - log based on problem type with NaN handling
                            if problem_type == "classification":
                                if 'auc' in row and not np.isnan(row['auc']):
                                    mlflow.log_metric("auc", float(row['auc']))
                                if 'logloss' in row and not np.isnan(row['logloss']):
                                    mlflow.log_metric("logloss", float(row['logloss']))
                                if 'mean_per_class_error' in row and not np.isnan(row['mean_per_class_error']):
                                    mlflow.log_metric("mean_per_class_error", float(row['mean_per_class_error']))
                            elif problem_type == "regression":
                                if 'rmse' in row and not np.isnan(row['rmse']):
                                    mlflow.log_metric("rmse", float(row['rmse']))
                                if 'mse' in row and not np.isnan(row['mse']):
                                    mlflow.log_metric("mse", float(row['mse']))
                                if 'mean_residual_deviance' in row and not np.isnan(row['mean_residual_deviance']):
                                    mlflow.log_metric("mean_residual_deviance", float(row['mean_residual_deviance']))
                            
                            # **Log all hyperparameters using enhanced function**
                            log_hyperparameters_to_mlflow(model, model.algo)
                            
                            # Additional model information
                            mlflow.log_param("total_models_trained", len(lb))
                            mlflow.log_param("model_rank", idx + 1)
                            mlflow.log_param("cross_validation_folds", cv_folds)
                            mlflow.log_param("train_split_ratio", train_split)
                            
                            # Log feature importance if available
                            try:
                                if hasattr(model, 'varimp') and model.varimp(use_pandas=True) is not None:
                                    feature_importance = model.varimp(use_pandas=True)
                                    # Log top 10 features
                                    for i, (feature, importance) in enumerate(zip(
                                        feature_importance['variable'].head(10), 
                                        feature_importance['relative_importance'].head(10)
                                    )):
                                        mlflow.log_param(f"top_feature_{i+1}", f"{feature}:{importance:.4f}")
                            except Exception as e:
                                logger.warning(f"Failed to log feature importance: {str(e)}")
                            
                            # Training time if available
                            if hasattr(model, 'training_time_ms') and model.training_time_ms:
                                mlflow.log_metric("training_time_ms", model.training_time_ms)
                            
                            # **Extract and log cross-validation metrics**
                            cv_metrics = extract_cv_metrics(model, problem_type)
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
            

            # Create response with enhanced model information, including feature importance and mlflow run id
            leaderboard_with_types = []
            for _, row in lb.iterrows():
                # Convert NaN values to None for JSON serialization
                model_dict = row.replace({np.nan: None}).to_dict()
                frontend_info = map_h2o_to_frontend(str(row['model_id']), selected_models)
                model_dict['model_type'] = frontend_info['display_name']
                model_dict['frontend_id'] = frontend_info['frontend_id']
                model_dict['mlflow_run_id'] = run_id  # parent run id for now

                # Add top 3 feature importances if available
                try:
                    model = h2o.get_model(str(row['model_id']))
                    if hasattr(model, 'varimp') and callable(getattr(model, 'varimp', None)):
                        varimp_df = model.varimp(use_pandas=True)
                        if varimp_df is not None and len(varimp_df) > 0:
                            model_dict['feature_importance'] = [
                                {"feature": r['variable'], "importance": float(r['relative_importance'])}
                                for _, r in varimp_df.head(3).iterrows()
                            ]
                        else:
                            model_dict['feature_importance'] = []
                    else:
                        model_dict['feature_importance'] = []
                except Exception as e:
                    logger.warning(f"Could not extract feature importance for {row['model_id']}: {str(e)}")
                    model_dict['feature_importance'] = []

                leaderboard_with_types.append(model_dict)

            logger.info(f"Training completed successfully. MLflow run ID: {run_id}")
            
            # **Log summary insights about top 3 models**
            try:
                top_3_summary = {
                    "top_3_algorithms": [str(lb.iloc[i]['model_id']).split('_')[0] for i in range(min(3, len(lb)))],
                }
                
                # Add performance metrics based on problem type with NaN handling
                if problem_type == "classification" and 'auc' in lb.columns:
                    if len(lb) >= 3 and not np.isnan(lb.iloc[0]['auc']) and not np.isnan(lb.iloc[min(2, len(lb)-1)]['auc']):
                        top_3_summary["performance_gap"] = float(lb.iloc[0]['auc'] - lb.iloc[min(2, len(lb)-1)]['auc'])
                    else:
                        top_3_summary["performance_gap"] = 0
                    if not np.isnan(lb.iloc[0]['auc']):
                        top_3_summary["best_auc"] = float(lb.iloc[0]['auc'])
                    if 'logloss' in lb.columns and not np.isnan(lb.iloc[0]['logloss']):
                        top_3_summary["best_logloss"] = float(lb.iloc[0]['logloss'])
                elif problem_type == "regression" and 'rmse' in lb.columns:
                    if len(lb) >= 3 and not np.isnan(lb.iloc[0]['rmse']) and not np.isnan(lb.iloc[min(2, len(lb)-1)]['rmse']):
                        top_3_summary["performance_gap"] = float(lb.iloc[min(2, len(lb)-1)]['rmse'] - lb.iloc[0]['rmse'])
                    else:
                        top_3_summary["performance_gap"] = 0
                    if not np.isnan(lb.iloc[0]['rmse']):
                        top_3_summary["best_rmse"] = float(lb.iloc[0]['rmse'])
                    if 'mse' in lb.columns and not np.isnan(lb.iloc[0]['mse']):
                        top_3_summary["best_mse"] = float(lb.iloc[0]['mse'])
                
                for key, value in top_3_summary.items():
                    if isinstance(value, list):
                        mlflow.log_param(f"summary_{key}", str(value))
                    else:
                        mlflow.log_metric(f"summary_{key}", value)
                
                logger.info(f"Top 3 models summary: {top_3_summary}")
            except Exception as summary_e:
                logger.warning(f"Could not log top 3 summary: {str(summary_e)}")
            
            # Create best_model info based on problem type with NaN handling
            best_model_info = {
                "id": str(lb.iloc[0]['model_id']),
                "type": leaderboard_with_types[0]['model_type']
            }
            
            # Add appropriate metrics based on problem type with NaN handling
            if problem_type == "classification":
                if 'auc' in lb.columns and not np.isnan(lb.iloc[0]['auc']):
                    best_model_info["auc"] = float(lb.iloc[0]['auc'])
                if 'logloss' in lb.columns and not np.isnan(lb.iloc[0]['logloss']):
                    best_model_info["logloss"] = float(lb.iloc[0]['logloss'])
            elif problem_type == "regression":
                if 'rmse' in lb.columns and not np.isnan(lb.iloc[0]['rmse']):
                    best_model_info["rmse"] = float(lb.iloc[0]['rmse'])
                if 'mse' in lb.columns and not np.isnan(lb.iloc[0]['mse']):
                    best_model_info["mse"] = float(lb.iloc[0]['mse'])
            
            return JSONResponse(content={
                "message": f"Training completed! {len(lb)} models trained from your selection.",
                "leaderboard": leaderboard_with_types,
                "best_model": best_model_info,
                "problem_type": problem_type,  # User-selected problem type
                "training_info": {
                    "duration_seconds": training_duration,
                    "selected_models": selected_models,
                    "h2o_algorithms": h2o_algorithms,
                    "mlflow_run_id": run_id,
                    "target_column": y_column,  # Include target column in response
                    "target_column_used": y_column,  # Clear confirmation of what target was used
                    "problem_type": problem_type  # User-selected problem type for clarity
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
        
        # Convert NaN values to None for JSON serialization
        preview_rows = df.head(10).replace({np.nan: None}).to_dict(orient='records')
        missing_values_dict = df.isnull().sum().replace({np.nan: None}).to_dict()
        
        # Convert numpy int64 to regular int for JSON serialization
        missing_values_dict = {k: int(v) if isinstance(v, np.integer) else v for k, v in missing_values_dict.items()}
        
        preview_data = {
            "columns": df.columns.tolist(),
            "rows": preview_rows,  # Show more rows with NaN converted to None
            "shape": df.shape,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": missing_values_dict,
            "target_column": df.columns[-1],
            "target_unique_values": int(df[df.columns[-1]].nunique()) if df.columns[-1] in df.columns else 0
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

@router.post("/predict-csv/")
async def predict_csv(file: UploadFile = File(...), model_id: str = Form(None)):
    """Make batch predictions on a CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Initialize H2O
        if not initialize_h2o():
            raise HTTPException(status_code=500, detail="Failed to initialize H2O cluster")
        
        logger.info(f"Batch prediction request for model: {model_id}")
        logger.info(f"File: {file.filename}")
        
        # Read uploaded file
        contents = await file.read()
        temp_input_path = None
        temp_output_path = None
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
                tmp_file.write(contents)
                temp_input_path = tmp_file.name
            
            # Read data into pandas
            df = pd.read_csv(temp_input_path)
            logger.info(f"Input data shape: {df.shape}")
            logger.info(f"Input columns: {list(df.columns)}")
            
            # Try to get the model from H2O
            try:
                if model_id:
                    model = h2o.get_model(model_id)
                    logger.info(f"Loaded model: {model_id}")
                else:
                    # If no model_id provided, try to get the latest model from the leaderboard
                    # This is a fallback for when model_id is not passed from frontend
                    models = h2o.ls()
                    if len(models) == 0:
                        raise HTTPException(status_code=404, detail="No models found in H2O cluster")
                    
                    # Get the most recent model
                    latest_model_id = models['key'].as_data_frame().iloc[-1]['key']
                    model = h2o.get_model(latest_model_id)
                    logger.info(f"Using latest model: {latest_model_id}")
                    
            except Exception as model_error:
                logger.error(f"Error loading model: {str(model_error)}")
                raise HTTPException(status_code=404, detail=f"Model not found: {str(model_error)}")
            
            # Convert to H2O frame
            h2o_frame = h2o.H2OFrame(df)
            logger.info(f"H2O frame shape: {h2o_frame.shape}")
            
            # Make predictions
            predictions = model.predict(h2o_frame)
            predictions_df = predictions.as_data_frame()
            
            logger.info(f"Predictions shape: {predictions_df.shape}")
            logger.info(f"Predictions columns: {list(predictions_df.columns)}")
            
            # Combine input data with predictions
            result_df = pd.concat([df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
            
            # Save result to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp_output:
                result_df.to_csv(tmp_output.name, index=False)
                temp_output_path = tmp_output.name
            
            logger.info(f"Batch prediction completed successfully. {len(result_df)} rows predicted.")
            
            # Return the CSV file
            return FileResponse(
                path=temp_output_path,
                media_type='application/octet-stream',
                filename=f"predictions_{file.filename}",
                headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
            )
            
        finally:
            # Clean up temporary files
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                except:
                    pass
            # Note: temp_output_path will be cleaned up by FastAPI after sending the response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
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

@router.get("/download-mojo/{model_id}")
async def download_mojo(model_id: str):
    try:
        logger.info(f"[MOJO] Initializing H2O for model_id: {model_id}")
        if not initialize_h2o():
            logger.error("[MOJO] Failed to initialize H2O cluster")
            raise HTTPException(status_code=500, detail="Failed to initialize H2O cluster")
        try:
            logger.info(f"[MOJO] Attempting to get model from H2O memory: {model_id}")
            model = h2o.get_model(model_id)
            logger.info(f"[MOJO] Model {model_id} found in memory.")
        except Exception:
            logger.warning(f"[MOJO] Model {model_id} not found in memory. Trying to load from disk.")
            model_path = f"/tmp/{model_id}"  # No .zip extension, matches h2o.save_model
            if os.path.exists(model_path):
                logger.info(f"[MOJO] Loading model from disk: {model_path}")
                model = h2o.load_model(model_path)
                logger.info(f"[MOJO] Model {model_id} loaded from disk.")
            else:
                logger.error(f"[MOJO] Model {model_id} not found in memory or on disk.")
                raise HTTPException(status_code=404, detail="Model not found in memory or on disk.")
        # Only allow MOJO for supported types
        logger.info(f"[MOJO] Checking if model {model_id} supports MOJO export (algo: {model.algo})")
        if model.algo.lower() not in ["drf", "gbm", "xgboost", "glm"]:
            logger.error(f"[MOJO] MOJO export not supported for model type: {model.algo}")
            raise HTTPException(status_code=400, detail="MOJO export not supported for this model type.")
        logger.info(f"[MOJO] Exporting MOJO for model {model_id}")
        mojo_path = model.download_mojo(path="/tmp", get_genmodel_jar=False)
        filename = os.path.basename(mojo_path)
        logger.info(f"[MOJO] MOJO file ready: {mojo_path}")
        return FileResponse(mojo_path, filename=filename, media_type="application/zip")
    except Exception as e:
        logger.error(f"MOJO download failed: {str(e)}")
        raise HTTPException(status_code=404, detail="MOJO not found for this model.")

@router.get("/get-hyperparameters/{run_id}")
async def get_hyperparameters(run_id: str):
    """Get hyperparameters for a specific MLflow run"""
    try:
        import mlflow
        
        # Get the run details
        run = mlflow.get_run(run_id)
        
        # Extract parameters
        params = run.data.params
        
        # Filter and organize hyperparameters
        hyperparams = {}
        for key, value in params.items():
            if key not in ['total_models_trained', 'model_rank', 'training_frame', 'validation_frame']:
                hyperparams[key] = value
        
        return {
            "hyperparameters": hyperparams,
            "run_id": run_id,
            "model_id": params.get('model_id', 'Unknown')
        }
        
    except Exception as e:
        logger.error(f"Failed to get hyperparameters for run {run_id}: {str(e)}")
        raise HTTPException(status_code=404, detail="Run not found or hyperparameters not available")