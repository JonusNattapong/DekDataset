#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment Tracking Module for DekDataset
==========================================

This module provides experiment tracking capabilities using MLflow and Weights & Biases (W&B)
for monitoring dataset generation, quality metrics, and system performance.

Features:
- MLflow integration for local experiment tracking
- Weights & Biases integration for cloud-based experiment monitoring
- Dataset generation metrics logging
- Quality control metrics tracking
- Cost and performance analytics
- Experiment comparison and visualization

Author: DekDataset Team
Created: 2025-01-28
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

# Optional imports with graceful fallbacks
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTracker:
    """Base class for experiment trackers."""
    
    def __init__(self, experiment_name: str = "DekDataset", project_name: str = "dataset-generation"):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.metrics = {}
        self.parameters = {}
        self.artifacts = []
        self.is_active = False
    
    def start_experiment(self, **kwargs):
        """Start a new experiment session."""
        self.is_active = True
        self.start_time = time.time()
        logger.info(f"Started experiment session: {self.session_id}")
    
    def end_experiment(self):
        """End the current experiment session."""
        if self.is_active:
            self.is_active = False
            duration = time.time() - self.start_time
            self.log_metric("experiment_duration_seconds", duration)
            logger.info(f"Ended experiment session: {self.session_id} (duration: {duration:.2f}s)")
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        self.parameters[key] = value
    
    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None):
        """Log a metric."""
        if step is None:
            self.metrics[key] = value
        else:
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append({"step": step, "value": value})
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file)."""
        self.artifacts.append({"file_path": file_path, "artifact_path": artifact_path})


class MLflowTracker(BaseTracker):
    """MLflow experiment tracker for local experiment management."""
    
    def __init__(self, experiment_name: str = "DekDataset", tracking_uri: Optional[str] = None):
        super().__init__(experiment_name)
        
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            self.enabled = False
            return
        
        self.enabled = True
          # Set tracking URI with connection testing
        if tracking_uri:
            try:
                # Test connection to remote MLflow server with timeout
                if tracking_uri.startswith(('http://', 'https://')):
                    logger.info(f"Testing connection to MLflow server at {tracking_uri}")
                    # Set a shorter timeout for connection testing
                    import requests
                    test_url = f"{tracking_uri.rstrip('/')}/health"
                    try:
                        response = requests.get(test_url, timeout=5)
                        if response.status_code == 200:
                            logger.info("MLflow server connection successful")
                        else:
                            logger.warning(f"MLflow server responded with status {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Cannot connect to MLflow server at {tracking_uri}: {e}")
                        logger.info("Falling back to local file storage")
                        # Fall back to local storage
                        project_root = Path(__file__).parent.parent.parent
                        mlruns_dir = project_root / "mlruns"
                        mlruns_dir.mkdir(exist_ok=True)
                        tracking_uri = f"file://{mlruns_dir}"
                
                mlflow.set_tracking_uri(tracking_uri)
                
            except Exception as e:
                logger.warning(f"Error setting MLflow tracking URI: {e}")
                # Fall back to local storage
                project_root = Path(__file__).parent.parent.parent
                mlruns_dir = project_root / "mlruns"
                mlruns_dir.mkdir(exist_ok=True)
                mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        else:
            # Default to local file storage
            project_root = Path(__file__).parent.parent.parent
            mlruns_dir = project_root / "mlruns"
            mlruns_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        
        # Set or create experiment with retry logic
        try:
            # Add timeout and retry for experiment operations
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(experiment_name)
                        logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
                    else:
                        experiment_id = experiment.experiment_id
                        logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
                    
                    mlflow.set_experiment(experiment_name)
                    break  # Success, exit retry loop
                    
                except Exception as retry_error:
                    if attempt < max_retries - 1:
                        logger.warning(f"MLflow experiment setup attempt {attempt + 1} failed: {retry_error}. Retrying...")
                        time.sleep(1)  # Wait before retry
                    else:
                        raise retry_error
                        
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment after retries: {e}")
            logger.info("MLflow tracking will be disabled for this session")
            self.enabled = False
    
    def _safe_mlflow_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute MLflow operation with error handling and timeout."""
        if not self.enabled:
            return None
            
        try:
            result = operation_func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(f"MLflow {operation_name} failed: {e}")            # Don't disable tracking for individual operation failures
            return None
    
    def start_experiment(self, run_name: Optional[str] = None, **kwargs):
        """Start a new MLflow run."""
        if not self.enabled:
            return
        
        super().start_experiment(**kwargs)
        
        run_name = run_name or f"dataset_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        def start_run():
            mlflow.start_run(run_name=run_name)
            # Log basic run information
            mlflow.log_param("session_id", self.session_id)
            mlflow.log_param("start_time", datetime.fromtimestamp(self.start_time).isoformat())
            mlflow.log_param("tracking_system", "MLflow")
            logger.info(f"Started MLflow run: {run_name}")
            self._safe_mlflow_operation("start_run", start_run)
    
    def end_experiment(self):
        """End the current MLflow run."""
        if not self.enabled:
            return
        
        super().end_experiment()
        
        def end_run():
            mlflow.end_run()
            logger.info("Ended MLflow run")
        
        self._safe_mlflow_operation("end_run", end_run)
    
    def log_param(self, key: str, value: Any):
        """Log a parameter to MLflow."""
        if not self.enabled:
            return
        
        super().log_param(key, value)
        
        def log_parameter():
            # Convert complex objects to JSON strings
            param_value = value
            if isinstance(param_value, (dict, list)):
                param_value = json.dumps(param_value)
            elif not isinstance(param_value, (str, int, float, bool)):
                param_value = str(param_value)
            
            mlflow.log_param(key, param_value)
        
        self._safe_mlflow_operation("log_param", log_parameter)
    
    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None):
        """Log a metric to MLflow."""
        if not self.enabled:
            return
        
        super().log_metric(key, value, step)
        
        def log_metric_func():
            mlflow.log_metric(key, value, step=step)
            self._safe_mlflow_operation("log_metric", log_metric_func)
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to MLflow."""
        if not self.enabled:
            return
        
        super().log_artifact(file_path, artifact_path)
        
        def log_artifact_func():
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_path)
            else:
                logger.warning(f"Artifact file not found: {file_path}")
        
        self._safe_mlflow_operation("log_artifact", log_artifact_func)
    
    def log_dataset_info(self, dataset_path: str, task_info: Dict, sample_count: int):
        """Log dataset-specific information."""
        if not self.enabled:
            return
        
        try:
            # Log dataset metadata
            self.log_param("dataset_path", dataset_path)
            self.log_param("task_name", task_info.get("name", "unknown"))
            self.log_param("task_type", task_info.get("type", "unknown"))
            self.log_param("sample_count", sample_count)
            
            # Log dataset file as artifact
            if os.path.exists(dataset_path):
                self.log_artifact(dataset_path, "datasets")
            
            # Log task configuration
            task_config_path = f"/tmp/task_config_{self.session_id}.json"
            with open(task_config_path, "w", encoding="utf-8") as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
            self.log_artifact(task_config_path, "configs")
            os.remove(task_config_path)  # Clean up temp file
            
        except Exception as e:
            logger.error(f"Failed to log dataset info: {e}")


class WandBTracker(BaseTracker):
    """Weights & Biases experiment tracker for cloud-based experiment management."""
    
    def __init__(self, project_name: str = "dekdataset", entity: Optional[str] = None):
        super().__init__(project_name=project_name)
        
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.enabled = False
            return
        
        self.enabled = True
        self.entity = entity
        self.run = None
    
    def start_experiment(self, run_name: Optional[str] = None, tags: Optional[List[str]] = None, **kwargs):
        """Start a new W&B run."""
        if not self.enabled:
            return
        
        super().start_experiment(**kwargs)
        
        try:
            run_name = run_name or f"dataset_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            config = {
                "session_id": self.session_id,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "tracking_system": "WandB"
            }
            config.update(kwargs)
            
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=config,
                tags=tags or ["dataset-generation"],
                reinit=True
            )
            
            logger.info(f"Started W&B run: {run_name}")
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            self.enabled = False
    
    def end_experiment(self):
        """End the current W&B run."""
        if not self.enabled or not self.run:
            return
        
        super().end_experiment()
        
        try:
            wandb.finish()
            self.run = None
            logger.info("Ended W&B run")
        except Exception as e:
            logger.error(f"Failed to end W&B run: {e}")
    
    def log_param(self, key: str, value: Any):
        """Log a parameter to W&B."""
        if not self.enabled or not self.run:
            return
        
        super().log_param(key, value)
        
        try:
            wandb.config[key] = value
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")
    
    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None):
        """Log a metric to W&B."""
        if not self.enabled or not self.run:
            return
        
        super().log_metric(key, value, step)
        
        try:
            log_dict = {key: value}
            if step is not None:
                log_dict["_step"] = step
            wandb.log(log_dict)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to W&B."""
        if not self.enabled or not self.run:
            return
        
        super().log_artifact(file_path, artifact_path)
        
        try:
            if os.path.exists(file_path):
                artifact_name = artifact_path or "dataset_artifacts"
                artifact = wandb.Artifact(artifact_name, type="dataset")
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)
            else:
                logger.warning(f"Artifact file not found: {file_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact {file_path}: {e}")
    
    def log_dataset_info(self, dataset_path: str, task_info: Dict, sample_count: int):
        """Log dataset-specific information to W&B."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Update config with dataset info
            wandb.config.update({
                "dataset_path": dataset_path,
                "task_name": task_info.get("name", "unknown"),
                "task_type": task_info.get("type", "unknown"),
                "sample_count": sample_count
            })
            
            # Log metrics
            self.log_metric("sample_count", sample_count)
            
            # Create dataset artifact
            if os.path.exists(dataset_path):
                dataset_artifact = wandb.Artifact("generated_dataset", type="dataset")
                dataset_artifact.add_file(dataset_path)
                wandb.log_artifact(dataset_artifact)
            
            # Create task config artifact
            task_artifact = wandb.Artifact("task_config", type="config")
            task_config_path = f"/tmp/task_config_{self.session_id}.json"
            with open(task_config_path, "w", encoding="utf-8") as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
            task_artifact.add_file(task_config_path)
            wandb.log_artifact(task_artifact)
            os.remove(task_config_path)  # Clean up temp file
            
        except Exception as e:
            logger.error(f"Failed to log dataset info: {e}")


class ExperimentTracker:
    """Unified experiment tracker that can use multiple backends simultaneously."""
    
    def __init__(self, 
                 use_mlflow: bool = True, 
                 use_wandb: bool = False,
                 mlflow_tracking_uri: Optional[str] = None,
                 wandb_project: str = "dekdataset",
                 wandb_entity: Optional[str] = None,
                 experiment_name: str = "DekDataset"):
        """
        Initialize unified experiment tracker.
        
        Args:
            use_mlflow: Enable MLflow tracking
            use_wandb: Enable Weights & Biases tracking
            mlflow_tracking_uri: MLflow tracking server URI
            wandb_project: W&B project name
            wandb_entity: W&B entity (username/team)
            experiment_name: Name for the experiment
        """
        self.trackers = []
        
        # Initialize MLflow tracker
        if use_mlflow:
            mlflow_tracker = MLflowTracker(experiment_name, mlflow_tracking_uri)
            if mlflow_tracker.enabled:
                self.trackers.append(mlflow_tracker)
        
        # Initialize W&B tracker
        if use_wandb:
            wandb_tracker = WandBTracker(wandb_project, wandb_entity)
            if wandb_tracker.enabled:
                self.trackers.append(wandb_tracker)
        
        if not self.trackers:
            logger.warning("No experiment trackers enabled. Install mlflow and/or wandb packages.")
        
        self.is_active = False
    
    def start_experiment(self, run_name: Optional[str] = None, **kwargs):
        """Start experiment tracking across all enabled trackers."""
        if not self.trackers:
            return
        
        self.is_active = True
        for tracker in self.trackers:
            tracker.start_experiment(run_name=run_name, **kwargs)
    
    def end_experiment(self):
        """End experiment tracking across all enabled trackers."""
        if not self.trackers:
            return
        
        for tracker in self.trackers:
            tracker.end_experiment()
        self.is_active = False
    
    def log_param(self, key: str, value: Any):
        """Log parameter across all enabled trackers."""
        for tracker in self.trackers:
            tracker.log_param(key, value)
    
    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None):
        """Log metric across all enabled trackers."""
        for tracker in self.trackers:
            tracker.log_metric(key, value, step)
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log artifact across all enabled trackers."""
        for tracker in self.trackers:
            tracker.log_artifact(file_path, artifact_path)
    
    def log_dataset_generation_start(self, task_info: Dict, target_count: int, **kwargs):
        """Log the start of dataset generation."""
        if not self.is_active:
            return
        
        # Log generation parameters
        self.log_param("generation_task", task_info.get("name", "unknown"))
        self.log_param("task_type", task_info.get("type", "unknown"))
        self.log_param("target_sample_count", target_count)
        self.log_param("generation_start_time", datetime.now().isoformat())
        
        # Log additional parameters
        for key, value in kwargs.items():
            self.log_param(f"generation_{key}", value)
        
        logger.info(f"Logged dataset generation start: {task_info.get('name')} ({target_count} samples)")
    
    def log_dataset_generation_progress(self, current_count: int, target_count: int, step: int):
        """Log dataset generation progress."""
        if not self.is_active:
            return
        
        progress_percent = (current_count / target_count) * 100 if target_count > 0 else 0
        
        self.log_metric("samples_generated", current_count, step)
        self.log_metric("generation_progress_percent", progress_percent, step)
        
        if step % 10 == 0:  # Log every 10 steps to avoid spam
            logger.info(f"Generation progress: {current_count}/{target_count} ({progress_percent:.1f}%)")
    
    def log_dataset_generation_complete(self, dataset_path: str, final_count: int, task_info: Dict, **metrics):
        """Log completion of dataset generation."""
        if not self.is_active:
            return
        
        # Log completion metrics
        self.log_param("generation_end_time", datetime.now().isoformat())
        self.log_metric("final_sample_count", final_count)
        self.log_metric("generation_success_rate", (final_count / task_info.get("target_count", final_count)) * 100)
        
        # Log additional metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(key, value)
            else:
                self.log_param(key, value)
        
        # Log dataset as artifact
        if os.path.exists(dataset_path):
            self.log_artifact(dataset_path, "generated_datasets")
        
        # Call tracker-specific dataset logging
        for tracker in self.trackers:
            if hasattr(tracker, 'log_dataset_info'):
                tracker.log_dataset_info(dataset_path, task_info, final_count)
        
        logger.info(f"Logged dataset generation complete: {dataset_path} ({final_count} samples)")
    
    def log_quality_metrics(self, quality_scores: Dict[str, float]):
        """Log data quality metrics."""
        if not self.is_active:
            return
        
        for metric_name, score in quality_scores.items():
            self.log_metric(f"quality_{metric_name}", score)
        
        # Calculate overall quality score
        if quality_scores:
            overall_quality = sum(quality_scores.values()) / len(quality_scores)
            self.log_metric("quality_overall", overall_quality)
        
        logger.info(f"Logged quality metrics: {list(quality_scores.keys())}")
    
    def log_cost_metrics(self, api_calls: int, tokens_used: int, estimated_cost: float):
        """Log cost-related metrics."""
        if not self.is_active:
            return
        
        self.log_metric("api_calls_total", api_calls)
        self.log_metric("tokens_used_total", tokens_used)
        self.log_metric("estimated_cost_usd", estimated_cost)
        
        if api_calls > 0:
            self.log_metric("avg_tokens_per_call", tokens_used / api_calls)
            self.log_metric("avg_cost_per_call", estimated_cost / api_calls)
        
        logger.info(f"Logged cost metrics: {api_calls} calls, {tokens_used} tokens, ${estimated_cost:.4f}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_experiment()


# Convenience function for easy usage
def create_experiment_tracker(**kwargs) -> "ExperimentTracker":
    """
    Create an experiment tracker with environment-based configuration.
    
    Environment variables:
    - MLFLOW_TRACKING_URI: MLflow tracking server URI
    - WANDB_PROJECT: W&B project name
    - WANDB_ENTITY: W&B entity name
    - DEKDATASET_USE_MLFLOW: Enable MLflow (default: true)
    - DEKDATASET_USE_WANDB: Enable W&B (default: false)
    """
    # Read configuration from environment
    use_mlflow = os.getenv("DEKDATASET_USE_MLFLOW", "true").lower() == "true"
    use_wandb = os.getenv("DEKDATASET_USE_WANDB", "false").lower() == "true"
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    wandb_project = os.getenv("WANDB_PROJECT", "dekdataset")
    wandb_entity = os.getenv("WANDB_ENTITY")
    
    # Override with explicit kwargs
    config = {
        "use_mlflow": use_mlflow,
        "use_wandb": use_wandb,
        "mlflow_tracking_uri": mlflow_uri,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity
    }
    config.update(kwargs)
    
    return ExperimentTracker(**config)


if __name__ == "__main__":
    # Example usage
    print("=== DekDataset Experiment Tracking Test ===")
    
    # Test with both trackers
    tracker = create_experiment_tracker(use_mlflow=True, use_wandb=False)
    
    with tracker:
        tracker.start_experiment("test_experiment")
        
        # Simulate dataset generation
        task_info = {
            "name": "sentiment_analysis",
            "type": "classification",
            "target_count": 100
        }
        
        tracker.log_dataset_generation_start(task_info, 100, batch_size=10)
        
        # Simulate progress
        for i in range(0, 101, 10):
            tracker.log_dataset_generation_progress(i, 100, i // 10)
        
        # Simulate completion
        tracker.log_dataset_generation_complete(
            "/tmp/test_dataset.jsonl", 
            100, 
            task_info,
            generation_time_seconds=45.2,
            api_provider="deepseek"
        )
        
        # Log quality metrics
        tracker.log_quality_metrics({
            "completeness": 0.98,
            "consistency": 0.95,
            "relevance": 0.92
        })
        
        # Log cost metrics
        tracker.log_cost_metrics(
            api_calls=10,
            tokens_used=5000,
            estimated_cost=0.25
        )
    
    print("Experiment tracking test completed!")
