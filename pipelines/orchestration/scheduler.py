"""
Pipeline scheduling and monitoring.
"""
import schedule
import time
from datetime import datetime
from typing import Callable, Dict, Any
from prefect import flow
from loguru import logger

from configs import get_config

class PipelineScheduler:
    """Schedules and manages periodic pipeline executions."""
    
    def __init__(self):
        self.config = get_config()
        self.jobs = {}
    
    def schedule_daily_processing(self, pipeline_func: Callable, time_str: str = "02:00"):
        """Schedule daily pipeline execution."""
        job = schedule.every().day.at(time_str).do(
            self._run_scheduled_pipeline, pipeline_func, "daily"
        )
        self.jobs["daily_processing"] = job
        logger.info(f"Scheduled daily pipeline for {time_str}")
    
    def schedule_hourly_processing(self, pipeline_func: Callable):
        """Schedule hourly pipeline execution."""
        job = schedule.every().hour.do(
            self._run_scheduled_pipeline, pipeline_func, "hourly"
        )
        self.jobs["hourly_processing"] = job
        logger.info("Scheduled hourly pipeline")
    
    def _run_scheduled_pipeline(self, pipeline_func: Callable, schedule_type: str):
        """Execute scheduled pipeline with logging."""
        logger.info(f"Starting {schedule_type} pipeline execution")
        try:
            result = pipeline_func()
            logger.success(f"{schedule_type} pipeline completed: {result}")
        except Exception as e:
            logger.error(f"{schedule_type} pipeline failed: {e}")
    
    def start_scheduler(self):
        """Start the scheduling loop."""
        logger.info("Starting pipeline scheduler")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Pipeline scheduler stopped")
