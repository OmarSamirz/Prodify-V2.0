from pathlib import Path
from dotenv import load_dotenv

import os
import sys
import shutil
import logging
import inspect
from datetime import datetime
from typing import Optional, Dict, Any


class Logger:

    _instance = None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize()

        return cls._instance
    
    def _initialize(self) -> None:
        env_path = os.path.join(Path(__file__).parents[2], "config", ".env")
        load_dotenv(env_path)

        log_dir = os.getenv("LOG_DIR", "logs")

        if os.path.exists(log_dir):
            if os.getenv("CLEAN_LOGS", "True").lower() == "true":
                try:
                    shutil.rmtree(log_dir)
                    print(f"Log directory cleaned: {os.path.abspath(log_dir)}")
                except Exception as e:
                    print(f"Warning: Could not clean log directory: {str(e)}")

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_path = os.path.join(log_dir, f"borai_{timestamp}.log")
        self.log_file_path = os.path.abspath(os.getenv("LOG_FILE_PATH", default_log_path))

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s] - %(module)s:%(funcName)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        logging.captureWarnings(True)

        print(f"Logger initialized. All logs will be saved to: {self.log_file_path}")

        sys.stdout = LoggerWriter(self.logger.info)
        sys.stderr = LoggerWriter(self.logger.error)

        self._log(logging.INFO, f"Logger initialized. Logs will be saved to {self.log_file_path}")

    def _get_caller_info(self) -> str:
        stack = inspect.stack()
        if len(stack) > 2:
            frame = stack[2]
            module = frame.frame.f_globals.get("__name__", "unknown")
            function = frame.function
            return f"{module}:{function}"

        return "unkown:unkown"
    
    def _log(self, level: int, message: str) -> None:
        self.logger.log(level, message)

    def info(self, message: str) -> None:
        self._log(logging.INFO, message)

    def debug(self, message: str) -> None:
        self._log(logging.DEBUG, message)

    def warning(self, message: str) -> None:
        self._log(logging.WARNING, message)

    def error(self, message: str) -> None:
        self._log(logging.ERROR, message)

    def critical(self, message: str) -> None:
        self._log(logging.CRITICAL, message)

    def exception(self, e: Exception, context: Optional[str] = None) -> None:
        message = f"Exception: {str(e)}"
        if context:
            message = f"{context} : {message}"
        self._log(logging.ERROR, message)

    def log_operation(self, operation: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        message = f"Operation '{operation}' {status}"

        if details:
            message += f" - Details: {details}"

        if status == "failed":
            self.error(message)
        else:
            self.info(message)
        
    def log_db_connection(self, status: str, db_type: str, connection_info: Optional[Dict[str, str]] = None) -> None:
        connection_details = connection_info or {}
        if "password" in connection_info:
            connection_details["password"] = "*****"

        message = f"Database {db_type} connection {status}"

        if connection_details:
            message += f" - Details: {connection_details}"

        if status == "failed":
            self.error(message)
        else:
            self.info(message)

    def log_model_loading(self, model_name: str, device_type: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        model_details = details or {}
        message = f"Model '{model_name}' loading on {device_type} {status}"

        if model_details:
            message += f" - Details: {model_details}"
        
        if status == "failed":
            self.error(message)
        else:
            self.info(message)


class LoggerWriter:

    def __init__(self, log_method):
        self.log_method = log_method
        self.buffer = ""

    def write(self, message):
        if message and not message.isspace():
            self.buffer += message
            if self.buffer.endswith("\n"):
                self.log_method(self.buffer.rstrip())
                self.buffer = ""
    
    def flush(self):
        if self.buffer:
            self.log_method(self.buffer.rstrip())
            self.buffer = ""


logger = Logger()