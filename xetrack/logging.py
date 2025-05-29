import sys
import json
import os
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import re
from xetrack.config import LOGURU_PARAMS

init = False


def validate_loguru():
    try:
        from loguru import logger
    except ImportError as e:
        raise ImportError(
            "loguru is not installed. Please install it using `pip install loguru`"
        ) from e
    return logger


class Logger:
    _current_config = None  # Track the current global configuration
    _logger_instance = None  # Shared logger instance

    def __init__(
        self,
        stdout: bool = True,
        logs_path: Optional[str] = None,
        file_format: Optional[str] = None,
        prettify: bool = True,
    ):
        self.logs_path = logs_path
        self.stdout = stdout
        self.file_format = file_format or LOGURU_PARAMS.LOG_FILE_FORMAT
        self.prettify = prettify
        
        # Get or create the shared logger instance
        if Logger._logger_instance is None:
            Logger._logger_instance = validate_loguru()
        self.logger = Logger._logger_instance
        
        # Configure the logger with this instance's settings
        self._configure_logger()

    def _configure_logger(self):
        """Configure the global logger with this instance's settings"""
        current_config = {
            'stdout': self.stdout,
            'logs_path': self.logs_path,
            'file_format': self.file_format,
            'prettify': self.prettify
        }
        
        # Only reconfigure if this is different from the current configuration
        if Logger._current_config != current_config:
            # Remove all existing handlers
            self.logger.remove()
            
            # Add custom levels (safe to call multiple times)
            self._add_levels()
            
            # Add stdout handler if requested
            if self.stdout:
                self.logger.add(
                    sys.stdout,
                    format=LOGURU_PARAMS.FORMAT,  # type: ignore
                    enqueue=False,
                )
            
            # Add file handler if logs_path is provided
            if self.logs_path:
                self.logger.add(
                    f"{self.logs_path}/{self.file_format}",
                    format=LOGURU_PARAMS.FORMAT,  # type: ignore
                    enqueue=False,
                    serialize=not self.prettify,
                )
            
            # Update the current configuration
            Logger._current_config = current_config.copy()

    def _add_levels(self):
        """Add custom log levels (safe to call multiple times)"""
        # Add custom log levels - loguru handles duplicates gracefully
        try:
            self.logger.level("MONITOR", no=26, color="<blue>")
            self.logger.level("TRACKING", no=27, color="<blue>")
            self.logger.level("EXPERIMENT", no=28, color="<blue>")
        except Exception:
            # Levels might already exist, which is fine
            pass

    def experiment(self, params: Dict[str, Any]):
        self.log(params, LOGURU_PARAMS.EXPERIMENT, indent=4)

    def track(self, data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]):
        return self.log(data, LOGURU_PARAMS.TRACKING)

    def monitor(
        self, data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
    ) -> Any:
        return self.log(data, LOGURU_PARAMS.MONITOR)

    def log(
        self,
        data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
        level: str = LOGURU_PARAMS.MONITOR,
        indent: Optional[int] = None,
    ) -> Any:
        """Create a monitoring log from a dictionary, a list of dictionaries or a pandas DataFrame"""
        if isinstance(data, dict):
            self.logger.log(level, json.dumps(data, indent=indent))
        elif isinstance(data, list):
            for row in data:
                self.monitor(row)
        else:
            data = data.to_dict(orient="records")  # type: ignore
            for row in data:
                self.monitor(row)  # type: ignore

    def warning(self, msg: str, *args: Any, **kwargs: Any):
        """Log a message with severity 'warning'."""
        self.logger.warning(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any):
        """Log a message with severity 'info'."""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any):
        """Log a message with severity 'debug'."""
        self.logger.debug(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any):
        """Log a message with severity 'error'."""
        self.logger.error(msg, *args, **kwargs)

    def _iter_logs(self, path: str) -> List[str]:
        result = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".log"):
                    result.append(os.path.join(root, file))
        result.sort(reverse=True)
        return result

    @staticmethod
    def is_tracked_entry(text: str):
        return re.findall(LOGURU_PARAMS.STRACTURED_REGEX, text, re.IGNORECASE)

    def read_structured_logs(self, path: str) -> List[Dict[str, Any]]:
        data_results = []
        with open(path, "r") as file:
            for line in file:
                entry = self._parse_structured_line(line)
                if entry:
                    data_results.append(entry)
        return data_results

    def _parse_structured_line(
        self, line: str, level: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """return a relevent log entry from a structured log line"""
        if line.startswith('{"text":') and self.is_tracked_entry(line):
            line_json = json.loads(line)
            log_text = line_json["text"]
            meta, text = log_text.split(LOGURU_PARAMS.DELIMITER)
            if level and level not in meta:
                return None
            return json.loads(text.strip())

    def read_log(self, path: str, level: Optional[str] = None) -> List[Dict[str, Any]]:
        data_list = []
        with open(path, "r") as file:
            json_string = ""
            for line in file:
                structured = self._parse_structured_line(line)
                if structured:
                    data_list.append(structured)
                    continue
                if LOGURU_PARAMS.DELIMITER in line:
                    # If there is an ongoing JSON string, parse it before starting a new one
                    if json_string:
                        try:
                            data = json.loads(json_string)
                            data_list.append(data)
                        except json.JSONDecodeError:
                            pass
                        json_string = ""

                    # Start a new JSON string
                    parts = line.split(LOGURU_PARAMS.DELIMITER)  # type: ignore
                    if len(parts) > 1:
                        json_string = parts[1].strip()
                else:
                    # If the line is a continuation of the JSON string
                    json_string += line.strip()

            # Parse the last JSON string if it exists
            if json_string:
                try:
                    data = json.loads(json_string)
                    data_list.append(data)
                except json.JSONDecodeError:
                    pass
        return data_list

    def read_logs(
        self, path: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        This function reads the logs from a directory. It returns a list of dictionaries.
        It reads latest logs first.

        Args:
        path (str, optional): Path to the directory where the logs are stored. Defaults to loggers' path.
        limit (int, optional): Maximum number of log files to read. Defaults to None.
        """
        if path is None:
            path = self.logs_path
        if path is None:
            raise ValueError("Please provide a directory to read the logs from")
        data = []
        for i, file in enumerate(self._iter_logs(path)):
            if limit is not None and i >= limit:
                break
            data.extend(self.read_log(file))
        return data
