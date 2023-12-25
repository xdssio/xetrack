import contextlib
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
    _initialized = False

    def __init__(self, stdout: bool = True,
                 logs_path: Optional[str] = None,
                 file_format: Optional[str] = None,
                 prettify: bool = True):
        self.logs_path = logs_path
        self.stdout = stdout
        self.file_format = file_format or LOGURU_PARAMS.LOG_FILE_FORMAT
        self.prettify = prettify
        self.logger = validate_loguru()
        self._init_logger()

    def _init_logger(self):
        if self.logs_path:
            self.logger.add(f"{self.logs_path}/{self.file_format}",
                            format=LOGURU_PARAMS.FORMAT, enqueue=False, serialize=not self.prettify)
        if Logger._initialized:
            return
        Logger._initialized = True
        self._add_levels()
        if self.stdout:
            self.logger.add(
                sys.stdout, format=LOGURU_PARAMS.FORMAT, enqueue=False)

    def _add_levels(self):
        """initialize custom loguru levels"""
        with contextlib.suppress(TypeError):
            self.logger.level(LOGURU_PARAMS.MONITOR, no=26, color="<yellow>")
            self.logger.level(LOGURU_PARAMS.EXPERIMENT, no=27, color="<red>")
            self.logger.level(LOGURU_PARAMS.TRACKING, no=28, color="<yellow>")
            self.logger.remove(0)

    def experiment(self, params: dict):
        self.log(params, LOGURU_PARAMS.EXPERIMENT, indent=4)

    def track(self, data: Union[Dict, pd.DataFrame, List[Dict]]):
        return self.log(data, LOGURU_PARAMS.TRACKING)

    def monitor(self, data: Union[Dict, pd.DataFrame, List[Dict]]) -> Any:
        return self.log(data, LOGURU_PARAMS.MONITOR)

    def log(self, data: Union[Dict, pd.DataFrame, List[Dict]], level=LOGURU_PARAMS.MONITOR, indent: Optional[int] = None) -> Any:
        """ Create a monitoring log from a dictionary, a list of dictionaries or a pandas DataFrame"""
        if isinstance(data, dict):
            self.logger.log(level,
                            json.dumps(data, indent=indent))
        elif isinstance(data, list):
            for row in data:
                self.monitor(row)
        elif isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
            for row in data:
                self.monitor(row)

    def warning(self, msg: str, *args, **kwargs):
        """ Log a message with severity 'warning'."""
        self.logger.warning(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """ Log a message with severity 'info'."""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """ Log a message with severity 'debug'."""
        self.logger.debug(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """ Log a message with severity 'error'."""
        self.logger.error(msg, *args, **kwargs)

    def _iter_logs(self, path: str) -> List[str]:
        result = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.log'):
                    result.append(os.path.join(root, file))
        result.sort(reverse=True)
        return result

    @staticmethod
    def is_tracked_entry(text: str):
        return re.findall(LOGURU_PARAMS.STRACTURED_REGEX, text, re.IGNORECASE)

    def read_structured_logs(self, path: str) -> List[Dict]:
        data_results = []
        with open(path, 'r') as file:
            for line in file:
                entry = self._parse_structured_line(line)
                if entry:
                    data_results.append(entry)
        return data_results

    def _parse_structured_line(self, line: str, level: Optional[str] = None) -> Optional[Dict]:
        """return a relevent log entry from a structured log line"""
        if line.startswith('{"text":') and self.is_tracked_entry(line):
            line_json = json.loads(line)
            log_text = line_json['text']
            meta, text = log_text.split(LOGURU_PARAMS.DELIMITER)
            if level and level not in meta:
                return None
            return json.loads(text.strip())

    def read_log(self, path: str, level: Optional[str] = None) -> List[Dict]:
        data_list = []
        with open(path, 'r') as file:
            json_string = ''
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
                        except json.JSONDecodeError as e:
                            pass
                        json_string = ''

                    # Start a new JSON string
                    json_string = line.split(
                        LOGURU_PARAMS.DELIMITER)[1].strip()
                else:
                    # If the line is a continuation of the JSON string
                    json_string += line.strip()

            # Parse the last JSON string if it exists
            if json_string:
                try:
                    data = json.loads(json_string)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    pass
        return data_list

    def read_logs(self, path: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
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
            raise ValueError(
                "Please provide a directory to read the logs from")
        data = []
        for i, file in enumerate(self._iter_logs(path)):
            if limit is not None and i >= limit:
                break
            data.extend(self.read_log(file))
        return data
