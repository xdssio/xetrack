import os


class ClassProperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


class SCHEMA_PARAMS:
    TABLE: str = "db.events"
    TRACK_ID: str = 'track_id'


class CONSTANTS:

    _DTYPES_TO_PYTHON = {
        'BOOLEAN': bool,
        'TINYINT': int,
        'SMALLINT': int,
        'INTEGER': int,
        'BIGINT': int,
        'FLOAT': float,
        'DOUBLE': float,
        'VARCHAR': str,
        'CHAR': str,
        'BLOB': bytearray,
        'DATE': str,
        'TIME': str,
        'TIMESTAMP': str,
        'DECIMAL': float,
        'INTERVAL': str,
        'UUID': str
    }
    TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


class TRACKER_CONSTANTS:
    FUNCTION_NAME: str = 'function_name'
    FUNCTION_TIME: str = 'function_time'
    FUNCTION_RESULT: str = 'function_result'
    ARGS: str = 'args'
    KWARGS: str = 'kwargs'
    ERROR: str = 'error'
    TIMESTAMP: str = 'timestamp'


class LOGURU_PARAMS():
    ROTATION: str = '1 day'
    EXPERIMENT: str = 'EXPERIMENT'
    MONITOR: str = 'MONITOR'
    TRACKING: str = 'TRACKING'
    LEVEL: str = 'level'
    TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_FILE_FORMAT = '{time:YYYY-MM-DD}.log'
    STRACTURED_REGEX = r'\b(MONITOR|EXPERIMENT|TRACKING)\b'

    @ClassProperty
    @classmethod
    def DELIMITER(cls) -> str:
        return os.getenv('LOGS_DELIMITER', '!📁!')

    @ClassProperty
    @classmethod
    def FORMAT(cls) -> str:
        delimiter = cls.DELIMITER
        prefix = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        suffix = "<level>{message}</level>"
        return f"{prefix}{delimiter}{suffix}"
