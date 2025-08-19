import os


class SCHEMA_PARAMS:
    DUCKDB_TABLE: str = "db.default"
    SQLITE_TABLE: str = "default"
    TRACK_ID: str = "track_id"
    DEFAULT_TABLE: str = "default"  # Default table name, can be overridden


class DEFAULTS:
    DB: str = "track.db"

class CONSTANTS:

    IN_MEMORY_DB: str = ":memory:"
    SKIP_INSERT: str = "SKIP_INSERT"

    DTYPES_TO_PYTHON = {
        "BOOLEAN": bool,
        "TINYINT": int,
        "SMALLINT": int,
        "INTEGER": int,
        "BIGINT": int,
        "FLOAT": float,
        "DOUBLE": float,
        "VARCHAR": str,
        "CHAR": str,
        "BLOB": bytearray,
        "DATE": str,
        "TIME": str,
        "TIMESTAMP": str,
        "DECIMAL": float,
        "INTERVAL": str,
        "UUID": str,
    }
    TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class TRACKER_CONSTANTS:
    FUNCTION_NAME: str = "function_name"
    FUNCTION_TIME: str = "function_time"
    FUNCTION_RESULT: str = "function_result"
    ARGS: str = "args"
    KWARGS: str = "kwargs"
    ERROR: str = "error"
    TIMESTAMP: str = "timestamp"
    GIT_COMMIT_KEY: str = "git_commit_hash"


class LoggerMeta(type):
    @property
    def DELIMITER(cls) -> str:
        """Get the delimiter for structured logs."""
        return os.getenv("LOGS_DELIMITER", "!📁!")

    @property
    def FORMAT(cls) -> str:
        """Get the log format with the delimiter."""
        delimiter = cls.DELIMITER
        prefix = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        suffix = "<level>{message}</level>"
        return f"{prefix}{delimiter}{suffix}"


class LOGURU_PARAMS(metaclass=LoggerMeta):
    ROTATION: str = "1 day"
    EXPERIMENT: str = "EXPERIMENT"
    MONITOR: str = "MONITOR"
    TRACKING: str = "TRACKING"
    LEVEL: str = "level"
    TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE_FORMAT: str = "{time:YYYY-MM-DD}.log"
    STRACTURED_REGEX: str = r"\b(MONITOR|EXPERIMENT|TRACKING)\b"
