TRACK_ID = 'track_id'
ID = '_id'
EVENTS = 'events'
DB = 'db'
TABLE = f"{DB}.{EVENTS}"
TIMESTAMP = 'timestamp'
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
