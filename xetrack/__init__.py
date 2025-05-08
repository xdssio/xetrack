import contextlib
from typing import Literal

from loguru import logger
from xetrack.config import SCHEMA_PARAMS, TRACKER_CONSTANTS
from xetrack.reader import Reader
from xetrack.tracker import Tracker

__version__ = "0.0.0"

with contextlib.suppress(ImportError):
    from importlib.metadata import version
    __version__ = version("xetrack")


def copy(source: str, target: str, assets: bool = True):
    import sqlite3
    from xetrack.tracker import Tracker
        
    source_tracker = Tracker(db=source)
    target_tracker = Tracker(db=target)
        
    for column, dtype in source_tracker.dtypes.items():
        if column not in target_tracker.columns:
            target_tracker.add_column(column, dtype)

    src_conn = sqlite3.connect(source)
    dest_conn = sqlite3.connect(target)
    
    pre_count = dest_conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    
    src_conn.execute(f"ATTACH DATABASE '{target}' AS dest")
    
    if assets:
        src_conn.execute('''
        INSERT OR IGNORE INTO dest.assets
        SELECT * FROM assets
        ''')  
        src_conn.execute('''
        INSERT OR IGNORE INTO dest.keys
        SELECT * FROM keys
        ''')  
        
        src_conn.execute('''
        INSERT OR IGNORE INTO dest.counts
        SELECT * FROM counts
        ''')
            
    src_cols = [row[1] for row in src_conn.execute("PRAGMA table_info(events)").fetchall()]
    dest_cols = [row[1] for row in dest_conn.execute("PRAGMA table_info(events)").fetchall()]
    common_cols = [col for col in src_cols if col in dest_cols]
        
    cols_str = ", ".join(common_cols)
    src_conn.execute(f'''
    INSERT OR IGNORE INTO dest.events ({cols_str})
    SELECT {cols_str} FROM events
    ''')
    
    src_conn.commit()    
    src_conn.execute("DETACH DATABASE dest")
    src_conn.close()
        
    post_count = dest_conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    dest_conn.close()
    
    return post_count - pre_count