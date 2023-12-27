import sqlite3
from typing import Any, Optional
import zlib
from sqlitedict import encode_key, decode_key
from sqlitedict import SqliteDict
import cloudpickle
import zlib
import xxhash


class AssetsManager():
    ASSETS_TABLE_NAME = 'assets'
    KEYS_TABLE_NAME = 'keys'
    COUNTS_TABLE_NAME = 'counts'

    def __init__(self, path: str, autocommit: bool = True, compress: bool = False):
        self.db = path
        self.compress = compress
        self.assets = SqliteDict(
            path, tablename=AssetsManager.ASSETS_TABLE_NAME,
            autocommit=autocommit, encode=self.encode_compress, decode=self.decode_compress,
            encode_key=encode_key, decode_key=decode_key)
        self.keys = SqliteDict(
            path, tablename=AssetsManager.KEYS_TABLE_NAME,
            autocommit=autocommit,
            encode_key=encode_key, decode_key=decode_key)
        self.counts = SqliteDict(
            path, tablename=AssetsManager.COUNTS_TABLE_NAME,
            autocommit=autocommit)

    def commit(self):
        self.assets.commit()
        self.keys.commit()
        self.counts.commit()

    @staticmethod
    def hash_bytes(b: bytes) -> str:
        return xxhash.xxh64(b).hexdigest()

    @staticmethod
    def _to_bytes(asset: Any) -> bytes:
        return cloudpickle.dumps(asset)

    @staticmethod
    def _from_bytes(asset_bytes: bytes) -> Optional[Any]:
        if asset_bytes:
            return cloudpickle.loads(asset_bytes)

    @staticmethod
    def _to_binary(asset_bytes: bytes) -> Optional[sqlite3.Binary]:
        if asset_bytes:
            return sqlite3.Binary(asset_bytes)

    def insert(self, key: str, asset: Any) -> str:
        asset_bytes = self._to_bytes(asset)
        asset_hash = self.hash_bytes(asset_bytes)
        if asset_hash not in self.assets:
            self.assets[asset_hash] = self._to_binary(asset_bytes)
        if asset_hash not in self.counts:
            self.counts[asset_hash] = 0
        self.counts[asset_hash] += 1
        self.keys[key] = asset_hash
        return asset_hash

    def get_from_hash(self, asset_hash: str, default=None) -> Any:
        return self._from_bytes(self.assets.get(asset_hash, None) or default)

    def get_from_key(self, key: str, default=None) -> Any:
        """Returns the asset stored in the database with the given key. If the key does not exist, returns the default value"""
        if hash_candidate := self.keys.get(key, None):
            return self.get_from_hash(hash_candidate, default=default)
        return default

    def get(self, key: str, default=None) -> Any:
        return self.get_from_key(key) or self.get_from_hash(key) or default

    def remove(self, key: str):
        """Removes the asset stored in the database with the given key"""
        asset_hash = self.keys.get(key)
        if asset_hash is None:
            return False
        self.counts[asset_hash] -= 1
        if self.counts[asset_hash] == 0:
            del self.assets[asset_hash]
            del self.counts[asset_hash]
        del self.keys[key]
        return True

    def remove_hash(self, asset_hash: str, remove_keys: bool = True):
        """Removes the asset stored in the database with the given hash
        Args:
            asset_hash (str): The hash of the asset to remove
            remove_keys (bool, optional): Whether to remove the keys associated with the asset. Defaults to True.
        """
        if asset_hash not in self.assets:
            return False
        del self.assets[asset_hash]
        if remove_keys:
            del self.counts[asset_hash]
            keys_to_drop = [keys_to_drop for keys_to_drop,
                            value in self.keys.items() if value == asset_hash]
            for key in keys_to_drop:
                del self.keys[key]
        return True

    def pop(self, key, default=None):
        ret = self.get(key, default)
        self.remove(key)
        return ret or default

    def encode_compress(self, obj_bytes: bytes) -> sqlite3.Binary:
        """deprecated"""
        if self.compress:
            obj_bytes = zlib.compress(obj_bytes)
        return sqlite3.Binary(obj_bytes)

    def decode_compress(self, obj: bytes) -> Any:
        """deprecated"""
        if obj:
            if self.compress:
                obj = zlib.decompress(obj)
            return obj

    @staticmethod
    def is_hash_candidate(value):
        return isinstance(value, str) and len(value) == 16
