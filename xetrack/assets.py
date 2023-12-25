import os
import sqlite3
from typing import Any
import zlib
from sqlitedict import encode_key, decode_key
from sqlitedict import SqliteDict
import cloudpickle
import zlib
import xxhash


class Assets():
    ASSETS_TABLE_NAME = 'assets'
    KEYS_TABLE_NAME = 'assets'
    COUNTS_TABLE_NAME = 'counts'

    def __init__(self, path: str, autocommit: bool = True, compress: bool = False):
        self.db = path
        self.compress = compress
        self.assets = SqliteDict(
            path, tablename=Assets.ASSETS_TABLE_NAME,
            autocommit=autocommit, encode=self.encode, decode=self.decode,
            encode_key=encode_key, decode_key=decode_key)
        self.keys = SqliteDict(
            path, tablename=Assets.KEYS_TABLE_NAME,
            autocommit=autocommit,
            encode_key=encode_key, decode_key=decode_key)
        self.counts = SqliteDict(
            path, tablename=Assets.COUNTS_TABLE_NAME,
            autocommit=autocommit)

    @staticmethod
    def hash_bytes(b: bytes) -> str:
        return xxhash.xxh64(b).hexdigest()

    def insert(self, key: str, asset: Any):
        asset_bytes = cloudpickle.dumps(asset)
        asset_hash = self.hash_bytes(asset_bytes)
        if asset_hash not in self.assets:
            self.assets[asset_hash] = asset_bytes
        if key not in self.keys:
            self.counts[asset_hash] = 1
        else:
            self.counts[asset_hash] += 1
        self.keys[key] = asset_hash

    def get(self, key: str, default=None) -> Any:
        """Returns the asset stored in the database with the given key. If the key does not exist, returns the default value"""
        return self.decode(self.assets.get(self.keys.get(key))) or default

    def remove(self, key: str):
        """Removes the asset stored in the database with the given key"""
        asset_hash = self.keys[key]
        self.counts[asset_hash] -= 1
        if self.counts[asset_hash] == 0:
            del self.assets[asset_hash]
            del self.counts[asset_hash]
        del self.keys[key]

    def pop(self, key, default=None):
        ret = self.get(key, default)
        self.remove(key)
        return ret or default

    def encode(self, obj) -> sqlite3.Binary:
        obj_bytes = cloudpickle.dumps(obj)
        if self.compress:
            obj_bytes = zlib.compress(obj_bytes)
        return sqlite3.Binary(obj_bytes)

    def decode(self, obj: bytes) -> Any:
        if obj:
            if self.compress:
                obj = zlib.decompress(obj)
            return cloudpickle.loads(obj)
