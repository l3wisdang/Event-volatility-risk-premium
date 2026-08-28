"""Point-in-time parquet cache.

A study you cannot re-run tomorrow against the same bytes is not a study.  The
source builds re-query IB on every launch and persist nothing, so no number in
either dashboard can be reproduced after the fact.  This stores each response
under a key derived from the *full* request, alongside a manifest recording when
it was retrieved and by whom.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .base import BarRequest

__all__ = ["ParquetCache"]


class ParquetCache:
    def __init__(self, root: str | Path = ".vrplab_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2, default=str))

    def _path(self, request: BarRequest) -> Path:
        return self.root / f"{request.cache_key()}.parquet"

    def get(self, request: BarRequest) -> pd.DataFrame | None:
        p = self._path(request)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        meta = self._manifest.get(request.cache_key(), {})
        df.attrs.update(meta)
        return df

    def put(self, request: BarRequest, df: pd.DataFrame, provider: str = "") -> None:
        key = request.cache_key()
        df.to_parquet(self._path(request))
        self._manifest[key] = {
            "symbol": request.symbol,
            "what": request.what,
            "bar_size": request.bar_size,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider or df.attrs.get("provider", ""),
            "rows": int(len(df)),
            "first": str(df.index.min()) if len(df) else None,
            "last": str(df.index.max()) if len(df) else None,
        }
        self._save_manifest()

    def entries(self) -> pd.DataFrame:
        return pd.DataFrame(self._manifest).T
