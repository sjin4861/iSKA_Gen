from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Dict, Any
from .file_system import FileSystem
from ..base.serializer import read_jsonl, append_jsonl_atomic, rewrite_jsonl_atomic

MANIFEST_DIR = "manifests"
CONTENTS_MANIFEST = "contents_manifest.jsonl"
STEMS_MANIFEST = "stems_manifest.jsonl"

@dataclass(frozen=True)
class ManifestEntry:
    kind: str                 # "passage" | "audio_script" | "image_caption" | "stem"
    date: str                 # YYYY-MM-DD
    model: str
    template_key: str
    benchmark_id: int
    benchmark_version: str
    path: str                 # data_store 내부 상대경로
    item_count: int
    sha256: Optional[str] = None
    status: str = "complete"  # "complete" | "partial" | "failed"
    created_at: Optional[str] = None  # ISO8601 (옵션)

class ManifestsFSStore:
    """contents/stems 결과물의 인덱스(카탈로그) 관리"""
    def __init__(self, fs: FileSystem | None = None):
        self.fs = fs or FileSystem()

    def _file(self, name: str) -> Path:
        return self.fs.path(MANIFEST_DIR, name)

    # ---- 조회 ----
    def list_contents(self, **filters) -> list[Dict[str, Any]]:
        rows = read_jsonl(self._file(CONTENTS_MANIFEST))
        return [r for r in rows if self._match(r, filters)]

    def list_stems(self, **filters) -> list[Dict[str, Any]]:
        rows = read_jsonl(self._file(STEMS_MANIFEST))
        return [r for r in rows if self._match(r, filters)]

    # ---- 추가/갱신 ----
    def upsert_content(self, entry: ManifestEntry) -> None:
        self._upsert(self._file(CONTENTS_MANIFEST), entry)

    def upsert_stem(self, entry: ManifestEntry) -> None:
        self._upsert(self._file(STEMS_MANIFEST), entry)

    # ---- 내부 유틸 ----
    def _match(self, row: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for k, v in filters.items():
            if row.get(k) != v:
                return False
        return True

    def _upsert(self, path: Path, entry: ManifestEntry) -> None:
        key_fields = ("kind","date","model","template_key","benchmark_id","benchmark_version")
        rows = read_jsonl(path)
        e = asdict(entry)
        replaced = False
        for i, r in enumerate(rows):
            if all(r.get(k) == e.get(k) for k in key_fields):
                rows[i] = e
                replaced = True
                break
        if replaced:
            rewrite_jsonl_atomic(path, rows)
        else:
            append_jsonl_atomic(path, e)
