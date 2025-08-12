import json, os, tempfile
from pathlib import Path

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def write_json_atomic(p: Path, obj) -> None:
    tmp = Path(tempfile.mkstemp(prefix=p.name, dir=str(p.parent))[1])
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    finally:
        if tmp.exists(): tmp.unlink(missing_ok=True)

def read_jsonl(p: Path) -> list[dict]:
    if not p.exists(): return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def append_jsonl_atomic(p: Path, row: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def rewrite_jsonl_atomic(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(prefix=p.name, dir=str(p.parent))[1])
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    finally:
        if tmp.exists(): tmp.unlink(missing_ok=True)
