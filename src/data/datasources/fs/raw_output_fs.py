# src/data/datasources/fs/raw_output_fs.py
from __future__ import annotations
import hashlib
from typing import Iterable, Iterator, Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
import json
import re

from src.domain.entities.outputs import CandidateOutput
from src.domain.entities.enums import ContentType
from src.domain.entities.output_query import OutputQuery

class RawOutputFSDataSource:
    """
    실제 폴더 구조(예시):
      data_store/
        raw_outputs/
          2025-08-08/
            passage/
              A.X-4.0-Light/
                passage_agent.create_dialogue_passage/
                  benchmark_3_v1.1.0_passage_agent.create_dialogue_passage.json
                ...
              EXAONE-3.5-7.8B-Instruct/...
            passage_processed/
            stem/
          2025-08-10/
            ...

    사용:
      ds = RawOutputFSDataSource("data_store")  # 또는 "data_store/raw_outputs"
      for c in ds.find_candidates(ContentType.passage, q): ...

    규칙:
      - 날짜 필터: YYYY-MM-DD 디렉터리명으로 1차 필터
      - kind: 동의어 매칭으로 디렉터리 선택
      - model_name 필터: 모델 디렉터리명과 매칭(슬래시는 언더스코어로 치환)
      - 파일명에서 benchmark_id/version/template 추출(가능하면)
      - JSON 파일은 [ ... ] 또는 {"items":[...]} / {"results":[...]} / {"data":[...]} 모두 지원
    """

    KIND_DIRS: Dict[ContentType, Tuple[str, ...]] = {
        ContentType.passage: ("passage",),
        ContentType.audio_script: ("audio", "audio_script", "dialog"),
        ContentType.image_caption: ("image", "image_caption", "picture", "photo"),
        ContentType.stem: ("stem",),
    }
    
    # 벤치마크 ID에 따른 실제 콘텐츠 타입 매핑
    # TODO: [데이터 구조 개선] 벤치마크 메타데이터 외부화
    # 현재는 하드코딩된 매핑이지만, 향후 벤치마크 설정 파일에서 로드하도록 개선 필요
    BENCHMARK_CONTENT_TYPE_MAP: Dict[int, ContentType] = {
        1: ContentType.passage,      # 읽고 말하기 - create_passage_rubric_aware
        2: ContentType.passage,      # 읽고 말하기 - create_domestic_passage
        3: ContentType.audio_script, # 듣고 말하기 - create_dialogue_passage
        4: ContentType.audio_script, # 듣고 말하기 - create_dialogue_passage
        5: ContentType.image_caption # 보고 말하기 - create_image_caption_and_situation
        # TODO: 새로운 벤치마크 추가 시 여기에 매핑 추가 필요
    }

    # benchmark_3_v1.1.0_passage_agent.create_dialogue_passage.json
    FILE_RE = re.compile(
        r"benchmark[_-](?P<bid>\d+)[_-]v(?P<ver>[0-9.]+)[_-](?P<tkey>.+)\.jsonl?$",
        re.IGNORECASE,
    )

    def __init__(self, root_dir: str | Path):
        root = Path(root_dir)
        # root가 data_store면 raw_outputs로 이동, 이미 raw_outputs면 그대로 사용
        self.base = root / "raw_outputs" if (root / "raw_outputs").exists() else root
        if not self.base.exists():
            raise FileNotFoundError(f"raw_outputs 경로를 찾을 수 없습니다: {self.base}")

    def _get_actual_content_type(self, benchmark_id: Optional[int]) -> ContentType:
        """
        벤치마크 ID에 따라 실제 content_type을 결정
        
        TODO: [로직 개선] 더 견고한 타입 결정 로직 필요
        현재는 단순 매핑이지만, 향후 다음 개선 사항들 고려:
        1. 벤치마크 메타데이터에서 동적으로 로드
        2. 파일 내용 분석을 통한 자동 타입 감지
        3. 템플릿 키 기반 타입 추론
        4. 폴백 메커니즘 (매핑 없는 경우 기본 처리)
        """
        if benchmark_id is not None and benchmark_id in self.BENCHMARK_CONTENT_TYPE_MAP:
            return self.BENCHMARK_CONTENT_TYPE_MAP[benchmark_id]
        return ContentType.passage  # 기본값

    # --------------------------------------------------------------------- #
    # Public
    # --------------------------------------------------------------------- #
    def find_candidates(self, kind: ContentType, q: OutputQuery) -> Iterable[CandidateOutput]:
        count = 0
        # breakpoint()
        for day_dir in self._iter_date_dirs(q):
            kind_dir = self._resolve_kind_dir(day_dir, kind)
            if not kind_dir:
                continue
            for model_dir in self._iter_model_dirs(kind_dir, q.model_name):
                for tdir in self._iter_template_dirs(model_dir):
                    for file in sorted(tdir.glob("*.json")) + sorted(tdir.glob("*.jsonl")):
                        # 파일명에서 benchmark_id가 추출되면 1차 필터
                        f_bid, f_ver, f_tkey = self._parse_file_meta(file.name)
                        if q.benchmark_id is not None and f_bid is not None and f_bid != q.benchmark_id:
                            continue
                        # 로드 & 라인/아이템 필터
                        for item in self._iter_items(file):
                            if not self._row_passes_filters(item, q):
                                continue
                            
                            # Stem 데이터인지 확인
                            is_stem_data = (
                                "stem" in str(file).lower() or 
                                any(key.startswith("stem_") for key in item.keys())
                            )
                            
                            if is_stem_data:
                                # Stem 데이터: 각 stem을 개별 후보로 생성
                                for candidate in self._create_stem_candidates(
                                    row=item,
                                    model_name=model_dir.name,
                                    date_str=day_dir.name,
                                    file_meta=(f_bid, f_ver, f_tkey),
                                    template_key=tdir.name,
                                    file_path=str(file),
                                ):
                                    yield candidate
                                    count += 1
                                    if q.limit and count >= q.limit:
                                        return
                            else:
                                # 일반 데이터
                                yield self._row_to_candidate(
                                    row=item,
                                    kind=kind,
                                    model_name=model_dir.name,
                                    date_str=day_dir.name,
                                    file_meta=(f_bid, f_ver, f_tkey),
                                    template_key=tdir.name,
                                    file_path=str(file),
                                )
                                count += 1
                                if q.limit and count >= q.limit:
                                    return

    # --------------------------------------------------------------------- #
    # FS walkers
    # --------------------------------------------------------------------- #
    def _iter_date_dirs(self, q: OutputQuery) -> Iterator[Path]:
        for d in sorted(self.base.iterdir()):
            if not d.is_dir():
                continue
            # 디렉터리명이 YYYY-MM-DD인지 검사
            try:
                ddate = datetime.strptime(d.name, "%Y-%m-%d")
            except ValueError:
                continue
            # 날짜 범위 필터
            if q.date_from and ddate < q.date_from:
                continue
            if q.date_to and ddate > q.date_to:
                continue
            yield d

    def _resolve_kind_dir(self, day_dir: Path, kind: ContentType) -> Optional[Path]:
        candidates = self.KIND_DIRS.get(kind, (kind.value,))
        for name in candidates:
            p = day_dir / name
            if p.exists() and p.is_dir():
                return p
        return None

    def _iter_model_dirs(self, kind_dir: Path, model_name: Optional[str]) -> Iterator[Path]:
        if not kind_dir.exists():
            return
        if model_name:
            mdir = kind_dir / model_name.replace("/", "_")
            if mdir.exists() and mdir.is_dir():
                yield mdir
            return
        # 모델 디렉토리 후보(템플릿 키/기타 폴더와 혼동되지 않도록 2단계 하위에 파일이 있는 폴더만)
        for p in sorted(kind_dir.iterdir()):
            if not p.is_dir():
                continue
            # 템플릿 디렉터리 유무로 모델 폴더인지 판단
            has_template = any((p / sub).is_dir() for sub in p.iterdir()) if any(True for _ in p.iterdir()) else False
            if has_template:
                yield p

    def _iter_template_dirs(self, model_dir: Path) -> Iterator[Path]:
        for tdir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            yield tdir

    # --------------------------------------------------------------------- #
    # JSON loaders
    # --------------------------------------------------------------------- #
    def _iter_items(self, file: Path) -> Iterator[Dict[str, Any]]:
        """
        파일 포맷 허용:
          1) [ { ... }, ... ]
          2) { "items": [ ... ] } / {"results":[...]} / {"data":[...]}
        """
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except Exception:
            return
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    yield it
            return
        if isinstance(data, dict):
            for k in ("items", "results", "data"):
                v = data.get(k)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            yield it
                    return
            # dict 단일 객체도 수용(단일 후보)
            yield data

    # --------------------------------------------------------------------- #
    # Filters & mapping
    # --------------------------------------------------------------------- #
    def _row_passes_filters(self, row: Dict[str, Any], q: OutputQuery) -> bool:
        # benchmark_id가 라인에 있으면 체크(없으면 파일/상위 메타로 커버)
        if q.benchmark_id is not None:
            try:
                rbid = int(row.get("benchmark_id")) if "benchmark_id" in row else None
                if rbid is not None and rbid != q.benchmark_id:
                    return False
            except Exception:
                pass
        if q.model_name and row.get("model_name") and row["model_name"] != q.model_name:
            return False
        if q.source_ids:
            sid = str(row.get("source_id") or "")
            if sid not in set(q.source_ids):
                return False
        return True

    # --------------------------------------------------------------------- #
    # Filters & mapping
    # --------------------------------------------------------------------- #

    def _create_stem_candidates(
        self,
        *,
        row: Dict[str, Any],
        model_name: str,
        date_str: str,
        file_meta: Tuple[Optional[int], Optional[str], Optional[str]],
        template_key: str,
        file_path: str,
    ) -> Iterator[CandidateOutput]:
        """
        Stem 데이터를 처리하여 통합된 후보 생성
        - content: source_passage (원본 지문)
        - content_type: passage (원본 타입)
        - stems: 모든 stem 문항을 리스트로 ([stem1, stem2, stem3, ...])
        
        변경사항: 개별 stem 후보 생성 → 통합된 하나의 후보 생성 (stems 배열에 모든 stem 포함)
        """
        source_passage = row.get("source_passage", "")
        f_bid, f_ver, f_tkey = file_meta
        benchmark_id = int(row.get("benchmark_id") or f_bid or -1)
        
        # 벤치마크 ID에 따라 실제 content_type 결정 (기본값은 passage)
        actual_content_type = ContentType.passage
        if benchmark_id in self.BENCHMARK_CONTENT_TYPE_MAP:
            actual_content_type = self.BENCHMARK_CONTENT_TYPE_MAP[benchmark_id]
        
        # stem_1, stem_2, stem_3 찾기
        stem_contents = []
        problem_types = []
        eval_goals = []
        
        # 최대 3개까지 stem 수집 (completeness_for_guidelines_binary에서 필요)
        for idx in range(1, 4):  # stem_1, stem_2, stem_3
            stem_content = row.get(f"stem_{idx}", "")
            problem_type = row.get(f"problem_type_{idx}", "")
            eval_goal = row.get(f"eval_goal_{idx}", "")
            
            if stem_content.strip():  # 비어있지 않은 stem만 추가
                stem_contents.append(stem_content)
                problem_types.append(problem_type)
                eval_goals.append(eval_goal)
        
        # stem이 하나도 없으면 건너뛰기
        if not stem_contents:
            return
        
        # === source_item 복원 ===
        # 1) row에 이미 들어있으면 그대로 사용
        source_item = row.get("source_item")
        # 2) 없으면 benchmark 규칙으로 구성
        if not source_item:
            # 벤치 1: 비교형(ko/foreign), 2/3/4: 단일형(topic/context), 5: 이미지(topic)
            if benchmark_id == 1:
                source_item = {
                    "source_kind": "passage_multi",
                    "korean_topic": row.get("korean_topic") or row.get("topic") or "",
                    "korean_context": row.get("korean_context") or row.get("context") or "",
                    "foreign_topic": row.get("foreign_topic"),
                    "foreign_context": row.get("foreign_context"),
                }
            elif benchmark_id in (2, 3, 4):
                source_item = {
                    "source_kind": "passage_single",
                    "topic": row.get("topic") or row.get("korean_topic") or "",
                    "context": row.get("context") or row.get("korean_context") or "",
                }
            elif benchmark_id == 5:
                source_item = {
                    "source_kind": "image_caption",
                    "topic": row.get("topic") or row.get("korean_topic") or "",
                }
            else:
                # 알 수 없는 경우라도 최소한의 정보 보존
                source_item = {
                    "source_kind": "unknown",
                    "topic": row.get("topic") or row.get("korean_topic"),
                    "context": row.get("context") or row.get("korean_context"),
                    "foreign_topic": row.get("foreign_topic"),
                    "foreign_context": row.get("foreign_context"),
                }
            

        # === source_id 생성(안전한 해시, 이미 있으면 사용) ===
        def _make_source_id() -> str:
            sid = row.get("source_id")
            if sid:
                return sid
            payload = None
            if source_item:
                try:
                    payload = json.dumps(source_item, ensure_ascii=False, sort_keys=True)
                except Exception:
                    payload = str(source_item)
            else:
                payload = source_passage or (f"{benchmark_id}:{file_path}")
            h = hashlib.sha1((payload or "").encode("utf-8")).hexdigest()[:10]
            # content_type 접두어로 구분도 가능하지만 bench 기준으로 식별
            return f"b{benchmark_id}_src_{h}"

        source_id = _make_source_id()
        candidate_id = f"{source_id}:{model_name}:{template_key}:{date_str}"

        # === meta 구성 ===
        meta: Dict[str, Any] = {
            "template_key": template_key,
            "date": date_str,
            "file_path": file_path,
            "problem_types": problem_types,
            "eval_goals": eval_goals,
            "stem_count": len(stem_contents),
        }
        if f_ver:
            meta["benchmark_version"] = f_ver
        if f_tkey and "template_key" not in meta:
            meta["template_key"] = f_tkey

        generated_at = datetime.strptime(date_str, "%Y-%m-%d")

        # === 최종 CandidateOutput ===
        yield CandidateOutput(
            source_id=source_id,
            benchmark_id=benchmark_id,
            model_name=model_name,
            candidate_id=candidate_id,
            content_type=actual_content_type,
            content=source_passage,
            stems=stem_contents,
            generated_at=generated_at,
            meta=meta,
            # ★ 추가: 평가에서 바로 참조할 수 있도록 박아둠
            source_item=source_item,
        )

    def _row_to_candidate(
        self,
        *,
        row: Dict[str, Any],
        kind: ContentType,
        model_name: str,
        date_str: str,
        file_meta: Tuple[Optional[int], Optional[str], Optional[str]],
        template_key: str,
        file_path: str,
    ) -> CandidateOutput:
        
        # 일반 데이터 처리
        content = (
            row.get("content")
            or row.get("passage")
            or row.get("output")
            or row.get("text")
            or ""
        )
        
        stems = row.get("stems")
        if isinstance(stems, str):
            stems = [s.strip() for s in stems.split(";;") if s.strip()]
        
        # source_id 없으면 파일+인덱스 기반으로 생성
        source_id = str(row.get("source_id") or row.get("id") or row.get("name") or row.get("source") or "")
        if not source_id:
            # 파일 기반 가명 ID
            source_id = f"src::{Path(file_path).stem}"

        f_bid, f_ver, f_tkey = file_meta
        benchmark_id = int(row.get("benchmark_id") or f_bid or -1)
        
        # 벤치마크 ID에 따라 실제 content_type 결정
        actual_content_type = kind  # 기본값은 전달받은 kind
        if benchmark_id in self.BENCHMARK_CONTENT_TYPE_MAP:
            actual_content_type = self.BENCHMARK_CONTENT_TYPE_MAP[benchmark_id]

        candidate_id = str(
            row.get("candidate_id")
            or f"{source_id}:{model_name}:{template_key}:{date_str}"
        )

        meta = dict(row.get("meta") or {})
        # 파일/디렉터리 메타 주입
        meta.update({
            "template_key": template_key,
            "date": date_str,
            "file_path": file_path,
        })
        if f_ver:
            meta.setdefault("benchmark_version", f_ver)
        if f_tkey and "template_key" not in meta:
            meta["template_key"] = f_tkey

        # generated_at: 라인 없으면 상위 날짜 사용
        gen_at = row.get("generated_at")
        if isinstance(gen_at, str):
            try:
                generated_at = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
            except Exception:
                generated_at = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            generated_at = datetime.strptime(date_str, "%Y-%m-%d")

        return CandidateOutput(
            source_id=source_id,
            benchmark_id=benchmark_id,
            model_name=model_name,
            candidate_id=candidate_id,
            content_type=actual_content_type,  # 결정된 실제 타입 사용
            content=str(content),
            stems=stems if stems else None,
            generated_at=generated_at,
            meta=meta,
        )

    @classmethod
    def _parse_file_meta(cls, filename: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        파일명에서 (benchmark_id, version, template_key) 추출.
        매칭 실패 시 (None, None, None)
        """
        m = cls.FILE_RE.match(filename)
        if not m:
            return (None, None, None)
        bid = int(m.group("bid"))
        ver = m.group("ver")
        tkey = m.group("tkey")
        return (bid, ver, tkey)
