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

BENCH_META = {
    1: {
        "problem_types": ["제목을 붙인 근거 설명하기", "자문화와 비교하기", "원인과 전망 예측하기"],
        "eval_goals": [
            "글의 전체적인 주제와 핵심 내용을 정확히 파악하는 능력을 평가한다.",
            "지문에 제시된 특정 문화 현상을 자신의 문화적 배경과 관련지어 공통점과 차이점을 구체적으로 비교 설명하는 능력을 평가한다.",
            "글에 제시된 사회/문화적 현상의 원인을 추론하고, 이를 근거로 미래에 나타날 변화나 결과를 논리적으로 설명하는 능력을 평가한다.",
        ],
    },
    2: {
    "problem_types": ["문제 상황 요약하기", "문제 해결 방안 제안하기", "기대 효과 및 부작용 설명하기"],
      "eval_goals": [
        "제시된 갈등 상황의 핵심 원인과 현재 상태를 정확히 분석하고, 이를 바탕으로 문제점을 간결하게 요약하는 능력을 평가한다.",
        "주어진 문제를 해결하기 위한 독창적이면서 실현 가능한 방안을 제시하는 능력을 평가한다.",
        "자신이 제안한 해결 방안이 가져올 긍정적인 기대 효과와 발생 가능한 부작용을 균형 있게 설명하는 능력을 평가한다."
      ],
    },
    3: {
        "problem_types": ["찬성/반대 입장 논거 파악하기", "논리적 근거 제시하기", "예상 반론에 재반박하기"],
      "eval_goals": [
        "제시된 지문에서 특정 입장(찬성 또는 반대)의 핵심 논거를 정확히 파악하고, 그 근거를 자신의 말로 요약하여 설명하는 능력을 평가한다.",
        "자신의 주장을 뒷받침하기 위해 타당한 이유와 구체적인 사례를 들어 논리적으로 설명하는 능력을 평가한다.",
        "자신의 주장과 반대되는 견해를 예상하고, 그에 대한 논리적인 재반박을 통해 자신의 주장을 강화하는 능력을 평가한다."
      ],
    },
    4: {
        "problem_types": ["두 대안의 핵심 차이점 파악하기", "주어진 기준에 따라 장단점 분석하기", "최종 선택 및 결정 이유 정당화하기"],
      "eval_goals": [
        "제시된 두 가지 선택지의 가장 본질적인 차이점이 무엇인지 정확히 파악하는 능력을 평가한다.",
        "가격, 시간, 편의성 등 주어진 특정 기준에 따라 각 옵션의 장점과 단점을 체계적으로 분석하는 능력을 평가한다.",
        "모든 정보를 종합하여 최종적으로 하나의 옵션을 선택하고, 자신의 선택을 논리적으로 정당화하는 능력을 평가한다."
      ],
    },
    5: {
        "problem_types": ["사진 속 문제 상황 묘사하기", "관련 개인 경험 이야기하기", "문제 해결을 위한 자신의 의견 제안하기"],
      "eval_goals": [
        "제시된 사진의 핵심 문제 상황을 구체적인 시각적 단서를 들어 묘사하는 능력을 평가한다.",
        "사진 속 상황과 관련된 자신의 실제 경험을 자연스럽게 설명하는 능력을 평가한다.",
        "묘사한 문제 상황을 해결하기 위한 합리적인 방안이나 제도를 제시하는 능력을 평가한다."
      ],
    }
}

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

    def _get_bench_meta(self, bench_id: int) -> Tuple[List[str], List[str]]:
        meta = BENCH_META.get(bench_id) or {}
        pts = list(meta.get("problem_types", []) or [])
        egs = list(meta.get("eval_goals", []) or [])
        return pts, egs

    # ... 기존 코드 유지 ...
    # --------------------------------------------------------------------- #
    # Public
    # --------------------------------------------------------------------- #
    def find_candidates(self, kind: ContentType, q: OutputQuery) -> Iterable[CandidateOutput]:
        count = 0
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
                                    index=count
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
        index:int=0
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
        benchmark_id = f_bid
        
        # 벤치마크 ID에 따라 실제 content_type 결정 (기본값은 passage)
        actual_content_type = ContentType.passage
        if benchmark_id in self.BENCHMARK_CONTENT_TYPE_MAP:
            actual_content_type = self.BENCHMARK_CONTENT_TYPE_MAP[benchmark_id]
        # stem_1, stem_2, stem_3 찾기
        stem_contents = [row.get(f"stem_{i}") for i in range(1, 4)]
        bench_pts, bench_egs = self._get_bench_meta(benchmark_id)

        # 길이 맞추기 (stem 개수 기준)
        def _align(lst: List[str], n: int) -> List[str]:
            if not lst:
                return [""] * n
            if len(lst) >= n:
                return lst[:n]
            return lst + [""] * (n - len(lst))

        n = len(stem_contents)
        problem_types = _align(bench_pts, n)
        eval_goals = _align(bench_egs, n)  
        # stem이 하나도 없으면 건너뛰기
        if not stem_contents:
            return
        if not any(problem_types):
            tmp = [row.get(f"problem_type_{i+1}", "") for i in range(n)]
            problem_types = _align(tmp, n)
        if not any(eval_goals):
            tmp = [row.get(f"eval_goal_{i+1}", "") for i in range(n)]
            eval_goals = _align(tmp, n)
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
            elif benchmark_id == 2:
                source_item = {
                    "source_kind": "passage_single",
                    "topic": row.get("topic") or row.get("korean_topic") or "",
                    "context": row.get("context") or row.get("korean_context") or "",
                }
            elif benchmark_id in (3, 4):
                source_item = {
                    "source_kind": "audio_script",
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

        source_id = str(index)#_make_source_id()
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
            row.get("source_passage")
            or row.get("passage")
            or row.get("output")
            or row.get("content")
            or ""
        )

        stems = [row.get(f"stem_{i}") for i in range(1, 4)]
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
