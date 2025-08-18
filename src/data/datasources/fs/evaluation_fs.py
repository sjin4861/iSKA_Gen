from __future__ import annotations
import re
from typing import Iterable, Iterator, Literal, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

from src.domain.entities.evaluation import EvaluationRecord
from src.domain.entities.evaluation_query import EvaluationQuery
from src.domain.entities.rubrics import RubricID
from src.domain.usecases.evaluation.evaluate import EvaluateInput, EvaluateOutput
from src.domain.entities.evaluation import EvaluationTarget, BinaryScore, LikertScore, JudgeMeta
from src.data.datasources.fs.templates_fs import TemplatesFSDataSource

# 루브릭별 평가 로직 선택
CONTENT_PLUS_STEM_RUBRIC = [
    RubricID.completeness_for_guidelines, 
    RubricID.l2_learner_suitability,
    RubricID.R1_GUIDELINE_COMPLETENESS,
    RubricID.R6_L2_APPROPRIATENESS
]

class EvaluationFSDataSource:
        """
        data_store/evaluations/ 아래 JSONL을 관리하는 FS 데이터소스.

        디렉토리 구조(쓰기/읽기 공통):
        evaluations/{rubric_id}/{content_type}/{evaluated_by}/{run_id or 'misc'}/YYYYMMDD.jsonl

        기능:
        - append(record, run_id?, date_str?) : 한 줄 추가
        - bulk_append(records, run_id?, date_str?)
        - find(query: EvaluationQuery) -> Iterable[EvaluationRecord]
        """

        def __init__(self, root: str | Path):
            root = Path(root)
            self.base = root / "evaluations" if (root / "evaluations").exists() else root
            self.base.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Write (append)
        # ------------------------------------------------------------------ #
        def append(self, record: EvaluationRecord, *, run_id: Optional[str] = None, date_str: Optional[str] = None) -> None:
            path = self._path_for(record, run_id=run_id, date_str=date_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")

        def bulk_append(self, records: Iterable[EvaluationRecord], *, run_id: Optional[str] = None, date_str: Optional[str] = None) -> None:
            # 같은 경로끼리 파일 핸들 재사용
            handles: Dict[Path, Any] = {}
            try:
                for rec in records:
                    path = self._path_for(rec, run_id=run_id, date_str=date_str)
                    if path not in handles:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        handles[path] = path.open("a", encoding="utf-8")
                    handles[path].write(rec.model_dump_json() + "\n")
            finally:
                for fh in handles.values():
                    try:
                        fh.close()
                    except Exception:
                        pass

        def _path_for(self, record: EvaluationRecord, *, run_id: Optional[str], date_str: Optional[str]) -> Path:
            rid = record.rubric_id.value if hasattr(record.rubric_id, "value") else str(record.rubric_id)
            ctype = record.target.content_type.value
            by = record.evaluated_by.value if hasattr(record.evaluated_by, "value") else str(record.evaluated_by)
            rdir = run_id or record.run_id or "misc"
            day = date_str or datetime.utcnow().strftime("%Y%m%d")
            
            # 평가 대상 모델과 벤치마크 정보 추출
            evaluated_model = "unknown_model"
            benchmark_id = "unknown_benchmark"
            
            # record의 target에서 정보 추출 시도
            if hasattr(record.target, 'content') and record.target.content:
                # EvaluationTarget의 경우 stems에서 원본 CandidateOutput 정보를 찾기 어려움
                # judge_meta나 notes에서 추출 시도
                pass
            
            # judge_meta에서 추출 시도 (judge_meta는 평가자 정보, target 모델 정보가 아님)
            if hasattr(record, 'judge_meta') and record.judge_meta:
                # judge_meta에는 평가자 모델 정보만 있음
                pass
            
            # notes에서 메타데이터 추출 시도
            if hasattr(record, 'notes') and record.notes:
                try:
                    import re
                    # 새로운 메타데이터 형식: [META:model=XXX,benchmark=YYY]
                    meta_pattern = r'\[META:model=([^,\]]+),benchmark=([^\]]+)\]'
                    match = re.search(meta_pattern, record.notes)
                    if match:
                        evaluated_model = match.group(1).replace("/", "_").replace(" ", "_")
                        benchmark_id = match.group(2)
                    else:
                        # 기존 방식으로도 시도 (구 버전 호환성)
                        if "model:" in record.notes.lower():
                            try:
                                model_part = record.notes.split("model:")[-1].split(",")[0].strip()
                                if model_part:
                                    evaluated_model = model_part.replace("/", "_").replace(" ", "_")
                            except:
                                pass
                except Exception as e:
                    pass
            
            # 2. judge_meta에서 모델명 추출 시도 (평가자 모델 정보, 백업용)
            if record.judge_meta and record.judge_meta.model_name and evaluated_model == "unknown_model":
                try:
                    judge_model = record.judge_meta.model_name.replace("/", "_").replace(" ", "_")
                    evaluated_model = judge_model
                except:
                    pass
            
            # 파일명 구성: YYYYMMDD_benchmark{ID}_{evaluated_model}.jsonl
            filename = f"{day}_benchmark{benchmark_id}_{evaluated_model}.jsonl"
            
            return self.base / rid / ctype / by / rdir / filename

        # ------------------------------------------------------------------ #
        # Read (find)
        # ------------------------------------------------------------------ #
        def find(self, query: EvaluationQuery) -> Iterable[EvaluationRecord]:
            count = 0
            for file in self._iter_files(query):
                # 파일 레벨 날짜 필터(YYYYMMDD.jsonl)
                for rec in self._iter_jsonl(file, query):
                    yield rec
                    count += 1
                    if query.limit and count >= query.limit:
                        return

        def _iter_files(self, q: EvaluationQuery) -> Iterator[Path]:
            base = self.base
            if not base.exists():
                return

            # 1) rubric 층
            rubric_dirs: List[Path]
            if q.rubric_ids:
                rubric_names = [r.value if hasattr(r, "value") else str(r) for r in q.rubric_ids]
                rubric_dirs = [base / r for r in rubric_names]
            else:
                rubric_dirs = [p for p in base.iterdir() if p.is_dir()]

            for rdir in rubric_dirs:
                if not rdir.exists():
                    continue

                # 2) content_type 층
                kind_dirs: List[Path]
                if q.content_types:
                    kind_dirs = [rdir / ct.value for ct in q.content_types]
                else:
                    kind_dirs = [p for p in rdir.iterdir() if p.is_dir()]

                for kdir in kind_dirs:
                    if not kdir.exists():
                        continue

                    # 3) evaluated_by 층
                    by_dirs: List[Path]
                    if q.evaluated_by:
                        by_dirs = [kdir / q.evaluated_by.value]
                    else:
                        by_dirs = [p for p in kdir.iterdir() if p.is_dir()]

                    for bdir in by_dirs:
                        if not bdir.exists():
                            continue

                        # 4) run_id 층
                        run_dirs: List[Path]
                        if q.run_ids:
                            run_dirs = [bdir / rid for rid in q.run_ids]
                        else:
                            run_dirs = [p for p in bdir.iterdir() if p.is_dir()]

                        for rid in run_dirs:
                            if not rid.exists():
                                continue

                            # 5) 파일(YYYYMMDD.jsonl)
                            for file in sorted(rid.glob("*.jsonl")):
                                if self._file_out_of_range(file.name, q.date_from, q.date_to):
                                    continue
                                yield file

        def _iter_jsonl(self, file: Path, q: EvaluationQuery) -> Iterator[EvaluationRecord]:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)

                    # 라인 레벨 정밀 필터
                    if q.rubric_ids:
                        rid = row.get("rubric_id")
                        valid = {r.value if hasattr(r, "value") else str(r) for r in q.rubric_ids}
                        if rid not in valid:
                            continue

                    if q.content_types:
                        ct = row.get("target", {}).get("content_type")
                        valid_ct = {c.value for c in q.content_types}
                        if ct not in valid_ct:
                            continue

                    if q.evaluated_by:
                        if row.get("evaluated_by") != q.evaluated_by.value:
                            continue

                    if q.run_ids:
                        if row.get("run_id") not in set(q.run_ids):
                            continue

                    if q.model_names:
                        jmeta = row.get("judge_meta") or {}
                        if jmeta.get("model_name") not in set(q.model_names):
                            continue

                    # created_at( judge_meta ) 시간 필터
                    if (q.date_from or q.date_to) and row.get("judge_meta", {}).get("created_at"):
                        try:
                            ts = self._parse_dt(row["judge_meta"]["created_at"])
                            if q.date_from and ts < q.date_from:  # type: ignore
                                continue
                            if q.date_to and ts > q.date_to:      # type: ignore
                                continue
                        except Exception:
                            pass

                    yield EvaluationRecord.model_validate(row)

        # ------------------------------------------------------------------ #
        # Utils
        # ------------------------------------------------------------------ #
        @staticmethod
        def _format_reference_from_source_item(source_item) -> str:
            """
            source_item(Union[PassageMultiSource, PassageSingleSource, AudioScriptSource, ImageCaptionSource] | dict)
            를 사람이 읽을 수 있는 참고 자료 문자열로 변환
            """
            try:
                # dict로 들어오는 경우와 pydantic 모델 둘 다 대응
                if isinstance(source_item, dict):
                    kt = source_item.get("korean_topic") or source_item.get("topic")
                    kc = source_item.get("korean_context") or source_item.get("context")
                    ft = source_item.get("foreign_topic")
                    fc = source_item.get("foreign_context")
                    kind = source_item.get("source_kind")
                else:
                    kind = getattr(source_item, "source_kind", None)
                    if kind == "passage_multi":
                        kt = source_item.korean_topic
                        kc = source_item.korean_context
                        ft = getattr(source_item, "foreign_topic", None)
                        fc = getattr(source_item, "foreign_context", None)
                    elif kind in ("passage_single", "audio_script"):
                        kt = source_item.topic
                        kc = source_item.context
                        ft = fc = None
                    elif kind == "image_caption":
                        kt = source_item.topic
                        kc = None
                        ft = fc = None
                    else:
                        kt = kc = ft = fc = None

                parts = []
                # 비교형(국내/해외) → 두 블록
                if kt or kc:
                    parts.append(f"- Home Topic ({kt or 'N/A'}): {kc or ''}".strip())
                if ft or fc:
                    parts.append(f"- Foreign Topic ({ft or 'N/A'}): {fc or ''}".strip())

                # 단일형/오디오 → topic/context만
                if not parts and (kt or kc):
                    parts.append(f"- Topic ({kt or 'N/A'}): {kc or ''}".strip())

                return "\n".join(p for p in parts if p) or ""
            except Exception:
                return ""
        
        @staticmethod
        def _file_out_of_range(filename: str, date_from: Optional[datetime], date_to: Optional[datetime]) -> bool:
            """
            파일명이 YYYYMMDD.jsonl일 때 날짜 범위와 맞지 않으면 True.
            """
            if not (date_from or date_to):
                return False
            stem = filename.split(".")[0]
            try:
                fdt = datetime.strptime(stem, "%Y%m%d")
            except Exception:
                return False
            if date_from and fdt < date_from:
                return True
            if date_to and fdt > date_to:
                return True
            return False

        @staticmethod
        def _parse_dt(val: str) -> datetime:
            """ISO8601(‘Z’ 포함) 또는 'YYYY-MM-DD' 보조 파서."""
            try:
                if "T" in val:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                return datetime.strptime(val, "%Y-%m-%d")
            except Exception:
                return datetime.utcnow()

        @classmethod
        def evaluate_with_client(cls, inp: EvaluateInput, client) -> EvaluateOutput:
            """
            기존에 생성된 클라이언트를 사용하여 평가를 수행하는 메서드
            (CUDA 재초기화 문제 해결용)
            """
           
            records = []
            success_count = 0
            failed_count = 0
            
            print(f"📝 클라이언트 재사용하여 '{inp.rubric_id.value}' 평가")
            
            # 모델명 추출 (파일명에 사용)
            model_id = getattr(client, 'model_name', None) or getattr(client, 'model_id', 'unknown_model')
            model_clean = model_id.replace("/", "_").replace(" ", "_")
            
            if inp.rubric_id in CONTENT_PLUS_STEM_RUBRIC:
                # Content + Stem 평가용 로직 (R1, R6)
                print(f"📝 '{inp.rubric_id.value}' - Content + Stem 평가 모드")
              
                for i, candidate in enumerate(inp.candidates):
                    try:
                        print(f"  📄 평가 중... ({i+1}/{len(inp.candidates)}) - Content + Stem")
                        evaluation_result = cls._evaluate_content_plus_stems(
                            client=client,
                            candidate=candidate,
                            rubric_id=inp.rubric_id
                        )
                        # EvaluationTarget 생성
                        eval_target = EvaluationTarget(
                            content_type=candidate.content_type,
                            content=candidate.content,
                            stems=candidate.stems or []
                        )
                        
                        # 메타데이터를 notes에 구조화하여 저장 (파일명 생성용)
                        benchmark_id = getattr(candidate, 'benchmark_id', 'unknown_benchmark')
                        notes_content = evaluation_result.get("feedback", "")
                        metadata_info = f"[META:model={model_clean},benchmark={benchmark_id}]"
                        full_notes = f"{metadata_info} {notes_content}".strip()
                        
                        # 점수 타입 결정
                        if inp.rubric_id in (RubricID.l2_learner_suitability, RubricID.R6_L2_APPROPRIATENESS):
                            # Likert (1~5)
                            likert_value = cls._first_likert_from_text(full_notes)
                            score_obj = LikertScore(value=likert_value)
                        else:
                            # Binary (PASS/FAIL)
                            is_pass = cls._binary_from_pass(full_notes)
                            score_obj = BinaryScore(value=is_pass)
                        # JudgeMeta 생성
                        judge_meta = JudgeMeta(
                            model_name=inp.model_name,
                            temperature=inp.temperature
                        )
                        record = EvaluationRecord(
                            target=eval_target,
                            rubric_id=inp.rubric_id,
                            score=score_obj,
                            evaluated_by=inp.evaluator_type,
                            judge_meta=judge_meta,
                            notes=full_notes,
                            run_id=inp.run_id or "default"
                        )
                        
                        records.append(record)
                        success_count += 1
                        
                    except Exception as e:
                        print(f"    ❌ 평가 실패: {e}")
                        
                        # 실패한 경우도 기록
                        eval_target = EvaluationTarget(
                            content_type=candidate.content_type,
                            content=candidate.content,
                            stems=candidate.stems or []
                        )
                        
                        # 실패 시에도 루브릭에 맞는 점수 타입 사용
                        if inp.rubric_id in [RubricID.l2_learner_suitability, RubricID.R6_L2_APPROPRIATENESS]:
                            error_score = LikertScore(value=1)  # 최저 점수
                        else:
                            error_score = BinaryScore(value=False)
                            
                        judge_meta = JudgeMeta(
                            model_name=inp.model_name,
                            temperature=inp.temperature
                        )
                        
                        # 실패 기록에도 메타데이터 추가
                        benchmark_id = getattr(candidate, 'benchmark_id', 'unknown_benchmark')
                        metadata_info = f"[META:model={model_clean},benchmark={benchmark_id}]"
                        error_notes = f"{metadata_info} 평가 실패: {str(e)}"
                        
                        error_record = EvaluationRecord(
                            target=eval_target,
                            rubric_id=inp.rubric_id,
                            score=error_score,
                            evaluated_by=inp.evaluator_type,
                            judge_meta=judge_meta,
                            notes=error_notes,
                            run_id=inp.run_id or "default"
                        )
                        
                        records.append(error_record)
                        failed_count += 1
            
            else:
                # Content Only 평가용 로직 (R2-R5)
                print(f"📝 '{inp.rubric_id.value}' - Content Only 평가 모드")
                for i, candidate in enumerate(inp.candidates):
                    try:
                        print(f"  📄 평가 중... ({i+1}/{len(inp.candidates)}) - Content Only")
                        
                        # Content Only 평가 수행 (stem 정보 제외)
                        evaluation_result = cls._evaluate_content_only(
                            candidate=candidate,
                            rubric_id=inp.rubric_id,
                            client=client
                        )
                        
                        # EvaluationTarget 생성 (Content만 포함)
                        eval_target = EvaluationTarget(
                            content_type=candidate.content_type,
                            content=candidate.content,
                            stems=[]  # Content Only 평가이므로 stems 제외
                        )
                        notes_text = (
                            evaluation_result.get("notes")
                            or evaluation_result.get("feedback")
                            or ""
                        )
                        is_pass = cls._binary_from_pass(notes_text)
                        score_obj = BinaryScore(value=is_pass)
                        
                        # JudgeMeta 생성
                        judge_meta = JudgeMeta(
                            model_name=inp.model_name,
                            temperature=inp.temperature
                        )
                        
                        record = EvaluationRecord(
                            target=eval_target,
                            rubric_id=inp.rubric_id,
                            score=score_obj,
                            evaluated_by=inp.evaluator_type,
                            judge_meta=judge_meta,
                            notes=evaluation_result.get("feedback", ""),
                            run_id=inp.run_id or "default"
                        )
                        
                        records.append(record)
                        success_count += 1
                        
                    except Exception as e:
                        print(f"    ❌ 평가 실패: {e}")
                        failed_count += 1
            
            return EvaluateOutput(
                records=records,
                success_count=success_count,
                failed_count=failed_count,
                total_count=len(inp.candidates)
            )
        
        @staticmethod
        def _binary_from_pass(text: str) -> bool:
            """
            notes/feedback 문자열에서 'PASS' 존재 여부를 단어 경계 기준으로 판정.
            - 대소문자 무시
            - COMPASS 등의 오탐을 막기 위해 \bPASS\b 사용
            """
            if not text:
                return False
            return re.search(r'(?i)\bPASS\b', text) is not None

        @staticmethod
        def _first_likert_from_text(text: str) -> int:
            """
            대괄호로 감싼 메타데이터([...])를 모두 제거한 뒤,
            최초로 등장하는 1~5 정수를 찾아 점수로 사용. 없으면 3 반환.
            """
            if not text:
                return 3
            # 메타데이터 제거: [ ... ] 블록 전부 삭제
            cleaned = re.sub(r"\[[^\]]*\]", " ", text)
            # 처음 등장하는 1~5 정수 추출 (단어 경계)
            m = re.search(r"\b([1-5])\b", cleaned)
            if not m:
                return 3
            val = int(m.group(1))
            return max(1, min(5, val))

        @classmethod
        def _evaluate_content_plus_stems(cls, candidate, rubric_id, client) -> Dict[str, Any]:
            """
            Content + Stems 평가 전용 메서드 (R1, R6)
            - candidate.content: 본문/지문
            - candidate.stems: List[str]
            - candidate.meta.problem_types / eval_goals: List[str]
            """
            content = (candidate.content or "").strip()
            stems = candidate.stems or []
            meta = getattr(candidate, "meta", {}) or {}
            problem_types = meta.get("problem_types", []) or []
            eval_goals = meta.get("eval_goals", []) or []

            # 프롬프트 생성 (템플릿 사용 가능하면 사용, 실패 시 Fallback)
            prompt = cls._build_content_plus_stems_prompt(
                content=content,
                stems=stems,
                problem_types=problem_types,
                eval_goals=eval_goals,
                rubric_id=rubric_id,
                candidate=candidate,
            )

            messages = [
                {"role": "user", "content": prompt}
            ]

            # LLM 호출
            resp_text = client.call(messages)

            # 응답 파싱 (PASS/FAIL 또는 점수)
            score, feedback = cls._parse_evaluation_response(resp_text)

            return {
                "score": score,
                "feedback": feedback,
                "notes": resp_text.strip(),
                "rubric_details": {
                    "has_stems": bool(stems),
                    "stem_count": len(stems),
                    "evaluation_type": "content_plus_stems",
                    "rubric_id": getattr(rubric_id, "value", str(rubric_id)),
                },
            }

        @classmethod
        def _evaluate_content_only(cls, candidate, rubric_id, client) -> Dict[str, Any]:
            """
            Content Only 평가 로직: content만 사용하여 LLM에게 평가 요청
            R2-R5 루브릭에서 사용 (stems 정보 제외)
            """
            content = candidate.content or ""
            
            if not content.strip():
                return {
                    "score": 0.0,
                    "feedback": "평가할 텍스트가 없습니다.",
                    "rubric_details": {"text_length": 0, "has_content": False, "evaluation_type": "content_only"}
                }
            
            # LLM에게 평가 요청
            try:
                if len(content) > 2000:  # 긴 텍스트인 경우 로그 출력
                    print(f"      📄 긴 텍스트 평가 중... ({len(content)}자)")
                
                evaluation_prompt = cls._build_content_only_prompt(content, rubric_id, candidate)
                
                messages = [
                    {"role": "user", "content": evaluation_prompt}
                ]
                
                print(f"      🤖 LLM 호출 중... (프롬프트 길이: {len(evaluation_prompt)}자)")
                response = client.call(messages)
                
                print(f"      ✅ LLM 응답 받음 (길이: {len(response)}자)")
                
                # 응답 파싱
                score, feedback = cls._parse_evaluation_response(response)
                
                return {
                    "score": score,
                    "feedback": feedback,
                    "rubric_details": {
                        "text_length": len(content),
                        "has_stems": False,
                        "evaluation_type": "content_only"
                    }
                }
                
            except Exception as e:
                print(f"⚠️ Content Only 평가 중 오류: {e}")

        @classmethod
        def _build_content_plus_stems_prompt(cls, *, content: str, stems: list[str], problem_types: list[str],
                                            eval_goals: list[str], rubric_id, candidate=None) -> str:
            """
            Content + Stems 평가용 프롬프트 빌더
            - 가능하면 템플릿을 사용하고, 없으면 Fallback 프롬프트로 구성
            """
            # ── 공통 준비
            rubric_key = getattr(rubric_id, "value", str(rubric_id))
            bench_id = getattr(candidate, "benchmark_id", None)
            stems_list = stems or []
            pt = problem_types or []
            eg = eval_goals or []

            # 최소 3개 stem/지침 정보 확인 (R1은 PT/EG도 3개 필요)
            need_pt_eg = rubric_key in ("completeness_for_guidelines", "R1_GUIDELINE_COMPLETENESS")
            has_3_stems = len(stems_list) >= 3
            has_3_guides = (len(pt) >= 3 and len(eg) >= 3) if need_pt_eg else True

            template_ds = TemplatesFSDataSource(agent="iska")

            # 벤치마크 분기
            is_listening = bench_id in (3, 4)
            is_visual = bench_id == 5
            var_name = None
            # ── 키 선택
            if is_listening:
                # 듣고 말하기
                tpl_map = {
                    "completeness_for_guidelines": "rubric_evaluation_listening.completeness_for_guidelines",
                    "R1_GUIDELINE_COMPLETENESS":  "rubric_evaluation_listening.completeness_for_guidelines",
                    "l2_learner_suitability":     "rubric_evaluation_listening.colloquial_response_suitability",
                    "R6_L2_APPROPRIATENESS":      "rubric_evaluation_listening.colloquial_response_suitability",
                }
                var_name = "audio_script"
            elif is_visual:
                # 보고 말하기
                tpl_map = {
                    "completeness_for_guidelines": "rubric_evaluation_visual.completeness_for_guidelines",
                    "R1_GUIDELINE_COMPLETENESS":  "rubric_evaluation_visual.completeness_for_guidelines",
                    "l2_learner_suitability":     "rubric_evaluation_visual.visual_cues_response_suitability",
                    "R6_L2_APPROPRIATENESS":      "rubric_evaluation_visual.visual_cues_response_suitability",
                }
                var_name = "image_caption"
            else:
                # 기본(passages, Bench 1/2)
                tpl_map = {
                    "completeness_for_guidelines": "rubric_evaluation.completeness_for_guidelines",
                    "R1_GUIDELINE_COMPLETENESS":  "rubric_evaluation.completeness_for_guidelines",
                    "l2_learner_suitability":     "rubric_evaluation.l2_learner_suitability",
                    "R6_L2_APPROPRIATENESS":      "rubric_evaluation.l2_learner_suitability",
                }
                var_name = "passage"

            tpl_key = tpl_map.get(rubric_key)

                # ── 템플릿 경로가 잡혔고, 필요한 정보가 충분하면 템플릿 사용
            if tpl_key and has_3_stems and has_3_guides:
                kwargs = {
                    var_name: content,
                    "stem1": stems_list[0],
                    "stem2": stems_list[1],
                    "stem3": stems_list[2],
                }
                if need_pt_eg:
                    kwargs.update({
                        "problem_type1": pt[0], "problem_type2": pt[1], "problem_type3": pt[2],
                        "eval_goal1": eg[0],   "eval_goal2": eg[1],   "eval_goal3": eg[2],
                    })
                try:
                    prompt = template_ds.get(tpl_key, **kwargs)
                    print(f"      📝 템플릿 '{tpl_key}' 사용 (Content+Stems)")
                    return prompt
                except Exception as e:
                    print(f"      ⚠️ 템플릿 로드 실패 ({tpl_key}): {e}")

        @classmethod
        def _build_content_only_prompt(cls, content: str, rubric_id, candidate=None) -> str:
            """
            Content Only 평가를 위한 프롬프트 생성 - YAML 템플릿 사용 (stems 정보 제외)
            벤치마크별로 다른 템플릿 사용
            """
            try:
                # 템플릿 데이터소스 초기화
                template_ds = TemplatesFSDataSource(agent="iska")
                
                # 벤치마크별 템플릿 선택
                benchmark_id = getattr(candidate, 'benchmark_id', None)
                is_listening_benchmark = benchmark_id in [3, 4]  # 듣고 말하기
                is_visual_benchmark = benchmark_id in [5]        # 보고 말하기
                
                # 루브릭별 템플릿 키 매핑 (벤치마크에 따라 다름)
                if is_listening_benchmark:
                    # 듣고 말하기 템플릿 (벤치마크 3, 4)
                    template_key_map = {
                        "clarity_of_core_theme": "rubric_evaluation_listening.conversation_topic_consistency",
                        "reference_groundedness": "rubric_evaluation_listening.background_consistency",
                        "logical_flow": "rubric_evaluation_listening.dialogue_flow_and_structure",
                        "korean_quality": "rubric_evaluation_listening.korean_quality",
                        "R2_TOPIC_CLARITY": "rubric_evaluation_listening.conversation_topic_consistency",
                        "R3_SOURCE_GROUNDEDNESS": "rubric_evaluation_listening.background_consistency",
                        "R4_LOGICAL_STRUCTURE": "rubric_evaluation_listening.dialogue_flow_and_structure",
                        "R5_KOREAN_QUALITY": "rubric_evaluation_listening.korean_quality"
                    }
                elif is_visual_benchmark:
                    # 보고 말하기 템플릿 (벤치마크 5)
                    template_key_map = {
                        "clarity_of_core_theme": "rubric_evaluation_visual.visual_theme_salience",
                        "reference_groundedness": "rubric_evaluation_visual.image_groundedness",
                        "logical_flow": "rubric_evaluation_visual.visual_reproducibility",
                        "korean_quality": "rubric_evaluation_visual.problem_korean_quality",
                        "R2_TOPIC_CLARITY": "rubric_evaluation_visual.visual_theme_salience",
                        "R3_SOURCE_GROUNDEDNESS": "rubric_evaluation_visual.image_groundedness",
                        "R4_LOGICAL_STRUCTURE": "rubric_evaluation_visual.visual_reproducibility",
                        "R5_KOREAN_QUALITY": "rubric_evaluation_visual.problem_korean_quality"
                    }
                else:
                    # 기본 템플릿 (벤치마크 1, 2)
                    template_key_map = {
                        "clarity_of_core_theme": "rubric_evaluation.core_theme_clarity",
                        "reference_groundedness": "rubric_evaluation.reference_groundedness",
                        "logical_flow": "rubric_evaluation.logical_flow_and_structure",
                        "korean_quality": "rubric_evaluation.korean_quality",
                        "R2_TOPIC_CLARITY": "rubric_evaluation.core_theme_clarity",
                        "R3_SOURCE_GROUNDEDNESS": "rubric_evaluation.reference_groundedness",
                        "R4_LOGICAL_STRUCTURE": "rubric_evaluation.logical_flow_and_structure",
                        "R5_KOREAN_QUALITY": "rubric_evaluation.korean_quality"
                    }
                
                # 루브릭 ID에 해당하는 템플릿 키 찾기
                rubric_key = rubric_id.value if hasattr(rubric_id, 'value') else str(rubric_id)
                template_key = template_key_map.get(rubric_key)
                
                if template_key:
                    try:
                        if is_listening_benchmark:
                            # 듣고 말하기 템플릿 사용 (audio_script 변수)
                            if template_key == "rubric_evaluation_listening.background_consistency":
                                # background_consistency는 background_info도 필요
                                background_info = ""
                                if candidate is not None:
                                    si = getattr(candidate, "source_item", None)
                                    if si:
                                        background_info = cls._format_reference_from_source_item(si)
                                
                                prompt = template_ds.get(template_key, 
                                    audio_script=content,
                                    background_info=background_info or "배경 정보가 제공되지 않았습니다."
                                )

                            else:
                                # 다른 듣고 말하기 루브릭들은 audio_script만 필요
                                prompt = template_ds.get(template_key, audio_script=content)
                            
                            print(f"      📝 템플릿 '{template_key}' 사용 (듣고 말하기 - Content Only)")
                        elif is_visual_benchmark:
                            # 보고 말하기 템플릿 사용 (source_passage 변수)
                            if template_key == "rubric_evaluation_visual.image_groundedness":
                                if candidate is not None:
                                    si = getattr(candidate, "source_item", None)
                                    if si:
                                        response_text = cls._format_reference_from_source_item(si)

                                prompt = template_ds.get(template_key, 
                                    response_text=response_text or "응답 텍스트가 제공되지 않았습니다.",
                                    image_caption=content  # Content Only이므로 같은 값 사용
                                )
                            else:
                                # 다른 보고 말하기 루브릭들은 image_caption 필요
                                prompt = template_ds.get(template_key, image_caption=content)
                            
                            print(f"      📝 템플릿 '{template_key}' 사용 (보고 말하기 - Content Only)")
                        else:
                            # 기본 템플릿 사용 (passage 변수)
                            if template_key == "rubric_evaluation.reference_groundedness":
                                # reference_groundedness는 reference도 필요
                                reference = ""
                                if candidate is not None:
                                    si = getattr(candidate, "source_item", None)
                                    if si:
                                        reference = cls._format_reference_from_source_item(si)
                                                                  
                                prompt = template_ds.get(template_key, 
                                    passage=content,
                                    reference=reference or "참고 자료가 제공되지 않았습니다."
                                )
                            else:
                                # 다른 루브릭들은 passage만 필요
                                prompt = template_ds.get(template_key, passage=content)
                            
                            print(f"      📝 템플릿 '{template_key}' 사용 (기본 - Content Only)")
                        
                        return prompt
                        
                    except Exception as template_error:
                        print(f"      ⚠️ 템플릿 로드 실패 ({template_key}): {template_error}")
                        # 템플릿 로드 실패 시 기본 프롬프트로 fallback
                
            except Exception as e:
                print(f"      ⚠️ 템플릿 시스템 오류: {e}")
                   
        @classmethod
        def _parse_evaluation_response(cls, response: str) -> tuple[float, str]:
            """
            LLM 응답을 파싱하여 점수와 피드백 추출
            - Binary: 응답이 PASS/FAIL로 시작하면 PASS=1.0, FAIL=0.0
            - Likert: 대괄호 메타데이터 제거 후, 처음 등장하는 1~5 정수를 점수로 사용
            - 피드백: '피드백:'/'Feedback:' 라인이 있으면 그 뒤, 없으면 전체 응답
            """
            import re

            try:
                raw = (response or "").strip()
                lines = raw.split("\n")
                feedback = raw if raw else "응답이 비어 있습니다."

                # 1) 피드백 라인 우선 추출
                for i, line in enumerate(lines):
                    s = line.strip()
                    if s.startswith("피드백:") or s.startswith("Feedback:"):
                        body = s.split(":", 1)[-1].strip()
                        # 나머지 줄도 이어붙임
                        tail = [l.strip() for l in lines[i + 1:] if l.strip()]
                        feedback = "\n".join([body] + tail) if body or tail else raw
                        break

                # 2) Binary: [답변] PASS/FAIL 로 시작하는지 체크 (대소문자 무시, 단어 경계)
                #   - 예: "PASS …", "FAIL …", "[답변] PASS …"
                m = re.match(r'^\s*(?:\[?\s*답변\s*\]?\s*)?\s*(PASS|FAIL)\b', raw, flags=re.IGNORECASE)
                if m:
                    tag = m.group(1).upper()
                    score = 1.0 if tag == "PASS" else 0.0
                    return score, feedback

                # 3) Likert: 대괄호 메타데이터 제거 후 처음 등장하는 1~5 정수
                cleaned = re.sub(r"\[[^\]]*\]", " ", raw)  # [ ... ] 블록 제거
                m2 = re.search(r"\b([1-5])\b", cleaned)
                if m2:
                    score = float(int(m2.group(1)))
                    return score, feedback

                # 4) Fallback: "점수:" / "Score:" 라인에 있는 숫자(1~5 범위로 클램프)
                for line in lines:
                    s = line.strip()
                    if s.startswith(("점수:", "Score:", "점수 ")):
                        nums = re.findall(r"\d+(?:\.\d+)?", s)
                        if nums:
                            val = float(nums[0])
                            val = max(1.0, min(5.0, val))
                            return val, feedback

                # 5) 최종 기본값
                return 3.0, feedback

            except Exception as e:
                print(f"      ⚠️ 응답 파싱 오류: {e}")
                safe = response.strip() if isinstance(response, str) and response.strip() else "평가 응답 파싱 오류"
                return 3.0, safe