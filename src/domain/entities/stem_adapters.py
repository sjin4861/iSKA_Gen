from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .stems import StemBundle, StemPrompt
from .content_types import ArtifactKind

# 레거시 키 이름 → (ArtifactKind, source_text_key)
_SOURCE_KEY_MAP: List[Tuple[str, ArtifactKind]] = [
    ("source_passage", ArtifactKind.passage),
    ("source_audio_script", ArtifactKind.audio_script),
    ("source_image_caption", ArtifactKind.image_caption),
]

def _detect_source(raw: Dict[str, Any]) -> Tuple[ArtifactKind, str]:
    for key, kind in _SOURCE_KEY_MAP:
        if key in raw and isinstance(raw[key], str):
            return kind, raw[key]
    raise ValueError("레거시 stem row에서 source_* 텍스트를 찾을 수 없습니다.")

def _collect_prompts(raw: Dict[str, Any]) -> List[StemPrompt]:
    """
    problem_type_1/eval_goal_1/stem_1, ... 순번 필드를 스캔해 리스트로 정규화.
    연속된 인덱스만 인정(중간 구멍 방지).
    """
    prompts: List[StemPrompt] = []
    idx = 1
    while True:
        pt_key = f"problem_type_{idx}"
        eg_key = f"eval_goal_{idx}"
        st_key = f"stem_{idx}"
        if not (pt_key in raw or eg_key in raw or st_key in raw):
            break
        # 최소 stem은 있어야 의미가 있으므로 stem을 중심으로 체크
        if st_key not in raw or not isinstance(raw[st_key], str) or not raw[st_key].strip():
            raise ValueError(f"stem_{idx} 값이 비어있습니다.")
        prompts.append(
            StemPrompt(
                problem_type=str(raw.get(pt_key, "")).strip(),
                eval_goal=str(raw.get(eg_key, "")).strip(),
                stem=str(raw[st_key]).strip(),
            )
        )
        idx += 1

    if not prompts:
        raise ValueError("레거시 stem row에서 프롬프트를 하나도 수집하지 못했습니다.")
    return prompts

def stem_bundle_from_legacy_row(raw: Dict[str, Any], *, benchmark_id: int | None = None,
                                benchmark_version: str | None = None) -> StemBundle:
    """
    레거시(flat) 한 행(dict)을 새 스키마 StemBundle로 변환.
    """
    kind, source_text = _detect_source(raw)
    prompts = _collect_prompts(raw)
    return StemBundle(
        source_kind=kind,
        source_text=source_text.strip(),
        prompts=prompts,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
    )

def stem_bundles_from_legacy_list(rows: List[Dict[str, Any]], *,
                                  benchmark_id: int | None = None,
                                  benchmark_version: str | None = None) -> List[StemBundle]:
    return [
        stem_bundle_from_legacy_row(r, benchmark_id=benchmark_id, benchmark_version=benchmark_version)
        for r in rows
    ]
