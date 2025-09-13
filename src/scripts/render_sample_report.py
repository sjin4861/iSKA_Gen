#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""생성(JSONL) + 평가(JSONL) 합쳐 HTML 리포트 렌더링 (Passage → Stems → Evaluation)

사용:
  uv run python -m src.scripts.render_sample_report \
    --sample outputs/samples/2025-09-13/sample_1_0.jsonl \
    --eval   outputs/samples/2025-09-13/sample_1_0_eval.jsonl

옵션:
  --eval 미지정 시, sample 파일 옆의 *_eval.jsonl 자동 탐색
  --output 미지정 시, sample 파일 옆에 *_report.html 저장
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from html import escape
from datetime import datetime
from typing import Any, Dict, List, Optional


# -------------------------
# IO helpers
# -------------------------
def _read_first_jsonl(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"빈 JSONL: {path}")
        return json.loads(line)

def _read_all_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _default_eval_path(sample_path: Path) -> Path:
    # sample_x_y.jsonl -> sample_x_y_eval.jsonl (sample과 같은 폴더)
    return sample_path.with_name(sample_path.stem + "_eval.jsonl")

def _default_output_path(sample_path: Path) -> Path:
    return sample_path.with_name(sample_path.stem + "_report.html")


# -------------------------
# Render helpers
# -------------------------
def _esc(x: Any) -> str:
    return escape("" if x is None else str(x), quote=False)

def _score_badge(score: Optional[int], rubric_id: str) -> str:
    """0/1은 PASS/FAIL, 1~5 Likert는 숫자 배지, None은 N/A."""
    if score is None:
        return "<span class='badge badge-na'>N/A</span>"
    # Binary rubrics
    binary = {
        "completeness_for_guidelines",
        "core_theme_clarity",
        "reference_groundedness",
        "logical_flow_and_structure",
        "korean_quality",
    }
    if rubric_id in binary:
        return (
            "<span class='badge badge-pass'>PASS</span>"
            if score == 1 else
            "<span class='badge badge-fail'>FAIL</span>"
        )
    # Likert (예: L2 적합성)
    return f"<span class='badge badge-num num-{score}'>{score}</span>"

def _render_evaluation_table(eval_records: List[Dict[str, Any]]) -> str:
    """모든 루브릭을 한 표로 렌더링 (R1 + R2~R6)."""
    order = [
        "completeness_for_guidelines",
        "core_theme_clarity",
        "reference_groundedness",
        "logical_flow_and_structure",
        "korean_quality",
        "l2_learner_suitability",
    ]
    rank = {rid: i for i, rid in enumerate(order)}

    rows = []
    for rec in eval_records:
        rid = rec.get("rubric_id")
        score = rec.get("score")
        just = (rec.get("justification") or "").strip()
        stem_idx = rec.get("stem_index", -1)

        if rid == "completeness_for_guidelines":
            used = rec.get("used_stem_indices")
            scope = f"All-stems (indices: {used})" if isinstance(used, list) else "All-stems"
        else:
            if stem_idx != -1:  # 혹시 stem 단위가 들어오면 무시
                continue
            scope = "Sample"

        rows.append({
            "rubric": rid,
            "scope": scope,
            "score": score,
            "just": just,
            "rank": rank.get(rid, 999),
        })

    rows.sort(key=lambda r: (r["rank"], r["rubric"]))

    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td class='rubric'>{_esc(r['rubric'])}</td>"
            f"<td class='scope'>{_esc(r['scope'])}</td>"
            f"<td class='score'>{_score_badge(r['score'], r['rubric'])}</td>"
            f"<td class='just'>{_esc(r['just'])}</td>"
            "</tr>"
        )

    return f"""
<section class="panel">
  <h2>Evaluation</h2>
  <table class="rubric-table">
    <thead>
      <tr>
        <th>Rubric</th>
        <th>Scope</th>
        <th>Score</th>
        <th>Justification</th>
      </tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</section>
""".strip()


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="생성+평가 HTML 리포트 생성 (Passage→Stems→Evaluation)")
    p.add_argument("--sample", required=True, help="생성 결과 JSONL (한 줄)")
    p.add_argument("--eval", default=None, help="평가 결과 JSONL (여러 줄). 미지정 시 sample 옆 *_eval.jsonl 자동 탐색")
    p.add_argument("--output", default=None, help="HTML 저장 경로 (기본: sample 옆 *_report.html)")
    return p.parse_args()


def main():
    args = parse_args()
    sample_path = Path(args.sample)

    # eval 경로 해석
    if args.eval:
        eval_path = Path(args.eval)
        if not eval_path.exists():
            fallback = _default_eval_path(sample_path)
            if fallback.exists():
                print(f"[info] 주어진 평가 파일이 없어 기본 경로로 대체: {fallback}")
                eval_path = fallback
            else:
                raise FileNotFoundError(f"평가 파일을 찾을 수 없음: {args.eval} (fallback도 없음: {fallback})")
    else:
        eval_path = _default_eval_path(sample_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"--eval 미지정 & 기본 경로에도 없음: {eval_path}")

    sample = _read_first_jsonl(sample_path)
    eval_records = _read_all_jsonl(eval_path)

    # 메타
    bench_id = sample.get("bench_id")
    spec_file = sample.get("spec_file") or sample.get("spec_path")
    template_key = sample.get("template_key")
    stem_template = sample.get("stem_template")
    model = sample.get("model")
    passage = sample.get("passage", "")
    stems: List[Dict[str, Any]] = sample.get("stems") or []

    # 섹션 렌더
    eval_section_html = _render_evaluation_table(eval_records)

    # stems
    stem_meta_frags, stem_text_frags = [], []
    for s in stems:
        idx = s.get("index", 0)
        stem_meta_frags.append(
            f"<div class='stem-meta-item'><h3>Stem {idx+1}</h3>"
            f"<p>문항 유형: {_esc(s.get('problem_type'))}<br>"
            f"평가 목표: {_esc(s.get('eval_goal'))}<br>"
            f"length: {_esc(s.get('length'))}</p></div>"
        )
        stem_text_frags.append(
            f"<div class='stem-text-item'><h3>Stem {idx+1}</h3><pre>{_esc(s.get('stem'))}</pre></div>"
        )
    stems_meta_section = "\n".join(stem_meta_frags) if stem_meta_frags else "<p><i>Stem 메타 없음</i></p>"
    stems_text_section = "\n".join(stem_text_frags) if stem_text_frags else "<p><i>생성된 stem 없음</i></p>"

    # HTML (Passage → Stems → Evaluation)
    html = f"""<html><head><meta charset='utf-8'><title>Sample {_esc(bench_id)}</title>
<style>
/* Layout */
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     line-height:1.55;max-width:1100px;margin:36px auto;padding:0 24px;background:#fafafa;color:#1f2937;}}
h1{{margin:0 0 16px;font-size:28px;font-weight:700;}}
h2{{margin-top:28px;border-bottom:1px solid #e5e7eb;padding-bottom:6px;font-size:20px;}}
h3{{margin:12px 0 4px;font-size:16px;}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:14px 16px;border-radius:8px;border:1px solid #e2e8f0;font-size:14px;}}
.panel{{border:1px solid #e5e7eb;background:#ffffff;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 1px 2px rgba(0,0,0,0.04);}}

/* Buttons */
button.toggle{{cursor:pointer;background:#2563eb;color:#fff;border:none;padding:6px 14px;margin:6px 0;border-radius:8px;font-size:13px;}}
button.toggle.secondary{{background:#475569;}}

/* Table */
.rubric-table{{width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;}}
.rubric-table thead th{{text-align:left;background:#f8fafc;border-bottom:1px solid #e2e8f0;padding:10px 8px;font-weight:600;color:#111827;}}
.rubric-table td{{padding:10px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top;}}
.rubric-table td.rubric{{white-space:nowrap;font-weight:600;color:#111827;}}
.rubric-table td.scope{{white-space:nowrap;color:#374151;}}
.rubric-table td.just{{color:#111827;}}
.rubric-table tr:hover td{{background:#fafbfc;}}

/* Badges */
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;line-height:1;}}
.badge-pass{{background:#16a34a;color:#fff;}}
.badge-fail{{background:#ef4444;color:#fff;}}
.badge-na{{background:#e5e7eb;color:#111;}}
.badge-num{{color:#fff;}}
.num-5{{background:#16a34a;}}
.num-4{{background:#22c55e;}}
.num-3{{background:#f59e0b;}}
.num-2{{background:#f97316;}}
.num-1{{background:#ef4444;}}

/* Stems */
.stem-box{{border:1px solid #e5e7eb;background:#fff;border-radius:12px;padding:16px;margin:18px 0;}}
.stem-text-item pre{{margin-top:6px;}}
.hidden{{display:none;}}
footer{{margin-top:36px;font-size:12px;color:#6b7280;}}
</style>
<script>
function toggle(id){{const el=document.getElementById(id); if(!el) return; el.classList.toggle('hidden');}}
</script>
</head><body>
<h1>Bench {_esc(bench_id)} Sample</h1>

<div><button class='toggle' onclick="toggle('meta')">메타데이터 보기/숨기기</button></div>
<div id='meta' class='panel hidden'>
  <ul style="margin:0;padding-left:18px;">
    <li><b>spec_file</b>: {_esc(spec_file)}</li>
    <li><b>template</b>: {_esc(template_key)}</li>
    <li><b>stem_template</b>: {_esc(stem_template)}</li>
    <li><b>model</b>: {_esc(model)}</li>
    <li><b>passage_length</b>: {_esc(sample.get('passage_length'))}</li>
  </ul>
</div>

<h2>Passage</h2>
<div class='panel'>
  <pre>{_esc(passage)}</pre>
</div>

<h2>Stems ({len(stems)})</h2>
<div class='stem-box'>
  <div style='margin-bottom:8px;'>
    <button class='toggle secondary' onclick="toggle('stem_meta')">Stem 메타 보기/숨기기</button>
  </div>
  <div id='stem_meta' class='hidden'>
    {stems_meta_section}
    <hr style='margin:20px 0;'>
  </div>
  <div id='stem_texts'>
    {stems_text_section}
  </div>
</div>

{eval_section_html}

<footer>Rendered: {datetime.now().isoformat(timespec='seconds')}</footer>
</body></html>"""

    out_path = Path(args.output) if args.output else _default_output_path(sample_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"💾 리포트 저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
