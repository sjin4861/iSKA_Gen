#!/usr/bin/env python3
"""단일 Passage + Stem 세트 생성 및 시각화 (HTML) 저장 스크립트."""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import json
import sys
from html import escape

# 경로 상위 추가 (필요시 유지)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modules.content_chain import build_content_chain
from src.modules.stem_chain import build_stem_chain
from src.modules.model_client import get_vllm_chat  # vLLM 8000 고정

# -----------------------------
# 스펙 로딩 유틸 (Repository 제거)
# -----------------------------
def load_spec_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"스펙 파일 없음: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"스펙 JSON은 리스트여야 합니다. got={type(data)}")
    return data

def get_benchmark(data: list[dict], bench_id: int) -> dict:
    for b in data:
        try:
            if int(b.get("id")) == int(bench_id):
                return b
        except Exception:
            continue
    raise KeyError(f"benchmark id {bench_id} 없음 (available: {[d.get('id') for d in data]})")

def get_items(bench: dict) -> list[dict]:
    return bench.get("items", [])

def get_problem_types(bench: dict) -> list[str]:
    return bench.get("problem_types", [])

def get_eval_goals(bench: dict) -> list[str]:
    return bench.get("eval_goals", [])

# -----------------------------
# 템플릿/기본값
# -----------------------------
DEFAULT_TEMPLATES = {
    1: "passage_agent.create_passage_rubric_aware",
    2: "passage_agent.create_domestic_passage",
    3: "audio_agent.create_dialogue_passage",
    4: "audio_agent.create_dialogue_passage",
    5: "image_agent.create_image_caption_and_situation",
}
STEM_TEMPLATE = "stem_agent.few_shot_new"

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="단일 샘플 Passage+Stem 생성")
    p.add_argument('--model', required=True, help='생성용 모델명 (예: A.X-4.0-Light)')
    p.add_argument('--bench-id', type=int, required=True)
    p.add_argument('--spec-type', choices=['small','train','test'], default='small')  # 🔧 유지 + choices로 제한
    p.add_argument('--output-dir', default='outputs/samples')
    p.add_argument('--gen-base-url', default='http://localhost:8000/v1',
                   help='생성용 vLLM base URL (기본: http://localhost:8000/v1)')
    p.add_argument('--passage-temperature', type=float, default=0.7)
    p.add_argument('--stem-temperature', type=float, default=0.3)
    p.add_argument('--max-tokens', type=int, default=None)
    # 🔧 추가: 스펙 위치/버전 접두사
    p.add_argument('--spec-dir', default='specifications',
                   help='스펙 파일들이 위치한 디렉터리 (기본: specifications)')
    p.add_argument('--spec-version', default='iSKA-Gen_Spec_v1.1.0',
                   help='스펙 파일 접두사 (기본: iSKA-Gen_Spec_v1.1.0)')
    return p.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # 🔧 spec-type -> 파일 경로 매핑
    spec_dir = Path(args.spec_dir)
    spec_basename = f"{args.spec_version}_{args.spec_type}.json"
    spec_path = (spec_dir / spec_basename).resolve()
    spec_data = load_spec_json(spec_path)  # ← 기존 load_spec_json 그대로 사용

    bench = get_benchmark(spec_data, args.bench_id)

    items = get_items(bench)
    if not items:
        print(f"[ERR] 벤치 {args.bench_id} 항목 없음")
        return 1
    item = items[0]

    problem_types = get_problem_types(bench)
    eval_goals = get_eval_goals(bench)

    template_key = DEFAULT_TEMPLATES.get(args.bench_id, 'passage_agent.create_passage')
    content_chain = build_content_chain(template_key)
    stem_chain = build_stem_chain(STEM_TEMPLATE)

    # === 생성용 LLM: 8000 포트에 고정 연결 ===
    llm_passage = get_vllm_chat(
        model_name=args.model,
        temperature=args.passage_temperature,
        max_tokens=args.max_tokens,
        base_url=args.gen_base_url,
    )
    llm_stem = get_vllm_chat(
        model_name=args.model,
        temperature=args.stem_temperature,
        max_tokens=args.max_tokens,
        base_url=args.gen_base_url,
    )

    domestic = ('domestic' in template_key) or ('dialogue' in template_key) or ('image_' in template_key)
    if domestic:
        topic = item.get('korean_topic') or item.get('topic')
        context = item.get('korean_context') or item.get('context')
        passage_input = {
            'llm': llm_passage,
            'topic': topic,
            'context': context,
            'problem_types': problem_types,
            'eval_goals': eval_goals,
        }
    else:
        passage_input = {
            'llm': llm_passage,
            'korean_topic': item.get('korean_topic'),
            'korean_context': item.get('korean_context'),
            'foreign_topic': item.get('foreign_topic'),
            'foreign_context': item.get('foreign_context'),
            'problem_types': problem_types,
            'eval_goals': eval_goals,
        }

    print(f"🚀 Passage 생성 (bench={args.bench_id}, template={template_key})")
    passage = content_chain.invoke(passage_input)
    print(f"  ✅ passage length={len(passage)}")

    # 최대 3개 문제유형/평가목표 조합으로 stem 생성
    max_pairs = min(3, len(problem_types), len(eval_goals))
    stems = []
    if max_pairs == 0:
        print("⚠️ problem_types / eval_goals 부족으로 stem 생성 불가")
    else:
        print("📝 Stem 생성 (총", max_pairs, "개)")
        for i in range(max_pairs):
            pt = problem_types[i]
            eg = eval_goals[i]
            try:
                stem_text = stem_chain.invoke({
                    'llm': llm_stem,
                    'passage': passage,
                    'problem_type': pt,
                    'eval_goal': eg,
                })
                stems.append({
                    'index': i,
                    'problem_type': pt,
                    'eval_goal': eg,
                    'stem': stem_text,
                    'length': len(stem_text),
                })
                print(f"  ✅ stem[{i}] {len(stem_text)} chars")
            except Exception as e:
                print(f"  ❌ stem[{i}] 오류: {e}")
        if max_pairs < 3:
            print(f"ℹ️ 이용 가능한 조합 {max_pairs}개만 생성")

    # 저장
    date_dir = ensure_dir(Path(args.output_dir) / datetime.now().strftime('%Y-%m-%d'))
    base_name = f"sample_{args.bench_id}_0"
    jsonl_path = date_dir / f"{base_name}.jsonl"
    html_path = date_dir / f"{base_name}.html"

    record = {
        'bench_id': args.bench_id,
        'spec_file': str(spec_path),
        'template_key': template_key,
        'stem_template': STEM_TEMPLATE,
        'model': args.model,
        'passage_length': len(passage),
        'passage': passage,
        'stems': stems,
    }

    # JSONL 저장
    with jsonl_path.open('w', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # HTML 시각화 (텍스트 이스케이프 적용)
    esc = lambda s: escape(s or "", quote=False)
    stems_meta_fragments = []
    stems_text_fragments = []
    for s in stems:
        stems_meta_fragments.append(
            f"<div class='stem-meta-item'><h3>Stem {s['index']+1}</h3>"
            f"<p>문항 유형: {esc(s['problem_type'])}<br>"
            f"평가 목표: {esc(s['eval_goal'])}<br>length: {s['length']}</p></div>"
        )
        stems_text_fragments.append(
            f"<div class='stem-text-item'><h3>Stem {s['index']+1}</h3><pre>{esc(s['stem'])}</pre></div>"
        )
    stems_meta_section = "\n".join(stems_meta_fragments) if stems_meta_fragments else "<p><i>Stem 메타 없음</i></p>"
    stems_text_section = "\n".join(stems_text_fragments) if stems_text_fragments else "<p><i>생성된 stem 없음</i></p>"

    html = f"""<html><head><meta charset='utf-8'><title>Sample {args.bench_id}</title>
<style>
body{{font-family:system-ui, sans-serif;line-height:1.5;max-width:1000px;margin:40px auto;padding:0 24px;background:#fafafa;color:#222;}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:14px 16px;border-radius:8px;border:1px solid #e2e8f0;font-size:14px;}}
button.toggle{{cursor:pointer;background:#2563eb;color:#fff;border:none;padding:6px 14px;margin:4px 0;border-radius:6px;font-size:13px;}}
button.toggle.secondary{{background:#475569;}}
.panel{{border:1px solid #d0d7de;background:#fff;border-radius:10px;padding:18px;margin:18px 0;box-shadow:0 1px 2px rgba(0,0,0,0.04);}}
.flex-row{{display:flex;gap:18px;flex-wrap:wrap;}}
.stem-box{{border:1px solid #cbd5e1;background:#fff;border-radius:10px;padding:16px;margin:24px 0;}}
.stem-text-item pre{{margin-top:4px;}}
.hidden{{display:none;}}
h1{{margin-top:0;font-size:26px;}}
h2{{margin-top:32px;border-bottom:1px solid #e2e8f0;padding-bottom:4px;}}
footer{{margin-top:40px;font-size:12px;color:#666;}}
</style>
<script>
function toggle(id){{const el=document.getElementById(id);if(!el)return;el.classList.toggle('hidden');}}
</script>
</head><body>
<h1>Bench {args.bench_id} Sample</h1>

<div><button class='toggle' onclick="toggle('meta')">메타데이터 보기/숨기기</button></div>
<div id='meta' class='panel hidden'>
    <ul>
        <li><b>spec_file</b>: {esc(str(spec_path))}</li>
        <li><b>template</b>: {esc(template_key)}</li>
        <li><b>model</b>: {esc(args.model)}</li>
        <li><b>passage_length</b>: {len(passage)}</li>
    </ul>
</div>

<h2>Passage</h2>
<div class='panel'>
<pre>{esc(passage)}</pre>
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

<footer>Generated: {datetime.now().isoformat(timespec='seconds')}</footer>
</body></html>"""

    with html_path.open('w', encoding='utf-8') as f:
        f.write(html)

    print(f"💾 저장됨: {jsonl_path}")
    print(f"💾 저장됨: {html_path}")
    print("🎉 완료")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
