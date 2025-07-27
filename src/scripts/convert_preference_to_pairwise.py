#!/usr/bin/env python3
"""
PreferenceRanking 벤치마크 데이터를 pairwise 형태로 변환하는 스크립트
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / 'modules'))
sys.path.append(str(Path.cwd().parent / 'utils'))

def convert_preference_to_pairwise(
    input_file: str,
    output_dir: str = "pairwise_data/test"
) -> None:
    """
    PreferenceRanking 데이터를 pairwise 형태로 변환
    
    Args:
        input_file (str): 입력 PreferenceRanking JSON 파일 경로
        output_dir (str): 출력 디렉토리
    """
    # 경로 설정
    data_dir = Path.cwd().parent / "data"
    input_path = data_dir / "benchmarks" / input_file
    output_path = data_dir / output_dir
    
    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Preference -> Pairwise 변환 시작...")
    print(f"   입력: {input_path}")
    print(f"   출력: {output_path}")
    
    # 원본 데이터 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        preference_data = json.load(f)
    
    print(f"📊 원본 데이터: {len(preference_data)}개 항목")
    
    # 고품질과 저품질 텍스트 분리
    high_quality = [item for item in preference_data if item.get("quality") == "high"]
    low_quality = [item for item in preference_data if item.get("quality") == "low"]
    
    print(f"   고품질 텍스트: {len(high_quality)}개")
    print(f"   저품질 텍스트: {len(low_quality)}개")
    
    # 주제별로 매칭
    pairs = []
    matched_topics = set()
    
    for high_item in high_quality:
        # 주제 추출 - "고품질_숫자_" 부분 제거
        high_name = high_item["name"]
        if high_name.startswith("고품질_"):
            # "고품질_01_회식_문화" -> "회식_문화"
            topic = "_".join(high_name.split("_")[2:])
            
            # 같은 주제의 저품질 텍스트 찾기
            matching_low = None
            for low_item in low_quality:
                low_name = low_item["name"]
                if low_name.startswith("저품질_"):
                    low_topic = "_".join(low_name.split("_")[2:])
                    if low_topic == topic:
                        matching_low = low_item
                        break
            
            if matching_low:
                pair = {
                    "pair_id": f"preference_pair_{topic}",
                    "topic": topic,
                    "chosen": high_item["text"],
                    "rejected": matching_low["text"],
                    "chosen_quality": "high",
                    "rejected_quality": "low",
                    "metadata": {
                        "chosen_name": high_item["name"],
                        "rejected_name": matching_low["name"]
                    }
                }
                pairs.append(pair)
                matched_topics.add(topic)
                print(f"✅ 매칭: {high_name} <-> {matching_low['name']}")
            else:
                print(f"⚠️ 주제 '{topic}'에 대한 저품질 텍스트를 찾을 수 없습니다.")
    
    print(f"\n📊 변환 결과: {len(pairs)}개의 pairwise 비교 쌍 생성")
    
    # 각 쌍을 개별 파일로 저장
    for i, pair in enumerate(pairs, 1):
        output_file = output_path / f"pair_{i:02d}_{pair['topic']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pair, f, ensure_ascii=False, indent=2)
        print(f"💾 저장: {output_file.name}")
    
    # 전체 pairwise 데이터도 저장
    all_pairs_file = output_path / "all_pairwise_data.json"
    with open(all_pairs_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"💾 전체 데이터 저장: {all_pairs_file.name}")
    
    # 요약 정보 저장
    summary = {
        "total_pairs": len(pairs),
        "matched_topics": list(matched_topics),
        "conversion_date": str(Path.cwd()),
        "input_file": str(input_path),
        "output_dir": str(output_path)
    }
    
    summary_file = output_path / "conversion_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📋 요약 정보 저장: {summary_file.name}")
    
    print(f"\n🎉 변환 완료! 총 {len(pairs)}개 쌍이 {output_path}에 저장되었습니다.")


if __name__ == "__main__":
    # PreferenceRanking 데이터를 pairwise로 변환
    convert_preference_to_pairwise(
        input_file="v1/iSKA-Gen_Benchmark_v1.0.0_20250726_PreferenceRanking.json",
        output_dir="pairwise_data/test"
    )
