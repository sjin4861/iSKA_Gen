import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / 'modules'))
sys.path.append(str(Path.cwd().parent / 'utils'))

from modules.iska.reward_model import RewardModel, compare_passage_pairs_with_reward_model
from modules.model_client import OpenAIModelClient


def load_preference_ranking_benchmark(
    benchmark_file: str
) -> List[Dict[str, Any]]:
    """
    Preference Ranking 벤치마크 데이터를 로드합니다.
    
    Args:
        benchmark_file (str): 벤치마크 파일명 (예: "v1/iSKA-Gen_Benchmark_v1.0.0_20250726_PreferenceRanking.json")
        
    Returns:
        List[Dict[str, Any]]: 로드된 벤치마크 데이터
    """
    try:
        # 프로젝트 루트 기준으로 경로 설정
        data_dir = Path.cwd().parent / "data" / "benchmarks"
        file_path = data_dir / benchmark_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"벤치마크 파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 리스트가 아닌 경우 리스트로 변환
        if not isinstance(raw_data, list):
            raw_data = [raw_data]
        
        print(f"✅ Preference Ranking 벤치마크 로드 완료: {len(raw_data)}개 항목")
        return raw_data
        
    except Exception as e:
        print(f"❌ 벤치마크 로드 실패: {e}")
        return []


def convert_to_pairwise_format(
    preference_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Preference Ranking 데이터를 pairwise 비교 형태로 변환합니다.
    
    Args:
        preference_data (List[Dict[str, Any]]): 원본 preference 데이터
        
    Returns:
        List[Dict[str, Any]]: pairwise 형태로 변환된 데이터
    """
    # 고품질과 저품질 텍스트 분리
    high_quality = [item for item in preference_data if item.get("quality") == "high"]
    low_quality = [item for item in preference_data if item.get("quality") == "low"]
    
    print(f"📊 고품질 텍스트: {len(high_quality)}개, 저품질 텍스트: {len(low_quality)}개")
    
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
            else:
                print(f"⚠️ 주제 '{topic}'에 대한 저품질 텍스트를 찾을 수 없습니다.")
    
    print(f"✅ {len(pairs)}개의 pairwise 비교 쌍 생성 완료.")
    return pairs


def evaluate_gpt4_preference_alignment(
    pairwise_data: List[Dict[str, Any]],
    openai_model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    GPT-4o를 사용하여 텍스트 품질을 평가하고 preference alignment를 측정합니다.
    
    Args:
        pairwise_data (List[Dict[str, Any]]): pairwise 형태의 평가 데이터
        openai_model (str): 사용할 OpenAI 모델 (기본값: "gpt-4o")
        
    Returns:
        Dict[str, Any]: GPT-4o 평가 결과
    """
    if not pairwise_data:
        return {"error": "평가할 pairwise 데이터가 없습니다."}
    
    print(f"🤖 GPT-4o 품질 평가 시작 (모델: {openai_model})...")
    
    try:
        # OpenAI 클라이언트 초기화
        client = OpenAIModelClient(
            model_name=openai_model,
            api_key=None  # 환경변수에서 자동 로드
        )
        
        evaluation_prompt = """
다음 두 텍스트를 비교하여 어느 것이 더 높은 품질을 가지는지 평가해주세요.

텍스트 A:
{text_a}

텍스트 B:
{text_b}

평가 기준:
1. 문법의 정확성
2. 표현의 자연스러움
3. 내용의 일관성과 논리성
4. 어휘 선택의 적절성

다음 형식으로만 답변해주세요:
선택: A 또는 B
이유: [간단한 이유]
"""
        
        results = []
        correct_predictions = 0
        
        for i, pair in enumerate(pairwise_data, 1):
            print(f"   📝 {i}/{len(pairwise_data)} 평가 중...")
            
            # 50% 확률로 순서 바꾸기 (bias 방지)
            import random
            if random.random() < 0.5:
                text_a, text_b = pair["chosen"], pair["rejected"]
                correct_choice = "A"
            else:
                text_a, text_b = pair["rejected"], pair["chosen"]
                correct_choice = "B"
            
            prompt = evaluation_prompt.format(text_a=text_a, text_b=text_b)
            
            try:
                response = client.generate_content(prompt)
                
                # GPT-4o 응답에서 선택 추출
                gpt_choice = None
                if "선택: A" in response or "선택:A" in response:
                    gpt_choice = "A"
                elif "선택: B" in response or "선택:B" in response:
                    gpt_choice = "B"
                
                is_correct = (gpt_choice == correct_choice)
                if is_correct:
                    correct_predictions += 1
                
                results.append({
                    "pair_id": pair["pair_id"],
                    "topic": pair["topic"],
                    "gpt_choice": gpt_choice,
                    "correct_choice": correct_choice,
                    "is_correct": is_correct,
                    "gpt_response": response,
                    "text_order": {"A": text_a[:50] + "...", "B": text_b[:50] + "..."}
                })
                
            except Exception as e:
                print(f"   ⚠️ GPT-4o 평가 실패: {e}")
                results.append({
                    "pair_id": pair["pair_id"],
                    "topic": pair["topic"],
                    "error": str(e)
                })
        
        accuracy = correct_predictions / len(pairwise_data) if pairwise_data else 0
        
        print(f"✅ GPT-4o 평가 완료: {correct_predictions}/{len(pairwise_data)} 정확도 = {accuracy:.2%}")
        
        return {
            "model": openai_model,
            "total_pairs": len(pairwise_data),
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "detailed_results": results
        }
        
    except Exception as e:
        print(f"❌ GPT-4o 평가 중 오류 발생: {e}")
        return {"error": str(e)}


def compare_reward_model_vs_gpt4(
    reward_model_results: Dict[str, Any],
    gpt4_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Reward Model과 GPT-4o의 평가 결과를 비교합니다.
    
    Args:
        reward_model_results: Reward Model 평가 결과
        gpt4_results: GPT-4o 평가 결과
        
    Returns:
        Dict[str, Any]: 비교 분석 결과
    """
    if "error" in reward_model_results or "error" in gpt4_results:
        return {"error": "평가 결과에 오류가 있습니다."}
    
    print(f"🔍 Reward Model vs GPT-4o 비교 분석...")
    
    # 결과 매칭
    rm_details = {result["pair_id"]: result for result in reward_model_results.get("detailed_results", [])}
    gpt_details = {result["pair_id"]: result for result in gpt4_results.get("detailed_results", [])}
    
    # 공통 pair_id 찾기
    common_pairs = set(rm_details.keys()) & set(gpt_details.keys())
    
    if not common_pairs:
        return {"error": "비교할 공통 데이터가 없습니다."}
    
    # 상세 비교
    detailed_comparisons = []
    agreements = 0
    
    for pair_id in common_pairs:
        rm_result = rm_details[pair_id]
        gpt_result = gpt_details[pair_id]
        
        # Reward Model이 chosen을 선호하는지 확인
        rm_prefers_chosen = rm_result.get("prefers_chosen", False)
        
        # GPT-4o가 chosen을 선호하는지 확인
        gpt_prefers_chosen = gpt_result.get("is_correct", False)
        
        # 두 모델이 일치하는지 확인
        agreement = (rm_prefers_chosen == gpt_prefers_chosen)
        if agreement:
            agreements += 1
        
        detailed_comparisons.append({
            "pair_id": pair_id,
            "topic": rm_result.get("topic", "Unknown"),
            "rm_prefers_chosen": rm_prefers_chosen,
            "gpt_prefers_chosen": gpt_prefers_chosen,
            "agreement": agreement,
            "rm_score_diff": rm_result.get("score_difference", 0),
            "gpt_choice": gpt_result.get("gpt_choice", "Unknown")
        })
    
    agreement_rate = agreements / len(common_pairs) if common_pairs else 0
    
    result = {
        "total_comparisons": len(common_pairs),
        "agreements": agreements,
        "disagreements": len(common_pairs) - agreements,
        "agreement_rate": agreement_rate,
        "reward_model_accuracy": reward_model_results.get("accuracy", 0),
        "gpt4_accuracy": gpt4_results.get("accuracy", 0),
        "detailed_comparisons": detailed_comparisons
    }
    
    print(f"✅ 비교 완료: {len(common_pairs)}개 쌍, 일치도 {agreement_rate:.2%}")
    
    return result


def run_preference_ranking_evaluation(
    reward_model_path: str,
    benchmark_file: str,
    openai_model: str = "gpt-4o",
    max_length: int = 512,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Preference Ranking 벤치마크를 사용하여 Reward Model과 GPT-4o의 alignment를 평가합니다.
    
    Args:
        reward_model_path (str): Reward Model 경로
        benchmark_file (str): 벤치마크 파일명
        openai_model (str): OpenAI 모델명
        max_length (int): 최대 토큰 길이
        device (str): 사용할 디바이스
        
    Returns:
        Dict[str, Any]: 전체 평가 결과
    """
    print(f"🚀 Preference Ranking 평가 파이프라인 시작...")
    
    try:
        # 1. 벤치마크 데이터 로드
        print(f"\n1️⃣ 벤치마크 데이터 로드:")
        preference_data = load_preference_ranking_benchmark(benchmark_file)
        
        if not preference_data:
            return {"error": "벤치마크 데이터를 로드할 수 없습니다."}
        
        # 2. Pairwise 형태로 변환
        print(f"\n2️⃣ Pairwise 형태로 변환:")
        pairwise_data = convert_to_pairwise_format(preference_data)
        
        if not pairwise_data:
            return {"error": "Pairwise 데이터 변환에 실패했습니다."}
        
        # 3. Reward Model 평가
        print(f"\n3️⃣ Reward Model 평가:")
        reward_model = RewardModel(
            model_path=reward_model_path,
            device=device,
            max_length=max_length
        )
        
        rm_results = compare_passage_pairs_with_reward_model(
            reward_model=reward_model,
            pairwise_data=pairwise_data
        )
        
        if "error" in rm_results:
            return {"error": f"Reward Model 평가 실패: {rm_results['error']}"}
        
        # 4. GPT-4o 평가
        print(f"\n4️⃣ GPT-4o 평가:")
        gpt4_results = evaluate_gpt4_preference_alignment(
            pairwise_data=pairwise_data,
            openai_model=openai_model
        )
        
        if "error" in gpt4_results:
            return {"error": f"GPT-4o 평가 실패: {gpt4_results['error']}"}
        
        # 5. 결과 비교
        print(f"\n5️⃣ 결과 비교:")
        comparison_result = compare_reward_model_vs_gpt4(
            reward_model_results=rm_results,
            gpt4_results=gpt4_results
        )
        
        if "error" in comparison_result:
            return {"error": f"결과 비교 실패: {comparison_result['error']}"}
        
        print(f"\n🎉 Preference Ranking 평가 완료!")
        
        return {
            "benchmark_file": benchmark_file,
            "reward_model_path": reward_model_path,
            "openai_model": openai_model,
            "total_pairs": len(pairwise_data),
            "reward_model_result": rm_results,
            "gpt4_result": gpt4_results,
            "comparison_result": comparison_result,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ 평가 파이프라인 실행 중 오류: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # 테스트 실행
    result = run_preference_ranking_evaluation(
        reward_model_path="~/models/train_2025-07-26-12-04-23",
        benchmark_file="v1/iSKA-Gen_Benchmark_v1.0.0_20250726_PreferenceRanking.json",
        openai_model="gpt-4o",
        max_length=256
    )
    
    if "error" not in result:
        print(f"\n📊 최종 결과:")
        print(f"   총 평가 쌍: {result['total_pairs']}개")
        comparison = result["comparison_result"]
        print(f"   Reward Model 정확도: {comparison.get('reward_model_accuracy', 0):.2%}")
        print(f"   GPT-4o 정확도: {comparison.get('gpt4_accuracy', 0):.2%}")
        print(f"   두 모델 일치도: {comparison.get('agreement_rate', 0):.2%}")
    else:
        print(f"❌ 평가 실패: {result['error']}")
