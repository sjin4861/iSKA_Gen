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
from utils.output_loader import load_passages
from utils.output_saver import save_model_output, DEFAULT_EVALUATION_DIR
from utils.benchmark_loader import load_benchmarks, get_benchmark_by_id


def evaluate_passage_preferences(
    reward_model_path: str,
    model_name: str,
    benchmark_file: str,
    benchmark_version: str = "v1.0.0",
    BENCH_ID_LIST: List[int] = [1, 2, 3, 4, 5],
    chosen_template: str = "passage_agent.create_passage",
    rejected_template: str = "passage_agent.create_passage_with_korean_errors",
    device: str = "auto",
    max_length: int = 512,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Reward Model을 사용하여 passage preference를 평가하는 함수
    
    Args:
        reward_model_path (str): Reward Model 경로
        model_name (str): 평가할 모델 이름
        benchmark_file (str): 벤치마크 파일명
        benchmark_version (str): 벤치마크 버전
        BENCH_ID_LIST (List[int]): 평가할 벤치마크 ID 리스트
        chosen_template (str): 선호되는 템플릿 키
        rejected_template (str): 거부되는 템플릿 키
        device (str): 사용할 디바이스
        max_length (int): 최대 토큰 길이
        seed (int): 랜덤 시드
        
    Returns:
        Dict[str, Any]: 평가 결과
    """
    print(f"🎯 Passage Preference 평가 시작...")
    print(f"   모델: {model_name}")
    print(f"   Reward Model: {reward_model_path}")
    print(f"   Chosen 템플릿: {chosen_template}")
    print(f"   Rejected 템플릿: {rejected_template}")
    
    # 전체 결과를 저장할 딕셔너리
    evaluation_results = {
        "model_name": model_name,
        "reward_model_path": reward_model_path,
        "chosen_template": chosen_template,
        "rejected_template": rejected_template,
        "benchmark_version": benchmark_version,
        "evaluation_timestamp": datetime.now().isoformat(),
        "benchmark_results": [],
        "overall_statistics": {}
    }
    
    all_comparisons = []
    total_pairs = 0
    total_correct = 0
    
    # 각 벤치마크별로 평가 수행
    for benchmark_id in BENCH_ID_LIST:
        print(f"\n📊 벤치마크 ID {benchmark_id} 평가 중...")
        
        # 1. 벤치마크 정보 로드
        benchmark = get_benchmark_by_id(benchmark_file, benchmark_id)
        if not benchmark:
            print(f"❌ 벤치마크 ID {benchmark_id}를 찾을 수 없습니다.")
            continue
        
        # 2. 두 가지 템플릿으로 생성된 passage 로드
        chosen_passages = load_passages(
            model_name=model_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key=chosen_template
        )
        
        rejected_passages = load_passages(
            model_name=model_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key=rejected_template
        )
        
        if not chosen_passages:
            print(f"❌ '{chosen_template}' 템플릿의 passage를 찾을 수 없습니다.")
            continue
            
        if not rejected_passages:
            print(f"❌ '{rejected_template}' 템플릿의 passage를 찾을 수 없습니다.")
            continue
        
        print(f"✅ Chosen passages: {len(chosen_passages)}개")
        print(f"✅ Rejected passages: {len(rejected_passages)}개")
        
        # 3. Passage pair 생성 및 평가
        benchmark_pairs = create_passage_pairs_for_evaluation(
            chosen_passages, 
            rejected_passages, 
            benchmark_id
        )
        
        if not benchmark_pairs:
            print(f"⚠️ 벤치마크 ID {benchmark_id}: 평가할 pair가 없습니다.")
            continue
        
        # 4. Reward Model로 평가
        comparison_result = compare_passage_pairs_with_reward_model(
            model_path=reward_model_path,
            passage_pairs=benchmark_pairs,
            device=device,
            max_length=max_length,
            seed=seed
        )
        
        # 5. 결과 집계
        benchmark_result = {
            "benchmark_id": benchmark_id,
            "benchmark_info": {
                "problem_types": benchmark.get("problem_types", []),
                "eval_goals": benchmark.get("eval_goals", [])
            },
            "total_pairs": comparison_result["total_pairs"],
            "correct_preferences": comparison_result["correct_preferences"],
            "accuracy": comparison_result["accuracy"],
            "comparisons": comparison_result["comparisons"]
        }
        
        evaluation_results["benchmark_results"].append(benchmark_result)
        all_comparisons.extend(comparison_result["comparisons"])
        total_pairs += comparison_result["total_pairs"]
        total_correct += comparison_result["correct_preferences"]
        
        print(f"✅ 벤치마크 ID {benchmark_id}: {comparison_result['accuracy']:.2%} 정확도")
    
    # 6. 전체 통계 계산
    overall_accuracy = total_correct / total_pairs if total_pairs > 0 else 0
    
    evaluation_results["overall_statistics"] = {
        "total_benchmarks": len(evaluation_results["benchmark_results"]),
        "total_pairs": total_pairs,
        "total_correct_preferences": total_correct,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_benchmark": {
            result["benchmark_id"]: result["accuracy"] 
            for result in evaluation_results["benchmark_results"]
        }
    }
    
    print(f"\n🎉 전체 평가 완료!")
    print(f"   전체 정확도: {overall_accuracy:.2%}")
    print(f"   총 평가 pair: {total_pairs}개")
    print(f"   정확한 선호도: {total_correct}개")
    
    return evaluation_results


def create_passage_pairs_for_evaluation(
    chosen_passages: List[Dict[str, Any]], 
    rejected_passages: List[Dict[str, Any]], 
    benchmark_id: int
) -> List[Dict[str, Any]]:
    """
    평가용 passage pair 생성
    
    Args:
        chosen_passages (List[Dict[str, Any]]): 선호되는 passage 리스트
        rejected_passages (List[Dict[str, Any]]): 거부되는 passage 리스트
        benchmark_id (int): 벤치마크 ID
        
    Returns:
        List[Dict[str, Any]]: 평가용 passage pair 리스트
    """
    pairs = []
    
    min_length = min(len(chosen_passages), len(rejected_passages))
    
    for i in range(min_length):
        chosen_data = chosen_passages[i]
        rejected_data = rejected_passages[i]
        
        # source_item이 동일한지 확인
        chosen_source = chosen_data.get("source_item", {})
        rejected_source = rejected_data.get("source_item", {})
        
        if chosen_source == rejected_source:
            pair = {
                "pair_id": f"benchmark_{benchmark_id}_item_{i}",
                "chosen": chosen_data["generated_passage"],
                "rejected": rejected_data["generated_passage"],
                "source_item": chosen_source,
                "benchmark_id": benchmark_id
            }
            pairs.append(pair)
        else:
            print(f"⚠️ Item {i}: source_item이 다릅니다. 건너뜀")
    
    return pairs


def evaluate_single_benchmark_preference(
    reward_model_path: str,
    model_name: str,
    benchmark_file: str,
    benchmark_id: int,
    benchmark_version: str = "v1.0.0",
    chosen_template: str = "passage_agent.create_passage",
    rejected_template: str = "passage_agent.create_passage_with_korean_errors",
    device: str = "auto",
    max_length: int = 512,
    seed: int = 42
) -> Dict[str, Any]:
    """
    단일 벤치마크에 대한 passage preference 평가
    
    Args:
        reward_model_path (str): Reward Model 경로
        model_name (str): 평가할 모델 이름
        benchmark_file (str): 벤치마크 파일명
        benchmark_id (int): 평가할 벤치마크 ID
        benchmark_version (str): 벤치마크 버전
        chosen_template (str): 선호되는 템플릿 키
        rejected_template (str): 거부되는 템플릿 키
        device (str): 사용할 디바이스
        max_length (int): 최대 토큰 길이
        seed (int): 랜덤 시드
        
    Returns:
        Dict[str, Any]: 평가 결과
    """
    print(f"🎯 단일 벤치마크 Preference 평가 시작...")
    print(f"   벤치마크 ID: {benchmark_id}")
    
    # 전체 평가 함수를 호출하되 단일 벤치마크만 평가
    result = evaluate_passage_preferences(
        reward_model_path=reward_model_path,
        model_name=model_name,
        benchmark_file=benchmark_file,
        benchmark_version=benchmark_version,
        BENCH_ID_LIST=[benchmark_id],
        chosen_template=chosen_template,
        rejected_template=rejected_template,
        device=device,
        max_length=max_length,
        seed=seed
    )
    
    # 단일 벤치마크 결과만 반환
    if result["benchmark_results"]:
        return result["benchmark_results"][0]
    else:
        return {"error": f"벤치마크 ID {benchmark_id} 평가 실패"}


def save_preference_evaluation_results(
    evaluation_results: Dict[str, Any],
    reward_model_name: str,
    benchmark_version: str = "v1.0.0"
) -> Path:
    """
    Preference 평가 결과를 저장
    
    Args:
        evaluation_results (Dict[str, Any]): 평가 결과
        reward_model_name (str): Reward Model 이름 (파일명용)
        benchmark_version (str): 벤치마크 버전
        
    Returns:
        Path: 저장된 파일 경로
    """
    model_name = evaluation_results.get("model_name", "unknown")
    chosen_template = evaluation_results.get("chosen_template", "unknown")
    rejected_template = evaluation_results.get("rejected_template", "unknown")
    
    # 파일명에 사용할 템플릿 정보 (간단하게)
    template_info = f"{chosen_template.split('.')[-1]}_vs_{rejected_template.split('.')[-1]}"
    
    saved_file = save_model_output(
        model_name=f"{model_name}_{reward_model_name}_preference",
        benchmark_id=0,  # 여러 벤치마크를 포함하므로 0
        benchmark_version=benchmark_version,
        template_key=f"preference_eval_{template_info}",
        data=evaluation_results,
        base_dir=DEFAULT_EVALUATION_DIR
    )
    
    return saved_file


def analyze_preference_patterns(
    evaluation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Preference 평가 결과에서 패턴 분석
    
    Args:
        evaluation_results (Dict[str, Any]): 평가 결과
        
    Returns:
        Dict[str, Any]: 분석 결과
    """
    analysis = {
        "score_distribution": {
            "chosen_scores": [],
            "rejected_scores": [],
            "score_differences": []
        },
        "performance_by_benchmark": {},
        "error_patterns": []
    }
    
    for benchmark_result in evaluation_results.get("benchmark_results", []):
        benchmark_id = benchmark_result["benchmark_id"]
        accuracy = benchmark_result["accuracy"]
        
        analysis["performance_by_benchmark"][benchmark_id] = {
            "accuracy": accuracy,
            "total_pairs": benchmark_result["total_pairs"],
            "correct_preferences": benchmark_result["correct_preferences"]
        }
        
        # 점수 분포 분석
        for comparison in benchmark_result.get("comparisons", []):
            chosen_score = comparison.get("chosen_score", 0)
            rejected_score = comparison.get("rejected_score", 0)
            score_diff = comparison.get("score_difference", 0)
            
            analysis["score_distribution"]["chosen_scores"].append(chosen_score)
            analysis["score_distribution"]["rejected_scores"].append(rejected_score)
            analysis["score_distribution"]["score_differences"].append(score_diff)
            
            # 오류 패턴 (chosen이 rejected보다 낮은 점수를 받은 경우)
            if not comparison.get("correct_preference", True):
                analysis["error_patterns"].append({
                    "benchmark_id": benchmark_id,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "score_difference": score_diff
                })
    
    # 통계 계산
    if analysis["score_distribution"]["chosen_scores"]:
        import numpy as np
        
        chosen_scores = analysis["score_distribution"]["chosen_scores"]
        rejected_scores = analysis["score_distribution"]["rejected_scores"]
        score_diffs = analysis["score_distribution"]["score_differences"]
        
        analysis["statistics"] = {
            "chosen_score_stats": {
                "mean": np.mean(chosen_scores),
                "std": np.std(chosen_scores),
                "min": np.min(chosen_scores),
                "max": np.max(chosen_scores)
            },
            "rejected_score_stats": {
                "mean": np.mean(rejected_scores),
                "std": np.std(rejected_scores),
                "min": np.min(rejected_scores),
                "max": np.max(rejected_scores)
            },
            "score_difference_stats": {
                "mean": np.mean(score_diffs),
                "std": np.std(score_diffs),
                "min": np.min(score_diffs),
                "max": np.max(score_diffs)
            }
        }
    
    return analysis


def load_preference_ranking_benchmark(
    benchmark_file: str = "v1/iSKA-Gen_Benchmark_v1.0.0_20250726_PreferenceRanking.json"
) -> List[Dict[str, Any]]:
    """
    Preference Ranking 벤치마크 데이터를 로드합니다.
    
    Args:
        benchmark_file (str): 벤치마크 파일명
        
    Returns:
        List[Dict[str, Any]]: 로드된 벤치마크 데이터
    """
    try:
        # 기존 benchmark_loader 유틸 함수 사용
        raw_data = load_benchmarks(benchmark_file)
        
        # 데이터가 리스트가 아닌 경우 (단일 객체인 경우) 리스트로 변환
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        elif not isinstance(raw_data, list):
            raise ValueError(f"예상치 못한 데이터 형식: {type(raw_data)}")
        
        print(f"✅ Preference Ranking 벤치마크 로드 완료: {len(raw_data)}개 항목")
        return raw_data
        
    except Exception as e:
        print(f"❌ 벤치마크 로드 실패: {e}")
        # 직접 파일 로드로 폴백
        benchmark_path = Path(__file__).resolve().parents[2] / "src" / "data" / "benchmarks" / benchmark_file
        if not benchmark_path.exists():
            raise FileNotFoundError(f"벤치마크 파일을 찾을 수 없습니다: {benchmark_path}")
        
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        
        print(f"✅ 직접 로드로 Preference Ranking 벤치마크 로드 완료: {len(raw_data)}개 항목")
        return raw_data


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
    
    print(f"✅ {len(pairs)}개의 pairwise 비교 쌍 생성 완료")
    return pairs


def evaluate_gpt4_preference_alignment(
    pairs: List[Dict[str, Any]],
    openai_model: str = "gpt-4o",
    max_tokens: int = 50
) -> Dict[str, Any]:
    """
    GPT-4o를 사용하여 preference alignment를 평가합니다.
    
    Args:
        pairs (List[Dict[str, Any]]): 평가할 pair 데이터
        openai_model (str): 사용할 OpenAI 모델
        max_tokens (int): 최대 토큰 수
        
    Returns:
        Dict[str, Any]: GPT-4o 평가 결과
    """
    print(f"🎯 GPT-4o Preference Alignment 평가 시작...")
    print(f"   모델: {openai_model}")
    print(f"   평가할 쌍: {len(pairs)}개")
    
    # OpenAI 클라이언트 초기화
    try:
        client = OpenAIModelClient()
    except Exception as e:
        print(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        return {"error": "OpenAI 클라이언트 초기화 실패"}
    
    # 평가 프롬프트 템플릿
    evaluation_prompt = """다음 두 한국어 텍스트 중 어느 것이 더 높은 품질인지 판단해주세요.

텍스트 A:
{text_a}

텍스트 B:
{text_b}

다음 기준으로 평가해주세요:
- 문법의 정확성
- 어휘 선택의 적절성
- 문체의 자연스러움
- 내용의 논리성

더 품질이 높다고 판단되는 텍스트를 선택하고, 간단한 이유를 제시해주세요.
답변은 "A" 또는 "B"로 시작하고, 그 다음에 이유를 설명해주세요."""
    
    results = []
    correct_count = 0
    
    for i, pair in enumerate(pairs):
        print(f"📊 Pair {i+1}/{len(pairs)} 평가 중...")
        
        # 랜덤하게 chosen/rejected 순서 결정 (bias 방지)
        import random
        if random.choice([True, False]):
            text_a = pair["chosen"]
            text_b = pair["rejected"]
            correct_answer = "A"
        else:
            text_a = pair["rejected"]
            text_b = pair["chosen"]
            correct_answer = "B"
        
        # 프롬프트 생성
        prompt = evaluation_prompt.format(text_a=text_a, text_b=text_b)
        
        try:
            # GPT-4o 호출
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=openai_model,
                max_tokens=max_tokens,
                temperature=0.1  # 일관된 평가를 위해 낮은 temperature
            )
            
            gpt_response = response.strip()
            gpt_choice = gpt_response[0].upper() if gpt_response else "?"
            
            # 정답 여부 확인
            is_correct = gpt_choice == correct_answer
            if is_correct:
                correct_count += 1
            
            result = {
                "pair_id": pair["pair_id"],
                "topic": pair["topic"],
                "text_a": text_a,
                "text_b": text_b,
                "correct_answer": correct_answer,
                "gpt_response": gpt_response,
                "gpt_choice": gpt_choice,
                "is_correct": is_correct
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ GPT-4o 평가 실패 (Pair {i+1}): {e}")
            result = {
                "pair_id": pair["pair_id"],
                "topic": pair["topic"],
                "error": str(e)
            }
            results.append(result)
    
    # 전체 결과 계산
    total_evaluated = len([r for r in results if "error" not in r])
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
    
    evaluation_result = {
        "model": openai_model,
        "total_pairs": len(pairs),
        "total_evaluated": total_evaluated,
        "correct_preferences": correct_count,
        "accuracy": accuracy,
        "evaluation_timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    print(f"✅ GPT-4o 평가 완료!")
    print(f"   평가된 쌍: {total_evaluated}/{len(pairs)}")
    print(f"   정확도: {accuracy:.2%}")
    
    return evaluation_result


def compare_reward_model_vs_gpt4(
    reward_model_path: str,
    pairs: List[Dict[str, Any]],
    openai_model: str = "gpt-4o",
    device: str = "auto",
    max_length: int = 512,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Reward Model과 GPT-4o의 preference alignment를 비교합니다.
    
    Args:
        reward_model_path (str): Reward Model 경로
        pairs (List[Dict[str, Any]]): 평가할 pair 데이터
        openai_model (str): 사용할 OpenAI 모델
        device (str): 사용할 디바이스
        max_length (int): 최대 토큰 길이
        seed (int): 랜덤 시드
        
    Returns:
        Dict[str, Any]: 비교 결과
    """
    print(f"🔄 Reward Model vs GPT-4o 비교 시작...")
    
    # 1. Reward Model 평가
    print("\n1️⃣ Reward Model 평가:")
    rm_comparison_pairs = [{"chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    rm_result = compare_passage_pairs_with_reward_model(
        model_path=reward_model_path,
        passage_pairs=rm_comparison_pairs,
        device=device,
        max_length=max_length,
        seed=seed
    )
    
    # 2. GPT-4o 평가
    print("\n2️⃣ GPT-4o 평가:")
    gpt_result = evaluate_gpt4_preference_alignment(pairs, openai_model)
    
    # 3. 결과 비교
    print("\n3️⃣ 결과 비교:")
    
    # 공통으로 평가된 pair만 비교
    rm_comparisons = rm_result.get("comparisons", [])
    gpt_results = gpt_result.get("results", [])
    
    agreement_count = 0
    total_comparable = min(len(rm_comparisons), len(gpt_results))
    
    detailed_comparisons = []
    
    for i in range(total_comparable):
        rm_comp = rm_comparisons[i]
        gpt_res = gpt_results[i]
        
        if "error" in gpt_res:
            continue
        
        # Reward Model의 선호도
        rm_prefers_chosen = rm_comp.get("correct_preference", False)
        
        # GPT-4o의 선호도 (정답과 일치 여부)
        gpt_prefers_chosen = gpt_res.get("is_correct", False)
        
        # 두 모델이 일치하는지 확인
        is_agreement = rm_prefers_chosen == gpt_prefers_chosen
        if is_agreement:
            agreement_count += 1
        
        detailed_comparison = {
            "pair_index": i,
            "topic": pairs[i]["topic"] if i < len(pairs) else f"pair_{i}",
            "rm_chosen_score": rm_comp.get("chosen_score", 0),
            "rm_rejected_score": rm_comp.get("rejected_score", 0),
            "rm_prefers_chosen": rm_prefers_chosen,
            "gpt_response": gpt_res.get("gpt_response", ""),
            "gpt_prefers_chosen": gpt_prefers_chosen,
            "agreement": is_agreement
        }
        detailed_comparisons.append(detailed_comparison)
    
    # 전체 비교 결과
    agreement_rate = agreement_count / total_comparable if total_comparable > 0 else 0
    
    comparison_result = {
        "comparison_timestamp": datetime.now().isoformat(),
        "reward_model_path": reward_model_path,
        "gpt_model": openai_model,
        "total_pairs": len(pairs),
        "comparable_pairs": total_comparable,
        "agreement_count": agreement_count,
        "agreement_rate": agreement_rate,
        "reward_model_accuracy": rm_result.get("accuracy", 0),
        "gpt4_accuracy": gpt_result.get("accuracy", 0),
        "detailed_comparisons": detailed_comparisons,
        "summary": {
            "rm_correct": rm_result.get("correct_preferences", 0),
            "rm_total": rm_result.get("total_pairs", 0),
            "gpt_correct": gpt_result.get("correct_preferences", 0),
            "gpt_total": gpt_result.get("total_evaluated", 0)
        }
    }
    
    print(f"✅ 비교 완료!")
    print(f"   비교 가능한 쌍: {total_comparable}개")
    print(f"   일치도: {agreement_rate:.2%}")
    print(f"   Reward Model 정확도: {rm_result.get('accuracy', 0):.2%}")
    print(f"   GPT-4o 정확도: {gpt_result.get('accuracy', 0):.2%}")
    
    return comparison_result


def run_preference_ranking_evaluation(
    reward_model_path: str,
    benchmark_file: str = "v1/iSKA-Gen_Benchmark_v1.0.0_20250726_PreferenceRanking.json",
    openai_model: str = "gpt-4o",
    device: str = "auto",
    max_length: int = 512,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Preference Ranking 벤치마크를 사용한 전체 평가 파이프라인
    
    Args:
        reward_model_path (str): Reward Model 경로
        benchmark_file (str): Preference Ranking 벤치마크 파일
        openai_model (str): 사용할 OpenAI 모델
        device (str): 사용할 디바이스
        max_length (int): 최대 토큰 길이
        seed (int): 랜덤 시드
        
    Returns:
        Dict[str, Any]: 전체 평가 결과
    """
    print("🚀 Preference Ranking 평가 파이프라인 시작...")
    
    try:
        # 1. 벤치마크 데이터 로드
        print("\n1️⃣ 벤치마크 데이터 로드:")
        preference_data = load_preference_ranking_benchmark(benchmark_file)
        
        # 2. Pairwise 형태로 변환
        print("\n2️⃣ Pairwise 형태로 변환:")
        pairs = convert_to_pairwise_format(preference_data)
        
        if not pairs:
            return {"error": "변환된 pair가 없습니다."}
        
        # 3. Reward Model vs GPT-4o 비교
        print("\n3️⃣ Reward Model vs GPT-4o 비교:")
        comparison_result = compare_reward_model_vs_gpt4(
            reward_model_path=reward_model_path,
            pairs=pairs,
            openai_model=openai_model,
            device=device,
            max_length=max_length,
            seed=seed
        )
        
        # 4. 결과 저장
        print("\n4️⃣ 결과 저장:")
        saved_file = save_model_output(
            model_name="PreferenceRanking_Comparison",
            benchmark_id=0,
            benchmark_version="v1.0.0",
            template_key="rm_vs_gpt4_alignment",
            data=comparison_result,
            base_dir=DEFAULT_EVALUATION_DIR
        )
        
        print(f"💾 결과 저장 완료: {saved_file}")
        
        # 5. 최종 결과 요약
        final_result = {
            "evaluation_type": "preference_ranking_alignment",
            "benchmark_file": benchmark_file,
            "total_pairs": len(pairs),
            "comparison_result": comparison_result,
            "saved_file": str(saved_file)
        }
        
        print("\n🎉 Preference Ranking 평가 완료!")
        print(f"   총 {len(pairs)}개 쌍 평가")
        print(f"   Reward Model vs GPT-4o 일치도: {comparison_result.get('agreement_rate', 0):.2%}")
        
        return final_result
        
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# --- 실행 예시 ---
if __name__ == "__main__":
    # 설정 값들
    REWARD_MODEL_PATH = "./saves/Qwen3-4B-Instruct/lora/train_2025-07-26-12-04-23"  # 실제 경로로 변경
    PREFERENCE_BENCHMARK_FILE = "v1/iSKA-Gen_Benchmark_v1.0.0_20250726_PreferenceRanking.json"
    OPENAI_MODEL = "gpt-4o"
    
    try:
        print("🚀 Preference Ranking vs GPT-4o Alignment 평가 시작...")
        
        # 새로운 Preference Ranking 평가 실행
        print("\n📊 Preference Ranking 벤치마크 평가:")
        preference_result = run_preference_ranking_evaluation(
            reward_model_path=REWARD_MODEL_PATH,
            benchmark_file=PREFERENCE_BENCHMARK_FILE,
            openai_model=OPENAI_MODEL,
            max_length=256  # 테스트용으로 짧게 설정
        )
        
        if "error" not in preference_result:
            comparison_data = preference_result["comparison_result"]
            print(f"\n📈 최종 결과:")
            print(f"   총 평가 쌍: {preference_result['total_pairs']}개")
            print(f"   Reward Model 정확도: {comparison_data.get('reward_model_accuracy', 0):.2%}")
            print(f"   GPT-4o 정확도: {comparison_data.get('gpt4_accuracy', 0):.2%}")
            print(f"   두 모델 일치도: {comparison_data.get('agreement_rate', 0):.2%}")
            
            # 상세 분석
            if comparison_data.get('detailed_comparisons'):
                agreements = [c for c in comparison_data['detailed_comparisons'] if c['agreement']]
                disagreements = [c for c in comparison_data['detailed_comparisons'] if not c['agreement']]
                
                print(f"\n📋 상세 분석:")
                print(f"   일치하는 경우: {len(agreements)}개")
                print(f"   불일치하는 경우: {len(disagreements)}개")
                
                if disagreements:
                    print(f"\n⚠️ 불일치 사례 (처음 3개):")
                    for i, disagreement in enumerate(disagreements[:3]):
                        print(f"   {i+1}. 주제: {disagreement.get('topic', 'Unknown')}")
                        print(f"      RM 선호: {'Chosen' if disagreement['rm_prefers_chosen'] else 'Rejected'}")
                        print(f"      GPT-4o 선호: {'Chosen' if disagreement['gpt_prefers_chosen'] else 'Rejected'}")
        
        # 기존 방식도 테스트 (호환성 확인)
        print("\n\n🔄 기존 방식 호환성 테스트:")
        OLD_MODEL_NAME = "A.X-4.0-Light"
        OLD_BENCHMARK_FILE = "v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial.json"
        BENCH_ID_LIST = [1]  # 테스트용으로 하나만
        
        try:
            evaluation_results = evaluate_passage_preferences(
                reward_model_path=REWARD_MODEL_PATH,
                model_name=OLD_MODEL_NAME,
                benchmark_file=OLD_BENCHMARK_FILE,
                benchmark_version="v1.0.0",
                BENCH_ID_LIST=BENCH_ID_LIST,
                max_length=256
            )
            
            print(f"✅ 기존 방식 호환성 확인 완료")
            if evaluation_results.get("overall_statistics"):
                stats = evaluation_results["overall_statistics"]
                print(f"   기존 방식 정확도: {stats.get('overall_accuracy', 0):.2%}")
            
        except Exception as e:
            print(f"⚠️ 기존 방식 테스트 실패 (정상적임): {e}")
        
        print("\n🎉 모든 평가 완료!")
        
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
