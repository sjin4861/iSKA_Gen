import os
import sys
import json
from pathlib import Path
import gc  # <-- 해결책 2: 가비지 컬렉터 모듈 임포트
import torch
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'modules'))

from modules.iska.passage_agent import PassageAgent
from modules.model_client import LocalModelClient
from utils.output_saver import save_model_output
from utils.benchmark_loader import load_benchmarks
import re

def clean_passage_text(text: str) -> str:
    """
    지문 텍스트에서 괄호와 그 안의 내용을 제거하는 후처리 함수
    
    Args:
        text (str): 원본 지문 텍스트
        
    Returns:
        str: 괄호 내용이 제거된 정리된 텍스트
    """
    if not text:
        return text
        
    # 모든 종류의 괄호 제거: (), [], {}, 【】, 『』 등
    # 중괄호, 대괄호, 소괄호, 한글 괄호 등 모든 괄호와 그 안의 내용 제거
    patterns = [
        r'\([^)]*\)',      # (내용)
        r'\[[^\]]*\]',     # [내용]
        r'\{[^}]*\}',      # {내용}
        r'【[^】]*】',       # 【내용】
        r'『[^』]*』',       # 『내용』
        r'「[^」]*」',       # 「내용」
        r'〈[^〉]*〉',       # 〈내용〉
        r'《[^》]*》',       # 《내용》
    ]
    
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    # 연속된 공백을 단일 공백으로 변환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # 앞뒤 공백 제거
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def generate_passage(benchmark_file : str, model_name : str,  template_key : str, benchmark_version: str = "v1.0.0", gpus : list = [2,3], BENCH_ID_LIST : list =  [1, 2, 3, 4, 5], date_str: str = None):
    benchmarks = load_benchmarks(benchmark_file)

    llm_client = LocalModelClient(model_name=model_name, gpus = gpus)
    passage_agent = PassageAgent(llm_client=llm_client)
    
    for id in BENCH_ID_LIST:
        benchmark = benchmarks[id - 1]  # id는 1부터 시작하므로 -1을 해줌
        problem_types = benchmark['problem_types']
        eval_goals = benchmark['eval_goals']
        passage_datas = []
        for item in benchmark['items']:
            korean_topic = item['korean_topic']
            korean_context = item['korean_context']
            foreign_topic = item['foreign_topic']
            foreign_context = item['foreign_context']
        
            source_item = {
                "korean_topic": korean_topic,
                "korean_context": korean_context,
                "foreign_topic": foreign_topic,
                "foreign_context": foreign_context
            }
            # 최대 10번까지 재시도하는 로직 추가
            max_retries = 10
            retry_count = 0
            generated_passage = None
            
            while generated_passage is None and retry_count < max_retries:
                temp_passage = passage_agent.generate_passage(korean_topic=korean_topic, korean_context=korean_context, foreign_topic=foreign_topic, foreign_context=foreign_context, problem_types=problem_types, eval_goals=eval_goals, template_key=template_key)
                
                if temp_passage is None:
                    retry_count += 1
                    print(f"Passage generation returned None. Retrying... ({retry_count}/{max_retries})")
                    # 메모리 정리
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    # 후처리: 괄호와 그 안의 내용 제거
                    # temp_passage = clean_passage_text(temp_passage)
                    print(f"Passage cleaned (removed brackets and their contents)")
                    
                    # 길이 검증 (공백 포함)
                    passage_length = len(temp_passage)
                    if passage_length < 300:
                        retry_count += 1
                        print(f"Passage too short ({passage_length} chars < 300). Retrying... ({retry_count}/{max_retries})")
                        # 메모리 정리
                        torch.cuda.empty_cache()
                        gc.collect()
                    elif passage_length > 500:
                        retry_count += 1
                        print(f"Passage too long ({passage_length} chars > 500). Retrying... ({retry_count}/{max_retries})")
                        # 메모리 정리
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        # 길이 조건을 만족하면 통과
                        generated_passage = temp_passage
                        print(f"Passage length validated ({passage_length} chars). Generation successful.")
                        break  # while 루프 탈출
            
            # 최종적으로 생성된 passage 사용
            passage_data = {
                "source_item": source_item,
                "generated_passage": generated_passage,
            }
            passage_datas.append(passage_data)
        
        # output_saver를 사용하여 결과 저장
        saved_file = save_model_output(
            model_name=model_name,
            benchmark_id=id,
            benchmark_version=benchmark_version,
            template_key=template_key,
            data=passage_datas,
            date_str=date_str
        )
        print(f"Generated passage for benchmark ID {id} and saved to {saved_file}")


def generate_single_passage(
    korean_topic: str,
    korean_context: str,
    foreign_topic: str,
    foreign_context: str,
    problem_types: list,
    eval_goals: list,
    model_name: str,
    template_key: str,
    gpus: list = [2, 3],
    max_retries: int = 10,
    min_length: int = 300,
    max_length: int = 500
) -> dict:
    """
    특정한 벤치마크 정보를 입력받아 지문 하나를 생성하는 함수
    
    Args:
        korean_topic (str): 한국 주제
        korean_context (str): 한국 컨텍스트
        foreign_topic (str): 외국 주제
        foreign_context (str): 외국 컨텍스트
        problem_types (list): 문제 유형 리스트 (3개)
        eval_goals (list): 평가 목표 리스트 (3개)
        model_name (str): 사용할 모델명
        template_key (str): 프롬프트 템플릿 키
        gpus (list): 사용할 GPU 리스트 (기본값: [2, 3])
        max_retries (int): 최대 재시도 횟수 (기본값: 10)
        min_length (int): 최소 지문 길이 (기본값: 300)
        max_length (int): 최대 지문 길이 (기본값: 500)
        
    Returns:
        dict: 생성된 지문 데이터 또는 오류 정보
    """
    print(f"\n🔧 단일 지문 생성을 시작합니다...")
    print(f"   📚 한국 주제: {korean_topic}")
    print(f"   🌍 외국 주제: {foreign_topic}")
    print(f"   🤖 모델: {model_name}")
    print(f"   📝 템플릿: {template_key}")
    
    try:
        # LLM 클라이언트 및 에이전트 초기화
        llm_client = LocalModelClient(model_name=model_name, gpus=gpus)
        passage_agent = PassageAgent(llm_client=llm_client)
        
        source_item = {
            "korean_topic": korean_topic,
            "korean_context": korean_context,
            "foreign_topic": foreign_topic,
            "foreign_context": foreign_context
        }
        
        # 지문 생성 재시도 로직
        retry_count = 0
        generated_passage = None
        
        while generated_passage is None and retry_count < max_retries:
            print(f"   🔄 시도 {retry_count + 1}/{max_retries}...")
            
            temp_passage = passage_agent.generate_passage(
                korean_topic=korean_topic,
                korean_context=korean_context,
                foreign_topic=foreign_topic,
                foreign_context=foreign_context,
                problem_types=problem_types,
                eval_goals=eval_goals,
                template_key=template_key
            )
            
            if temp_passage is None:
                retry_count += 1
                print(f"   ⚠️ 지문 생성 실패. 재시도 중... ({retry_count}/{max_retries})")
                # 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()
            else:
                # 후처리: 괄호와 그 안의 내용 제거 (필요시)
                # temp_passage = clean_passage_text(temp_passage)
                
                # 길이 검증
                passage_length = len(temp_passage)
                print(f"   📏 생성된 지문 길이: {passage_length}자")
                
                if passage_length < min_length:
                    retry_count += 1
                    print(f"   ⚠️ 지문이 너무 짧습니다 ({passage_length}자 < {min_length}자). 재시도...")
                    # 메모리 정리
                    torch.cuda.empty_cache()
                    gc.collect()
                elif passage_length > max_length:
                    retry_count += 1
                    print(f"   ⚠️ 지문이 너무 깁니다 ({passage_length}자 > {max_length}자). 재시도... ({retry_count}/{max_retries})")
                    # 메모리 정리
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    # 길이 조건 만족
                    generated_passage = temp_passage
                    print(f"   ✅ 지문 길이 검증 통과: {passage_length}자")
                    break
        
        # 결과 반환
        if generated_passage is not None:
            result = {
                "success": True,
                "source_item": source_item,
                "generated_passage": generated_passage,
                "passage_length": len(generated_passage),
                "retries_used": retry_count,
                "generation_info": {
                    "model_name": model_name,
                    "template_key": template_key,
                    "gpus": gpus
                }
            }
            print(f"   🎉 지문 생성 성공! (재시도 {retry_count}회)")
            print(f"   📝 생성된 지문 미리보기: {generated_passage[:100]}...")
            return result
        else:
            result = {
                "success": False,
                "error": f"최대 재시도 횟수({max_retries})를 초과했습니다.",
                "source_item": source_item,
                "retries_used": retry_count,
                "generation_info": {
                    "model_name": model_name,
                    "template_key": template_key,
                    "gpus": gpus
                }
            }
            print(f"   ❌ 지문 생성 실패: 최대 재시도 횟수 초과")
            return result
            
    except Exception as e:
        result = {
            "success": False,
            "error": f"지문 생성 중 오류 발생: {str(e)}",
            "source_item": {
                "korean_topic": korean_topic,
                "korean_context": korean_context,
                "foreign_topic": foreign_topic,
                "foreign_context": foreign_context
            },
            "generation_info": {
                "model_name": model_name,
                "template_key": template_key,
                "gpus": gpus
            }
        }
        print(f"   ❌ 지문 생성 중 예외 발생: {e}")
        return result


def generate_single_passage_from_benchmark(
    benchmark_file: str,
    benchmark_id: int,
    item_index: int,
    model_name: str,
    template_key: str,
    benchmark_version: str = "v1.0.0",
    gpus: list = [2, 3],
    max_retries: int = 10,
    min_length: int = 300,
    max_length: int = 500
) -> dict:
    """
    벤치마크 파일에서 특정 아이템을 선택하여 지문 하나를 생성하는 함수
    
    Args:
        benchmark_file (str): 벤치마크 파일명
        benchmark_id (int): 벤치마크 ID (1-5)
        item_index (int): 벤치마크 내 아이템 인덱스 (0부터 시작)
        model_name (str): 사용할 모델명
        template_key (str): 프롬프트 템플릿 키
        benchmark_version (str): 벤치마크 버전 (기본값: "v1.0.0")
        gpus (list): 사용할 GPU 리스트 (기본값: [2, 3])
        max_retries (int): 최대 재시도 횟수 (기본값: 10)
        min_length (int): 최소 지문 길이 (기본값: 300)
        max_length (int): 최대 지문 길이 (기본값: 500)
        
    Returns:
        dict: 생성된 지문 데이터 또는 오류 정보
    """
    print(f"\n🔧 벤치마크에서 단일 지문 생성을 시작합니다...")
    print(f"   📄 벤치마크 파일: {benchmark_file}")
    print(f"   🆔 벤치마크 ID: {benchmark_id}")
    print(f"   📍 아이템 인덱스: {item_index}")
    
    try:
        # 벤치마크 로드
        benchmarks = load_benchmarks(benchmark_file)
        
        if benchmark_id < 1 or benchmark_id > len(benchmarks):
            return {
                "success": False,
                "error": f"잘못된 벤치마크 ID: {benchmark_id} (유효 범위: 1-{len(benchmarks)})"
            }
        
        benchmark = benchmarks[benchmark_id - 1]  # ID는 1부터 시작
        problem_types = benchmark['problem_types']
        eval_goals = benchmark['eval_goals']
        items = benchmark['items']
        
        if item_index < 0 or item_index >= len(items):
            return {
                "success": False,
                "error": f"잘못된 아이템 인덱스: {item_index} (유효 범위: 0-{len(items)-1})"
            }
        
        item = items[item_index]
        
        print(f"   📚 선택된 아이템: {item['korean_topic']} vs {item['foreign_topic']}")
        
        # 단일 지문 생성 호출
        result = generate_single_passage(
            korean_topic=item['korean_topic'],
            korean_context=item['korean_context'],
            foreign_topic=item['foreign_topic'],
            foreign_context=item['foreign_context'],
            problem_types=problem_types,
            eval_goals=eval_goals,
            model_name=model_name,
            template_key=template_key,
            gpus=gpus,
            max_retries=max_retries,
            min_length=min_length,
            max_length=max_length
        )
        
        # 벤치마크 정보 추가
        if result["success"]:
            result["benchmark_info"] = {
                "benchmark_file": benchmark_file,
                "benchmark_id": benchmark_id,
                "item_index": item_index,
                "benchmark_version": benchmark_version,
                "problem_types": problem_types,
                "eval_goals": eval_goals
            }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"벤치마크 로드 중 오류 발생: {str(e)}",
            "benchmark_info": {
                "benchmark_file": benchmark_file,
                "benchmark_id": benchmark_id,
                "item_index": item_index
            }
        }

 