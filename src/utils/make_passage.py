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
from utils.output_saver import save_model_output, DEFAULT_RAW_OUTPUT_DIR
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

def generate_passage(
    benchmark_file: str,
    model_name: str,
    template_key: str,
    benchmark_version: str = "v1.0.0",
    gpus: list = [2, 3],
    BENCH_ID_LIST: list = [1, 2, 3, 4, 5],
    date_str: str = None,
    min_length: int = 300,
    max_length: int = 800,
    only_indices: list | None = None,
):
    """
    벤치마크에 따라 지문을 생성합니다. 필요 시 특정 인덱스만 부분 재생성하고 기존 파일과 병합 저장합니다.

    Args:
        benchmark_file: 벤치마크 JSON 경로/이름
        model_name: 생성 모델명
        template_key: 사용 프롬프트 키 (디렉터리/파일명에 포함)
        benchmark_version: 벤치마크 버전
        gpus: 사용할 GPU 목록
        BENCH_ID_LIST: 처리할 벤치마크 ID 목록
        date_str: 저장 날짜 디렉터리 (None이면 오늘 날짜)
        min_length: 최소 길이
        max_length: 최대 길이
        only_indices: 지정 시 해당 인덱스만 재생성하고 기존 결과를 유지
    """
    benchmarks = load_benchmarks(benchmark_file)

    llm_client = LocalModelClient(model_name=model_name, gpus = gpus)
    passage_agent = PassageAgent(llm_client=llm_client)
    
    for id in BENCH_ID_LIST:
        benchmark = benchmarks[id - 1]  # id는 1부터 시작하므로 -1을 해줌
        problem_types = benchmark['problem_types']
        eval_goals = benchmark['eval_goals']
        total_items = len(benchmark['items'])

        # 전체 생성 모드 vs 부분 재생성 모드 분기
        passage_datas = []

        if only_indices is not None:
            # 부분 재생성: 선택된 인덱스만 생성하고, 기존 파일과 병합 저장
            print(f"   🎯 부분 재생성 모드: indices={sorted(only_indices)}")

            # 기존 파일 경로 계산 및 로드 시도
            output_type = 'passage' if 'passage' in template_key else 'misc'
            # 날짜 결정 (save_model_output과 동일 규칙); 파일 병합을 의도할 때는 동일 date_str 사용 권장
            from datetime import datetime as _dt
            eff_date = date_str if date_str is not None else _dt.now().strftime("%Y-%m-%d")
            model_dir = DEFAULT_RAW_OUTPUT_DIR / eff_date / output_type / model_name / template_key
            file_name = f"benchmark_{id}_{benchmark_version}_{template_key}.json"
            output_path = model_dir / file_name

            existing_data = None
            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as rf:
                        existing_data = json.load(rf)
                    if not isinstance(existing_data, list) or len(existing_data) != total_items:
                        print("   ⚠️ 기존 파일 구조가 예상과 다릅니다. 새로 구성하여 저장합니다.")
                        existing_data = None
                except Exception as e:
                    print(f"   ⚠️ 기존 파일 로드 실패: {e}. 새로 구성하여 저장합니다.")

            # 새로 생성한 항목들을 담는 맵
            regenerated: dict[int, dict] = {}

            for idx, item in enumerate(benchmark['items']):
                if idx not in set(only_indices):
                    continue

                if id == 5:
                    topic = item.get('topic', '')
                    print(f"   🖼️ [#{idx}] 이미지 캡션 생성 중: {topic}")

                    source_item = {"topic": topic}

                    max_retries = 3
                    retry_count = 0
                    generated_passage = None

                    while generated_passage is None and retry_count < max_retries:
                        try:
                            temp_passage = passage_agent.generate_image_caption_and_situation(topic)

                            if temp_passage and temp_passage.strip():
                                generated_passage = temp_passage
                                print(f"   ✅ [#{idx}] 이미지 캡션 생성 성공")
                                break
                            else:
                                retry_count += 1
                                print(f"   ⚠️ [#{idx}] 빈 결과, 재시도... ({retry_count}/{max_retries})")
                                torch.cuda.empty_cache()
                                gc.collect()

                        except Exception as e:
                            retry_count += 1
                            print(f"   ❌ [#{idx}] 생성 오류: {e}, 재시도... ({retry_count}/{max_retries})")
                            torch.cuda.empty_cache()
                            gc.collect()

                    regenerated[idx] = {
                        "source_item": source_item,
                        "generated_passage": generated_passage,
                    }

                else:
                    # ID 1, 2, 3, 4
                    if 'korean_topic' in item:
                        korean_topic = item['korean_topic']
                        korean_context = item['korean_context']
                        foreign_topic = item['foreign_topic']
                        foreign_context = item['foreign_context']
                        source_item = {
                            "topic": korean_topic,
                            "context": korean_context,
                            "foreign_topic": foreign_topic,
                            "foreign_context": foreign_context,
                        }
                    else:
                        korean_topic = item.get('topic', '')
                        korean_context = item.get('context', '')
                        foreign_topic = ""
                        foreign_context = ""
                        source_item = {
                            "topic": korean_topic,
                            "context": korean_context,
                        }

                    max_retries = 10
                    retry_count = 0
                    generated_passage = None

                    while generated_passage is None and retry_count < max_retries:
                        temp_passage = passage_agent.generate_passage(
                            korean_topic=korean_topic,
                            korean_context=korean_context,
                            foreign_topic=foreign_topic,
                            foreign_context=foreign_context,
                            problem_types=problem_types,
                            eval_goals=eval_goals,
                            template_key=template_key,
                        )

                        if temp_passage is None:
                            retry_count += 1
                            print(f"[#{idx}] None 반환. 재시도... ({retry_count}/{max_retries})")
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            passage_length = len(temp_passage)
                            if passage_length < min_length:
                                retry_count += 1
                                print(f"[#{idx}] 지문 짧음 {passage_length} < {min_length}. 재시도... ({retry_count}/{max_retries})")
                                torch.cuda.empty_cache()
                                gc.collect()
                            elif passage_length > max_length:
                                retry_count += 1
                                print(f"[#{idx}] 지문 김 {passage_length} > {max_length}. 재시도... ({retry_count}/{max_retries})")
                                torch.cuda.empty_cache()
                                gc.collect()
                            else:
                                generated_passage = temp_passage
                                print(f"[#{idx}] 길이 OK: {passage_length}")
                                break

                    regenerated[idx] = {
                        "source_item": source_item,
                        "generated_passage": generated_passage,
                    }

            # 병합 결과 구성
            if existing_data is not None:
                for k, v in regenerated.items():
                    existing_data[k] = v
                passage_datas = existing_data
            else:
                # 기존 파일이 없으면 전체 길이에 맞게 재구성 (선택 인덱스만 생성됨)
                passage_datas = []
                for idx, item in enumerate(benchmark['items']):
                    if idx in regenerated:
                        passage_datas.append(regenerated[idx])
                    else:
                        # 기본 source_item 채우기
                        if id == 5:
                            src = {"topic": item.get('topic', '')}
                        elif 'korean_topic' in item:
                            src = {
                                "topic": item['korean_topic'],
                                "context": item['korean_context'],
                                "foreign_topic": item['foreign_topic'],
                                "foreign_context": item['foreign_context'],
                            }
                        else:
                            src = {
                                "topic": item.get('topic', ''),
                                "context": item.get('context', ''),
                            }
                        passage_datas.append({"source_item": src, "generated_passage": None})

        else:
            # 전체 생성 모드 (기존 동작 유지)
            for item in benchmark['items']:
                if id == 5:
                    # ID 5: 이미지 캡션 및 상황 설명 생성
                    topic = item.get('topic', '')
                    print(f"   🖼️ 이미지 캡션 생성 중: {topic}")

                    source_item = {
                        "topic": topic
                    }

                    # 최대 3번까지 재시도
                    max_retries = 3
                    retry_count = 0
                    generated_passage = None

                    while generated_passage is None and retry_count < max_retries:
                        try:
                            temp_passage = passage_agent.generate_image_caption_and_situation(topic)

                            if temp_passage and temp_passage.strip():
                                generated_passage = temp_passage
                                print(f"   ✅ 이미지 캡션 생성 성공: {topic}")
                                break
                            else:
                                retry_count += 1
                                print(f"   ⚠️ 이미지 캡션 생성 실패, 재시도 중... ({retry_count}/{max_retries})")
                                torch.cuda.empty_cache()
                                gc.collect()

                        except Exception as e:
                            retry_count += 1
                            print(f"   ❌ 이미지 캡션 생성 오류: {e}, 재시도 중... ({retry_count}/{max_retries})")
                            torch.cuda.empty_cache()
                            gc.collect()

                    if generated_passage is None:
                        print(f"   ❌ 이미지 캡션 생성 최종 실패: {topic}")

                    passage_data = {
                        "source_item": source_item,
                        "generated_passage": generated_passage
                    }
                    passage_datas.append(passage_data)

                else:
                    # ID 1, 2, 3, 4: 기존 지문 생성 로직
                    # 벤치마크 구조에 따라 필드명 처리
                    if 'korean_topic' in item:
                        # ID 1: korean/foreign 구조
                        korean_topic = item['korean_topic']
                        korean_context = item['korean_context']
                        foreign_topic = item['foreign_topic']
                        foreign_context = item['foreign_context']

                        # source_item: ID 1은 비교형이므로 korean과 foreign 모두 포함
                        source_item = {
                            "topic": korean_topic,
                            "context": korean_context,
                            "foreign_topic": foreign_topic,
                            "foreign_context": foreign_context
                        }
                    else:
                        # ID 2, 3, 4: topic/context 구조 (domestic)
                        korean_topic = item.get('topic', '')
                        korean_context = item.get('context', '')
                        foreign_topic = ""  # domestic은 foreign 정보 없음
                        foreign_context = ""  # domestic은 foreign 정보 없음

                        # source_item: ID 2,3,4는 domestic이므로 topic/context만 포함
                        source_item = {
                            "topic": korean_topic,
                            "context": korean_context
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
                            if passage_length < min_length:
                                retry_count += 1
                                print(f"Passage too short ({passage_length} chars < {min_length}). Retrying... ({retry_count}/{max_retries})")
                                # 메모리 정리
                                torch.cuda.empty_cache()
                                gc.collect()
                            elif passage_length > max_length:
                                retry_count += 1
                                print(f"Passage too long ({passage_length} chars > {max_length}). Retrying... ({retry_count}/{max_retries})")
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

        # output_saver를 사용하여 결과 저장 (부분 재생성의 경우 기존 파일에 덮어쓰기 형태로 저장)
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
            }
        }

def generate_domestic_passage_from_benchmark(
    benchmark_file: str,
    benchmark_id: int,
    item_index: int,
    model_name: str,
    template_key: str,
    benchmark_version: str = "v1.0.1",
    gpus: list = [2, 3],
    max_retries: int = 10,
    min_length: int = 300,
    max_length: int = 500
) -> dict:
    """
    v1.0.1 벤치마크 파일에서 특정 아이템을 선택하여 domestic 지문 하나를 생성하는 함수
    
    Args:
        benchmark_file (str): 벤치마크 파일명
        benchmark_id (int): 벤치마크 ID (1-5)
        item_index (int): 벤치마크 내 아이템 인덱스 (0부터 시작)
        model_name (str): 사용할 모델명
        template_key (str): 프롬프트 템플릿 키 (domestic/dialogue 용)
        benchmark_version (str): 벤치마크 버전 (기본값: "v1.0.1")
        gpus (list): 사용할 GPU 리스트 (기본값: [2, 3])
        max_retries (int): 최대 재시도 횟수 (기본값: 10)
        min_length (int): 최소 지문 길이 (기본값: 300)
        max_length (int): 최대 지문 길이 (기본값: 500)
        
    Returns:
        dict: 생성된 지문 데이터 또는 오류 정보
    """
    print(f"\n🔧 벤치마크에서 domestic 지문 생성을 시작합니다...")
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
        
        # v1.0.1 벤치마크는 topic/context 구조를 사용
        topic = item.get('topic', item.get('korean_topic', ''))
        context = item.get('context', item.get('korean_context', ''))
        
        print(f"   📚 선택된 아이템: {topic}")
        
        # domestic 지문 생성 호출 - 통일된 generate_single_passage 함수 사용
        result = generate_single_passage(
            korean_topic=topic,
            korean_context=context,
            foreign_topic="",  # domestic은 foreign 정보 없음
            foreign_context="",  # domestic은 foreign 정보 없음
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

def generate_image_caption_and_situation_for_benchmark(
    benchmark_file: str,
    benchmark_id: int = 5,
    model_name: str = "EXAONE-3.0-7.8B-Instruct",
    benchmark_version: str = "v1.1.0",
    gpus: list = [2, 3],
    date_str: str = None
) -> dict:
    """
    ID 5 (보고 말하기) 벤치마크의 모든 주제에 대해 이미지 캡션과 상황 설명을 생성하는 함수
    
    Args:
        benchmark_file (str): 벤치마크 파일명
        benchmark_id (int): 벤치마크 ID (기본값: 5)
        model_name (str): 사용할 모델명
        benchmark_version (str): 벤치마크 버전 (기본값: "v1.1.0")
        gpus (list): 사용할 GPU 리스트 (기본값: [2, 3])
        date_str (str): 날짜 문자열 (None이면 현재 날짜 사용)
        
    Returns:
        dict: 생성 결과 정보
    """
    print(f"\n🖼️ ID {benchmark_id} 벤치마크의 이미지 캡션 및 상황 설명 생성을 시작합니다...")
    print(f"   📄 벤치마크 파일: {benchmark_file}")
    print(f"   🤖 모델: {model_name}")
    
    try:
        # 벤치마크 로드
        benchmarks = load_benchmarks(benchmark_file)
        
        if benchmark_id < 1 or benchmark_id > len(benchmarks):
            return {
                "success": False,
                "error": f"잘못된 벤치마크 ID: {benchmark_id} (유효 범위: 1-{len(benchmarks)})"
            }
        
        benchmark = benchmarks[benchmark_id - 1]  # ID는 1부터 시작
        items = benchmark['items']
        
        print(f"   📊 처리할 아이템 수: {len(items)}")
        
        # LLM 클라이언트 및 에이전트 초기화
        llm_client = LocalModelClient(model_name=model_name, gpus=gpus)
        passage_agent = PassageAgent(llm_client=llm_client)
        
        # 결과 저장용 리스트
        generated_data = []
        success_count = 0
        total_count = len(items)
        
        for idx, item in enumerate(items):
            topic = item.get('topic', '')
            
            print(f"\n   🔄 진행률: {idx + 1}/{total_count} - 처리 중: {topic}")
            
            try:
                # 이미지 캡션 및 상황 설명 생성
                result = passage_agent.generate_image_caption_and_situation(topic)
                
                if result:
                    generated_item = {
                        "source_item": {
                            "topic": topic
                        },
                        "generated_passage": result,
                        "generation_status": "success"
                    }
                    success_count += 1
                    print(f"   ✅ 성공: {topic}")
                else:
                    generated_item = {
                        "source_item": {
                            "topic": topic
                        },
                        "generated_passage": None,
                        "generation_status": "failed",
                        "error": "생성 결과가 없습니다."
                    }
                    print(f"   ❌ 실패: {topic}")
                
                generated_data.append(generated_item)
                
            except Exception as e:
                generated_item = {
                    "source_item": {
                        "topic": topic
                    },
                    "generated_passage": None,
                    "generation_status": "error",
                    "error": str(e)
                }
                generated_data.append(generated_item)
                print(f"   ❌ 오류: {topic} - {e}")
        
        # 결과 저장
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        
        saved_file = save_model_output(
            model_name=model_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key="create_image_caption_and_situation",
            data=generated_data,
            date_str=date_str
        )
        
        # 최종 결과 반환
        result_summary = {
            "success": True,
            "total_items": total_count,
            "success_count": success_count,
            "failure_count": total_count - success_count,
            "success_rate": f"{(success_count/total_count)*100:.1f}%",
            "saved_file": saved_file,
            "benchmark_info": {
                "benchmark_file": benchmark_file,
                "benchmark_id": benchmark_id,
                "benchmark_version": benchmark_version
            },
            "generation_info": {
                "model_name": model_name,
                "template_key": "create_image_caption_and_situation",
                "gpus": gpus,
                "date": date_str
            }
        }
        
        print(f"\n🎉 이미지 캡션 및 상황 설명 생성 완료!")
        print(f"   📊 성공률: {result_summary['success_rate']} ({success_count}/{total_count})")
        print(f"   💾 저장 위치: {saved_file}")
        
        return result_summary
        
    except Exception as e:
        return {
            "success": False,
            "error": f"벤치마크 처리 중 오류 발생: {str(e)}",
            "benchmark_info": {
                "benchmark_file": benchmark_file,
                "benchmark_id": benchmark_id
            }
        }

def generate_single_image_caption_and_situation(
    topic: str,
    model_name: str = "EXAONE-3.0-7.8B-Instruct",
    gpus: list = [2, 3],
    max_retries: int = 3
) -> dict:
    """
    단일 주제에 대해 이미지 캡션과 상황 설명을 생성하는 함수
    
    Args:
        topic (str): 주제 (예: "{쓰레기 분리배출}" 또는 "쓰레기 분리배출")
        model_name (str): 사용할 모델명
        gpus (list): 사용할 GPU 리스트 (기본값: [2, 3])
        max_retries (int): 최대 재시도 횟수 (기본값: 3)
        
    Returns:
        dict: 생성된 결과 또는 오류 정보
    """
    # 중괄호 제거
    clean_topic = topic.strip('{}')
    
    print(f"\n🖼️ '{clean_topic}' 주제의 이미지 캡션과 상황 설명 생성을 시작합니다...")
    print(f"   🤖 모델: {model_name}")
    
    try:
        # LLM 클라이언트 및 에이전트 초기화
        llm_client = LocalModelClient(model_name=model_name, gpus=gpus)
        passage_agent = PassageAgent(llm_client=llm_client)
        
        # 생성 재시도 로직
        retry_count = 0
        generated_result = None
        
        while generated_result is None and retry_count < max_retries:
            print(f"   🔄 시도 {retry_count + 1}/{max_retries}...")
            
            try:
                temp_result = passage_agent.generate_image_caption_and_situation(topic)
                
                if temp_result and temp_result.strip():
                    generated_result = temp_result
                    break
                else:
                    print(f"   ⚠️ 시도 {retry_count + 1} 실패: 빈 결과")
                    retry_count += 1
                    
            except Exception as e:
                print(f"   ⚠️ 시도 {retry_count + 1} 오류: {e}")
                retry_count += 1
        
        # 결과 반환
        if generated_result is not None:
            result = {
                "success": True,
                "source_item": {
                    "topic": clean_topic
                },
                "generated_passage": generated_result,
                "retries_used": retry_count,
                "generation_info": {
                    "model_name": model_name,
                    "template_key": "create_image_caption_and_situation",
                    "gpus": gpus
                }
            }
            print(f"   🎉 생성 성공! (재시도 {retry_count}회)")
            print(f"   📝 생성된 내용 미리보기: {generated_result[:80]}...")
            return result
        else:
            result = {
                "success": False,
                "source_item": {
                    "topic": clean_topic
                },
                "generated_passage": None,
                "error": f"최대 재시도 횟수({max_retries})를 초과했습니다.",
                "retries_used": retry_count,
                "generation_info": {
                    "model_name": model_name,
                    "template_key": "create_image_caption_and_situation",
                    "gpus": gpus
                }
            }
            print(f"   ❌ 생성 실패: 최대 재시도 횟수 초과")
            return result
            
    except Exception as e:
        result = {
            "success": False,
            "source_item": {
                "topic": clean_topic
            },
            "generated_passage": None,
            "error": f"생성 중 오류 발생: {str(e)}",
            "generation_info": {
                "model_name": model_name,
                "template_key": "create_image_caption_and_situation",
                "gpus": gpus
            }
        }
        print(f"   ❌ 생성 중 예외 발생: {e}")
        return result
