import json
import torch
from pathlib import Path
import re
from tqdm import tqdm
import sys

# --- 프로젝트 경로 설정 및 클라이언트 임포트 ---
# 이 스크립트 파일의 위치를 기준으로 프로젝트 루트를 추정합니다.
# (예: /path/to/iSKA_Gen/src/scripts/scoring.py)
# 필요시 경로를 직접 수정해주세요.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3] # iSKA_Gen 디렉토리
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.modules.model_client import LocalModelClient
    print("✅ LocalModelClient를 성공적으로 import 했습니다.")
except (ImportError, IndexError, ModuleNotFoundError):
    print("🚨 오류: LocalModelClient를 import할 수 없습니다.")
    print("프로젝트 구조를 확인하거나, 스크립트를 iSKA_Gen 프로젝트 내 올바른 위치에서 실행해주세요.")
    # 스크립트 실행을 중단합니다.
    sys.exit(1)


# --- 0. 설정 ---
# TODO: 평가에 사용할 로컬 모델 이름을 지정하세요 (LocalModelClient가 인식하는 이름).
MODEL_NAME = "EXAONE-4.0-32B" # 예시: LocalModelClient에 전달될 모델 이름
INPUT_FILE = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/final_prompted_dataset/v4/final_prompted_dataset_chosen_2.jsonl"
OUTPUT_FILE = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/final_prompted_dataset/llm/v4/final_prompted_dataset_chosen_2_llm_scored.jsonl"

# GPU 사용 설정 (chModelClient 내부에서 처리되므로 여기서는 참고용)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")


# --- 1. LLM 평가 함수 ---

def get_llm_score(prompt_text: str, chosen_text: str, client: LocalModelClient) -> int or None:
    """LLM을 사용하여 주어진 콘텐츠의 점수를 5점 척도로 평가합니다."""
    
    # LLM에게 전달할 평가 프롬프트 템플릿
    evaluation_prompt_content = f"""당신은 한국어 교육 자료 평가 전문가입니다. 주어진 [평가 지침]에 따라 아래 [자료 및 문항 세트]가 L2 한국어 학습자에게 얼마나 적합한지 평가해 주세요.

[평가 지침]
{prompt_text}

[자료 및 문항 세트]
{chosen_text}

---
위 내용을 바탕으로 1점에서 5점 사이의 정수 점수 하나만 응답해 주십시오. 
5 매우 적절 / 4 대체로 적절(일부 어려움) / 3 가능하나 방해 요소 있음 / 2 어려운 요소 다수 / 1 매우 어려움

오직 숫자 하나만 출력해야 합니다.

점수:"""

    # LocalModelClient의 인터페이스에 맞게 messages 형식으로 전달
    messages = [{"role": "user", "content": evaluation_prompt_content}]

    # 클라이언트를 통해 모델 추론
    response_text = client.call(messages, max_new_tokens=5, temperature=0.1)
    
    # 출력된 텍스트에서 숫자(점수)만 파싱
    match = re.search(r'\d+', response_text)
    if match:
        return int(match.group(0))
    else:
        print(f"⚠️ 경고: 응답에서 점수를 파싱할 수 없습니다. 응답: '{response_text}'")
        return None

# --- 2. 메인 실행 블록 ---

if __name__ == "__main__":
    try:
        input_path = Path(INPUT_FILE)
        output_path = Path(OUTPUT_FILE)
        
        if not input_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

        # 1. LocalModelClient 인스턴스 생성
        # with 문을 사용하여 스크립트 종료 시 자동으로 리소스가 정리되도록 함
        with LocalModelClient(model_name=MODEL_NAME) as client:
            # 2. 입력 파일을 읽고 결과를 저장할 리스트 준비
            with open(input_path, 'r', encoding='utf-8') as f:
                all_data = [json.loads(line) for line in f]

            print(f"\n📄 총 {len(all_data)}개 항목에 대한 LLM 평가를 시작합니다...")

            # 3. 각 항목에 대해 스코어링 진행
            for item in tqdm(all_data, desc="LLM 평가 진행 중"):
                prompt = item.get("prompt")
                chosen = item.get("chosen")
                
                if prompt and chosen:
                    score = get_llm_score(prompt, chosen, client)
                    item["llm_score"] = score
                else:
                    item["llm_score"] = None
            
            # 4. 결과 파일 저장
            print(f"\n💾 평가 결과를 '{output_path}' 파일에 저장합니다...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print("\n" + "=" * 50)
        print("🎉 모든 작업이 성공적으로 완료되었습니다.")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류가 발생했습니다: {e}")
