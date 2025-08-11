#!/usr/bin/env python3
"""
이미지 캡션 및 상황 설명 생성 기능 간단 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

def test_image_caption_generation():
    """이미지 캡션 생성 기능 테스트"""
    try:
        from src.modules.iska.passage_agent import PassageAgent
        from src.modules.model_client import LocalModelClient
        
        print("🧪 이미지 캡션 생성 기능 테스트를 시작합니다...\n")
        
        # 더미 LLM 클라이언트로 테스트 (실제 모델 호출 없이)
        class DummyLLMClient:
            def call(self, messages, temperature=0.7):
                topic = "쓰레기 분리배출"  # 예시용
                return f"""**이미지 설명:** 아파트 단지 안의 분리수거장. 플라스틱, 비닐, 종이 등 종류별로 나뉜 여러 개의 큰 분리수거함이 놓여 있다. 한 젊은 여성이 내용물을 비우지 않은 채 라벨이 붙어 있는 페트병을 플라스틱 수거함에 버리려고 하고 있고, 다른 한쪽에서는 나이 든 경비원 아저씨가 그 모습을 보며 손가락으로 페트병을 가리키며 난처한 표정을 짓고 있다.

**문제 상황:** 이 이미지는 한국의 일상적인 쓰레기 분리배출 상황을 보여주며, 올바른 분리배출 방법에 대한 이해가 부족한 사람과 이를 관리하는 사람 사이의 잠재적인 갈등 상황을 담고 있다."""
        
        # PassageAgent 초기화
        dummy_client = DummyLLMClient()
        passage_agent = PassageAgent(llm_client=dummy_client)
        
        # 테스트할 주제
        test_topic = "{쓰레기 분리배출}"
        
        print(f"📝 테스트 주제: {test_topic}")
        print("-" * 60)
        
        # 이미지 캡션 및 상황 설명 생성
        result = passage_agent.generate_image_caption_and_situation(
            topic=test_topic,
            template_key='passage_agent.create_image_caption_and_situation'
        )
        
        if result:
            print("\n✅ 테스트 성공!")
            print("📋 생성된 결과:")
            print("-" * 40)
            print(result)
            print("-" * 40)
        else:
            print("\n❌ 테스트 실패: 결과가 없습니다.")
            
    except ImportError as e:
        print(f"❌ 모듈 임포트 오류: {e}")
        print("💡 모듈 경로를 확인하고 다시 시도해주세요.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    test_image_caption_generation()
