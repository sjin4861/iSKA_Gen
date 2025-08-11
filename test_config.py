#!/usr/bin/env python
# coding: utf-8

"""
ConfigManager 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src' / 'scripts'))

def test_config_manager():
    """ConfigManager 테스트"""
    try:
        from src.scripts.managers.config_manager import ConfigManager
        print('✅ ConfigManager import 성공!')
        
        config = ConfigManager()
        print('✅ ConfigManager 초기화 성공!')
        
        config.print_config_summary()
        
        # 간단한 기능 테스트
        print(f"\n🧪 기능 테스트:")
        print(f"passage 타입 ID들: {config.get_benchmark_ids_for_type('passage')}")
        print(f"ID 1의 템플릿 키: {config.get_template_key_for_id(1)}")
        print(f"ID 2의 길이 제한: {config.get_length_limits_for_id(2)}")
        
    except Exception as e:
        print(f'❌ 오류: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_manager()
