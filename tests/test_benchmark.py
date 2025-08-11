"""
벤치마크 데이터셋 검증 테스트 스크립트

벤치마크 파일의 구조, 필수 키, 아이템 개수 등을 검증합니다.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Set

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.benchmark_loader import load_benchmarks, list_available_benchmarks

class BenchmarkValidator:
    """벤치마크 데이터 검증 클래스"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_tests = []
        
    def log_error(self, message: str):
        """에러 메시지 기록"""
        self.errors.append(f"❌ ERROR: {message}")
        
    def log_warning(self, message: str):
        """경고 메시지 기록"""
        self.warnings.append(f"⚠️  WARNING: {message}")
        
    def log_pass(self, message: str):
        """성공 메시지 기록"""
        self.passed_tests.append(f"✅ PASS: {message}")
    
    def validate_benchmark_structure(self, benchmark: Dict[str, Any], expected_keys: Set[str]) -> bool:
        """벤치마크 구조 검증"""
        benchmark_id = benchmark.get('id', 'Unknown')
        
        # 필수 키 확인
        missing_keys = expected_keys - set(benchmark.keys())
        if missing_keys:
            self.log_error(f"Benchmark ID {benchmark_id}: Missing required keys: {missing_keys}")
            return False
            
        # 타입 검증
        if not isinstance(benchmark.get('id'), int):
            self.log_error(f"Benchmark ID {benchmark_id}: 'id' must be integer")
            return False
            
        if not isinstance(benchmark.get('problem_types'), list):
            self.log_error(f"Benchmark ID {benchmark_id}: 'problem_types' must be list")
            return False
            
        if not isinstance(benchmark.get('eval_goals'), list):
            self.log_error(f"Benchmark ID {benchmark_id}: 'eval_goals' must be list")
            return False
            
        if not isinstance(benchmark.get('items'), list):
            self.log_error(f"Benchmark ID {benchmark_id}: 'items' must be list")
            return False
            
        self.log_pass(f"Benchmark ID {benchmark_id}: Structure validation passed")
        return True
    
    def validate_item_structure_v1_0_0(self, item: Dict[str, Any], benchmark_id: int, item_idx: int) -> bool:
        """v1.0.0 아이템 구조 검증 (비교형: korean_topic/foreign_topic)"""
        required_keys = {'korean_topic', 'korean_context', 'foreign_topic', 'foreign_context'}
        
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            self.log_error(f"Benchmark ID {benchmark_id}, Item {item_idx}: Missing keys: {missing_keys}")
            return False
            
        # 값이 비어있는지 확인
        for key in required_keys:
            if not item.get(key) or not isinstance(item[key], str) or not item[key].strip():
                self.log_error(f"Benchmark ID {benchmark_id}, Item {item_idx}: '{key}' is empty or invalid")
                return False
                
        return True
    
    def validate_item_structure_v1_1_0_domestic(self, item: Dict[str, Any], benchmark_id: int, item_idx: int) -> bool:
        """v1.1.0 단일 주제형 아이템 구조 검증 (topic/context)"""
        required_keys = {'topic', 'context'}
        
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            self.log_error(f"Benchmark ID {benchmark_id}, Item {item_idx}: Missing keys: {missing_keys}")
            return False
            
        # 값이 비어있는지 확인
        for key in required_keys:
            if not item.get(key) or not isinstance(item[key], str) or not item[key].strip():
                self.log_error(f"Benchmark ID {benchmark_id}, Item {item_idx}: '{key}' is empty or invalid")
                return False
                
        return True
    
    def validate_item_structure_v1_1_0_simple(self, item: Dict[str, Any], benchmark_id: int, item_idx: int) -> bool:
        """v1.1.0 단순형 아이템 구조 검증 (topic만)"""
        required_keys = {'topic'}
        
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            self.log_error(f"Benchmark ID {benchmark_id}, Item {item_idx}: Missing keys: {missing_keys}")
            return False
            
        # 값이 비어있는지 확인
        if not item.get('topic') or not isinstance(item['topic'], str) or not item['topic'].strip():
            self.log_error(f"Benchmark ID {benchmark_id}, Item {item_idx}: 'topic' is empty or invalid")
            return False
                
        return True
        
    def validate_items(self, benchmark: Dict[str, Any], is_v1_1_0: bool = False) -> bool:
        """아이템들 검증"""
        benchmark_id = benchmark.get('id')
        items = benchmark.get('items', [])
        
        if not items:
            self.log_error(f"Benchmark ID {benchmark_id}: No items found")
            return False
        
        # 아이템 개수 확인 (모든 벤치마크는 50개 아이템을 가져야 함)
        expected_count = 50
        if len(items) != expected_count:
            if len(items) < expected_count:
                self.log_error(f"Benchmark ID {benchmark_id}: Expected {expected_count} items, got {len(items)}")
            else:
                self.log_warning(f"Benchmark ID {benchmark_id}: Expected {expected_count} items, got {len(items)} (more than expected)")
        else:
            self.log_pass(f"Benchmark ID {benchmark_id}: Item count validation passed ({len(items)} items)")
        
        # 각 아이템 구조 검증
        valid_items = 0
        for idx, item in enumerate(items):
            if is_v1_1_0:
                # v1.1.0 버전 검증
                if benchmark_id == 1:
                    # ID 1은 여전히 비교형
                    if self.validate_item_structure_v1_0_0(item, benchmark_id, idx):
                        valid_items += 1
                elif benchmark_id in [2, 3, 4]:
                    # ID 2, 3, 4는 단일 주제형 (topic + context)
                    if self.validate_item_structure_v1_1_0_domestic(item, benchmark_id, idx):
                        valid_items += 1
                elif benchmark_id == 5:
                    # ID 5는 단순형 (topic만)
                    if self.validate_item_structure_v1_1_0_simple(item, benchmark_id, idx):
                        valid_items += 1
            else:
                # v1.0.0 버전 검증 (모두 비교형)
                if self.validate_item_structure_v1_0_0(item, benchmark_id, idx):
                    valid_items += 1
        
        if valid_items == len(items):
            self.log_pass(f"Benchmark ID {benchmark_id}: All {valid_items} items passed validation")
            return True
        else:
            self.log_error(f"Benchmark ID {benchmark_id}: Only {valid_items}/{len(items)} items passed validation")
            return False
    
    def validate_problem_types_and_eval_goals(self, benchmark: Dict[str, Any]) -> bool:
        """문제 유형과 평가 목표 검증"""
        benchmark_id = benchmark.get('id')
        problem_types = benchmark.get('problem_types', [])
        eval_goals = benchmark.get('eval_goals', [])
        
        # 모든 벤치마크는 3개의 problem_types와 eval_goals를 가져야 함
        if len(problem_types) != 3:
            self.log_error(f"Benchmark ID {benchmark_id}: Expected 3 problem_types, got {len(problem_types)}")
            return False
            
        if len(eval_goals) != 3:
            self.log_error(f"Benchmark ID {benchmark_id}: Expected 3 eval_goals, got {len(eval_goals)}")
            return False
        
        # 내용이 비어있는지 확인
        for i, pt in enumerate(problem_types):
            if not pt or not pt.strip():
                self.log_error(f"Benchmark ID {benchmark_id}: problem_types[{i}] is empty")
                return False
                
        for i, eg in enumerate(eval_goals):
            if not eg or not eg.strip():
                self.log_error(f"Benchmark ID {benchmark_id}: eval_goals[{i}] is empty")
                return False
        
        self.log_pass(f"Benchmark ID {benchmark_id}: problem_types and eval_goals validation passed")
        return True
    
    def validate_file(self, file_name: str) -> bool:
        """벤치마크 파일 전체 검증"""
        print(f"\n{'='*60}")
        print(f"🔍 벤치마크 파일 검증: {file_name}")
        print(f"{'='*60}")
        
        try:
            benchmarks = load_benchmarks(file_name)
        except Exception as e:
            self.log_error(f"Failed to load benchmark file '{file_name}': {e}")
            return False
        
        if not benchmarks:
            self.log_error(f"Benchmark file '{file_name}' is empty")
            return False
        
        # 버전 확인
        is_v1_1_0 = 'v1.1.0' in file_name
        print(f"📋 버전: {'v1.1.0' if is_v1_1_0 else 'v1.0.0'}")
        print(f"📊 벤치마크 세트 개수: {len(benchmarks)}")
        
        # 기본 구조 검증
        expected_keys = {'id', 'problem_types', 'eval_goals', 'items'}
        all_valid = True
        
        # ID 중복 확인
        ids = [b.get('id') for b in benchmarks]
        if len(ids) != len(set(ids)):
            self.log_error(f"Duplicate benchmark IDs found in '{file_name}'")
            all_valid = False
        
        # 각 벤치마크 검증
        for benchmark in benchmarks:
            # 구조 검증
            if not self.validate_benchmark_structure(benchmark, expected_keys):
                all_valid = False
                continue
            
            # 문제 유형 및 평가 목표 검증
            if not self.validate_problem_types_and_eval_goals(benchmark):
                all_valid = False
            
            # 아이템 검증
            if not self.validate_items(benchmark, is_v1_1_0):
                all_valid = False
        
        return all_valid
    
    def print_summary(self):
        """검증 결과 요약 출력"""
        print(f"\n{'='*60}")
        print("📋 검증 결과 요약")
        print(f"{'='*60}")
        
        if self.passed_tests:
            print(f"\n✅ 성공한 테스트 ({len(self.passed_tests)}개):")
            for test in self.passed_tests:
                print(f"  {test}")
        
        if self.warnings:
            print(f"\n⚠️  경고 ({len(self.warnings)}개):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.errors:
            print(f"\n❌ 오류 ({len(self.errors)}개):")
            for error in self.errors:
                print(f"  {error}")
        else:
            print(f"\n🎉 모든 검증을 통과했습니다!")
        
        print(f"\n📊 전체 결과:")
        print(f"  - 성공: {len(self.passed_tests)}")
        print(f"  - 경고: {len(self.warnings)}")
        print(f"  - 오류: {len(self.errors)}")

def main():
    """메인 함수"""
    print("🚀 iSKA-Gen 벤치마크 검증 시작")
    
    # 사용 가능한 벤치마크 파일 목록 가져오기
    try:
        available_files = list_available_benchmarks()
        if not available_files:
            print("❌ 사용 가능한 벤치마크 파일이 없습니다.")
            return
        
        print(f"\n📂 발견된 벤치마크 파일 ({len(available_files)}개):")
        for i, file_name in enumerate(available_files, 1):
            print(f"  {i}. {file_name}")
        
    except Exception as e:
        print(f"❌ 벤치마크 파일 목록을 가져오는 중 오류 발생: {e}")
        return
    
    # 검증기 초기화
    validator = BenchmarkValidator()
    
    # 모든 파일 검증
    all_files_valid = True
    for file_name in available_files:
        if not validator.validate_file(file_name):
            all_files_valid = False
    
    # 결과 요약
    validator.print_summary()
    
    # 전체 결과
    if all_files_valid and not validator.errors:
        print(f"\n🎊 모든 벤치마크 파일이 성공적으로 검증되었습니다!")
        return 0
    else:
        print(f"\n💥 일부 벤치마크 파일에서 문제가 발견되었습니다.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
