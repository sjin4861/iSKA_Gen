from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from src.domain.entities.outputs import CandidateOutput
from src.domain.entities.enums import ContentType
from src.utils.settings_loader import get_settings
from .file_system import read_json, write_json_atomic, ensure_dir

class DataStoreFSDataSource:
    """
    data_store 디렉토리를 기반으로 한 파일시스템 데이터소스
    - raw_outputs: 날짜별/모델별/콘텐츠타입별 생성 결과 저장
    - benchmarks: 벤치마크 정의 파일들
    - evaluations: 평가 결과들
    """
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            settings = get_settings()
            base_path = settings.get('data_store_path', './data_store')
        
        self.base_path = Path(base_path)
        self.raw_outputs_path = self.base_path / "raw_outputs"
        self.benchmarks_path = self.base_path / "benchmarks"
        self.evaluations_path = self.base_path / "evaluations"
        
        # 디렉토리 생성
        ensure_dir(self.raw_outputs_path)
        ensure_dir(self.benchmarks_path)
        ensure_dir(self.evaluations_path)

    # === CandidateOutput 관련 ===
    
    def exists(self, *, source_id: str, benchmark_id: int, model_name: str, 
               kind: ContentType, date_str: Optional[str] = None) -> bool:
        """특정 소스에 대한 후보가 이미 존재하는지 확인"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # 파일 경로 구성: raw_outputs/{date}/{kind}/{model_name}/...
        candidates = self._load_candidates_from_date_model_kind(date_str, model_name, kind)
        
        for candidate in candidates:
            if (candidate.get('source_id') == source_id and 
                candidate.get('benchmark_id') == benchmark_id):
                return True
        return False
    
    def append_candidate(self, candidate: CandidateOutput, date_str: Optional[str] = None) -> None:
        """새로운 후보를 추가"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # 파일 경로 구성
        file_path = self._get_candidate_file_path(
            date_str, candidate.model_name, candidate.content_type, 
            candidate.benchmark_id, candidate.meta.get('benchmark_version', 'v1.0.0') if candidate.meta else 'v1.0.0',
            candidate.meta.get('template_key', 'default_template') if candidate.meta else 'default_template'
        )
        
        # 기존 데이터 로드
        existing_data = []
        if file_path.exists():
            existing_data = read_json(file_path) or []
        
        # 새 데이터 추가
        candidate_dict = {
            "source_id": candidate.source_id,  # 최소 필드
            "generated_passage": candidate.content,  # 호환성을 위해
            "content": candidate.content,
            "source_id": candidate.source_id,
            "benchmark_id": candidate.benchmark_id,
            "model_name": candidate.model_name,
            "candidate_id": candidate.candidate_id,
            "content_type": candidate.content_type.value,
            "generated_at": candidate.generated_at.isoformat() if candidate.generated_at else None,
            "stems": candidate.stems,
            "meta": candidate.meta,
            "source_item": candidate.source_item
        }
        
        existing_data.append(candidate_dict)
        
        # 파일 저장
        write_json_atomic(file_path, existing_data)
    
    def _load_candidates_from_date_model_kind(self, date_str: str, model_name: str, 
                                            kind: ContentType) -> List[Dict[str, Any]]:
        """특정 날짜/모델/종류에서 모든 후보들을 로드"""
        candidates = []
        
        # raw_outputs/{date}/{kind}/{model_name}/ 디렉토리 탐색
        model_dir = self.raw_outputs_path / date_str / kind.value / model_name
        if not model_dir.exists():
            return candidates
        
        # 모든 JSON 파일들을 탐색
        for json_file in model_dir.rglob("*.json"):
            data = read_json(json_file)
            if data:
                if isinstance(data, list):
                    candidates.extend(data)
                else:
                    candidates.append(data)
        
        return candidates
    
    def _get_candidate_file_path(self, date_str: str, model_name: str, content_type: ContentType,
                                benchmark_id: int, benchmark_version: str, template_key: str = None) -> Path:
        """후보 저장을 위한 파일 경로 생성"""
        # 예: raw_outputs/2025-08-08/passage/A.X-4.0-Light/passage_agent.create_dialogue_passage/
        #     benchmark_3_v1.1.0_passage_agent.create_dialogue_passage.json

        template_key = template_key or "default_template"  # 기본값, 필요시 매개변수로 받을 수 있음

        dir_path = (self.raw_outputs_path / date_str / content_type.value /
                   model_name / template_key)
        
        filename = f"benchmark_{benchmark_id}_{benchmark_version}_{template_key}.json"
        return dir_path / filename

    # === 구조화된 데이터 로드/저장 ===
    
    def load_passage_list(self, model: str, benchmark_id: int, version: str, 
                         template_key: str, date_str: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """지문 리스트 로드 (content_repository에서 사용)"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        file_path = self._get_passage_list_path(model, benchmark_id, version, template_key, date_str)
        return read_json(file_path)
    
    def save_passage_list(self, data: List[Dict[str, Any]], model: str, benchmark_id: int,
                         version: str, template_key: str, date_str: Optional[str]) -> None:
        """지문 리스트 저장"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        file_path = self._get_passage_list_path(model, benchmark_id, version, template_key, date_str)
        write_json_atomic(file_path, data)
    
    def patch_by_indices(self, model: str, benchmark_id: int, version: str,
                        template_key: str, patch: Dict[int, Dict[str, Any]], 
                        date_str: Optional[str]) -> None:
        """인덱스별로 데이터 패치"""
        data = self.load_passage_list(model, benchmark_id, version, template_key, date_str)
        if data is None:
            return
        
        for idx, patch_data in patch.items():
            if 0 <= idx < len(data):
                data[idx].update(patch_data)
        
        self.save_passage_list(data, model, benchmark_id, version, template_key, date_str)
    
    def find_null_indices(self, items: List[Dict[str, Any]]) -> List[int]:
        """None이거나 빈 값을 가진 인덱스들 찾기"""
        null_indices = []
        for i, item in enumerate(items):
            # 주요 필드들이 비어있거나 None인지 확인
            if (not item.get('generated_passage') and 
                not item.get('content') and
                not item.get('text')):
                null_indices.append(i)
        return null_indices
    
    def _get_passage_list_path(self, model: str, benchmark_id: int, version: str,
                              template_key: str, date_str: str) -> Path:
        """지문 리스트 파일 경로 생성"""
        # 예: raw_outputs/2025-08-08/passage_processed/A.X-4.0-Light/
        #     benchmark_3_v1.1.0_passage_agent.create_dialogue_passage.json
        
        dir_path = (self.raw_outputs_path / date_str / "passage_processed" / model)
        filename = f"benchmark_{benchmark_id}_{version}_{template_key}.json"
        return dir_path / filename
    
    # === 벤치마크 관련 ===
    
    def load_benchmark(self, benchmark_id: int, version: str) -> Optional[Dict[str, Any]]:
        """벤치마크 정의 로드"""
        # benchmarks/v1/iSKA-Gen_Benchmark_v1.1.0_20250808.json 같은 형태
        benchmark_files = list(self.benchmarks_path.rglob(f"*{version}*.json"))
        
        for file_path in benchmark_files:
            data = read_json(file_path)
            if data and data.get('benchmark_id') == benchmark_id:
                return data
        
        return None
    
    def save_benchmark(self, benchmark_data: Dict[str, Any], version: str, 
                      date_str: Optional[str] = None) -> None:
        """벤치마크 정의 저장"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        
        filename = f"iSKA-Gen_Benchmark_{version}_{date_str}.json"
        file_path = self.benchmarks_path / "v1" / filename
        write_json_atomic(file_path, benchmark_data)
    
    # === 평가 결과 관련 ===
    
    def save_evaluation_result(self, result_data: Dict[str, Any], rubric_id: str,
                              date_str: Optional[str] = None) -> None:
        """평가 결과 저장"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        filename = f"{rubric_id}_evaluation.json"
        file_path = self.evaluations_path / date_str / filename
        write_json_atomic(file_path, result_data)
    
    def load_evaluation_result(self, rubric_id: str, 
                              date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """평가 결과 로드"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        file_path = self.evaluations_path / date_str / f"{rubric_id}_evaluation.json"
        return read_json(file_path)
    
    # === 유틸리티 메서드들 ===
    
    def create_file_path(self, content_type: ContentType, model_name: str, 
                        template_key: str, benchmark_id: int, benchmark_version: str,
                        date_str: Optional[str] = None) -> Path:
        """표준화된 파일 경로 생성"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        return self._get_candidate_file_path(
            date_str, model_name, content_type, benchmark_id, benchmark_version, template_key
        )
    
    def get_file_name_pattern(self, benchmark_id: int, benchmark_version: str, 
                             template_key: str) -> str:
        """표준화된 파일명 패턴"""
        return f"benchmark_{benchmark_id}_{benchmark_version}_{template_key}.json"