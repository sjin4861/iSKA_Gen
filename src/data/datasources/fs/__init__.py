"""
File System DataSources Module

이 모듈은 파일시스템 기반의 데이터소스들을 제공합니다.

핵심 구조:
- DataStoreFSDataSource: 통합 데이터 저장소 (핵심 구현체)
- 레거시 호환 래퍼들: ContentStoreFSDataSource, StemStoreFSDataSource, BenchmarkFSStore
- 전용 DataSources: RawOutputFSDataSource, EvaluationFSDataSource, TemplatesFSDataSource
- 공통 유틸리티: file_system, text_generation
"""

from __future__ import annotations

# 핵심 통합 데이터소스
from .data_store_fs import DataStoreFSDataSource

# 레거시 호환 래퍼들
from .content_store import ContentStoreFSDataSource
from .stem_store import StemStoreFSDataSource 
from .benchmark_store import BenchmarkFSStore

# 전용 데이터소스들
from .raw_output_fs import RawOutputFSDataSource
from .evaluation_fs import EvaluationFSDataSource
from .templates_fs import TemplatesFSDataSource
from .text_generation import TextGenerationDataSource

# 공통 유틸리티
from .file_system import (
    ensure_dir,
    read_json,
    write_json_atomic,
    list_files,
    merge_list_by_indices
)

__all__ = [
    # 핵심 통합 데이터소스
    "DataStoreFSDataSource",
    
    # 레거시 호환 래퍼들
    "ContentStoreFSDataSource", 
    "StemStoreFSDataSource",
    "BenchmarkFSStore",
    
    # 전용 데이터소스들
    "RawOutputFSDataSource",
    "EvaluationFSDataSource", 
    "TemplatesFSDataSource",
    "TextGenerationDataSource",
    
    # 공통 유틸리티
    "ensure_dir",
    "read_json", 
    "write_json_atomic",
    "list_files",
    "merge_list_by_indices",
]

# 타입 별칭들
ContentStore = ContentStoreFSDataSource  # 하위 호환성
StemStore = StemStoreFSDataSource        # 하위 호환성