# src/domain/entities/benchmark.py
from typing import List, Optional
from pydantic import AliasChoices, ConfigDict, Field
from src.domain.entities.base import DomainModel


class BenchmarkItem(DomainModel):
    """
    벤치마크의 개별 문항 엔티티
    """
    topic: Optional[str] = Field(None, description="문항 주제")
    context: Optional[str] = Field(None, description="문항 지문 내용")
    korean_topic: Optional[str] = Field(None, description="비교 문화의 한국어 주제")
    korean_context: Optional[str] = Field(None, description="비교 문화의 한국어 지문 내용")
    foreign_topic: Optional[str] = Field(None, description="비교 문화의 외국어 주제")
    foreign_context: Optional[str] = Field(None, description="비교 문화의 외국어 지문 내용")

    # Pydantic v2 방식
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "korean_topic": "회식 문화",
                "korean_context": "회식은 한국 직장 문화의 중요한 부분...",
                "foreign_topic": "Happy Hour Culture",
                "foreign_context": "Happy hour is a social tradition..."
            }
        },
        extra="forbid"
    )
        
class BenchmarkItemFlat(DomainModel):
    """벤치마크 아이템을 단일/복합 구분 없이 평탄화한 DTO"""
    topic: str = Field(..., description="메인 주제 (한국어 기준)")
    context: str = Field(..., description="메인 지문 본문 (한국어 기준)")
    foreign_topic: Optional[str] = Field(None, description="외국어 주제")
    foreign_context: Optional[str] = Field(None, description="외국어 본문")

    model_config = ConfigDict(extra="forbid")

class BenchmarkSet(DomainModel):
    """
    벤치마크 세트 엔티티 (유형, 평가 목표, 문항 포함)
    """
    id: int = Field(..., description="벤치마크 세트 고유 ID")
    problem_types: List[str] = Field(..., description="문제 유형 리스트")
    eval_goals: List[str] = Field(..., description="평가 목표 리스트")
    items: List[BenchmarkItem] = Field(..., description="세트에 포함된 개별 문항들")

    model_config = ConfigDict(extra="forbid")

class BenchmarkCollection(DomainModel):
    """
    전체 벤치마크 데이터 컬렉션
    """
    benchmarks: List[BenchmarkSet] = Field(
            validation_alias=AliasChoices("benchmark", "benchmarks"),
            serialization_alias="benchmarks",
            description="여러 벤치마크 세트",
        )
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

# ===== 사용 예시 =====
if __name__ == "__main__":
    import json
    from pathlib import Path

    # data 디렉토리의 benchmark.json 읽기
    benchmark_path = Path("src/data/benchmark.json")
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))

    benchmark_collection = BenchmarkCollection(benchmarks=data)

    # 첫 번째 벤치마크 세트 정보 출력
    print(benchmark_collection.benchmarks[0].problem_types)
    print(benchmark_collection.benchmarks[0].items[0].korean_topic)
