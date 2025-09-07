from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, TYPE_CHECKING
from src.domain.entities.evaluation import EvaluationRecord
from src.domain.entities.evaluation_query import EvaluationQuery

if TYPE_CHECKING:
    from src.domain.usecases.evaluation.evaluate import EvaluateInput, EvaluateOutput
    from src.domain.usecases.evaluation.evaluate_stems import EvaluateStemsInput, EvaluateStemsOutput

class EvaluationRepository(ABC):
    """
    채점 결과(EvaluationRecord)를 영속화하는 추상 포트.
    파일(JSONL), RDB, 벡터DB 등 구현체는 data 계층에 둔다.
    """
    
    # TODO: [ARCHITECTURE] 다음 메서드들을 domain/usecases/evaluation/로 분리 필요:
    # - save -> SaveEvaluationRecordUseCase
    # - bulk_save -> BulkSaveEvaluationRecordsUseCase  
    # - find -> FindEvaluationRecordsUseCase
    # - count -> CountEvaluationRecordsUseCase
    # evaluate, evaluate_stems는 이미 UseCase로 존재함 (evaluate.py, evaluate_stems.py)
    # Repository는 순수한 데이터 접근 인터페이스만 제공해야 함
    
    @abstractmethod
    def save(self, record: EvaluationRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def bulk_save(self, records: Iterable[EvaluationRecord]) -> None:
        raise NotImplementedError
    
    # ✅ 조회 추가
    @abstractmethod
    def find(self, query: EvaluationQuery) -> Iterable[EvaluationRecord]:
        """
        조건에 맞는 EvaluationRecord를 스트리밍으로 반환한다.
        구현체는 저장소별 최적화된 스캔/인덱싱 전략을 사용할 수 있다.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> Optional[int]:
        return None

    @abstractmethod
    def evaluate(self, inp: 'EvaluateInput') -> 'EvaluateOutput':
        """
        평가 입력에 따라 평가 결과를 생성한다.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_stems(self, inp: 'EvaluateStemsInput') -> 'EvaluateStemsOutput':
        """
        여러 루브릭에 대해 stem 평가를 수행한다.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_stems_with_shared_client(self, inp: 'EvaluateStemsInput', shared_client) -> 'EvaluateStemsOutput':
        """
        공유 클라이언트를 사용하여 여러 루브릭에 대해 stem 평가를 수행한다. (CUDA 재초기화 방지)
        """
        raise NotImplementedError