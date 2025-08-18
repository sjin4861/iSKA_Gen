from __future__ import annotations
from typing import Iterable, Optional

from src.domain.repositories.evaluation_repository import EvaluationRepository
from src.domain.entities.evaluation import EvaluationRecord
from src.domain.entities.evaluation_query import EvaluationQuery
from src.domain.usecases.evaluation.evaluate import EvaluateInput, EvaluateOutput
from src.domain.usecases.evaluation.evaluate_stems import EvaluateStemsInput, EvaluateStemsOutput
from src.domain.entities.enums import EvaluatorType

from src.data.datasources.fs.evaluation_fs import EvaluationFSDataSource


class EvaluationRepositoryImpl(EvaluationRepository):
    """
    도메인 리포 인터페이스 구현체.
    내부적으로 FS 데이터소스(EvaluationFSDataSource)를 사용해
    data_store/evaluations/ 경로에 JSONL append/scan을 수행한다.
    """

    def __init__(self, root_dir: str = "data_store"):
        self.ds = EvaluationFSDataSource(root_dir)

    # --- 쓰기 ---
    def save(self, record: EvaluationRecord) -> None:
        # record.run_id가 없으면 'misc' 폴더로 떨어진다.
        self.ds.append(record, run_id=record.run_id)

    def bulk_save(self, records: Iterable[EvaluationRecord]) -> None:
        # 서로 다른 run_id가 섞여 있어도 DS가 경로를 나눠서 append한다.
        self.ds.bulk_append(records)

    # --- 읽기 ---
    def find(self, query: EvaluationQuery):
        return self.ds.find(query)

    # --- 선택: 전체 개수(간단 구현 - full scan) ---
    def count(self) -> Optional[int]:
        # 주의: 데이터가 많으면 느릴 수 있음. 필요 시 인덱스/메타 유지로 최적화.
        total = 0
        for _ in self.ds.find(EvaluationQuery()):
            total += 1
        return total

    def evaluate(self, inp: EvaluateInput) -> EvaluateOutput:
        """
        평가 입력에 따라 실제 평가를 수행하고 결과를 저장소에 저장
        """
        # 실제 평가 수행
        output = self.ds.evaluate(inp)
        
        # 평가 결과를 저장소에 저장
        if output.records:
            self.bulk_save(output.records)
            print(f"✅ {len(output.records)}개의 평가 결과가 저장되었습니다.")
        
        return output

    def evaluate_stems(self, inp):
        """
        여러 루브릭에 대해 stem 평가를 수행
        """
        from src.domain.usecases.evaluation.evaluate_stems import EvaluateStemsOutput
        from src.domain.entities.enums import EvaluatorType
        
        # 🔧 모델 클라이언트를 한 번만 생성하여 모든 루브릭에서 재사용
        print(f"🤖 평가 모델 클라이언트 생성: {inp.evaluator_model}")
        try:
            from src.modules.model_client import create_model_client
            
            # 모델명에 따라 클라이언트 타입 결정
            if inp.evaluator_model.startswith(("gpt-4", "gpt-3.5", "claude-", "gemini-")):
                client_type = "openai"
            elif inp.evaluator_model.startswith("http"):
                client_type = "vllm"
            else:
                client_type = "local"
            
            print(f"   📱 클라이언트 타입: {client_type}")
            
            shared_client = create_model_client(
                client_type=client_type,
                model_name=inp.evaluator_model,
                temperature=inp.temperature,
                max_new_tokens=inp.max_tokens,
                gpus=[0, 1, 2]
            )
            
            print(f"✅ 평가 모델 준비 완료 - 모든 루브릭에서 재사용")
            
        except Exception as e:
            print(f"❌ 평가 모델 클라이언트 생성 실패: {e}")
            return EvaluateStemsOutput(
                evaluations=[],
                total_success=0,
                total_failed=len(inp.rubric_ids) * len(inp.stem_candidates),
                total_count=len(inp.rubric_ids) * len(inp.stem_candidates)
            )
        
        evaluations = []
        total_success = 0
        total_failed = 0
        total_count = 0
        
        for rubric_id in inp.rubric_ids:
            print(f"\n🔍 루브릭 '{rubric_id.value}' 평가 시작...")
            
            evaluate_input = EvaluateInput(
                candidates=inp.stem_candidates,
                rubric_id=rubric_id,
                evaluator_type=EvaluatorType.LLM,
                model_name=inp.evaluator_model,
                run_id=inp.run_id,
                temperature=inp.temperature,
                max_tokens=inp.max_tokens
            )
            
            # 🔧 공유 클라이언트를 사용하여 평가 수행
            evaluation_output = self.ds.evaluate_with_client(evaluate_input, shared_client)
            
            # 평가 결과 저장
            if evaluation_output.records:
                self.bulk_save(evaluation_output.records)
                print(f"   💾 {len(evaluation_output.records)}개 결과 저장됨")
            
            evaluations.append(evaluation_output)
            
            total_success += evaluation_output.success_count
            total_failed += evaluation_output.failed_count
            total_count += evaluation_output.total_count
            
            print(f"✅ 루브릭 '{rubric_id.value}' 완료: {evaluation_output.success_count}/{evaluation_output.total_count} 성공")
        
        return EvaluateStemsOutput(
            evaluations=evaluations,
            total_success=total_success,
            total_failed=total_failed,
            total_count=total_count
        )
    
    def evaluate_stems_with_shared_client(self, inp: EvaluateStemsInput, shared_client) -> EvaluateStemsOutput:
        """
        외부에서 전달받은 공유 클라이언트를 사용하여 여러 루브릭에 대해 stem 평가를 수행
        (CUDA 재초기화 방지)
        """
        print(f"\n🔧 공유 클라이언트 사용하여 {len(inp.rubric_ids)}개 루브릭 평가 수행")
        
        evaluations = []
        total_success = 0
        total_failed = 0
        total_count = 0
        
        for rubric_id in inp.rubric_ids:
            print(f"\n🔍 루브릭 '{rubric_id.value}' 평가 시작...")
            
            evaluate_input = EvaluateInput(
                candidates=inp.stem_candidates,
                rubric_id=rubric_id,
                evaluator_type=EvaluatorType.LLM,
                model_name=inp.evaluator_model,
                run_id=inp.run_id,
                temperature=inp.temperature,
                max_tokens=inp.max_tokens
            )
            
            # 🔧 전달받은 공유 클라이언트를 사용하여 평가 수행
            evaluation_output = self.ds.evaluate_with_client(evaluate_input, shared_client)
            
            # 평가 결과 저장
            if evaluation_output.records:
                self.bulk_save(evaluation_output.records)
                print(f"   💾 {len(evaluation_output.records)}개 결과 저장됨")
            
            evaluations.append(evaluation_output)
            
            total_success += evaluation_output.success_count
            total_failed += evaluation_output.failed_count
            total_count += evaluation_output.total_count
            
            print(f"✅ 루브릭 '{rubric_id.value}' 완료: {evaluation_output.success_count}/{evaluation_output.total_count} 성공")
        
        print(f"\n🏁 전체 평가 완료 - 공유 클라이언트 사용")
        
        return EvaluateStemsOutput(
            evaluations=evaluations,
            total_success=total_success,
            total_failed=total_failed,
            total_count=total_count
        )
        
