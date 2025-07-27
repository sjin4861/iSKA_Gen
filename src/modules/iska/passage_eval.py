"""
지문 평가 모듈 (Passage Evaluator)

생성된 지문의 품질을 다각도로 평가하는 모듈:
- 이진 루브릭: 5가지 항목 Pass/Fail 평가 (빠른 검수)
- 상세 점수: 일관성, 일치성, 자연스러움, 한국어 품질 (1-5점)
- OpenAI 기반, 추후 Reward Model 지원 예정
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

# 간단한 레벤슈타인 거리 구현
def levenshtein_distance(s1: str, s2: str) -> int:
    """간단한 레벤슈타인 거리 계산"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

# 런타임에 import
try:
    from ..model_client import BaseModelClient, OpenAIModelClient
    from ..client_factory import ModelClientFactory
except ImportError:
    try:
        from src.modules.model_client import BaseModelClient, OpenAIModelClient
        from src.modules.client_factory import ModelClientFactory
    except ImportError:
        from modules.model_client import BaseModelClient, OpenAIModelClient
        from modules.client_factory import ModelClientFactory

try:
    from src.utils.prompt_loader import get_prompt
    from src.utils.settings_loader import get_settings
except ImportError:
    from utils.prompt_loader import get_prompt
    from utils.settings_loader import get_settings

class PassageEvaluator:
    """
    지문 평가를 위한 클래스 - OpenAI 기반 평가 시스템
    
    주요 기능:
    - 일관성, 사실 일치성, 자연스러움, 한국어 품질 평가
    - 레벤슈타인 거리 기반 정량적 평가
    - 추후 Reward Model로 교체 가능한 구조
    """
    
    def __init__(self, llm_client: Optional[BaseModelClient] = None, template_key: str = "iska", **kwargs):
        """
        PassageEvaluator 초기화
        
        Args:
            llm_client: 평가에 사용할 LLM 클라이언트 (None이면 기본 OpenAI 클라이언트 생성)
            template_key: 프롬프트 템플릿 키 (기본: "iska")
            **kwargs: 클라이언트 생성 시 추가 파라미터
        """
        self.template_key = template_key
        
        if llm_client is None:
            print("🔧 기본 OpenAI 클라이언트를 생성합니다...")
            try:
                # 설정에서 기본 모델 정보 가져오기
                try:
                    settings = get_settings()
                    openai_config = settings.get('external_services', {}).get('openai', {})
                    model_name = openai_config.get('model', 'gpt-4o-mini')
                except:
                    model_name = 'gpt-4o-mini'
                
                self.llm_client = ModelClientFactory.create_openai_client(
                    model_name=model_name,
                    **kwargs
                )
                print(f"✅ OpenAI 클라이언트 생성 완료: {model_name}")
            except Exception as e:
                print(f"⚠️ OpenAI 클라이언트 생성 실패: {e}")
                print("🔄 기본 설정으로 재시도...")
                try:
                    self.llm_client = ModelClientFactory.create_openai_client()
                except Exception as e2:
                    print(f"❌ 기본 클라이언트 생성도 실패: {e2}")
                    raise RuntimeError(f"클라이언트 생성 실패: {e}")
        else:
            self.llm_client = llm_client
            print("✅ 지문 평가기가 사용자 제공 클라이언트로 초기화되었습니다.")
        
        # 평가 지표별 기본 설정
        self.evaluation_config = {
            "coherence": {"weight": 0.3, "max_score": 5},
            "consistency": {"weight": 0.3, "max_score": 5},  # factual_consistency -> consistency
            "naturalness": {"weight": 0.2, "max_score": 5},
            "korean_quality": {"weight": 0.2, "max_score": 5}
        }

    def calculate_normalized_score(self, original_text: str, corrected_text: str) -> float:
        """
        레벤슈타인 거리를 이용해 0~1 사이의 정규화된 점수를 계산합니다.
        1.0에 가까울수록 완벽한 문장입니다.
        """
        dist = levenshtein_distance(original_text, corrected_text)
        max_len = max(len(original_text), len(corrected_text))
        if max_len == 0:
            return 1.0
        return 1.0 - (dist / max_len)

    def scale_to_1_5(self, normalized_score: float) -> int:
        """0~1 사이의 점수를 1~5점 척도로 변환합니다."""
        if normalized_score >= 0.99:
            return 5  # 거의 완벽
        elif normalized_score >= 0.95:
            return 4  # 사소한 오류
        elif normalized_score >= 0.90:
            return 3  # 눈에 띄는 오류
        elif normalized_score >= 0.80:
            return 2  # 많은 오류
        else:
            return 1  # 심각한 오류

    def evaluate_coherence(self, passage: str, problem_types: List[str], eval_goals: List[str], 
                          template_key: Optional[str] = None) -> int:
        """
        일관성 평가를 수행합니다.
        
        Args:
            passage: 평가할 지문
            problem_types: 문제 유형 리스트
            eval_goals: 평가 목표 리스트  
            template_key: 프롬프트 템플릿 키 (None이면 기본값 사용)
            
        Returns:
            int: 일관성 점수 (1-5점)
        """
        template_key = template_key or self.template_key
        
        try:
            prompt = get_prompt(
                'passage_eval.coherence',
                agent='iska',
                problem_type1=problem_types[0], eval_goal1=eval_goals[0],
                problem_type2=problem_types[1], eval_goal2=eval_goals[1],
                problem_type3=problem_types[2], eval_goal3=eval_goals[2],
                passage=passage
            )
            
            response = self.llm_client.call([{"role": "user", "content": prompt}])
            
            # 점수 파싱
            score = self._parse_score_from_response(response, "coherence")
            
            return score if score is not None else 3
            
        except Exception as e:
            print(f"❌ 일관성 평가 중 오류 발생: {e}")
            return 3

    def evaluate_consistency(self, passage: str, home_context: Optional[str] = None, 
                           foreign_context: Optional[str] = None, template_key: Optional[str] = None) -> int:
        """
        일치성 평가를 수행합니다 (컨텍스트 기반).
        
        Args:
            passage: 평가할 지문
            home_context: 한국 컨텍스트 (None 가능)
            foreign_context: 외국 컨텍스트 (None 가능)
            template_key: 프롬프트 템플릿 키
            
        Returns:
            int: 일치성 점수 (1-5점)
        """
        template_key = template_key or self.template_key
        
        try:
            prompt = get_prompt(
                'passage_eval.consistency',
                agent='iska',
                home_context=home_context or "N/A",
                foreign_context=foreign_context or "N/A",
                passage=passage
            )
            
            response = self.llm_client.call([{"role": "user", "content": prompt}])
            
            # 점수 파싱
            score = self._parse_score_from_response(response, "consistency")
            
            # 점수 범위 제한
            if score is not None:
                score = min(score, 5)
                score = max(score, 1)
                return score
            else:
                return 3  # 기본 점수
            
        except Exception as e:
            print(f"❌ 일치성 평가 중 오류 발생: {e}")
            return 3  # 오류 시 기본 점수

    def evaluate_naturalness(self, passage: str, home_topic: str, foreign_topic: str, 
                           template_key: Optional[str] = None) -> int:
        """
        자연스러움 평가를 수행합니다.
        
        Args:
            passage: 평가할 지문
            home_topic: 한국 주제
            foreign_topic: 외국 주제
            template_key: 프롬프트 템플릿 키
            
        Returns:
            int: 자연스러움 점수 (1-5점)
        """
        template_key = template_key or self.template_key
        
        try:
            prompt = get_prompt(
                'passage_eval.naturalness',
                agent='iska',
                home_topic=home_topic,
                foreign_topic=foreign_topic,
                passage=passage
            )
            
            response = self.llm_client.call([{"role": "user", "content": prompt}])
            
            # 점수 파싱
            score = self._parse_score_from_response(response, "naturalness")
            
            return score if score is not None else 3
            
        except Exception as e:
            print(f"❌ 자연스러움 평가 중 오류 발생: {e}")
            return 3

    def evaluate_korean_quality(self, passage: str, template_key: Optional[str] = None) -> int:
        """
        한국어 품질 평가를 수행합니다.
        
        Args:
            passage: 평가할 지문
            template_key: 프롬프트 템플릿 키
            
        Returns:
            int: 한국어 품질 점수 (1-5점)
        """
        template_key = template_key or self.template_key
        
        try:
            prompt = get_prompt(
                'passage_eval.korean_quality',
                agent='iska',
                passage=passage
            )
            
            response = self.llm_client.call([{"role": "user", "content": prompt}])
            
            # 구조화된 텍스트 응답에서 교정된 문장 추출
            original_text = passage.strip()
            corrected_text = self._extract_corrected_passage(response, original_text)
            
            # Calculate normalized score and convert to 1-5 scale
            normalized_score = self.calculate_normalized_score(original_text, corrected_text)
            grammar_score = self.scale_to_1_5(normalized_score)
            
            return grammar_score
            
        except Exception as e:
            print(f"❌ 한국어 품질 평가 중 오류 발생: {e}")
            return 5  # 오류 시 만점으로 간주 (오류 없음으로 판단)

    def evaluate_binary_rubric(self, passage: str, problem_types: List[str], eval_goals: List[str],
                              korean_topic: str, foreign_topic: str, 
                              korean_context: str, foreign_context: str,
                              template_key: Optional[str] = None) -> Dict:
        """
        이진 루브릭을 사용한 종합 평가를 수행합니다.
        
        Args:
            passage: 평가할 지문
            problem_types: 문제 유형 리스트 (3개)
            eval_goals: 평가 목표 리스트 (3개)
            korean_topic: 한국 주제
            foreign_topic: 외국 주제
            korean_context: 한국 참고 자료
            foreign_context: 외국 참고 자료
            template_key: 프롬프트 템플릿 키
            
        Returns:
            Dict: 이진 평가 결과 (true/false + feedback)
        """
        template_key = template_key or self.template_key
        
        try:
            print("🔍 이진 루브릭 평가를 시작합니다...")
            
            # 3개의 문제 유형과 평가 목표가 필요
            if len(problem_types) < 3 or len(eval_goals) < 3:
                print("⚠️ 문제 유형과 평가 목표가 각각 최소 3개씩 필요합니다.")
                return {
                    "scores": {
                        "is_guideline_compliant": False,
                        "is_factually_consistent": False,
                        "is_natural": False,
                        "has_high_korean_quality": False,
                        "is_learner_appropriate": False
                    },
                    "feedback": "평가 메타데이터가 불충분합니다. 문제 유형과 평가 목표가 각각 3개씩 필요합니다."
                }
            
            prompt = get_prompt(
                template_key,
                agent='iska',
                problem_type1=problem_types[0], eval_goal1=eval_goals[0],
                problem_type2=problem_types[1], eval_goal2=eval_goals[1],
                problem_type3=problem_types[2], eval_goal3=eval_goals[2],
                korean_topic=korean_topic, korean_context=korean_context,
                foreign_topic=foreign_topic, foreign_context=foreign_context,
                passage=passage
            )
            
            response = self.llm_client.call([{"role": "user", "content": prompt}])
            # print(response)
            # JSON 응답 파싱
            result = self._parse_binary_rubric_response(response)
            
            print("✅ 이진 루브릭 평가 완료!")
            return result
            
        except Exception as e:
            print(f"❌ 이진 루브릭 평가 중 오류 발생: {e}")
            return {
                "scores": {
                    "is_guideline_compliant": False,
                    "is_factually_consistent": False,
                    "is_natural": False,
                    "has_high_korean_quality": False,
                    "is_learner_appropriate": False
                },
                "feedback": f"평가 중 오류가 발생했습니다: {str(e)}"
            }

    def _parse_binary_rubric_response(self, response: str) -> Dict:
        """이진 루브릭 응답에서 JSON을 파싱합니다."""
        import json
        import re
        
        try:
            # JSON 부분만 추출
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # 필수 평가 항목 키들
                required_keys = [
                    "is_guideline_compliant", "is_factually_consistent", "is_natural", 
                    "has_high_korean_quality", "is_learner_appropriate"
                ]
                
                # Case 1: scores 키가 있는 경우 (중첩 구조)
                if "scores" in result and isinstance(result["scores"], dict):
                    scores = result["scores"]
                    if all(key in scores for key in required_keys):
                        # boolean 값 확인 및 변환
                        for key in required_keys:
                            if isinstance(scores[key], str):
                                scores[key] = scores[key].lower() == "true"
                            elif not isinstance(scores[key], bool):
                                scores[key] = False
                        
                        return {
                            "scores": scores,
                            "feedback": result.get("feedback", "")
                        }
                
                # Case 2: 평가 항목이 최상위에 직접 있는 경우 (플랫 구조)
                elif all(key in result for key in required_keys):
                    scores = {}
                    for key in required_keys:
                        if isinstance(result[key], str):
                            scores[key] = result[key].lower() == "true"
                        elif isinstance(result[key], bool):
                            scores[key] = result[key]
                        else:
                            scores[key] = False
                    
                    return {
                        "scores": scores,
                        "feedback": result.get("feedback", "")
                    }
            
            # JSON 파싱 실패 시 기본값 반환
            print("⚠️ JSON 파싱에 실패했습니다. 기본값을 반환합니다.")
            print(f"🔍 파싱 시도한 JSON: {json_str if 'json_str' in locals() else 'JSON 추출 실패'}")
            return self._get_default_binary_result("JSON 파싱 실패")
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 디코딩 오류: {e}")
            return self._get_default_binary_result("JSON 형식 오류")
        except Exception as e:
            print(f"⚠️ 응답 파싱 중 예외 발생: {e}")
            return self._get_default_binary_result(f"파싱 오류: {str(e)}")

    def _get_default_binary_result(self, error_message: str) -> Dict:
        """기본 이진 평가 결과를 반환합니다."""
        return {
            "scores": {
                "is_guideline_compliant": False,
                "is_factually_consistent": False,
                "is_natural": False,
                "has_high_korean_quality": False,
                "is_learner_appropriate": False
            },
            "feedback": error_message
        }

    def evaluate_passage_metrics(self, passage: str, problem_types: List[str], eval_goals: List[str], 
                                home_context: Optional[str] = None, foreign_context: Optional[str] = None,
                                home_topic: str = "", foreign_topic: str = "",
                                template_key: Optional[str] = None) -> Dict:
        """
        모든 평가 지표를 종합적으로 실행합니다.
        
        Args:
            passage: 평가할 지문
            problem_types: 문제 유형 리스트
            eval_goals: 평가 목표 리스트
            home_context: 한국 컨텍스트 (선택적)
            foreign_context: 외국 컨텍스트 (선택적)
            home_topic: 한국 주제
            foreign_topic: 외국 주제
            template_key: 프롬프트 템플릿 키
            
        Returns:
            Dict: 종합 평가 결과
        """
        template_key = template_key or self.template_key
        
        print("🔍 지문 종합 평가를 시작합니다...")
        
        results = {}
        
        # 각 평가 지표별 실행
        print("   📊 일관성 평가 중...")
        results['coherence_score'] = self.evaluate_coherence(passage, problem_types, eval_goals, template_key)
        
        print("   📋 일치성 평가 중...")
        results['consistency_score'] = self.evaluate_consistency(passage, home_context, foreign_context, template_key)
        
        print("   🌊 자연스러움 평가 중...")
        results['naturalness_score'] = self.evaluate_naturalness(passage, home_topic, foreign_topic, template_key)
        
        print("   🇰🇷 한국어 품질 평가 중...")
        results['korean_quality_score'] = self.evaluate_korean_quality(passage, template_key)
        
        # 종합 점수 계산
        overall_score = self._calculate_overall_score_simple(results)
        
        # 메타 정보 추가
        results['overall_score'] = overall_score
        results['template_used'] = template_key
        results['evaluation_timestamp'] = self._get_timestamp()
        
        print("✅ 지문 종합 평가 완료!")
        return results

    def evaluate_passage_comprehensive(self, passage: str, problem_types: List[str], eval_goals: List[str],
                                      korean_topic: str, foreign_topic: str,
                                      korean_context: str, foreign_context: str,
                                      use_binary_rubric: bool = True,
                                      template_key: Optional[str] = None) -> Dict:
        """
        지문에 대한 종합적인 평가를 수행합니다 (이진 루브릭 + 기존 점수 평가).
        
        Args:
            passage: 평가할 지문
            problem_types: 문제 유형 리스트 (3개)
            eval_goals: 평가 목표 리스트 (3개)
            korean_topic: 한국 주제
            foreign_topic: 외국 주제
            korean_context: 한국 참고 자료
            foreign_context: 외국 참고 자료
            use_binary_rubric: 이진 루브릭 사용 여부
            template_key: 프롬프트 템플릿 키
            
        Returns:
            Dict: 종합 평가 결과
        """
        template_key = template_key or self.template_key
        
        print("🔍 지문 종합 평가를 시작합니다...")
        
        results = {
            "evaluation_type": "comprehensive",
            "template_used": template_key,
            "evaluation_timestamp": self._get_timestamp()
        }
        
        # 1. 이진 루브릭 평가 (우선 실행)
        if use_binary_rubric:
            print("   📋 이진 루브릭 평가 중...")
            binary_result = self.evaluate_binary_rubric(
                passage=passage,
                problem_types=problem_types,
                eval_goals=eval_goals,
                korean_topic=korean_topic,
                foreign_topic=foreign_topic,
                korean_context=korean_context,
                foreign_context=foreign_context,
                template_key=template_key
            )
            results["binary_evaluation"] = binary_result
            
            # 이진 평가가 모두 true인 경우에만 상세 점수 평가 진행
            all_passed = all(binary_result["scores"].values())
            if not all_passed:
                print("⚠️ 이진 루브릭에서 실패 항목이 있어 상세 평가를 건너뜁니다.")
                results["detailed_evaluation"] = {
                    "skipped": True,
                    "reason": "이진 루브릭 평가에서 실패 항목 존재"
                }
                results["overall_passed"] = False
                return results
        
        # 2. 상세 점수 평가 (이진 루브릭 통과 시에만)
        print("   📊 상세 점수 평가 중...")
        detailed_result = self.evaluate_passage_metrics(
            passage=passage,
            problem_types=problem_types,
            eval_goals=eval_goals,
            home_context=korean_context,
            foreign_context=foreign_context,
            home_topic=korean_topic,
            foreign_topic=foreign_topic,
            template_key=template_key
        )
        results["detailed_evaluation"] = detailed_result
        results["overall_passed"] = True
        
        print("✅ 지문 종합 평가 완료!")
        return results

    def get_binary_evaluation_summary(self, results: Dict) -> Dict:
        """이진 평가 결과 요약을 생성합니다."""
        if "binary_evaluation" not in results:
            return {"error": "이진 평가 결과가 없습니다."}
        
        binary_eval = results["binary_evaluation"]
        scores = binary_eval.get("scores", {})
        
        # 통과/실패 개수 계산
        passed_count = sum(1 for value in scores.values() if value)
        total_count = len(scores)
        failed_items = [key for key, value in scores.items() if not value]
        
        # 한국어 항목명 매핑
        korean_labels = {
            "is_guideline_compliant": "평가 지침 부합성",
            "is_factually_consistent": "사실 기반성",
            "is_natural": "연결 자연스러움",
            "has_high_korean_quality": "한국어 품질",
            "is_learner_appropriate": "L2 학습자 적합성"
        }
        
        summary = {
            "전체 통과율": f"{passed_count}/{total_count}",
            "통과 여부": passed_count == total_count,
            "검수 항목별 결과": {
                korean_labels.get(key, key): "✅ 통과" if value else "❌ 실패"
                for key, value in scores.items()
            },
            "실패 항목": [korean_labels.get(item, item) for item in failed_items],
            "피드백": binary_eval.get("feedback", ""),
            "평가 시간": results.get("evaluation_timestamp", "N/A")
        }
        
        return summary

    def _parse_score_from_response(self, response: str, evaluation_type: str) -> Optional[int]:
        """응답에서 점수를 파싱합니다."""
        import re
        
        # 다양한 점수 패턴 시도
        score_patterns = [
            r'(\d+)점\s*\([^)]+\)',  # "4점 (자연스러움)" format
            r'(\d+)점',  # "4점" format
            r':\s*(\d+)',  # ": 4" format
            r'(\d+)\s*\(',  # "4 (매우 적합)" format
            r'(\d+)$'  # Just number at end
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    score = int(matches[-1] if evaluation_type == "coherence" else matches[0])
                    if 1 <= score <= 5:  # 유효한 점수 범위 확인
                        return score
                except ValueError:
                    continue
        
        # 마지막 시도: 첫 번째 또는 마지막 줄에서 숫자 찾기
        try:
            if evaluation_type == "coherence":
                score_line = response.strip().split('\n')[-1]
            else:
                score_line = response.strip().split('\n')[0]
            
            score_match = re.search(r'(\d+)', score_line)
            if score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 5:
                    return score
        except:
            pass
        
        return None

    def _extract_corrected_passage(self, response: str, original_text: str) -> str:
        """구조화된 텍스트 응답에서 교정된 문장을 추출합니다."""
        import re
        
        # "교정된 문장:" 뒤의 내용 추출
        corrected_match = re.search(r'\*\*교정된 문장:\*\*\s*\n?(.*?)(?=\n\*\*|$)', response, re.DOTALL)
        if corrected_match:
            corrected_text = corrected_match.group(1).strip()
            # 괄호나 부연설명 제거
            corrected_text = re.sub(r'\(.*?\)', '', corrected_text).strip()
            if corrected_text and corrected_text != original_text:
                return corrected_text
        
        # "오류 없음" 체크
        if "오류 없음" in response or "오류가 없" in response:
            return original_text
        
        # 추출 실패 시 원본 반환
        return original_text

    def _extract_explanation(self, response: str, score: Optional[int]) -> str:
        """응답에서 상세 설명을 추출합니다."""
        if score is None:
            return response.strip()
        
        score_text = f"{score}점"
        if score_text in response:
            parts = response.split(score_text, 1)
            if len(parts) > 1:
                return parts[1].strip().lstrip(':').strip()
        
        return response.strip()

    def _calculate_overall_score_simple(self, results: Dict) -> float:
        """가중치를 적용하여 전체 점수를 계산합니다 (단순 버전)."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        score_keys = {
            "coherence_score": 0.3,
            "consistency_score": 0.3,
            "naturalness_score": 0.2,
            "korean_quality_score": 0.2
        }
        
        for score_key, weight in score_keys.items():
            if score_key in results and isinstance(results[score_key], (int, float)):
                total_weighted_score += results[score_key] * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _get_timestamp(self) -> str:
        """현재 타임스탬프를 반환합니다."""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_evaluation_summary(self, results: Dict) -> Dict:
        """평가 결과 요약을 생성합니다."""
        if 'overall_score' not in results:
            return {"error": "전체 평가 결과가 없습니다."}
        
        summary = {
            "총점": results['overall_score'],
            "평가 항목별 점수": {
                "일관성": results.get('coherence_score', 'N/A'),
                "일치성": results.get('consistency_score', 'N/A'),
                "자연스러움": results.get('naturalness_score', 'N/A'),
                "한국어 품질": results.get('korean_quality_score', 'N/A')
            },
            "평가 시간": results.get('evaluation_timestamp', 'N/A'),
            "사용된 템플릿": results.get('template_used', 'N/A')
        }
        
        return summary

# ========================= 편의 기능들 =========================

def create_passage_evaluator(model_type: str = "openai", model_name: Optional[str] = None, 
                            template_key: str = "iska", **kwargs) -> PassageEvaluator:
    """
    지문 평가기를 생성하는 편의 함수
    
    Args:
        model_type: 모델 타입 ("openai", "local")
        model_name: 모델 이름 (None이면 기본값 사용)
        template_key: 프롬프트 템플릿 키
        **kwargs: 추가 파라미터
        
    Returns:
        PassageEvaluator: 초기화된 평가기
    """
    if model_type == "openai":
        client = ModelClientFactory.create_openai_client(
            model_name=model_name or "gpt-4o-mini", **kwargs
        )
    elif model_type == "local":
        # 추후 Reward Model 사용 시
        client = ModelClientFactory.create_local_client(
            model_name=model_name or "reward-model-korean", **kwargs
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    return PassageEvaluator(llm_client=client, template_key=template_key)


def create_binary_passage_evaluator(model_type: str = "openai", model_name: Optional[str] = None, 
                                   template_key: str = "iska", **kwargs) -> PassageEvaluator:
    """
    이진 루브릭 전용 지문 평가기를 생성하는 편의 함수
    
    Args:
        model_type: 모델 타입 ("openai", "local")
        model_name: 모델 이름 (None이면 기본값 사용)
        template_key: 프롬프트 템플릿 키
        **kwargs: 추가 파라미터
        
    Returns:
        PassageEvaluator: 이진 루브릭용 평가기
    """
    print("📋 이진 루브릭 전용 평가기를 생성합니다...")
    return create_passage_evaluator(model_type, model_name, template_key, **kwargs)


def evaluate_passage_with_binary_rubric(passage: str, problem_types: List[str], eval_goals: List[str],
                                       korean_topic: str, foreign_topic: str,
                                       korean_context: str, foreign_context: str,
                                       model_type: str = "openai", model_name: Optional[str] = None) -> Dict:
    """
    이진 루브릭으로 지문을 평가하는 원스톱 함수
    
    Args:
        passage: 평가할 지문
        problem_types: 문제 유형 리스트 (3개)
        eval_goals: 평가 목표 리스트 (3개)
        korean_topic: 한국 주제
        foreign_topic: 외국 주제
        korean_context: 한국 참고 자료
        foreign_context: 외국 참고 자료
        model_type: 모델 타입
        model_name: 모델 이름
        
    Returns:
        Dict: 평가 결과
    """
    evaluator = create_binary_passage_evaluator(model_type, model_name)
    
    return evaluator.evaluate_binary_rubric(
        passage=passage,
        problem_types=problem_types,
        eval_goals=eval_goals,
        korean_topic=korean_topic,
        foreign_topic=foreign_topic,
        korean_context=korean_context,
        foreign_context=foreign_context
    )


def create_reward_model_evaluator(model_name: str = "reward-model-korean", 
                                gpus: Optional[List[int]] = None,
                                template_key: str = "iska") -> PassageEvaluator:
    """
    Reward Model 기반 평가기를 생성하는 편의 함수 (추후 사용)
    
    Args:
        model_name: Reward Model 이름
        gpus: 사용할 GPU 리스트
        template_key: 프롬프트 템플릿 키
        
    Returns:
        PassageEvaluator: Reward Model 기반 평가기
    """
    print("🎯 Reward Model 기반 평가기를 생성합니다...")
    
    client = ModelClientFactory.create_local_client(
        model_name=model_name,
        gpus=gpus
    )
    
    evaluator = PassageEvaluator(llm_client=client, template_key=template_key)
    print(f"✅ Reward Model 평가기 생성 완료: {model_name}")
    
    return evaluator


# ========================= 사용 예시 =========================

if __name__ == "__main__":
    print("🔍 PassageEvaluator 사용 예시")
    
    # 기본 OpenAI 평가기 생성
    evaluator = create_passage_evaluator("openai", "gpt-4o-mini")
    
    # 샘플 데이터를 더 간결하게
    sample_passage = "한국의 추석은 가족이 모여 조상을 기리는 전통 명절입니다. 미국의 추수감사절과 비슷하게 가족의 화합을 중시하지만, 종교적 색채보다는 유교적 전통이 강합니다."
    
    sample_problem_types = ["제목을 붙인 근거 설명하기", "자문화와 비교하기", "원인과 전망 예측하기"]
    sample_eval_goals = [
        "글의 핵심 내용을 요약하고, 제목의 타당성을 설명하는 능력을 평가한다.",
        "문화 현상을 자신의 문화와 비교 설명하는 능력을 평가한다.",
        "사회/문화적 현상의 원인과 미래 변화를 예측하는 능력을 평가한다."
    ]
    
    # 종합 평가 실행
    try:
        results = evaluator.evaluate_passage_metrics(
            passage=sample_passage,
            problem_types=sample_problem_types,
            eval_goals=sample_eval_goals,
            home_context="한국의 추석 관련 전통과 의미에 대한 상세 정보",
            foreign_context="미국의 추수감사절 관련 전통과 의미에 대한 상세 정보",
            home_topic="한국의 추석",
            foreign_topic="미국의 추수감사절"
        )
        
        # 결과 출력
        print(f"📊 평가 결과:")
        print(f"   일관성: {results.get('coherence_score', 'N/A')}점")
        print(f"   일치성: {results.get('consistency_score', 'N/A')}점")
        print(f"   자연스러움: {results.get('naturalness_score', 'N/A')}점")
        print(f"   한국어 품질: {results.get('korean_quality_score', 'N/A')}점")
        print(f"   총점: {results.get('overall_score', 'N/A'):.2f}점")
        
        # 결과 요약
        summary = evaluator.get_evaluation_summary(results)
        print(f"� 평가 요약:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ 평가 실행 실패: {e}")
