import sys
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent.parent))
sys.path.append(str(Path.cwd().parent / 'modules'))

from modules.model_client import ModelClient
from utils.prompt_loader import load_prompt

class StemEvaluator:
    def __init__(self, llm_client: ModelClient):
        self.llm_client = llm_client

    def evaluate_stem(
        self,
        passage: str,
        stem: str,
        problem_type: str,
        eval_goal: str,
        guideline_num: int,
    ) -> Dict[str, Any]:
        """
        Generates an evaluation for a single stem based on a guideline.
        """
        prompt_template = load_prompt(f'iska/rubric_evaluation.completeness_for_guideline_{guideline_num}')
        
        prompt = prompt_template.format(
            passage=passage,
            stem=stem,
            problem_type=problem_type,
            eval_goal=eval_goal
        )
        
        response = self.llm_client.chat(prompt)
        
        # Parse the response to get PASS/FAIL and the reasoning
        try:
            result_text = response.choices[0].message.content
            parts = result_text.split('\n', 1)
            judgment = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else ""
        except Exception as e:
            judgment = "ERROR"
            reason = str(e)
            
        return {
            "judgment": judgment,
            "reason": reason,
            "raw_response": response.choices[0].message.content
        }