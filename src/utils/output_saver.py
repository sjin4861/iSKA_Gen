import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(str(Path.cwd().parent.parent))

# --- Set base paths relative to the project root ---
# This file (output_saver.py) is assumed to be in src/utils/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "raw_outputs"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "src" / "data" / "evaluations"
DEFAULT_PAIRWISE_DATA_DIR = PROJECT_ROOT / "src" / "data" / "pairwise_data"


def save_model_output(
    model_name: str,
    benchmark_id: int,
    benchmark_version: str,
    template_key: str,
    data: List[Dict[str, Any]],
    base_dir: Path = DEFAULT_RAW_OUTPUT_DIR,
    date_str: str = None
) -> Path:
    """
    Saves the raw output from a model to a structured directory.

    Args:
        model_name (str): The name of the model that generated the results.
        benchmark_id (int): The ID of the benchmark that was run.
        benchmark_version (str): The benchmark version (e.g., "v1.0.0").
        template_key (str): The key for the template used (e.g., "create_passage").
        data (List[Dict[str, Any]]): The data to save (must be JSON serializable).
        base_dir (Path): The base directory where all results will be stored.

    Returns:
        Path: The path object of the file that was ultimately saved.
        
    Raises:
        IOError: If there is an error writing the file.
    """
    try:
        # 1. Create date-based directory (e.g., data/raw_outputs/2025-07-28/)
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # 2. Determine output_type (passage, stem, option)
        if 'passage' in template_key:
            output_type = 'passage'
        elif 'stem' in template_key:
            output_type = 'stem'
        elif 'option' in template_key:
            output_type = 'option'
        else:
            output_type = 'misc' # Fallback for other types

        # 3. Create the full directory path: base_dir/date/type/model
        model_dir = base_dir / date_str / output_type / model_name / template_key
        model_dir.mkdir(parents=True, exist_ok=True)

        # 4. Create the filename (without timestamp)
        file_name = f"benchmark_{benchmark_id}_{benchmark_version}_{template_key}.json"
        output_path = model_dir / file_name

        # 5. Save the data as a JSON file
        print(f"  💾 Saving results to '{output_path}'...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"  ✅ Save complete.")
        return output_path

    except (IOError, TypeError) as e:
        print(f"❌ Error saving file: {e}")
        # Re-raise the exception so the caller can handle it
        raise IOError(f"Failed to save data for model {model_name} to {output_path}") from e


# --- Example Usage ---
if __name__ == "__main__":
    # This code will run when the script is executed directly.
    
    # 1. Sample data for testing
    sample_model_name = "MyTestModel-v1"
    sample_benchmark_id = 99
    sample_benchmark_version = "v1.2.3"
    sample_template_key = "create_passage_test"
    sample_data = [
        {"source_item": {"korean_topic": "테스트 주제"}, "generated_passage": "이것은 테스트 결과입니다."},
    ]

    print("--- Testing save_model_output function ---")
    try:
        saved_file = save_model_output(
            model_name=sample_model_name,
            benchmark_id=sample_benchmark_id,
            benchmark_version=sample_benchmark_version,
            template_key=sample_template_key,
            data=sample_data
        )
        print(f"\nTest results successfully saved to:\n{saved_file}")
        
        # Optional: Verify the saved file content
        with open(saved_file, 'r', encoding='utf-8') as f:
            read_data = json.load(f)
        print("\nVerifying saved file content (first item):")
        print(json.dumps(read_data[0], ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
