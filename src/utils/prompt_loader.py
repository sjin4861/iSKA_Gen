# src/utils/prompt_loader.py
"""Utility for loading & formatting prompt templates stored in YAML.

Rationale
---------
*   Keeps long prompt strings out of the source code.
*   Allows non‑engineers to tweak prompts without touching Python.
*   Supports hot‑reloading in Streamlit by relying on pathlib stat time.
*   Very small dependency footprint (only `pyyaml`, which is already
    required by Streamlit).
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Environment variable to point to the prompts YAML file.
_ENV_KEY = "ISKA_PROMPT_FILE"

# Default location relative to repo root if the env‑var is unset.
_DEFAULT_PATH = Path("config/prompts.yaml")

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _resolve_yaml_path(user_path: str | os.PathLike | None = None, agent: str | None = None) -> Path:
    """Return an absolute `Path` to the YAML file or directory.

    Resolution order:
    1. Explicit *user_path* argument.
    2. Agent-specific directory (e.g., "iska", "ska") in src/config/prompts/
    3. `ISKA_PROMPT_FILE` environment variable.
    4. `config/prompts.yaml` relative to the project root.
    5. `src/config/prompts/` directory relative to the project root.
    """
    # If user_path is explicitly provided, use it
    if user_path is not None:
        candidate = Path(user_path)
        if candidate.is_absolute():
            resolved = candidate.expanduser()
            if resolved.exists():
                return resolved
        else:
            repo_root = Path(__file__).resolve().parents[2]
            resolved = (repo_root / candidate).expanduser()
            if resolved.exists():
                return resolved
    
    # If agent is specified, try agent-specific directory first
    if agent is not None:
        repo_root = Path(__file__).resolve().parents[2]
        agent_dir = repo_root / "src/config/prompts" / agent
        if agent_dir.exists() and agent_dir.is_dir():
            return agent_dir
    
    # Fallback to environment variable or default
    candidate = Path(os.getenv(_ENV_KEY, _DEFAULT_PATH))

    # If the path is already absolute we are done.
    if candidate.is_absolute():
        resolved = candidate.expanduser()
        if resolved.exists():
            return resolved
    else:
        # Otherwise treat it as *relative to* the repository root
        # (i.e. two levels above utils/ → project/src/.. → project/)
        repo_root = Path(__file__).resolve().parents[2]
        resolved = (repo_root / candidate).expanduser()
        if resolved.exists():
            return resolved
    
    # Fallback: try src/config/prompts/ directory
    repo_root = Path(__file__).resolve().parents[2]
    prompts_dir = repo_root / "src/config/prompts"
    if prompts_dir.exists() and prompts_dir.is_dir():
        return prompts_dir
    
    # If nothing exists, return the original candidate for error reporting
    return candidate if candidate.is_absolute() else (repo_root / candidate).expanduser()


@functools.lru_cache(maxsize=10)  # Increased cache size for multiple agents
def _load_yaml(abs_path: Path, agent: str | None = None) -> Mapping[str, str]:
    """Load and memoise YAML at *abs_path*. 
    
    If abs_path is a directory, load all .yaml files in it and merge them.
    If abs_path is a file, load just that file.
    """
    if abs_path.is_dir():
        # Load all .yaml files in the directory
        merged = {}
        yaml_files = sorted(abs_path.glob("*.yaml"))
        
        if not yaml_files:
            raise FileNotFoundError(f"No .yaml files found in directory: {abs_path}")
            
        for yaml_file in yaml_files:
            with yaml_file.open(encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
                merged.update(content)
        
        return merged
    
    elif abs_path.is_file():
        # Load single file
        with abs_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    else:
        raise FileNotFoundError(f"Prompt file or directory not found: {abs_path}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def get_prompt(name: str, /, agent: str | None = None, **format_kwargs: Any) -> str:
    """Return a formatted prompt template by *name*.

    Parameters
    ----------
    name:
        Key in the YAML file. Supports dot notation for nested keys (e.g., "topic_gen.genqa").
    agent:
        Agent name to specify which prompts directory to use (e.g., "iska", "ska", "geval").
        If None, uses default resolution order.
    **format_kwargs:
        Will be passed to `str.format`.  Leave empty to get the raw template.

    Raises
    ------
    KeyError
        If *name* is not present in the YAML.
    """
    # Clear cached YAML to pick up any file changes immediately
    _load_yaml.cache_clear()
    abs_path = _resolve_yaml_path(agent=agent)
    templates = _load_yaml(abs_path, agent=agent)
    
    # Support dot notation for nested keys
    keys = name.split('.')
    current = templates
    
    try:
        for key in keys:
            current = current[key]
        template = current
    except (KeyError, TypeError):
        raise KeyError(
            f"Prompt '{name}' not found in {abs_path}. "
            "Check the YAML keys or file path."
        )

    if not isinstance(template, str):
        raise ValueError(f"Prompt '{name}' is not a string template.")

    # If only injecting problem_json, bypass formatting to preserve literal braces in template
    if list(format_kwargs.keys()) == ['problem_json']:
        return template.replace('{problem_json}', format_kwargs['problem_json'])
    # Default formatting for other use cases
    try:
        return template.format(**format_kwargs)
    except KeyError as err:
        missing = err.args[0]
        # Debug: print raw template and provided args
        print(f"[DEBUG] Failed formatting prompt '{name}'")
        print(f"[DEBUG] Raw template:\n{template}")
        print(f"[DEBUG] Provided args: {format_kwargs}")
        raise KeyError(
            f"Missing placeholder '{missing}' when formatting prompt '{name}'. "
            "Provided args: "
            + ", ".join(format_kwargs.keys())
        ) from None


# --------------------------------------------------------------------------- #
# Agent-specific convenience functions
# --------------------------------------------------------------------------- #

def get_iska_prompt(name: str, /, **format_kwargs: Any) -> str:
    """Get prompt from iSKA agent prompts directory."""
    return get_prompt(name, agent="iska", **format_kwargs)

def get_ska_prompt(name: str, /, **format_kwargs: Any) -> str:
    """Get prompt from SKA agent prompts directory."""
    return get_prompt(name, agent="ska", **format_kwargs)

def get_geval_prompt(name: str, /, **format_kwargs: Any) -> str:
    """Get prompt from G-Eval prompts directory."""
    return get_prompt(name, agent="geval", **format_kwargs)

def get_naco_prompt(name: str, /, **format_kwargs: Any) -> str:
    """Get prompt from NACO prompts directory."""
    return get_prompt(name, agent="naco", **format_kwargs)

def list_available_agents() -> list[str]:
    """List all available agent directories."""
    repo_root = Path(__file__).resolve().parents[2]
    prompts_dir = repo_root / "src/config/prompts"
    
    if not prompts_dir.exists():
        return []
    
    agents = []
    for item in prompts_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
            agents.append(item.name)
    
    return sorted(agents)

def list_agent_prompts(agent: str) -> list[str]:
    """List all available prompt files for a specific agent."""
    repo_root = Path(__file__).resolve().parents[2]
    agent_dir = repo_root / "src/config/prompts" / agent
    
    if not agent_dir.exists():
        return []
    
    yaml_files = list(agent_dir.glob("*.yaml"))
    return sorted([f.stem for f in yaml_files])