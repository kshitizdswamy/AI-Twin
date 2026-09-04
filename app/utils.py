import logging
import re
import yaml
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_text(text: str) -> str:
    """Remove unwanted special characters and clean whitespace."""
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text).strip()

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Load YAML configuration safely."""
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Failed to load config from {filepath}: {e}")
        return {}
