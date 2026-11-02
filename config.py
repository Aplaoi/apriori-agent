"""Application configuration, loaded from .env (never hard-code secrets)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
WORK_DIR = BASE_DIR / "work"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

for _directory in (UPLOAD_DIR, OUTPUT_DIR, WORK_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

WEKA_JAVA = os.getenv("WEKA_JAVA", "java").strip() or "java"


def _find_weka_jar() -> Path | None:
    configured = os.getenv("WEKA_JAR", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            Path(r"D:\Weka-3-8-7\weka.jar"),
            Path(r"D:\Weka-3-8-6\weka.jar"),
            Path(r"C:\Program Files\Weka-3-8-7\weka.jar"),
            Path(r"C:\Program Files\Weka-3-8-6\weka.jar"),
            Path(r"C:\Program Files (x86)\Weka-3-8-7\weka.jar"),
            BASE_DIR / "vendor" / "weka.jar",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


WEKA_JAR = _find_weka_jar()

DEFAULT_MIN_SUPPORT = 0.10
DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_MAX_RULES = 30
DEFAULT_BINS = 5
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
WEKA_TIMEOUT_SECONDS = 180


def java_executable() -> str:
    """Return the configured Java executable, falling back to PATH lookup."""
    if Path(WEKA_JAVA).is_file():
        return WEKA_JAVA
    return shutil.which(WEKA_JAVA) or "java"