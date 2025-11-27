import json
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger("fraud_db")

DB_PATH = Path(__file__).parent / "fraud_cases.json"

def _load_all_cases() -> list:
    if not DB_PATH.exists():
        logger.warning("DB file not found, returning empty list.")
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_all_cases(cases: list):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    logger.info("Wrote updated cases to disk.")

def find_case_by_username(user_name: str) -> Optional[Dict]:
    cases = _load_all_cases()
    for case in cases:
        if case.get("userName", "").lower() == user_name.lower():
            return case
    return None

def update_case(user_name: str, updates: Dict) -> bool:
    cases = _load_all_cases()
    for i, case in enumerate(cases):
        if case.get("userName", "").lower() == user_name.lower():
            cases[i].update(updates)
            _write_all_cases(cases)
            return True
    return False
