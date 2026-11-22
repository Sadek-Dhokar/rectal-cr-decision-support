"""
Audit logging: pseudonymized, local-only, no raw clinical values.
"""
import hashlib
import json
import datetime
from pathlib import Path
from src.config import AUDIT_SALT, REPORTS_DIR, MODEL_VERSION


def hash_patient_id(patient_id: str) -> str:
    """Salted SHA-256 hash of patient_id."""
    salted = f"{AUDIT_SALT}:{patient_id}"
    return hashlib.sha256(salted.encode()).hexdigest()[:16]


def log_decision(patient_id: str, result: dict, log_path: Path = None):
    """
    Append a decision to the audit log.
    Stores: timestamp, hashed ID, decision, probability, source, rule, model version.
    Never stores raw clinical values.
    """
    if log_path is None:
        log_path = REPORTS_DIR / "audit_log.jsonl"

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "patient_hash": hash_patient_id(patient_id),
        "decision": result.get("decision"),
        "probability": result.get("probability"),
        "confidence": result.get("confidence"),
        "source": result.get("source"),
        "rule_fired": result.get("rule"),
        "model_version": result.get("model_version", MODEL_VERSION),
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
