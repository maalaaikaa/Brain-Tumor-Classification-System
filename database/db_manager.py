from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from config import MODEL_DIR

DB_PATH = MODEL_DIR / "history.db"

def get_connection() -> sqlite3.Connection:
    """Get sqlite3 connection and set row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize the SQLite database schema if not exists."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_age INTEGER,
                patient_gender TEXT,
                prediction_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                image_path TEXT NOT NULL,
                physician_notes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

def save_record(
    patient_id: str,
    patient_age: int | None,
    patient_gender: str | None,
    prediction_class: str,
    confidence: float,
    probabilities_dict: dict,
    image_path: str,
    physician_notes: str | None = None,
) -> int:
    """Save a scan record and return its auto-incremented ID."""
    init_db()
    probabilities_json = json.dumps(probabilities_dict)
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO scan_records (
                patient_id, patient_age, patient_gender, prediction_class,
                confidence, probabilities_json, image_path, physician_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                patient_age,
                patient_gender,
                prediction_class,
                confidence,
                probabilities_json,
                str(image_path),
                physician_notes,
            ),
        )
        conn.commit()
        return cursor.lastrowid

def get_all_records() -> list[dict]:
    """Retrieve all logged scan records, sorted by timestamp descending."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM scan_records ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        records = []
        for r in rows:
            record_dict = dict(r)
            try:
                record_dict["probabilities"] = json.loads(record_dict["probabilities_json"])
            except Exception:
                record_dict["probabilities"] = {}
            records.append(record_dict)
        return records

def get_record_by_id(record_id: int) -> dict | None:
    """Retrieve a single scan record by ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM scan_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        record_dict = dict(row)
        try:
            record_dict["probabilities"] = json.loads(record_dict["probabilities_json"])
        except Exception:
            record_dict["probabilities"] = {}
        return record_dict

def update_notes(record_id: int, notes: str) -> None:
    """Update physician notes for a record."""
    init_db()
    with get_connection() as conn:
        conn.execute(
            "UPDATE scan_records SET physician_notes = ? WHERE id = ?",
            (notes, record_id),
        )
        conn.commit()

def delete_record(record_id: int) -> None:
    """Delete a scan record from database."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM scan_records WHERE id = ?", (record_id,))
        conn.commit()
