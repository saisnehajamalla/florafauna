import sqlite3
import os
from datetime import datetime
import json

DB_PATH = "models/predictions.db"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_path TEXT,
    predicted_class TEXT,
    confidence REAL,
    top_k_json TEXT,
    temperature REAL
);
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(CREATE_SQL)
    conn.commit()
    conn.close()


def log_prediction(image_path: str, predicted_class: str, confidence: float, top_k: dict, temperature: float = None):
    """Save a prediction record.
    top_k should be a dict like {label: confidence, ...}
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO predictions (timestamp, image_path, predicted_class, confidence, top_k_json, temperature) VALUES (?,?,?,?,?,?)",
        (ts, image_path, predicted_class, confidence, json.dumps(top_k), temperature)
    )
    conn.commit()
    conn.close()


def fetch_recent_predictions(limit: int = 10):
    """Return recent prediction rows as a list of tuples:
    (id, timestamp, predicted_class, confidence_percent, top_k_json)
    confidence_percent is a float in 0..100 for display convenience.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, predicted_class, confidence, top_k_json FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    # convert confidence to percent for display
    out = []
    for r in rows:
        id_, ts, cls, conf, topk = r
        try:
            conf_pct = float(conf) * 100.0 if conf is not None else None
        except Exception:
            conf_pct = conf
        out.append((id_, ts, cls, conf_pct, topk))
    return out


if __name__ == '__main__':
    # quick test
    init_db()
    log_prediction('tests/sample.jpg', 'Pepper__bell___healthy', 0.87, {'Pepper__bell___healthy': 0.87, 'Pepper__bell___Bacterial_spot': 0.10}, 1.0)
    print('Wrote test record to', DB_PATH)
