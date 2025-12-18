import sqlite3
from datetime import datetime

DB_PATH = "detections.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            defect_count INTEGER,
            avg_confidence REAL,
            defect_details TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_detection(filename, defect_count, avg_confidence, defect_details):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (image_name, defect_count, avg_confidence, defect_details, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, defect_count, avg_confidence, defect_details, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_all_detections():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_detection_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(defect_count), AVG(avg_confidence) FROM detections")
    total_images, total_defects, avg_confidence = cursor.fetchone()
    conn.close()
    return {
        "total_images": total_images or 0,
        "total_defects": total_defects or 0,
        "avg_confidence": avg_confidence or 0.0
    }

def clear_all():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
