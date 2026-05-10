import sqlite3
from datetime import datetime
import pandas as pd
import json

DB_PATH = "detections.db"

# =========================================================
# DATABASE CONNECTION
# =========================================================
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# =========================================================
# INITIALIZE DATABASE
# =========================================================
def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            defect_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            defect_details TEXT,
            processing_time REAL DEFAULT 0,
            model_version TEXT DEFAULT 'YOLOv12',
            timestamp TEXT
        )
        """)

        conn.commit()

# =========================================================
# ADD DETECTION
# =========================================================
def add_detection(
    filename,
    defect_count,
    avg_confidence,
    defect_details,
    processing_time=0,
    model_version="YOLOv12"
):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert dict/list to JSON string if needed
    if isinstance(defect_details, (dict, list)):
        defect_details = json.dumps(defect_details)

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO detections (
            image_name,
            defect_count,
            avg_confidence,
            defect_details,
            processing_time,
            model_version,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            filename,
            defect_count,
            avg_confidence,
            defect_details,
            processing_time,
            model_version,
            timestamp
        ))

        conn.commit()

# =========================================================
# GET ALL DETECTIONS
# =========================================================
def get_all_detections():

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT *
        FROM detections
        ORDER BY id DESC
        """)

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

# =========================================================
# GET RECENT DETECTIONS
# =========================================================
def get_recent_detections(limit=10):

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT *
        FROM detections
        ORDER BY id DESC
        LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

# =========================================================
# GET STATS
# =========================================================
def get_detection_stats():

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT
            COUNT(*) as total_images,
            SUM(defect_count) as total_defects,
            AVG(avg_confidence) as avg_confidence,
            AVG(processing_time) as avg_processing_time,
            MAX(defect_count) as max_defects
        FROM detections
        """)

        row = cursor.fetchone()

        return {
            "total_images": row["total_images"] or 0,
            "total_defects": row["total_defects"] or 0,
            "avg_confidence": round(row["avg_confidence"] or 0, 2),
            "avg_processing_time": round(row["avg_processing_time"] or 0, 3),
            "max_defects": row["max_defects"] or 0
        }

# =========================================================
# GET DEFECT DISTRIBUTION
# =========================================================
def get_defect_distribution():

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT defect_details
        FROM detections
        """)

        rows = cursor.fetchall()

    defect_counter = {}

    for row in rows:

        if row["defect_details"]:

            try:
                defects = json.loads(row["defect_details"])

                for defect in defects:

                    defect_type = defect.get("Type") or defect.get("type")

                    if defect_type:
                        defect_counter[defect_type] = (
                            defect_counter.get(defect_type, 0) + 1
                        )

            except:
                pass

    return defect_counter

# =========================================================
# SEARCH DETECTIONS
# =========================================================
def search_detections(keyword):

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT *
        FROM detections
        WHERE image_name LIKE ?
        ORDER BY id DESC
        """, (f"%{keyword}%",))

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

# =========================================================
# EXPORT TO CSV
# =========================================================
def export_csv(filename="detections_export.csv"):

    data = get_all_detections()

    if not data:
        return None

    df = pd.DataFrame(data)

    df.to_csv(filename, index=False)

    return filename

# =========================================================
# DELETE SINGLE RECORD
# =========================================================
def delete_detection(record_id):

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        DELETE FROM detections
        WHERE id = ?
        """, (record_id,))

        conn.commit()

# =========================================================
# CLEAR DATABASE
# =========================================================
def clear_all():

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("DELETE FROM detections")

        conn.commit()

# =========================================================
# RESET DATABASE
# =========================================================
def reset_database():

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS detections")

        conn.commit()

    init_db()

# =========================================================
# GET DATAFRAME
# =========================================================
def get_dataframe():

    data = get_all_detections()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)

# =========================================================
# HEALTH CHECK
# =========================================================
def database_health_check():

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT 1")

            return True

    except Exception as e:
        print("Database Error:", e)
        return False