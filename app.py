"""Flask web application for six-class industrial defect detection."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database" / "database.db"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
DEFECT_CLASSES = (
    "crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches",
)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY="change-this-secret-key-in-production",
        UPLOAD_FOLDER=str(UPLOAD_FOLDER),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    initialize_database()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/predict")
    def predict() -> Any:
        image = request.files.get("image")
        if image is None or not image.filename:
            flash("Please choose an image before submitting.", "warning")
            return redirect(url_for("index"))
        if not allowed_file(image.filename):
            flash("Only JPG, JPEG, and PNG images are supported.", "danger")
            return redirect(url_for("index"))

        unique_name = f"{uuid.uuid4().hex}_{secure_filename(image.filename)}"
        image_path = UPLOAD_FOLDER / unique_name
        try:
            image.save(image_path)
            from predict import ModelUnavailableError, predict_image
            prediction, confidence, all_probs = predict_image(image_path)
        except ModelUnavailableError as error:
            image_path.unlink(missing_ok=True)
            flash(str(error), "warning")
            return redirect(url_for("index"))
        except Exception:
            image_path.unlink(missing_ok=True)
            app.logger.exception("Prediction failed")
            flash("We could not process this image. Please try another image.", "danger")
            return redirect(url_for("index"))

        status = "DEFECT DETECTED" if confidence >= 60.0 else "PASSED"
        record_id, created_at = save_prediction(unique_name, prediction, confidence, status)
        return render_template(
            "result.html", record_id=record_id, image_name=unique_name,
            prediction=prediction, confidence=confidence, all_probs=all_probs, status=status, created_at=created_at,
        )

    @app.get("/dashboard")
    def dashboard() -> str:
        return render_template(
            "dashboard.html", stats=get_dashboard_stats(), recent_predictions=get_predictions(limit=6)
        )

    @app.get("/history")
    def history() -> str:
        page = max(request.args.get("page", 1, type=int), 1)
        query = request.args.get("q", "").strip()
        per_page = 8
        records, total = get_paginated_predictions(query, page, per_page)
        total_pages = max((total + per_page - 1) // per_page, 1)
        return render_template("history.html", records=records, page=min(page, total_pages), total_pages=total_pages, query=query)

    @app.post("/history/<int:record_id>/delete")
    def delete_history_record(record_id: int) -> Any:
        delete_prediction(record_id)
        flash("Prediction record deleted.", "success")
        return redirect(url_for("history"))

    @app.get("/about")
    def about() -> str:
        return render_template("about.html")

    @app.errorhandler(413)
    def file_too_large(_: Exception) -> Any:
        flash("Image size must be 10 MB or smaller.", "danger")
        return redirect(url_for("index"))
    return app


def allowed_file(filename: str) -> bool:
    """Return whether a file name has a permitted image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection exposing rows as dictionaries."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_predictions_table(connection: sqlite3.Connection) -> None:
    """Create the unconstrained prediction table used for the six defect labels."""
    connection.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'DEFECT DETECTED',
            created_at TEXT NOT NULL
        )
    """)


def initialize_database() -> None:
    """Create or migrate the prediction history schema for multiclass labels."""
    with get_connection() as connection:
        exists = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone()
        if not exists:
            create_predictions_table(connection)
            return
        
        # Check if status column exists
        cursor = connection.execute("PRAGMA table_info(predictions)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'status' not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN status TEXT NOT NULL DEFAULT 'DEFECT DETECTED'")
            
        schema = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone()[0]
        if "CHECK" in schema.upper():
            connection.execute("ALTER TABLE predictions RENAME TO predictions_legacy")
            create_predictions_table(connection)
            placeholders = ", ".join("?" for _ in DEFECT_CLASSES)
            connection.execute(
                f"""INSERT INTO predictions (id, image_name, prediction, confidence, status, created_at)
                    SELECT id, image_name, prediction, confidence, 'DEFECT DETECTED', created_at
                    FROM predictions_legacy WHERE prediction IN ({placeholders})""",
                DEFECT_CLASSES,
            )
            connection.execute("DROP TABLE predictions_legacy")


def save_prediction(image_name: str, prediction: str, confidence: float, status: str) -> tuple[int, str]:
    """Persist an inference result and return its identifier and UTC timestamp."""
    if prediction not in DEFECT_CLASSES:
        raise ValueError("Prediction is not one of the configured defect classes.")
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO predictions (image_name, prediction, confidence, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (image_name, prediction, confidence, status, created_at),
        )
        return int(cursor.lastrowid), created_at


def get_predictions(limit: int | None = None) -> list[dict[str, Any]]:
    """Fetch newest prediction records."""
    sql, parameters = "SELECT * FROM predictions ORDER BY id DESC", ()
    if limit is not None:
        sql, parameters = f"{sql} LIMIT ?", (limit,)
    with get_connection() as connection:
        return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


def get_paginated_predictions(query: str, page: int, per_page: int) -> tuple[list[dict[str, Any]], int]:
    """Fetch a searchable page of prediction records."""
    where_clause, parameters = "", []
    if query:
        where_clause = "WHERE image_name LIKE ? OR prediction LIKE ?"
        parameters = [f"%{query}%", f"%{query}%"]
    with get_connection() as connection:
        total = connection.execute(f"SELECT COUNT(*) FROM predictions {where_clause}", parameters).fetchone()[0]
        rows = connection.execute(
            f"SELECT * FROM predictions {where_clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*parameters, per_page, (page - 1) * per_page],
        ).fetchall()
    return [dict(row) for row in rows], int(total)


def delete_prediction(record_id: int) -> None:
    """Delete one prediction record."""
    with get_connection() as connection:
        connection.execute("DELETE FROM predictions WHERE id = ?", (record_id,))


def get_dashboard_stats() -> dict[str, Any]:
    """Return totals, average confidence, one count for each defect class, and status counts."""
    with get_connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS total, AVG(confidence) AS average_confidence FROM predictions").fetchone()
        
        counts = {name: 0 for name in DEFECT_CLASSES}
        for item in connection.execute("SELECT prediction, COUNT(*) AS count FROM predictions GROUP BY prediction"):
            if item["prediction"] in counts:
                counts[item["prediction"]] = int(item["count"])
                
        status_counts = {"DEFECT DETECTED": 0, "PASSED": 0}
        for item in connection.execute("SELECT status, COUNT(*) AS count FROM predictions GROUP BY status"):
            if item["status"] in status_counts:
                status_counts[item["status"]] = int(item["count"])
                
    # Read model accuracy from classification_report.txt if available
    accuracy = "Not evaluated"
    report_path = BASE_DIR / "model" / "classification_report.txt"
    if report_path.exists():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "accuracy" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        accuracy = f"{float(parts[1]) * 100:.0f}%"
                    except ValueError:
                        pass
                break
                
    return {
        "total": int(row["total"]), 
        "average_confidence": round(float(row["average_confidence"] or 0), 1), 
        "class_counts": counts,
        "defects_found": status_counts["DEFECT DETECTED"],
        "passed": status_counts["PASSED"],
        "accuracy": accuracy
    }


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
