"""
SQLite database layer with WAL mode for safe concurrent writes.

Uses append-only records with revision tracking for traceability.
"""
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Database path
DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "lab.db"


def ensure_db_dir():
    """Ensure the data directory exists."""
    DB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    """Get a database connection with WAL mode enabled."""
    ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Initialize the database schema."""
    with get_connection() as conn:
        conn.executescript("""
            -- Recipes table
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                components_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                revision INTEGER DEFAULT 1
            );
            
            -- Batches table
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id INTEGER NOT NULL REFERENCES recipes(id),
                operator TEXT NOT NULL,
                batch_date TEXT NOT NULL,
                process_params_json TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                revision INTEGER DEFAULT 1
            );
            
            -- Samples table
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL REFERENCES batches(id),
                sample_code TEXT NOT NULL UNIQUE,
                thickness_cm REAL,
                area_cm2 REAL,
                intended_test TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Measurements table
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER REFERENCES samples(id),
                measurement_type TEXT NOT NULL,
                raw_file_path TEXT NOT NULL,
                raw_file_hash TEXT NOT NULL,
                import_mapping_json TEXT,
                params_json TEXT,
                results_json TEXT,
                plot_refs_json TEXT,
                software_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                revision INTEGER DEFAULT 1
            );
            
            -- Traceability log
            CREATE TABLE IF NOT EXISTS traceability_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_id INTEGER REFERENCES measurements(id),
                action TEXT,
                details_json TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_samples_batch ON samples(batch_id);
            CREATE INDEX IF NOT EXISTS idx_measurements_sample ON measurements(sample_id);
            CREATE INDEX IF NOT EXISTS idx_measurements_type ON measurements(measurement_type);
        """)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# =============================================================================
# Recipe CRUD
# =============================================================================

def create_recipe(name: str, components: Dict[str, Any], 
                  description: str = "") -> int:
    """Create a new recipe and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO recipes (name, description, components_json)
               VALUES (?, ?, ?)""",
            (name, description, json.dumps(components))
        )
        return cursor.lastrowid


def get_recipe(recipe_id: int) -> Optional[Dict]:
    """Get a recipe by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM recipes WHERE id = ?", (recipe_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["components"] = json.loads(d.pop("components_json"))
            return d
        return None


def list_recipes() -> List[Dict]:
    """List all recipes."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, description, created_at FROM recipes ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]


# =============================================================================
# Batch CRUD
# =============================================================================

def create_batch(recipe_id: int, operator: str, batch_date: str,
                 process_params: Optional[Dict] = None, notes: str = "") -> int:
    """Create a new batch and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO batches (recipe_id, operator, batch_date, process_params_json, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (recipe_id, operator, batch_date, 
             json.dumps(process_params) if process_params else None, notes)
        )
        return cursor.lastrowid


def get_batch(batch_id: int) -> Optional[Dict]:
    """Get a batch by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if row:
            d = dict(row)
            if d.get("process_params_json"):
                d["process_params"] = json.loads(d.pop("process_params_json"))
            else:
                d.pop("process_params_json", None)
                d["process_params"] = {}
            return d
        return None


def list_batches(recipe_id: Optional[int] = None) -> List[Dict]:
    """List batches, optionally filtered by recipe."""
    with get_connection() as conn:
        if recipe_id:
            rows = conn.execute(
                """SELECT id, recipe_id, operator, batch_date, notes, created_at 
                   FROM batches WHERE recipe_id = ? ORDER BY id DESC""",
                (recipe_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, recipe_id, operator, batch_date, notes, created_at 
                   FROM batches ORDER BY id DESC"""
            ).fetchall()
        return [dict(row) for row in rows]


# =============================================================================
# Sample CRUD
# =============================================================================

def create_sample(batch_id: int, sample_code: str, thickness_cm: Optional[float] = None,
                  area_cm2: Optional[float] = None, intended_test: str = "",
                  notes: str = "") -> int:
    """Create a new sample and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO samples (batch_id, sample_code, thickness_cm, area_cm2, intended_test, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (batch_id, sample_code, thickness_cm, area_cm2, intended_test, notes)
        )
        return cursor.lastrowid


def get_sample(sample_id: int) -> Optional[Dict]:
    """Get a sample by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM samples WHERE id = ?", (sample_id,)
        ).fetchone()
        return dict(row) if row else None


def get_sample_by_code(sample_code: str) -> Optional[Dict]:
    """Get a sample by its code."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM samples WHERE sample_code = ?", (sample_code,)
        ).fetchone()
        return dict(row) if row else None


def list_samples(batch_id: Optional[int] = None) -> List[Dict]:
    """List samples, optionally filtered by batch."""
    with get_connection() as conn:
        if batch_id:
            rows = conn.execute(
                """SELECT * FROM samples WHERE batch_id = ? ORDER BY id DESC""",
                (batch_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM samples ORDER BY id DESC"
            ).fetchall()
        return [dict(row) for row in rows]


# =============================================================================
# Measurement CRUD
# =============================================================================

def create_measurement(measurement_type: str, raw_file_path: str, raw_file_hash: str,
                       import_mapping: Optional[Dict] = None, params: Optional[Dict] = None,
                       results: Optional[Dict] = None, plot_refs: Optional[List[str]] = None,
                       software_version: str = "", sample_id: Optional[int] = None) -> int:
    """Create a new measurement and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO measurements 
               (sample_id, measurement_type, raw_file_path, raw_file_hash,
                import_mapping_json, params_json, results_json, plot_refs_json, software_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sample_id, measurement_type, raw_file_path, raw_file_hash,
             json.dumps(import_mapping) if import_mapping else None,
             json.dumps(params) if params else None,
             json.dumps(results) if results else None,
             json.dumps(plot_refs) if plot_refs else None,
             software_version)
        )
        measurement_id = cursor.lastrowid
        
        # Log creation
        conn.execute(
            """INSERT INTO traceability_log (measurement_id, action, details_json)
               VALUES (?, ?, ?)""",
            (measurement_id, "created", json.dumps({
                "file_hash": raw_file_hash,
                "params": params
            }))
        )
        
        return measurement_id


def get_measurement(measurement_id: int) -> Optional[Dict]:
    """Get a measurement by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM measurements WHERE id = ?", (measurement_id,)
        ).fetchone()
        if row:
            d = dict(row)
            for json_field in ["import_mapping_json", "params_json", "results_json", "plot_refs_json"]:
                if d.get(json_field):
                    key = json_field.replace("_json", "")
                    d[key] = json.loads(d.pop(json_field))
                else:
                    d.pop(json_field, None)
            return d
        return None


def update_measurement_results(measurement_id: int, results: Dict, 
                               plot_refs: Optional[List[str]] = None):
    """Update measurement results (creates a new revision)."""
    with get_connection() as conn:
        # Get current revision
        row = conn.execute(
            "SELECT revision FROM measurements WHERE id = ?", (measurement_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Measurement {measurement_id} not found")
        
        new_revision = row["revision"] + 1
        
        conn.execute(
            """UPDATE measurements 
               SET results_json = ?, plot_refs_json = ?, revision = ?
               WHERE id = ?""",
            (json.dumps(results), 
             json.dumps(plot_refs) if plot_refs else None,
             new_revision, measurement_id)
        )
        
        # Log update
        conn.execute(
            """INSERT INTO traceability_log (measurement_id, action, details_json)
               VALUES (?, ?, ?)""",
            (measurement_id, "results_updated", json.dumps({
                "revision": new_revision,
                "results": results
            }))
        )


def list_measurements(sample_id: Optional[int] = None, 
                      measurement_type: Optional[str] = None) -> List[Dict]:
    """List measurements with optional filters."""
    with get_connection() as conn:
        query = "SELECT id, sample_id, measurement_type, raw_file_path, created_at, revision FROM measurements WHERE 1=1"
        params = []
        
        if sample_id:
            query += " AND sample_id = ?"
            params.append(sample_id)
        if measurement_type:
            query += " AND measurement_type = ?"
            params.append(measurement_type)
        
        query += " ORDER BY id DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


# Initialize database on import
init_database()
