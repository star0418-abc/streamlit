"""
SQLite database layer with WAL mode for safe concurrent writes.

Uses append-only records with revision tracking for traceability.
Cloud-aware: Uses /tmp on Streamlit Cloud for writable storage.
"""
import sqlite3
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# =============================================================================
# Cloud Environment Detection & Path Configuration
# =============================================================================

def is_cloud_environment() -> bool:
    """
    Detect if running on Streamlit Community Cloud.
    
    Cloud indicators:
    - STREAMLIT_SHARING env var is set
    - Running from /mount or /app paths (Cloud container paths)
    - HOME is /home/appuser (Cloud default)
    """
    # Check environment variables set by Streamlit Cloud
    if os.environ.get("STREAMLIT_SHARING"):
        return True
    if os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud":
        return True
    
    # Check typical Cloud container paths
    cwd = os.getcwd()
    if cwd.startswith("/mount/") or cwd.startswith("/app"):
        return True
    
    # Check home directory (Cloud uses /home/appuser)
    home = os.environ.get("HOME", "")
    if home == "/home/appuser":
        return True
    
    return False


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_db_path() -> Path:
    """
    Get the database path, using /tmp on Cloud for writable storage.
    
    On Cloud: /tmp/lab.db (ephemeral but writable)
    On Local: {project_root}/data/lab.db
    """
    if is_cloud_environment():
        return Path("/tmp/lab.db")
    else:
        return get_project_root() / "data" / "lab.db"


def get_db_dir() -> Path:
    """Get the directory containing the database."""
    return get_db_path().parent


# =============================================================================
# Lazy Initialization
# =============================================================================

_initialized = False
_init_error: Optional[str] = None


def _ensure_initialized():
    """Ensure database is initialized (lazy init on first use)."""
    global _initialized, _init_error
    if _initialized:
        return
    if _init_error:
        # Already tried and failed
        return
    try:
        init_database()
        _initialized = True
    except Exception as e:
        _init_error = str(e)
        # Don't re-raise; let operations handle the error gracefully


def ensure_db_dir():
    """Ensure the data directory exists."""
    db_dir = get_db_dir()
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # On Cloud, /tmp should be writable; project dirs may not be
        pass


@contextmanager
def get_connection():
    """Get a database connection with WAL mode enabled."""
    _ensure_initialized()
    
    if _init_error:
        raise RuntimeError(f"Database initialization failed: {_init_error}")
    
    ensure_db_dir()
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=30.0)
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
    ensure_db_dir()
    db_path = get_db_path()
    
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
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
        conn.commit()
    finally:
        conn.close()


# =============================================================================
# Diagnostics
# =============================================================================

def get_diagnostics() -> Dict[str, Any]:
    """
    Get environment and database diagnostics for debugging.
    
    Returns dict with:
    - runtime_env: "cloud" or "local"
    - cwd: current working directory
    - project_root: detected project root
    - db_path: database file path
    - db_exists: whether DB file exists
    - db_writable: whether DB location is writable
    - tables: dict of table_name -> row_count (empty if DB not accessible)
    - init_error: any initialization error message
    """
    db_path = get_db_path()
    project_root = get_project_root()
    
    diagnostics = {
        "runtime_env": "cloud" if is_cloud_environment() else "local",
        "cwd": os.getcwd(),
        "project_root": str(project_root),
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "db_writable": False,
        "tables": {},
        "init_error": _init_error,
    }
    
    # Check if directory is writable
    db_dir = get_db_dir()
    try:
        if db_dir.exists():
            test_file = db_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            diagnostics["db_writable"] = True
    except (PermissionError, OSError):
        pass
    
    # Get table row counts if DB exists
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            cur = conn.cursor()
            tables = [r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )]
            for table in tables:
                try:
                    count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    diagnostics["tables"][table] = count
                except Exception:
                    diagnostics["tables"][table] = "error"
            conn.close()
        except Exception as e:
            diagnostics["tables"] = {"error": str(e)}
    
    return diagnostics


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


# NOTE: No import-time init_database() call!
# Database is initialized lazily on first connection via _ensure_initialized()
