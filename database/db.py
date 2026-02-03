"""
SQLite database layer with WAL mode for safe concurrent writes.

Uses append-only records with revision tracking for traceability.
Cloud-aware: Uses /tmp on Streamlit Cloud for writable storage.

=============================================================================
Schema Versioning Policy
=============================================================================
This module uses PRAGMA user_version to track schema versions and applies
incremental migrations on init. This ensures existing databases are upgraded
without data loss when the schema evolves.

Version History:
  - v0: Original schema (implicit, no user_version set)
  - v1: Initial tracked version (current schema)

Migration safety:
  - Migrations are additive (ALTER TABLE ADD COLUMN, CREATE INDEX)
  - No destructive changes (DROP COLUMN not used)
  - column_exists() checks prevent duplicate ALTER TABLE errors

=============================================================================
JSON Serialization Policy
=============================================================================
All JSON fields use safe_json_dumps() which handles:
  - datetime/date/time -> ISO 8601 string
  - Path -> str(path)
  - numpy scalars (np.float32, np.int64, etc.) -> Python scalars
  - pandas Timestamp/Timedelta -> ISO string
  - Decimal -> float
  - Enum -> value
  - set/tuple -> list
  - Unknown types -> str(obj) with warning logged

This prevents crashes from common scientific Python objects.
"""
import sqlite3
import json
import hashlib
import os
import threading
import logging
import warnings
from pathlib import Path
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# Schema Version Constant
# =============================================================================

CURRENT_SCHEMA_VERSION = 1

# =============================================================================
# Safe JSON Serialization
# =============================================================================

def _json_default(obj: Any) -> Any:
    """
    Default handler for json.dumps to convert non-serializable types.
    
    Handles:
      - datetime/date/time -> ISO string
      - Path -> str
      - numpy scalars -> Python scalars via item()
      - pandas Timestamp/Timedelta -> ISO/str
      - Decimal -> float
      - Enum -> value
      - set/tuple -> list
      
    Falls back to str(obj) for unknown types (with warning).
    """
    # datetime types
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.isoformat()
    
    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)
    
    # Decimal
    if isinstance(obj, Decimal):
        return float(obj)  # Could also use str() if precision matters
    
    # Enum
    if isinstance(obj, Enum):
        return obj.value
    
    # set/frozenset/tuple -> list
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    
    # numpy types (check by module name to avoid import dependency)
    obj_type = type(obj)
    if obj_type.__module__ == 'numpy':
        # numpy scalars have item() method
        if hasattr(obj, 'item'):
            return obj.item()
        # numpy arrays -> lists
        if hasattr(obj, 'tolist'):
            return obj.tolist()
    
    # pandas types
    if obj_type.__module__.startswith('pandas'):
        type_name = obj_type.__name__
        if type_name in ('Timestamp', 'DatetimeTZDtype'):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        if type_name in ('Timedelta', 'NaT'):
            return str(obj)
    
    # Fallback: convert to string and warn
    warnings.warn(
        f"JSON serialization: unknown type {obj_type.__name__} converted to str",
        category=UserWarning,
        stacklevel=3
    )
    logger.warning(f"JSON fallback: type={obj_type.__name__} value={repr(obj)[:100]}")
    return str(obj)


def safe_json_dumps(obj: Any) -> Optional[str]:
    """
    Safely serialize object to JSON string, handling common scientific types.
    
    Returns None if obj is None, otherwise returns JSON string.
    Uses _json_default for non-standard types.
    """
    if obj is None:
        return None
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


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
# Lazy Initialization with Thread Safety
# =============================================================================

_init_lock = threading.Lock()
_initialized = False
_init_error: Optional[str] = None


def _ensure_initialized():
    """Ensure database is initialized (lazy init on first use, thread-safe)."""
    global _initialized, _init_error
    
    # Fast path: already initialized
    if _initialized:
        return
    
    with _init_lock:
        # Double-check after acquiring lock
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
            logger.error(f"Database initialization failed: {e}")
            # Don't re-raise; let operations handle the error gracefully


def ensure_db_dir() -> Path:
    """
    Ensure the data directory exists.
    
    Returns the db directory Path on success.
    Raises RuntimeError with actionable message on failure.
    """
    db_dir = get_db_dir()
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir
    except PermissionError as e:
        # On Cloud, /tmp should always be writable
        if is_cloud_environment():
            raise RuntimeError(
                f"Cannot create database directory on Cloud: {db_dir}. "
                f"This is unexpected for /tmp. Error: {e}"
            )
        # On local, suggest alternatives
        raise RuntimeError(
            f"Cannot create database directory: {db_dir}. "
            f"Permission denied. Consider:\n"
            f"  1. Run with appropriate permissions\n"
            f"  2. Set a custom DB path via environment variable\n"
            f"  3. Check if the path is on a read-only filesystem\n"
            f"Original error: {e}"
        )


# =============================================================================
# Standard PRAGMAs
# =============================================================================

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply consistent PRAGMAs to a connection."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds


# =============================================================================
# Schema Migration Utilities
# =============================================================================

def _get_user_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from PRAGMA user_version."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return row[0] if row else 0


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    """Set the schema version via PRAGMA user_version."""
    conn.execute(f"PRAGMA user_version = {version}")


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """
    Check if a column exists in a table.
    
    Uses PRAGMA table_info for safe column existence check.
    """
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cursor.fetchone() is not None


def _create_base_schema(conn: sqlite3.Connection) -> None:
    """Create the base schema (v0/v1) if tables don't exist."""
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


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    """
    Migration from v0 (unversioned) to v1.
    
    v1 is the baseline tracked version. No schema changes needed,
    just sets the version marker.
    """
    # v1 is current base schema - no changes needed
    # Future migrations (v1->v2, etc.) would add columns here:
    #
    # Example future migration:
    # if not column_exists(conn, 'measurements', 'new_column'):
    #     conn.execute("ALTER TABLE measurements ADD COLUMN new_column TEXT")
    #
    pass


# Add future migrations here:
# def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
#     """Migration from v1 to v2."""
#     if not column_exists(conn, 'batches', 'status'):
#         conn.execute("ALTER TABLE batches ADD COLUMN status TEXT DEFAULT 'active'")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure the database schema is up to date.
    
    Reads user_version and applies incremental migrations.
    Safe to call on every connection; no-op if already current.
    """
    current_version = _get_user_version(conn)
    
    # Create base schema if needed (fresh DB or v0)
    _create_base_schema(conn)
    
    # Apply migrations incrementally
    if current_version < 1:
        logger.info(f"Migrating schema from v{current_version} to v1")
        _migrate_v0_to_v1(conn)
        _set_user_version(conn, 1)
        current_version = 1
    
    # Future migrations:
    # if current_version < 2:
    #     logger.info(f"Migrating schema from v{current_version} to v2")
    #     _migrate_v1_to_v2(conn)
    #     _set_user_version(conn, 2)
    #     current_version = 2
    
    # Verify we're at the expected version
    if current_version != CURRENT_SCHEMA_VERSION:
        logger.warning(
            f"Schema version mismatch: DB has v{current_version}, "
            f"code expects v{CURRENT_SCHEMA_VERSION}"
        )


# =============================================================================
# Connection Management
# =============================================================================

@contextmanager
def get_connection():
    """Get a database connection with WAL mode and proper PRAGMAs."""
    _ensure_initialized()
    
    if _init_error:
        raise RuntimeError(f"Database initialization failed: {_init_error}")
    
    ensure_db_dir()
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """
    Initialize the database schema (with migrations).
    
    Thread-safe via _init_lock. Safe to call multiple times.
    """
    ensure_db_dir()
    db_path = get_db_path()
    
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    try:
        ensure_schema(conn)
        conn.commit()
        logger.info(f"Database initialized at {db_path}")
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
    - schema_version: current PRAGMA user_version
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
        "schema_version": None,
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
    
    # Get table row counts and schema version if DB exists
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            cur = conn.cursor()
            
            # Schema version
            diagnostics["schema_version"] = _get_user_version(conn)
            
            # Table counts
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
            (name, description, safe_json_dumps(components))
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
             safe_json_dumps(process_params), notes)
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
             safe_json_dumps(import_mapping),
             safe_json_dumps(params),
             safe_json_dumps(results),
             safe_json_dumps(plot_refs),
             software_version)
        )
        measurement_id = cursor.lastrowid
        
        # Log creation
        conn.execute(
            """INSERT INTO traceability_log (measurement_id, action, details_json)
               VALUES (?, ?, ?)""",
            (measurement_id, "created", safe_json_dumps({
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
            (safe_json_dumps(results), 
             safe_json_dumps(plot_refs),
             new_revision, measurement_id)
        )
        
        # Log update
        conn.execute(
            """INSERT INTO traceability_log (measurement_id, action, details_json)
               VALUES (?, ?, ?)""",
            (measurement_id, "results_updated", safe_json_dumps({
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
