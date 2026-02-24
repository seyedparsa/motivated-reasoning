"""Single SQLite database for probe evaluation metrics."""

import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.getenv("MOTIVATION_HOME", "outputs"), "probe_metrics.db")
DEFAULT_LLM_DB_PATH = os.path.join(os.getenv("MOTIVATION_HOME", "outputs"), "llm_metrics.db")

_PRIMARY_KEY = [
    "model", "dataset", "split", "bias", "probe",
    "universal_probe", "balanced", "filter_mentions", "n_ckpts", "ckpt_mode",
    "layer", "step", "tag", "n_questions", "n_test_questions", "classifier",
]

_COLUMNS = _PRIMARY_KEY + [
    "test_examples",
    "n_zeros", "n_ones",
    "accuracy", "auc",
    "updated_at",
]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS probe_metrics (
    model           TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    split           TEXT NOT NULL,
    bias            TEXT NOT NULL,
    probe           TEXT NOT NULL,
    universal_probe INTEGER NOT NULL,
    balanced        INTEGER NOT NULL,
    filter_mentions INTEGER NOT NULL,
    n_ckpts         INTEGER NOT NULL,
    ckpt_mode       TEXT NOT NULL,
    layer           INTEGER NOT NULL,
    step            INTEGER NOT NULL,
    tag             TEXT NOT NULL DEFAULT '',
    n_questions     INTEGER NOT NULL,
    n_test_questions INTEGER NOT NULL,
    classifier      TEXT NOT NULL DEFAULT 'rfm',
    test_examples   INTEGER,
    n_zeros         INTEGER,
    n_ones          INTEGER,
    accuracy        REAL,
    auc             REAL,
    updated_at      TEXT,
    PRIMARY KEY (model, dataset, split, bias, probe,
                 universal_probe, balanced, filter_mentions, n_ckpts, ckpt_mode,
                 layer, step, tag, n_questions, n_test_questions, classifier)
)
"""

_MIGRATE_PROBE_COLUMNS = [
    "ALTER TABLE probe_metrics ADD COLUMN n_zeros INTEGER",
    "ALTER TABLE probe_metrics ADD COLUMN n_ones INTEGER",
]


def _migrate_probe_n_questions_pk(conn):
    """Add n_questions and n_test_questions to primary key.

    This allows small and large scale runs to be stored separately.
    Existing rows with NULL values get n_questions=0, n_test_questions=0.
    """
    # Check current primary key by looking at table schema
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='probe_metrics'")
    result = cursor.fetchone()
    if result is None:
        return  # Table doesn't exist yet

    create_sql = result[0]
    if "n_questions, n_test_questions)" in create_sql:
        return  # Already migrated

    cursor = conn.execute("PRAGMA table_info(probe_metrics)")
    old_cols = [row[1] for row in cursor.fetchall()]

    select_parts = []
    for c in _COLUMNS:
        if c == "n_questions" and c in old_cols:
            select_parts.append("COALESCE(n_questions, 0) AS n_questions")
        elif c == "n_test_questions" and c in old_cols:
            select_parts.append("COALESCE(n_test_questions, 0) AS n_test_questions")
        elif c in old_cols:
            select_parts.append(c)
        else:
            select_parts.append(f"NULL AS {c}")

    insert_cols = ", ".join(_COLUMNS)
    select_expr = ", ".join(select_parts)
    create_new = _CREATE_TABLE.replace("IF NOT EXISTS ", "").replace("probe_metrics", "probe_metrics_new", 1)

    conn.execute("BEGIN")
    try:
        conn.execute(create_new)
        conn.execute(f"INSERT INTO probe_metrics_new ({insert_cols}) SELECT {select_expr} FROM probe_metrics")
        conn.execute("DROP TABLE probe_metrics")
        conn.execute("ALTER TABLE probe_metrics_new RENAME TO probe_metrics")
        conn.commit()
        print("Migrated probe_metrics: added n_questions, n_test_questions to primary key")
    except Exception:
        conn.rollback()
        raise


def _migrate_probe_tag(conn):
    """Add tag column and rebuild table with updated primary key.

    Existing rows get tag = '' (empty string = no tag / production run).
    All operations run inside a single transaction so the migration is atomic.
    """
    cursor = conn.execute("PRAGMA table_info(probe_metrics)")
    columns = [row[1] for row in cursor.fetchall()]
    if "tag" in columns:
        return  # Already migrated

    old_cols = list(columns)
    select_parts = []
    for c in _COLUMNS:
        if c == "tag":
            select_parts.append("'' AS tag")
        elif c in old_cols:
            select_parts.append(c)
        else:
            select_parts.append(f"NULL AS {c}")
    insert_cols = ", ".join(_COLUMNS)
    select_expr = ", ".join(select_parts)
    create_sql = _CREATE_TABLE.replace("IF NOT EXISTS ", "").replace("probe_metrics", "probe_metrics_new", 1)

    conn.execute("BEGIN")
    try:
        conn.execute(create_sql)
        conn.execute(f"INSERT INTO probe_metrics_new ({insert_cols}) SELECT {select_expr} FROM probe_metrics")
        conn.execute("DROP TABLE probe_metrics")
        conn.execute("ALTER TABLE probe_metrics_new RENAME TO probe_metrics")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _migrate_probe_filter_mentions(conn):
    """Add filter_mentions column and rebuild table with updated primary key.

    Existing rows get filter_mentions = 1 (i.e. filtering was enabled).
    """
    # Check if column already exists
    cursor = conn.execute("PRAGMA table_info(probe_metrics)")
    columns = [row[1] for row in cursor.fetchall()]
    if "filter_mentions" in columns:
        return  # Already migrated

    conn.executescript("""
        ALTER TABLE probe_metrics RENAME TO _probe_metrics_old;
    """)
    conn.execute(_CREATE_TABLE.replace("IF NOT EXISTS ", ""))
    # Copy old data, setting filter_mentions = 1 for all existing rows
    old_cols = [c for c in columns]
    select_parts = []
    for c in _COLUMNS:
        if c == "filter_mentions":
            select_parts.append("1 AS filter_mentions")
        elif c in old_cols:
            select_parts.append(c)
        else:
            select_parts.append(f"NULL AS {c}")
    insert_cols = ", ".join(_COLUMNS)
    select_expr = ", ".join(select_parts)
    conn.execute(f"INSERT INTO probe_metrics ({insert_cols}) SELECT {select_expr} FROM _probe_metrics_old")
    conn.execute("DROP TABLE _probe_metrics_old")
    conn.commit()


def _migrate_probe_classifier(conn):
    """Normalize wide-format rows (rfm_accuracy, linear_accuracy, ...) into
    per-classifier rows with a single accuracy/auc pair.

    Each old row becomes up to two new rows (classifier='rfm' and 'linear').
    Rows that already have the 'classifier' column are left untouched.
    """
    cursor = conn.execute("PRAGMA table_info(probe_metrics)")
    columns = [row[1] for row in cursor.fetchall()]
    if "classifier" in columns:
        return  # Already migrated

    old_cols = list(columns)
    shared_cols = [c for c in old_cols if c not in (
        "rfm_accuracy", "rfm_auc", "linear_accuracy", "linear_auc",
    )]

    create_sql = _CREATE_TABLE.replace("IF NOT EXISTS ", "").replace(
        "probe_metrics", "probe_metrics_new", 1
    )

    conn.execute("BEGIN")
    try:
        conn.execute(create_sql)
        shared_select = ", ".join(shared_cols)
        new_cols = ", ".join(shared_cols + ["classifier", "accuracy", "auc"])

        for clf, acc_col, auc_col in [
            ("rfm", "rfm_accuracy", "rfm_auc"),
            ("linear", "linear_accuracy", "linear_auc"),
        ]:
            if acc_col not in old_cols:
                continue
            conn.execute(
                f"INSERT INTO probe_metrics_new ({new_cols}) "
                f"SELECT {shared_select}, '{clf}', {acc_col}, {auc_col} "
                f"FROM probe_metrics WHERE {acc_col} IS NOT NULL"
            )

        conn.execute("DROP TABLE probe_metrics")
        conn.execute("ALTER TABLE probe_metrics_new RENAME TO probe_metrics")
        conn.commit()
        print("Migrated probe_metrics: normalized to per-classifier rows")
    except Exception:
        conn.rollback()
        raise


def get_db(path=None):
    """Open (or create) the probe metrics database and return a connection."""
    path = path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_CREATE_TABLE)
    # Run column migrations (ignore errors if columns already exist)
    for sql in _MIGRATE_PROBE_COLUMNS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    # Rebuild table to add filter_mentions to primary key (preserves existing rows)
    _migrate_probe_filter_mentions(conn)
    # Rebuild table to add tag to primary key (preserves existing rows)
    _migrate_probe_tag(conn)
    # Rebuild table to add n_questions, n_test_questions to primary key
    _migrate_probe_n_questions_pk(conn)
    # Normalize wide-format rows to per-classifier rows
    _migrate_probe_classifier(conn)
    return conn


def upsert_rows(rows, db_path=None):
    """Insert or replace rows into the probe_metrics table."""
    if not rows:
        return
    conn = get_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ", ".join("?" for _ in _COLUMNS)
    col_names = ", ".join(_COLUMNS)
    sql = f"INSERT OR REPLACE INTO probe_metrics ({col_names}) VALUES ({placeholders})"
    for row in rows:
        row["updated_at"] = now
        values = [row.get(col) for col in _COLUMNS]
        conn.execute(sql, values)
    conn.commit()
    conn.close()
    print(f"Upserted {len(rows)} rows into {db_path or DEFAULT_DB_PATH}")


def query_df(sql, params=(), db_path=None):
    """Run a SQL query and return results as a pandas DataFrame."""
    conn = get_db(db_path)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# ============================================================================
# LLM Metrics Table
# ============================================================================

_LLM_PRIMARY_KEY = [
    "model", "dataset", "split", "bias", "probe", "balanced", "filter_mentions", "llm", "tag",
]

_LLM_COLUMNS = _LLM_PRIMARY_KEY + [
    "n_questions", "n_test_questions", "test_examples",
    "n_zeros", "n_ones",
    "llm_accuracy", "llm_auc",
    "updated_at",
]

_CREATE_LLM_TABLE = """
CREATE TABLE IF NOT EXISTS llm_metrics (
    model            TEXT NOT NULL,
    dataset          TEXT NOT NULL,
    split            TEXT NOT NULL,
    bias             TEXT NOT NULL,
    probe            TEXT NOT NULL,
    balanced         INTEGER NOT NULL,
    filter_mentions  INTEGER NOT NULL,
    llm              TEXT NOT NULL,
    tag              TEXT NOT NULL DEFAULT '',
    n_questions      INTEGER,
    n_test_questions INTEGER,
    test_examples    INTEGER,
    n_zeros          INTEGER,
    n_ones           INTEGER,
    llm_accuracy     REAL,
    llm_auc          REAL,
    updated_at       TEXT,
    PRIMARY KEY (model, dataset, split, bias, probe, balanced, filter_mentions, llm, tag)
)
"""

_MIGRATE_LLM_COLUMNS = [
    "ALTER TABLE llm_metrics ADD COLUMN n_zeros INTEGER",
    "ALTER TABLE llm_metrics ADD COLUMN n_ones INTEGER",
    "ALTER TABLE llm_metrics ADD COLUMN llm TEXT",
]


def _migrate_llm_tag(conn):
    """Add tag column and rebuild table with updated primary key.

    Existing rows get tag = '' (empty string = no tag / production run).
    All operations run inside a single transaction so the migration is atomic.
    """
    cursor = conn.execute("PRAGMA table_info(llm_metrics)")
    columns = [row[1] for row in cursor.fetchall()]
    if "tag" in columns:
        return  # Already migrated

    old_cols = list(columns)
    select_parts = []
    for c in _LLM_COLUMNS:
        if c == "tag":
            select_parts.append("'' AS tag")
        elif c in old_cols:
            select_parts.append(c)
        else:
            select_parts.append(f"NULL AS {c}")
    insert_cols = ", ".join(_LLM_COLUMNS)
    select_expr = ", ".join(select_parts)
    create_sql = _CREATE_LLM_TABLE.replace("IF NOT EXISTS ", "").replace("llm_metrics", "llm_metrics_new", 1)

    conn.execute("BEGIN")
    try:
        conn.execute(create_sql)
        conn.execute(f"INSERT INTO llm_metrics_new ({insert_cols}) SELECT {select_expr} FROM llm_metrics")
        conn.execute("DROP TABLE llm_metrics")
        conn.execute("ALTER TABLE llm_metrics_new RENAME TO llm_metrics")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _migrate_llm_filter_mentions(conn):
    """Add filter_mentions column and rebuild table with updated primary key.

    Existing rows get filter_mentions = 1 (i.e. filtering was enabled).
    """
    cursor = conn.execute("PRAGMA table_info(llm_metrics)")
    columns = [row[1] for row in cursor.fetchall()]
    if "filter_mentions" in columns:
        return  # Already migrated

    conn.executescript("""
        ALTER TABLE llm_metrics RENAME TO _llm_metrics_old;
    """)
    conn.execute(_CREATE_LLM_TABLE.replace("IF NOT EXISTS ", ""))
    old_cols = [c for c in columns]
    select_parts = []
    for c in _LLM_COLUMNS:
        if c == "filter_mentions":
            select_parts.append("1 AS filter_mentions")
        elif c in old_cols:
            select_parts.append(c)
        else:
            select_parts.append(f"NULL AS {c}")
    insert_cols = ", ".join(_LLM_COLUMNS)
    select_expr = ", ".join(select_parts)
    conn.execute(f"INSERT INTO llm_metrics ({insert_cols}) SELECT {select_expr} FROM _llm_metrics_old")
    conn.execute("DROP TABLE _llm_metrics_old")
    conn.commit()


def get_llm_db(path=None):
    """Open (or create) the LLM metrics database and return a connection."""
    path = path or DEFAULT_LLM_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_CREATE_LLM_TABLE)
    # Run column migrations (ignore errors if columns already exist)
    for sql in _MIGRATE_LLM_COLUMNS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    # Rebuild table to add filter_mentions to primary key (preserves existing rows)
    _migrate_llm_filter_mentions(conn)
    # Rebuild table to add tag to primary key (preserves existing rows)
    _migrate_llm_tag(conn)
    return conn


def upsert_llm_rows(rows, db_path=None):
    """Insert or replace rows into the llm_metrics table."""
    if not rows:
        return
    conn = get_llm_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ", ".join("?" for _ in _LLM_COLUMNS)
    col_names = ", ".join(_LLM_COLUMNS)
    sql = f"INSERT OR REPLACE INTO llm_metrics ({col_names}) VALUES ({placeholders})"
    for row in rows:
        row["updated_at"] = now
        values = [row.get(col) for col in _LLM_COLUMNS]
        conn.execute(sql, values)
    conn.commit()
    conn.close()
    print(f"Upserted {len(rows)} rows into {db_path or DEFAULT_LLM_DB_PATH}")
