from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp

from tools.vector_search import ALL_TABLE_NAMES
from agents.state import QueryIntent

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_SUBQUERY_DEPTH: int = 4
# For ALL_TABLES queries: 9 tables → 8 UNION / UNION ALL connectors
EXPECTED_UNION_COUNT_ALL_TABLES: int = 8

# sqlglot expression types that are unconditionally forbidden
# (structural DDL / admin commands that should never appear in a T2S query)
_FORBIDDEN_NODE_TYPES: tuple[type, ...] = (
    exp.Drop,
    exp.Create,
    exp.Attach,
    exp.Pragma,
    exp.Command,   # catches VACUUM, REINDEX, and other raw commands
)

# Write-DML node types that ARE allowed when sql_command_type declares them
_WRITE_DML_TYPES: dict[type, str] = {
    exp.Delete: "DELETE",
    exp.Update: "UPDATE",
    exp.Insert: "INSERT",
}

# SQLite internal / system table name prefixes — case-insensitive
_SYSTEM_TABLE_PREFIXES: tuple[str, ...] = (
    "sqlite_",          # sqlite_master, sqlite_sequence, sqlite_temp_master
    "information_schema",
)

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid:      bool
    risk_level: str                   # "SAFE" | "WARN" | "BLOCK"
    violations: list[str] = field(default_factory=list)
    fixed_sql:  str        = ""

    def __str__(self) -> str:
        status = "✅ SAFE" if self.risk_level == "SAFE" else (
                 "⚠️  WARN" if self.risk_level == "WARN" else "🚫 BLOCK")
        lines  = [f"[{status}]  valid={self.valid}"]
        for v in self.violations:
            lines.append(f"  • {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-parse auto-fix
# ─────────────────────────────────────────────────────────────────────────────

def _auto_fix(sql: str) -> str:
    """
    Apply cheap, safe text-level fixes before AST parsing.

    1. Normalise line endings.
    2. Strip markdown code fences (``` or ```sql … ```).
    3. Collapse >1 trailing semicolons into exactly one.
    """
    sql = sql.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Strip leading ```sql or ``` fence
    sql = re.sub(r"^```(?:sql)?\s*\n?", "", sql, flags=re.IGNORECASE)
    # Strip trailing ``` fence
    sql = re.sub(r"\n?```\s*$", "", sql).strip()

    # Collapse multiple consecutive semicolons  e.g.  "...;;  ;" → "...;"
    sql = re.sub(r";(\s*;)+", ";", sql).strip()

    # Ensure the query ends with exactly one semicolon
    if sql and not sql.endswith(";"):
        sql += ";"

    return sql


# ─────────────────────────────────────────────────────────────────────────────
# Individual check functions
# Each returns (is_violation: bool, message: str)
# ─────────────────────────────────────────────────────────────────────────────

def _check_parse(sql: str) -> tuple[bool, str, list[sqlglot.Expression]]:
    """
    Layer 1 — parse with sqlglot (SQLite dialect).

    sqlglot is intentionally permissive and will parse almost any token
    sequence without raising.  We therefore do two checks:
      a) sqlglot.parse() succeeds (catches truly unparseable input)
      b) Every top-level statement has a recognised SQL root node type
         (Select, Union, Insert, Update, Delete, Drop, Create, Alter,
          Attach, Pragma, Command).  Anything else (e.g. the 'Not' node
          produced for "THIS IS NOT SQL") is treated as invalid.

    Returns (failed, message, parsed_statements).
    """
    # Root node types that represent real SQL statements
    VALID_ROOT_TYPES: frozenset[str] = frozenset({
        "Select", "Union", "Insert", "Update", "Delete",
        "Drop", "Create", "Alter", "Attach", "Pragma", "Command",
    })

    try:
        stmts = sqlglot.parse(sql, dialect="sqlite", error_level=sqlglot.ErrorLevel.RAISE)
    except sqlglot.errors.ParseError as e:
        return True, f"SQL failed to parse: {e}", []

    if not stmts:
        return True, "SQL produced no parseable statements.", []

    invalid_roots = [
        type(s).__name__ for s in stmts
        if type(s).__name__ not in VALID_ROOT_TYPES
    ]
    if invalid_roots:
        return True, (
            f"SQL does not contain valid SQL statements "
            f"(unexpected root node types: {', '.join(invalid_roots)}). "
            "The generated output may not be a SQL query."
        ), []

    return False, "", stmts


def _check_multi_statement(stmts: list) -> tuple[bool, str]:
    """Layer 2 — multiple statements = SQL injection vector."""
    if len(stmts) > 1:
        types = [type(s).__name__ for s in stmts]
        return True, (
            f"Query contains {len(stmts)} statements ({', '.join(types)}). "
            "Only a single statement is allowed."
        )
    return False, ""


def _check_forbidden_nodes(stmts: list) -> tuple[bool, str]:
    """
    Layer 3 — DDL / admin node types that must never appear.
    Walks the entire AST of every statement.
    """
    for stmt in stmts:
        for node in stmt.walk():
            node_type = type(node)
            if isinstance(node, _FORBIDDEN_NODE_TYPES):
                return True, (
                    f"Forbidden operation detected: {node_type.__name__}. "
                    "DDL (DROP, CREATE), ATTACH, PRAGMA, and admin commands "
                    "are not permitted."
                )
    return False, ""


def _check_write_intent(
    stmts: list, declared_command_type: str
) -> tuple[bool, str]:
    """
    Layer 4 — DELETE / UPDATE / INSERT are only allowed when thought_node
    explicitly declared that sql_command_type in the AgentState.

    This stops an LLM hallucinating a DELETE inside what should be a SELECT.
    """
    declared = declared_command_type.upper()
    for stmt in stmts:
        for node in stmt.walk():
            for node_type, expected_cmd in _WRITE_DML_TYPES.items():
                if isinstance(node, node_type) and declared != expected_cmd:
                    return True, (
                        f"Query contains a {expected_cmd} statement but the "
                        f"declared command type is {declared!r}. "
                        "Undeclared write operations are blocked."
                    )
    return False, ""


def _check_system_tables(stmts: list) -> tuple[bool, str]:
    """
    Layer 5 — SQLite internal tables must never be queried.
    Covers sqlite_master, sqlite_sequence, information_schema, etc.
    """
    for stmt in stmts:
        for table_node in stmt.find_all(exp.Table):
            name = table_node.name.lower()
            for prefix in _SYSTEM_TABLE_PREFIXES:
                if name.startswith(prefix):
                    return True, (
                        f"Access to system table {table_node.name!r} is forbidden."
                    )
    return False, ""


def _check_table_whitelist(stmts: list) -> tuple[bool, str]:
    """
    Layer 6 — every table referenced in the query must be in ALL_TABLE_NAMES.

    Catches:
      • Hallucinated table names from the LLM
      • Attempts to query external / injected table names
    """
    allowed = {t.lower() for t in ALL_TABLE_NAMES}
    violations: list[str] = []

    for stmt in stmts:
        for table_node in stmt.find_all(exp.Table):
            name = table_node.name
            if name.lower() not in allowed:
                violations.append(name)

    if violations:
        unknown = ", ".join(f"{n!r}" for n in sorted(set(violations)))
        return True, (
            f"Query references unknown table(s): {unknown}. "
            f"Allowed tables: {', '.join(sorted(ALL_TABLE_NAMES))}."
        )
    return False, ""


def _check_subquery_depth(stmts: list) -> tuple[bool, str]:
    """
    Layer 7 (WARN) — deeply nested subqueries hurt readability and
    can indicate runaway generation.  We count total Select nodes
    (each level of nesting adds one).
    """
    for stmt in stmts:
        select_nodes = list(stmt.find_all(exp.Select))
        depth        = len(select_nodes)       # 1 = flat, 2 = one subquery, etc.
        if depth > MAX_SUBQUERY_DEPTH:
            return True, (
                f"Query contains {depth} nested SELECT levels "
                f"(max recommended: {MAX_SUBQUERY_DEPTH}). "
                "Consider simplifying the query."
            )
    return False, ""


def _check_union_completeness(
    stmts: list, query_intent: str
) -> tuple[bool, str]:
    """
    Layer 8 (WARN) — when the intent is ALL_TABLES we expect 8 UNION
    connectors (9 tables − 1).  Fewer means the LLM truncated the query.

    Uses AST Union node count, not the old fragile string-count heuristic.
    """
    if query_intent != QueryIntent.ALL_TABLES:
        return False, ""

    total_unions = 0
    for stmt in stmts:
        total_unions += len(list(stmt.find_all(exp.Union)))

    if total_unions < EXPECTED_UNION_COUNT_ALL_TABLES:
        return True, (
            f"ALL_TABLES query has only {total_unions} UNION connector(s) "
            f"(expected {EXPECTED_UNION_COUNT_ALL_TABLES} for all 9 tables). "
            "The query may be incomplete — consider regenerating."
        )
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Public validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_sql(
    sql: str,
    sql_command_type: str = "SELECT",
    query_intent:     str = QueryIntent.SINGLE_TABLE,
) -> ValidationResult:
    """
    Parse *sql* into an AST and run all 8 validation layers.

    Parameters
    ──────────
    sql              Raw SQL string from Codestral (may have markdown fences).
    sql_command_type Declared command type from AgentState (e.g. "SELECT").
    query_intent     QueryIntent constant from AgentState.

    Returns
    ───────
    ValidationResult with:
      valid=True  + risk_level="SAFE"  → safe to execute
      valid=True  + risk_level="WARN"  → execute but log warnings
      valid=False + risk_level="BLOCK" → do NOT execute; trigger retry
    """
    violations: list[str] = []
    risk_level = "SAFE"

    # ── Pre-parse auto-fix ───────────────────────────────────────────────
    fixed_sql = _auto_fix(sql)
    logger.info(f"validate_sql: checking {len(fixed_sql)} chars  "
                f"cmd={sql_command_type!r}  intent={query_intent!r}")

    # ── Layer 1: parse ───────────────────────────────────────────────────
    failed, msg, stmts = _check_parse(fixed_sql)
    if failed:
        violations.append(msg)
        logger.warning(f"[BLOCK] parse failed: {msg}")
        return ValidationResult(
            valid=False, risk_level="BLOCK",
            violations=violations, fixed_sql=fixed_sql
        )

    # ── Layer 2: multi-statement (BLOCK) ─────────────────────────────────
    failed, msg = _check_multi_statement(stmts)
    if failed:
        violations.append(msg)
        logger.warning(f"[BLOCK] multi-statement: {msg}")
        return ValidationResult(
            valid=False, risk_level="BLOCK",
            violations=violations, fixed_sql=fixed_sql
        )

    # ── Layer 3: forbidden nodes (BLOCK) ─────────────────────────────────
    failed, msg = _check_forbidden_nodes(stmts)
    if failed:
        violations.append(msg)
        logger.warning(f"[BLOCK] forbidden node: {msg}")
        return ValidationResult(
            valid=False, risk_level="BLOCK",
            violations=violations, fixed_sql=fixed_sql
        )

    # ── Layer 4: undeclared write DML (BLOCK) ────────────────────────────
    failed, msg = _check_write_intent(stmts, sql_command_type)
    if failed:
        violations.append(msg)
        logger.warning(f"[BLOCK] write intent mismatch: {msg}")
        return ValidationResult(
            valid=False, risk_level="BLOCK",
            violations=violations, fixed_sql=fixed_sql
        )

    # ── Layer 5: system tables (BLOCK) ───────────────────────────────────
    failed, msg = _check_system_tables(stmts)
    if failed:
        violations.append(msg)
        logger.warning(f"[BLOCK] system table: {msg}")
        return ValidationResult(
            valid=False, risk_level="BLOCK",
            violations=violations, fixed_sql=fixed_sql
        )

    # ── Layer 6: table whitelist (BLOCK) ─────────────────────────────────
    failed, msg = _check_table_whitelist(stmts)
    if failed:
        violations.append(msg)
        logger.warning(f"[BLOCK] table whitelist: {msg}")
        return ValidationResult(
            valid=False, risk_level="BLOCK",
            violations=violations, fixed_sql=fixed_sql
        )

    # ── Layer 7: subquery depth (WARN) ───────────────────────────────────
    failed, msg = _check_subquery_depth(stmts)
    if failed:
        violations.append(msg)
        risk_level = "WARN"
        logger.warning(f"[WARN] subquery depth: {msg}")

    # ── Layer 8: union completeness (WARN) ───────────────────────────────
    failed, msg = _check_union_completeness(stmts, query_intent)
    if failed:
        violations.append(msg)
        risk_level = "WARN"
        logger.warning(f"[WARN] union completeness: {msg}")

    is_valid = risk_level != "BLOCK"
    result   = ValidationResult(
        valid=is_valid, risk_level=risk_level,
        violations=violations, fixed_sql=fixed_sql
    )
    logger.info(f"validate_sql result: {result}")
    return result


# from typing import TypedDict
# from utils.llm_client import init_codestral_client, generate_sql_with_codestral
# from langchain_core.messages import SystemMessage
# import logging
# from .state import AgentState  # Import from shared state module

# logger = logging.getLogger(__name__)

# def sql_query_generator_node(state: AgentState) -> AgentState:
#     """Generate SQL query based on NL text, schema, and command type."""
#     logger.info("Entering sql_query_generator_node")
#     try:
#         api_key = init_codestral_client("config/settings.yaml")
#         nl_text = state["messages"][-1].content
#         schema = state["schema"]
#         sql_command_type = state["sql_command_type"]
#         extracted_entities = state["extracted_entities"]
        
#         if not schema:
#             raise ValueError("No schema available for SQL generation")
        
#         sql_query = generate_sql_with_codestral(
#             api_key, nl_text, schema, sql_command_type, extracted_entities
#         )
        
#         state["generated_sql"] = sql_query
#         state["messages"].append(SystemMessage(content=f"Generated SQL: {sql_query}"))
#         logger.info(f"Generated SQL: {sql_query}")
#     except Exception as e:
#         logger.error(f"Error in sql_query_generator_node: {e}")
#         state["generated_sql"] = ""
#         state["messages"].append(SystemMessage(content=f"Error generating SQL: {e}"))
#     return state
