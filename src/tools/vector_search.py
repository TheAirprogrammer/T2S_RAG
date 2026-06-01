from __future__ import annotations

import logging
import re
from typing import Any

import chromadb
import yaml
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
from chromadb.utils import embedding_functions

from agents.state import QueryIntent, SQLPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain constants — single source of truth for the 9-table schema
# ─────────────────────────────────────────────────────────────────────────────

JOIN_KEY = "regno"

# Columns that exist in every one of the 9 tables
SHARED_COLUMNS: list[str] = ["regno", "name", "semester", "avg gpa"]

# Every table in the database
ALL_TABLE_NAMES: list[str] = [
    "1_batch_AIML_Results",
    "1_batch_CSD_Results",
    "1_batch_AIDS_Results",
    "2_batch_AIML_Results",
    "2_batch_CSD_Results",
    "2_batch_AIDS_Results",
    "3_batch_AIML_Results",
    "3_batch_CSD_Results",
    "3_batch_AIDS_Results",
]

# batch prefix  →  list of tables in that batch
BATCH_MAP: dict[str, list[str]] = {
    "1_batch": [t for t in ALL_TABLE_NAMES if t.startswith("1_batch")],
    "2_batch": [t for t in ALL_TABLE_NAMES if t.startswith("2_batch")],
    "3_batch": [t for t in ALL_TABLE_NAMES if t.startswith("3_batch")],
}

# branch suffix  →  list of tables with that branch
BRANCH_MAP: dict[str, list[str]] = {
    "AIML": [t for t in ALL_TABLE_NAMES if "AIML" in t],
    "CSD":  [t for t in ALL_TABLE_NAMES if "CSD"  in t],
    "AIDS": [t for t in ALL_TABLE_NAMES if "AIDS" in t],
}

# Map QueryIntent → SQLPattern
_INTENT_TO_PATTERN: dict[str, str] = {
    QueryIntent.SINGLE_TABLE : SQLPattern.SIMPLE_SELECT,
    QueryIntent.SAME_BATCH   : SQLPattern.UNION_ALL,
    QueryIntent.CROSS_BATCH  : SQLPattern.INNER_JOIN_REGNO,
    QueryIntent.ALL_TABLES   : SQLPattern.UNION_ALL,
    QueryIntent.SUBQUERY     : SQLPattern.SUBQUERY_AVG,
    QueryIntent.AGGREGATION  : SQLPattern.GROUP_BY_AGG,
    QueryIntent.UNCERTAIN    : SQLPattern.SIMPLE_SELECT,
}


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB client factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_collection(config: dict) -> Any:
    """Return the ChromaDB collection described by *config*."""
    client = chromadb.PersistentClient(
        path=config["persist_dir"],
        settings=Settings(is_persistent=True),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    return client.get_collection(
        name=config["collection_name"],
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config["embedding_model"]
        ),
    )


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level fetch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_schema_for_table(collection: Any, table: str) -> str:
    """
    Fetch the raw schema document for a single table name from ChromaDB.
    Returns an empty string if nothing is found.
    """
    results = collection.query(
        query_texts=[f"Table: {table}"],
        n_results=1,
        where={"table_name": table},
    )
    if results["documents"] and results["documents"][0]:
        return results["documents"][0][0]
    logger.warning(f"No schema document found for table: {table!r}")
    return ""


def _fetch_schemas_for_tables(
    collection: Any, tables: list[str]
) -> dict[str, str]:
    """
    Fetch raw schema documents for a list of table names.
    Returns {table_name: schema_text}.  Missing tables map to "".
    """
    return {t: _fetch_schema_for_table(collection, t) for t in tables}


# ─────────────────────────────────────────────────────────────────────────────
# Schema compression
# ─────────────────────────────────────────────────────────────────────────────

def _extract_unique_columns(schema_text: str) -> list[str]:
    """
    Parse column lines from a schema string like:
        Table: X
        Columns:
        - col_name (TYPE)
    Returns a list of "col_name (TYPE)" strings.
    """
    return re.findall(r"-\s+(.+)", schema_text)


def _build_compressed_schema(schemas: dict[str, str]) -> str:
    """
    Given {table_name: raw_schema_text}, produce a compressed representation
    that deduplicates columns shared across all tables.

    Output format
    ─────────────
    == SHARED COLUMNS (present in all tables) ==
    - regno (TEXT)
    - name  (TEXT)
    ...

    == TABLE-SPECIFIC COLUMNS ==
    [1_batch_AIML_Results]
      Unique: "JUNE 2022 gpa" (REAL), "SEPTEMBER 2022 gpa" (REAL), ...

    [2_batch_AIML_Results]
      Unique: "APRIL 2023 gpa" (REAL), ...

    == JOIN KEY ==
    All tables share: regno (TEXT, PRIMARY KEY)
    Use INNER JOIN ... ON t1.regno = t2.regno
    """
    if not schemas:
        return ""

    # Collect columns per table
    all_cols: dict[str, list[str]] = {}
    for table, schema_text in schemas.items():
        if schema_text:
            all_cols[table] = _extract_unique_columns(schema_text)

    if not all_cols:
        return "\n\n".join(s for s in schemas.values() if s)

    # Find columns that appear in every table (by name prefix before the space)
    def col_name(col_entry: str) -> str:
        return col_entry.split("(")[0].strip().lower()

    # Count how many tables each column name appears in
    col_presence: dict[str, int] = {}
    for cols in all_cols.values():
        seen_here: set[str] = set()
        for c in cols:
            cn = col_name(c)
            if cn not in seen_here:
                col_presence[cn] = col_presence.get(cn, 0) + 1
                seen_here.add(cn)

    total_tables = len(all_cols)
    shared_names = {cn for cn, count in col_presence.items() if count == total_tables}

    lines: list[str] = []

    # ── Shared columns block ────────────────────────────────────────────
    lines.append("== SHARED COLUMNS (present in all tables) ==")
    for table, cols in all_cols.items():
        for c in cols:
            if col_name(c) in shared_names:
                lines.append(f"  - {c}")
        break  # Only need one table's version; they're identical

    # ── Table-specific columns block ────────────────────────────────────
    lines.append("\n== TABLE-SPECIFIC EXAM / UNIQUE COLUMNS ==")
    for table, cols in all_cols.items():
        unique_cols = [c for c in cols if col_name(c) not in shared_names]
        if unique_cols:
            lines.append(f"\n[{table}]")
            for c in unique_cols:
                lines.append(f"  - {c}")
        else:
            lines.append(f"\n[{table}]  — no table-specific columns")

    # ── Join key reminder ───────────────────────────────────────────────
    lines.append(
        f"\n== JOIN KEY ==\n"
        f"  All tables share: {JOIN_KEY} (TEXT, PRIMARY KEY per table)\n"
        f"  Cross-table join: t1.\"{JOIN_KEY}\" = t2.\"{JOIN_KEY}\""
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Join context builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_join_context(tables: list[str], intent: str) -> str:
    """
    Return a human-readable join-hint string for the SQL generator.
    Empty string for single-table queries.
    """
    if len(tables) <= 1:
        return ""

    # Determine shared batch / branch for the hint header
    batches  = sorted({t.split("_")[0] + "_batch" for t in tables if "_" in t})
    branches = sorted({t.split("_")[-2] for t in tables if "_" in t})

    lines: list[str] = ["== JOIN / UNION CONTEXT =="]

    if intent == QueryIntent.SAME_BATCH:
        lines.append(
            f"Tables are all from the SAME BATCH ({', '.join(batches)}) "
            f"covering branches: {', '.join(branches)}.\n"
            "  Recommended pattern: UNION ALL (same schema, combine rows)\n"
            "  Example:\n"
            f"    SELECT \"regno\", \"name\", \"avg gpa\", '{tables[0]}' AS source\n"
            f"    FROM \"{tables[0]}\"\n"
            "    UNION ALL\n"
            f"    SELECT \"regno\", \"name\", \"avg gpa\", '{tables[1]}' AS source\n"
            f"    FROM \"{tables[1]}\"\n"
            "    -- repeat for remaining tables"
        )

    elif intent == QueryIntent.CROSS_BATCH:
        lines.append(
            f"Tables span MULTIPLE BATCHES ({', '.join(batches)}) "
            f"for branches: {', '.join(branches)}.\n"
            f"  Join key: \"{JOIN_KEY}\" (same student may appear in multiple batches)\n"
            "  Recommended pattern: INNER JOIN on regno for direct comparison\n"
            "  Example:\n"
            f"    SELECT a.\"{JOIN_KEY}\", a.\"name\",\n"
            f"           a.\"avg gpa\" AS batch1_gpa,\n"
            f"           b.\"avg gpa\" AS batch2_gpa\n"
            f"    FROM \"{tables[0]}\" a\n"
            f"    INNER JOIN \"{tables[1]}\" b ON a.\"{JOIN_KEY}\" = b.\"{JOIN_KEY}\""
        )

    elif intent == QueryIntent.ALL_TABLES:
        lines.append(
            "Query spans ALL 9 TABLES across all batches and branches.\n"
            "  Recommended pattern: UNION ALL with a literal 'source' column\n"
            "  Template (repeat for each of the 9 tables):\n"
            "    SELECT \"regno\", \"name\", \"avg gpa\",\n"
            "           '1_batch_AIML_Results' AS batch_branch\n"
            "    FROM \"1_batch_AIML_Results\"\n"
            "    UNION ALL\n"
            "    SELECT \"regno\", \"name\", \"avg gpa\",\n"
            "           '1_batch_CSD_Results' AS batch_branch\n"
            "    FROM \"1_batch_CSD_Results\"\n"
            "    -- ... continue for all 9 tables"
        )

    elif intent in (QueryIntent.SUBQUERY, QueryIntent.AGGREGATION):
        lines.append(
            f"Tables: {', '.join(tables)}\n"
            "  Subquery / aggregation pattern:\n"
            "  - Use a subquery in WHERE for per-table comparisons:\n"
            "      WHERE \"avg gpa\" > (SELECT AVG(\"avg gpa\") FROM \"<same_table>\")\n"
            "  - Use GROUP BY for branch/batch-level aggregation:\n"
            "      SELECT '<branch>' AS branch, AVG(\"avg gpa\") FROM \"<table>\"\n"
            "      UNION ALL ...\n"
            "  - Wrap in an outer SELECT if ranking is needed"
        )

    else:
        lines.append(
            f"Tables involved: {', '.join(tables)}\n"
            f"  Join key: \"{JOIN_KEY}\"\n"
            "  Use UNION ALL to combine rows or INNER JOIN for per-student comparison."
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Per-intent fetch strategies
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_single_table(
    collection: Any, table_name: str
) -> tuple[list[str], dict[str, str]]:
    """
    Returns (retrieved_tables, {table: schema}).
    Handles comma-separated table_name for legacy callers.
    """
    tables = [t.strip() for t in table_name.split(",") if t.strip()]
    if not tables:
        return [], {}
    schemas = _fetch_schemas_for_tables(collection, tables)
    found   = [t for t, s in schemas.items() if s]
    return found, schemas


def _fetch_same_batch_tables(
    collection: Any, table_name: str, entities: list[str]
) -> tuple[list[str], dict[str, str]]:
    """
    Infer the batch from the table_name (or entities), then fetch all 3
    branch tables for that batch.
    """
    # Determine which batch
    batch_prefix = None
    for prefix in BATCH_MAP:
        if table_name.startswith(prefix) or any(prefix in e for e in entities):
            batch_prefix = prefix
            break

    # Fallback: take explicit tables from table_name
    explicit = [t.strip() for t in table_name.split(",") if t.strip()]

    if batch_prefix:
        target_tables = BATCH_MAP[batch_prefix]
        logger.info(f"SAME_BATCH: fetching batch {batch_prefix!r} → {target_tables}")
    else:
        target_tables = explicit or []
        logger.warning(f"SAME_BATCH: could not infer batch, using explicit: {target_tables}")

    schemas = _fetch_schemas_for_tables(collection, target_tables)
    found   = [t for t, s in schemas.items() if s]
    return found, schemas


def _fetch_cross_batch_tables(
    collection: Any, table_name: str, entities: list[str]
) -> tuple[list[str], dict[str, str]]:
    """
    Infer the branch from table_name (or entities), then fetch that branch
    table from every batch (up to 3 tables).
    """
    # Determine which branch
    branch = None
    for b in BRANCH_MAP:
        if b in table_name or any(b in e for e in entities):
            branch = b
            break

    explicit = [t.strip() for t in table_name.split(",") if t.strip()]

    if branch:
        target_tables = BRANCH_MAP[branch]
        logger.info(f"CROSS_BATCH: fetching branch {branch!r} → {target_tables}")
    else:
        target_tables = explicit or []
        logger.warning(f"CROSS_BATCH: could not infer branch, using explicit: {target_tables}")

    schemas = _fetch_schemas_for_tables(collection, target_tables)
    found   = [t for t, s in schemas.items() if s]
    return found, schemas


def _fetch_all_tables(
    collection: Any,
) -> tuple[list[str], dict[str, str]]:
    """Fetch all 9 tables."""
    logger.info("ALL_TABLES: fetching all 9 tables")
    schemas = _fetch_schemas_for_tables(collection, ALL_TABLE_NAMES)
    found   = [t for t, s in schemas.items() if s]
    return found, schemas


def _fetch_subquery_tables(
    collection: Any, table_name: str, entities: list[str]
) -> tuple[list[str], dict[str, str]]:
    """
    For SUBQUERY / AGGREGATION intents the primary table is always needed.
    If the query compares across tables, we also pull related ones.
    The heuristic: if entity text mentions 'batch' or 'branch' (plural /
    comparative), include the full batch; otherwise just the primary table.
    """
    entity_text = " ".join(entities).lower()
    comparative = any(kw in entity_text for kw in
                      ("compare", "highest", "lowest", "best", "rank",
                       "across", "all branches", "all batches"))

    explicit = [t.strip() for t in table_name.split(",") if t.strip()]

    if comparative and explicit:
        # Try to expand to the full batch of the first explicit table
        batch_prefix = None
        for prefix in BATCH_MAP:
            if explicit[0].startswith(prefix):
                batch_prefix = prefix
                break
        target_tables = BATCH_MAP.get(batch_prefix, explicit) if batch_prefix else explicit
    else:
        target_tables = explicit if explicit else [table_name]

    logger.info(f"SUBQUERY/AGG: fetching {target_tables}")
    schemas = _fetch_schemas_for_tables(collection, target_tables)
    found   = [t for t, s in schemas.items() if s]
    return found, schemas


# ─────────────────────────────────────────────────────────────────────────────
# Public strategy router
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_schemas_for_intent(
    table_name: str,
    intent: str,
    entities: list[str],
    config_path: str,
) -> dict:
    """
    Strategy-aware schema retrieval.  Called by action_node.

    Parameters
    ──────────
    table_name   Comma-separated table name(s) from thought_node.
    intent       A QueryIntent constant.
    entities     Extracted entities from thought_node.
    config_path  Path to settings.yaml.

    Returns
    ───────
    {
      "raw_schema":        str,
      "compressed_schema": str,
      "join_context":      str,
      "retrieved_tables":  list[str],
      "sql_pattern":       str,
    }
    """
    config     = _load_config(config_path)
    collection = _get_collection(config)

    # ── Route to the right fetch strategy ───────────────────────────────
    if intent == QueryIntent.SINGLE_TABLE or intent == QueryIntent.UNCERTAIN:
        found, schemas = _fetch_single_table(collection, table_name)

    elif intent == QueryIntent.SAME_BATCH:
        found, schemas = _fetch_same_batch_tables(collection, table_name, entities)

    elif intent == QueryIntent.CROSS_BATCH:
        found, schemas = _fetch_cross_batch_tables(collection, table_name, entities)

    elif intent == QueryIntent.ALL_TABLES:
        found, schemas = _fetch_all_tables(collection)

    elif intent in (QueryIntent.SUBQUERY, QueryIntent.AGGREGATION):
        found, schemas = _fetch_subquery_tables(collection, table_name, entities)

    else:
        # Unknown intent — best-effort single-table fetch
        logger.warning(f"Unknown intent {intent!r} — falling back to single-table fetch")
        found, schemas = _fetch_single_table(collection, table_name)

    if not found:
        logger.error(f"retrieve_schemas_for_intent: no schemas found for {table_name!r}")
        return {
            "raw_schema":        "",
            "compressed_schema": "",
            "join_context":      "",
            "retrieved_tables":  [],
            "sql_pattern":       SQLPattern.NONE,
        }

    # ── Build outputs ────────────────────────────────────────────────────
    raw_schema        = "\n\n".join(s for s in schemas.values() if s)
    compressed_schema = _build_compressed_schema(schemas)
    join_context      = _build_join_context(found, intent)
    sql_pattern       = _INTENT_TO_PATTERN.get(intent, SQLPattern.SIMPLE_SELECT)

    logger.info(
        f"retrieve_schemas_for_intent → intent={intent}  "
        f"tables={found}  pattern={sql_pattern}"
    )

    return {
        "raw_schema":        raw_schema,
        "compressed_schema": compressed_schema,
        "join_context":      join_context,
        "retrieved_tables":  found,
        "sql_pattern":       sql_pattern,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy / utility functions  (kept with identical signatures)
# ─────────────────────────────────────────────────────────────────────────────

def search_table_schema(table_name: str, config_path: str) -> str | None:
    """
    Original single-table schema lookup.
    Kept for backward compatibility; prefer retrieve_schemas_for_intent().

    Handles comma-separated table_name (multiple tables → concatenated).
    """
    logger.info(f"search_table_schema: {table_name!r}")
    config     = _load_config(config_path)
    collection = _get_collection(config)

    tables  = [t.strip() for t in table_name.split(",") if t.strip()]
    schemas = []
    for table in tables:
        schema = _fetch_schema_for_table(collection, table)
        schemas.append(schema if schema else f"No schema found for table: {table}")

    combined = "\n\n".join(schemas)
    return combined if schemas else None


def search_relevant_tables_by_content(
    entities: list[str],
    nl_text: str,
    config_path: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search for tables that might contain columns/content relevant to
    *entities* and *nl_text*.  Used by semantic_search_node when the LLM
    could not identify a table name.

    Returns list of {table_name, confidence_score, reason, schema_preview}.
    """
    logger.info(f"search_relevant_tables_by_content: entities={entities}")
    config     = _load_config(config_path)
    collection = _get_collection(config)

    candidate_tables: list[dict] = []

    # Build a diverse set of search queries
    search_queries: list[str] = []
    for entity in entities:
        search_queries.append(f"column {entity}")
        search_queries.append(f"field {entity}")
    if len(entities) > 1:
        search_queries.append(f"table with {' and '.join(entities)}")
    search_queries.append(nl_text)

    for query in search_queries:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            if not (results["documents"] and results["documents"][0]):
                continue

            for i, doc in enumerate(results["documents"][0]):
                metadata   = (results["metadatas"][0][i]  if results["metadatas"]  else {})
                distance   = (results["distances"][0][i]  if results["distances"]  else 1.0)
                table_name = metadata.get("table_name", "Unknown")
                confidence = max(0.0, 1.0 - distance)

                existing = next(
                    (t for t in candidate_tables if t["table_name"] == table_name),
                    None,
                )
                if existing:
                    if confidence > existing["confidence_score"]:
                        existing["confidence_score"] = confidence
                        existing["reason"]           = f"Found relevant content for: {query!r}"
                else:
                    candidate_tables.append({
                        "table_name":       table_name,
                        "confidence_score": confidence,
                        "reason":           f"Contains content matching: {query!r}",
                        "schema_preview":   doc[:200] + "…" if len(doc) > 200 else doc,
                    })

        except Exception as e:
            logger.warning(f"Error searching with query {query!r}: {e}")
            continue

    # Sort by confidence, drop low-signal results
    candidate_tables.sort(key=lambda x: x["confidence_score"], reverse=True)
    candidate_tables = [t for t in candidate_tables if t["confidence_score"] > 0.2]

    logger.info(f"search_relevant_tables_by_content → {len(candidate_tables)} candidates")
    return candidate_tables[:top_k]


def get_all_table_names(config_path: str) -> list[str]:
    """Return all table names currently stored in the vector store."""
    config     = _load_config(config_path)
    collection = _get_collection(config)

    try:
        all_data   = collection.get(include=["metadatas"])
        table_names: set[str] = set()
        if all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                if "table_name" in meta:
                    table_names.add(meta["table_name"])
        return list(table_names)
    except Exception as e:
        logger.error(f"get_all_table_names error: {e}")
        return []



# import chromadb
# from chromadb.utils import embedding_functions
# from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
# import yaml
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def search_table_schema(table_name, config_path):
#     """Search for table schema in Chroma vector store for one or more table names."""
#     logger.info(f"Searching schema for table(s): {table_name}")
#     config = yaml.safe_load(open(config_path))
    
#     # Initialize PersistentClient with persist_dir
#     chroma_client = chromadb.PersistentClient(
#         path=config['persist_dir'],
#         settings=Settings(is_persistent=True),
#         tenant=DEFAULT_TENANT,
#         database=DEFAULT_DATABASE
#     )
    
#     try:
#         # Debug: List all collections
#         collections = chroma_client.list_collections()
#         logger.info(f"Available collections: {[col.name for col in collections]}")
        
#         collection = chroma_client.get_collection(
#             name=config['collection_name'],
#             embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#                 model_name=config['embedding_model']
#             )
#         )
        
#         # Split table_name string into individual table names
#         table_names = [name.strip() for name in table_name.split(',')]
#         logger.info(f"Processing table names: {table_names}")
        
#         # Query schemas for each table
#         schemas = []
#         for table in table_names:
#             logger.info(f"Querying schema for table: {table}")
#             results = collection.query(
#                 query_texts=[f"Table: {table}"],
#                 n_results=1,
#                 where={"table_name": table}
#             )
#             logger.info(f"Query results for {table}: {results['documents']}")
#             if results['documents'] and results['documents'][0]:
#                 schemas.append(results['documents'][0][0])
#             else:
#                 logger.warning(f"No schema found for table: {table}")
#                 schemas.append(f"No schema found for table: {table}")
        
#         # Combine schemas into a single string
#         combined_schema = "\n\n".join(schemas)
#         return combined_schema if schemas else None
        
#     except Exception as e:
#         logger.error(f"Error querying collection: {e}")
#         raise
# # Add these functions to the existing file

# def search_relevant_tables_by_content(entities, nl_text, config_path, top_k=5):
#     """Search for tables that might contain relevant columns based on entities."""
#     logger.info(f"Searching for tables containing entities: {entities}")
#     config = yaml.safe_load(open(config_path))
    
#     chroma_client = chromadb.PersistentClient(
#         path=config['persist_dir'],
#         settings=Settings(is_persistent=True),
#         tenant=DEFAULT_TENANT,
#         database=DEFAULT_DATABASE
#     )
    
#     try:
#         collection = chroma_client.get_collection(
#             name=config['collection_name'],
#             embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#                 model_name=config['embedding_model']
#             )
#         )
        
#         candidate_tables = []
        
#         # Search using different strategies
#         search_queries = []
        
#         # 1. Search by individual entities
#         for entity in entities:
#             search_queries.append(f"column {entity}")
#             search_queries.append(f"field {entity}")
        
#         # 2. Search by combined context
#         if len(entities) > 1:
#             search_queries.append(f"table with {' and '.join(entities)}")
        
#         # 3. Search by natural language context
#         search_queries.append(nl_text)
        
#         for query in search_queries:
#             try:
#                 results = collection.query(
#                     query_texts=[query],
#                     n_results=top_k,
#                     include=['documents', 'metadatas', 'distances']
#                 )
                
#                 if results['documents'] and results['documents'][0]:
#                     for i, doc in enumerate(results['documents'][0]):
#                         metadata = results['metadatas'][0][i] if results['metadatas'] else {}
#                         distance = results['distances'][0][i] if results['distances'] else 1.0
                        
#                         table_name = metadata.get('table_name', 'Unknown')
#                         confidence = max(0, 1 - distance)  # Convert distance to confidence
                        
#                         # Check if this table already exists in candidates
#                         existing = next((t for t in candidate_tables if t['table_name'] == table_name), None)
#                         if existing:
#                             # Update confidence if this is better
#                             if confidence > existing['confidence_score']:
#                                 existing['confidence_score'] = confidence
#                                 existing['reason'] = f"Found relevant content for query: {query}"
#                         else:
#                             candidate_tables.append({
#                                 'table_name': table_name,
#                                 'confidence_score': confidence,
#                                 'reason': f"Contains content matching: {query}",
#                                 'schema_preview': doc[:200] + "..." if len(doc) > 200 else doc
#                             })
                            
#             except Exception as e:
#                 logger.warning(f"Error searching with query '{query}': {e}")
#                 continue
        
#         # Sort by confidence and remove duplicates
#         candidate_tables = sorted(candidate_tables, key=lambda x: x['confidence_score'], reverse=True)
        
#         # Filter out very low confidence results
#         candidate_tables = [t for t in candidate_tables if t['confidence_score'] > 0.2]
        
#         logger.info(f"Found {len(candidate_tables)} candidate tables")
#         return candidate_tables[:top_k]  # Return top results
        
#     except Exception as e:
#         logger.error(f"Error in semantic search: {e}")
#         return []

# def get_all_table_names(config_path):
#     """Get all available table names from the vector store."""
#     config = yaml.safe_load(open(config_path))
    
#     chroma_client = chromadb.PersistentClient(
#         path=config['persist_dir'],
#         settings=Settings(is_persistent=True),
#         tenant=DEFAULT_TENANT,
#         database=DEFAULT_DATABASE
#     )
    
#     try:
#         collection = chroma_client.get_collection(
#             name=config['collection_name'],
#             embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#                 model_name=config['embedding_model']
#             )
#         )
        
#         # Get all documents to extract table names
#         all_data = collection.get(include=['metadatas'])
#         table_names = set()
        
#         if all_data['metadatas']:
#             for metadata in all_data['metadatas']:
#                 if 'table_name' in metadata:
#                     table_names.add(metadata['table_name'])
        
#         return list(table_names)
        
#     except Exception as e:
#         logger.error(f"Error getting table names: {e}")
#         return []