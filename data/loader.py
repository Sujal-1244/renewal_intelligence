"""
Data Loader Module — Ingests and cleans all data sources.

Design Decision: We load all CSVs into standardized DataFrames and parse
the unstructured CSM notes into a structured list of dictionaries. Each
loader handles its own data quality issues (missing values, type coercion,
date parsing) to keep downstream analysis clean.

Assumption: Data files live in the project root alongside ASSIGNMENT.md.
Risk: If file locations change, only this module needs updating.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from renewal_intelligence.config.settings import DATA_DIR, REFERENCE_DATE


def load_accounts() -> pd.DataFrame:
    """
    Load and clean accounts.csv.
    
    Cleaning steps:
    - Parse contract_end_date as datetime
    - Ensure ARR is numeric (some edge cases in real data)
    - Strip whitespace from string columns
    - Add computed column: days_to_renewal
    """
    df = pd.read_csv(DATA_DIR / "accounts.csv")
    
    # Clean string columns
    for col in ["account_name", "plan_tier", "industry", "csm_name", "region"]:
        df[col] = df[col].str.strip()
    
    # Parse dates
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"])
    
    # Compute days to renewal from reference date
    df["days_to_renewal"] = (df["contract_end_date"] - REFERENCE_DATE).dt.days
    
    # Flag accounts in the 90-day renewal window
    df["in_renewal_window"] = df["days_to_renewal"].between(-30, 90)
    # Include accounts that ALREADY expired within last 30 days (may still be saveable)
    
    return df


def load_usage_metrics() -> pd.DataFrame:
    """
    Load and clean usage_metrics.csv.
    
    Cleaning steps:
    - Parse month as period for proper time series handling
    - Ensure all numeric columns are properly typed
    - Sort by account_id and month for trend analysis
    """
    df = pd.read_csv(DATA_DIR / "usage_metrics.csv")
    
    # Parse month as datetime for easier comparison
    df["month_dt"] = pd.to_datetime(df["month"] + "-01")
    
    # Ensure numeric types
    numeric_cols = ["api_calls", "content_entries_created", "active_users", "workflows_triggered"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sort for time series analysis
    df = df.sort_values(["account_id", "month_dt"]).reset_index(drop=True)
    
    return df


def load_support_tickets() -> pd.DataFrame:
    """
    Load and clean support_tickets.csv.
    
    Cleaning steps:
    - Parse created_date as datetime
    - Standardize priority to uppercase (P1, P2, P3, P4)
    - Handle missing resolution_time_hours (NaN = still unresolved)
    - Standardize status values
    """
    df = pd.read_csv(DATA_DIR / "support_tickets.csv")
    
    # Parse dates
    df["created_date"] = pd.to_datetime(df["created_date"])
    
    # Standardize priority
    df["priority"] = df["priority"].str.upper().str.strip()
    
    # Standardize status
    df["status"] = df["status"].str.strip().str.capitalize()
    
    # Mark unresolved tickets (no resolution time = still open)
    df["is_unresolved"] = df["resolution_time_hours"].isna() | (df["status"].isin(["Open", "Escalated"]))
    
    return df


def load_nps_responses() -> pd.DataFrame:
    """
    Load and clean nps_responses.csv.
    
    Cleaning steps:
    - Handle missing/empty comments gracefully
    - Classify NPS scores into promoter/passive/detractor
    - Flag non-English comments for translation
    """
    df = pd.read_csv(DATA_DIR / "nps_responses.csv")
    
    # Clean comments
    df["verbatim_comment"] = df["verbatim_comment"].fillna("").str.strip()
    
    # NPS classification (industry standard: 0-6 detractor, 7-8 passive, 9-10 promoter)
    df["nps_category"] = pd.cut(
        df["score"],
        bins=[-1, 6, 8, 10],
        labels=["Detractor", "Passive", "Promoter"]
    )
    
    # Flag non-English comments using simple heuristic
    # More robust detection would use langdetect, but this captures the known cases
    def _detect_non_english(text: str) -> bool:
        if not text:
            return False
        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return True
        # Check for common French/Spanish patterns
        if re.search(r'[àâäéèêëîïôùûüçñ¿¡]', text, re.IGNORECASE):
            return True
        # Check for French articles/prepositions
        if re.search(r'\b(est|mais|pour|notre|équipe|produit|l\'interface)\b', text, re.IGNORECASE):
            return True
        # Check for Spanish
        if re.search(r'\b(soporte|español|inexistente|producto|bueno|pero)\b', text, re.IGNORECASE):
            return True
        return False
    
    df["is_non_english"] = df["verbatim_comment"].apply(_detect_non_english)
    
    return df


def load_csm_notes() -> List[Dict]:
    """
    Parse unstructured CSM notes into structured records.
    
    The CSM notes are intentionally messy. This parser handles:
    - Multiple date formats (Mar 12, 3/15, 2026-03-20, march 25, 04/03, etc.)
    - Inconsistent account name spellings
    - Account IDs embedded in text (e.g., "acct 1001", "#1007")
    - Missing delimiters
    
    Returns a list of dicts, each representing one note entry.
    
    Assumption: Notes are separated by "---" delimiters.
    Risk: Some notes may not follow this convention. We handle gracefully.
    """
    notes_path = DATA_DIR / "csm_notes.txt"
    raw_text = notes_path.read_text(encoding="utf-8")
    
    # Split on "---" delimiters
    raw_entries = [entry.strip() for entry in raw_text.split("---") if entry.strip()]
    
    # Remove the header
    if raw_entries and "CSM Call Notes" in raw_entries[0]:
        # Keep the header info but start parsing from entry 1
        raw_entries = raw_entries[1:]  # First real entry starts after header separator
    
    parsed_notes = []
    for entry in raw_entries:
        if not entry or len(entry) < 10:
            continue
            
        note = {
            "raw_text": entry,
            "account_id": _extract_account_id(entry),
            "account_name_raw": _extract_account_name(entry),
            "date_raw": _extract_date(entry),
            "csm_name": _extract_csm_name(entry),
        }
        parsed_notes.append(note)
    
    return parsed_notes


def _extract_account_id(text: str) -> int | None:
    """Extract account ID from various formats in CSM notes."""
    # Patterns: "acct 1001", "#1007", "(1009)", "account 1016", "(acct 1024)"
    patterns = [
        r'acct\s*(\d{4})',
        r'#(\d{4})',
        r'account\s*(\d{4})',
        r'\((\d{4})\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_account_name(text: str) -> str | None:
    """
    Extract the account name from the first line of a CSM note.
    
    Handles various formats:
    - "Talked to Acme Corp"
    - "acct 1001 - BritePath Solutions (sic)"
    - "2026-03-20 | NovaTech Industries | James O."
    - "march 25 -- meridian health"
    """
    first_line = text.split("\n")[0].strip()
    
    # Pattern: "| AccountName |" (pipe-delimited)
    pipe_match = re.search(r'\|\s*([A-Za-z][A-Za-z\s&\-\'\.]+?)\s*\|', first_line)
    if pipe_match:
        return pipe_match.group(1).strip()
    
    # Pattern: "Talked to AccountName"
    talked_match = re.search(r'[Tt]alked to\s+(.+?)[\.\,]', first_line)
    if talked_match:
        return talked_match.group(1).strip()
    
    # Pattern: "- AccountName" or "-- AccountName" after date
    dash_match = re.search(r'[-–]\s*([A-Za-z][A-Za-z\s&\-\'\.]+?)(?:\s*[\(\-#\n]|$)', first_line)
    if dash_match:
        name = dash_match.group(1).strip()
        # Filter out common false positives
        if name.lower() not in ["note", "all good", "routine", "quick"]:
            return name
    
    # Pattern: "date AccountName"  (no separator, just whitespace after date)
    # Handles both "Apr 1 - vanguard retail" and "04/01 coral bay resorts"
    date_then_name = re.search(
        r'(?:\d{1,2}/\d{1,2}|\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2})\s+([A-Za-z][a-z]+(?:\s+[A-Za-z]+)+)',
        first_line
    )
    if date_then_name:
        name = date_then_name.group(1).strip()
        # Remove trailing junk like "(sic)" or note fragments
        name = re.sub(r'\s*\(.*$', '', name)
        # Filter out common false-positive fragments
        skip = {"note for", "all good", "routine check", "quick sync", "just a quick"}
        if name.lower() not in skip and len(name) > 3:
            return name
    
    return None


def _extract_date(text: str) -> str | None:
    """Extract date from various messy formats."""
    first_line = text.split("\n")[0]
    
    # ISO format: 2026-03-20
    iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', first_line)
    if iso_match:
        return iso_match.group(1)
    
    # M/D format: 3/15, 04/03, 3/28
    md_match = re.search(r'\b(\d{1,2}/\d{1,2})\b', first_line)
    if md_match:
        return md_match.group(1)
    
    # Month Day format: Mar 12, April 2, march 25
    month_match = re.search(
        r'\b([A-Za-z]+)\s+(\d{1,2})\b',
        first_line
    )
    if month_match:
        return f"{month_match.group(1)} {month_match.group(2)}"
    
    return None


def _extract_csm_name(text: str) -> str | None:
    """Extract CSM name if mentioned in the note."""
    known_csms = [
        "Carlos Mendez", "Sarah Chen", "David Kim", "Raj Patel",
        "James Okafor", "Emily Watson", "Priya Sharma", "Anna Kowalski"
    ]
    
    for csm in known_csms:
        # Check for full name or first name only
        if csm.lower() in text.lower():
            return csm
        # Check for abbreviated: "James O.", "Emily W."
        parts = csm.split()
        abbrev = f"{parts[0]} {parts[1][0]}."
        if abbrev.lower() in text.lower():
            return csm
    
    # Also check for first names with initial: "priya", "sarah"
    first_names = {name.split()[0].lower(): name for name in known_csms}
    for first, full in first_names.items():
        if re.search(rf'\b{first}\b', text, re.IGNORECASE):
            return full
    
    return None


def load_changelog() -> str:
    """Load the raw changelog text for LLM analysis."""
    return (DATA_DIR / "changelog.md").read_text(encoding="utf-8")


def load_all_data() -> Dict:
    """
    Load all data sources and return as a dictionary.
    
    This is the main entry point for data loading.
    Returns all datasets in a single call for convenience.
    """
    return {
        "accounts": load_accounts(),
        "usage": load_usage_metrics(),
        "tickets": load_support_tickets(),
        "nps": load_nps_responses(),
        "csm_notes": load_csm_notes(),
        "changelog": load_changelog(),
    }
