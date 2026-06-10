"""
Entity Reconciliation Module — Fuzzy matching for account name resolution.

Design Decision: CSM notes contain misspelled account names (BritePath vs
BrightPath, Pinacle vs Pinnacle, etc.) and sometimes reference accounts
by ID. We use RapidFuzz for fast, accurate fuzzy matching to reconcile
these inconsistencies.

Why RapidFuzz over FuzzyWuzzy?
- 10x faster (C++ backend vs pure Python)
- Same API surface
- Better handling of partial matches
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from rapidfuzz import fuzz, process

from renewal_intelligence.config.settings import FUZZY_MATCH_THRESHOLD


def reconcile_csm_notes(
    csm_notes: List[Dict],
    accounts_df: pd.DataFrame
) -> List[Dict]:
    """
    Match CSM notes to accounts using fuzzy name matching and ID lookup.
    
    Strategy (in priority order):
    1. If account_id is explicitly mentioned → direct match
    2. If account_name is present → fuzzy match against accounts.csv
    3. If neither → flag as unmatched for manual review
    
    Args:
        csm_notes: Parsed CSM note entries
        accounts_df: The accounts DataFrame
        
    Returns:
        CSM notes enriched with matched account_id and match confidence
    """
    # Build lookup structures
    account_names = dict(zip(accounts_df["account_name"], accounts_df["account_id"]))
    name_list = list(account_names.keys())
    id_to_name = dict(zip(accounts_df["account_id"], accounts_df["account_name"]))
    
    enriched_notes = []
    for note in csm_notes:
        enriched = note.copy()
        
        # Strategy 1: Direct ID match
        if note.get("account_id") and note["account_id"] in id_to_name:
            enriched["matched_account_id"] = note["account_id"]
            enriched["matched_account_name"] = id_to_name[note["account_id"]]
            enriched["match_method"] = "id_direct"
            enriched["match_confidence"] = 100
        
        # Strategy 2: Fuzzy name match
        elif note.get("account_name_raw"):
            match = _fuzzy_match_account(note["account_name_raw"], name_list)
            if match:
                matched_name, score = match
                enriched["matched_account_id"] = account_names[matched_name]
                enriched["matched_account_name"] = matched_name
                enriched["match_method"] = "fuzzy_name"
                enriched["match_confidence"] = score
            else:
                enriched["matched_account_id"] = None
                enriched["matched_account_name"] = None
                enriched["match_method"] = "unmatched"
                enriched["match_confidence"] = 0
        
        else:
            enriched["matched_account_id"] = None
            enriched["matched_account_name"] = None
            enriched["match_method"] = "no_identifier"
            enriched["match_confidence"] = 0
        
        enriched_notes.append(enriched)
    
    return enriched_notes


def _fuzzy_match_account(
    raw_name: str,
    canonical_names: List[str]
) -> Optional[Tuple[str, float]]:
    """
    Fuzzy match a raw account name against the canonical list.
    
    Uses multiple matching strategies and takes the best:
    - Token sort ratio: Handles word reordering
    - Partial ratio: Handles abbreviations/truncations
    - Weighted ratio: Standard overall similarity
    
    Args:
        raw_name: The potentially misspelled name from CSM notes
        canonical_names: List of official account names
        
    Returns:
        Tuple of (matched_name, confidence_score) or None if no good match
    """
    if not raw_name or not canonical_names:
        return None
    
    # Clean the input name
    clean_name = raw_name.strip().lower()
    
    # Try multiple matching strategies
    best_match = None
    best_score = 0
    
    for name in canonical_names:
        # Weighted combination of different matching strategies
        token_sort = fuzz.token_sort_ratio(clean_name, name.lower())
        partial = fuzz.partial_ratio(clean_name, name.lower())
        weighted = fuzz.WRatio(clean_name, name.lower())
        
        # Take the max of all strategies (best representation of similarity)
        score = max(token_sort, partial, weighted)
        
        if score > best_score:
            best_score = score
            best_match = name
    
    if best_score >= FUZZY_MATCH_THRESHOLD:
        return (best_match, best_score)
    
    return None


def build_account_data_map(
    accounts_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    nps_df: pd.DataFrame,
    csm_notes: List[Dict]
) -> Dict[int, Dict]:
    """
    Build a unified data map keyed by account_id.
    
    This aggregates all data sources into a single lookup structure
    that downstream analysis modules can use without re-joining data.
    
    Args:
        accounts_df: Accounts DataFrame
        usage_df: Usage metrics DataFrame
        tickets_df: Support tickets DataFrame
        nps_df: NPS responses DataFrame
        csm_notes: Reconciled CSM notes
        
    Returns:
        Dictionary mapping account_id → aggregated data dict
    """
    account_map = {}
    
    for _, row in accounts_df.iterrows():
        aid = row["account_id"]
        
        # Get usage data for this account
        acct_usage = usage_df[usage_df["account_id"] == aid].sort_values("month_dt")
        
        # Get tickets for this account
        acct_tickets = tickets_df[tickets_df["account_id"] == aid]
        
        # Get NPS for this account
        acct_nps = nps_df[nps_df["account_id"] == aid]
        
        # Get CSM notes for this account
        acct_notes = [n for n in csm_notes if n.get("matched_account_id") == aid]
        
        account_map[aid] = {
            "account": row.to_dict(),
            "usage": acct_usage,
            "tickets": acct_tickets,
            "nps": acct_nps,
            "csm_notes": acct_notes,
        }
    
    return account_map
