"""Utility functions for UMLS mapping"""

import re
from typing import Dict

# Medical abbreviation dictionary
MEDICAL_ABBREV = {
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "dm": "diabetes mellitus",
    "mi": "myocardial infarction",
    "htn": "hypertension",
    "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "ckd": "chronic kidney disease",
}

def normalize_text(text: str) -> str:
    """Normalize entity text"""
    text = text.lower()
    text = ' '.join(text.split())  # Normalize whitespace
    text = text.strip('.,;:()[]{}')
    return text

def expand_abbreviations(text: str, abbrev_dict: Dict[str, str] = None) -> str:
    """Expand medical abbreviations"""
    if abbrev_dict is None:
        abbrev_dict = MEDICAL_ABBREV
    
    words = text.lower().split()
    expanded = []
    for word in words:
        expanded.append(abbrev_dict.get(word, word))
    return ' '.join(expanded)
