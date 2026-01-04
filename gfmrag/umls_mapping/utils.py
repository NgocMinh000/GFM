"""Utility functions for UMLS mapping"""

import re
from typing import Dict

# Medical abbreviation dictionary (from requirements)
MEDICAL_ABBREV = {
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "dm": "diabetes mellitus",
    "mi": "myocardial infarction",
    "copd": "chronic obstructive pulmonary disease",
    "htn": "hypertension",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "ckd": "chronic kidney disease",
    "ra": "rheumatoid arthritis",
    "oa": "osteoarthritis",
    "af": "atrial fibrillation",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "cva": "cerebrovascular accident",
    "gerd": "gastroesophageal reflux disease",
    "ibs": "irritable bowel syndrome",
    "uti": "urinary tract infection",
    "hiv": "human immunodeficiency virus",
    "aids": "acquired immunodeficiency syndrome",
}

def normalize_text(text: str) -> str:
    """
    Normalize entity text

    Steps:
    1. Lowercase
    2. Convert Roman numerals to Arabic (i→1, ii→2, etc.)
    3. Normalize whitespace
    4. Strip punctuation
    """
    text = text.lower()

    # Roman numeral to Arabic conversion (must wrap with spaces to avoid partial matches)
    text = ' ' + text + ' '
    roman_to_arabic = [
        (' i ', ' 1 '),
        (' ii ', ' 2 '),
        (' iii ', ' 3 '),
        (' iv ', ' 4 '),
        (' v ', ' 5 '),
    ]
    for roman, arabic in roman_to_arabic:
        text = text.replace(roman, arabic)
    text = text.strip()

    # Normalize whitespace
    text = ' '.join(text.split())

    # Strip common punctuation
    text = text.strip('.,;:()[]{}"\'-')

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
