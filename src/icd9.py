"""
ICD-9-CM and anti-diabetic drug ontologies.

Only public, standard classification structure is encoded here:

* ICD-9-CM chapter ranges (the 17 numeric chapters plus the V and E
  supplementary classifications).
* The implicit ICD-9 code hierarchy: full code (e.g. ``250.83``) ->
  four-character subcategory (``250.8``) -> three-character category
  (``250``) -> chapter -> root.
* Pharmacological classes of the 23 drugs recorded in the Diabetes
  130-US Hospitals dataset.

No values are learned or estimated in this module.
"""
from __future__ import annotations

import math

ROOT = "ROOT"

# (low, high, chapter id, description) for numeric ICD-9 codes
ICD9_CHAPTERS = [
    (1, 139, "C01", "Infectious and parasitic diseases"),
    (140, 239, "C02", "Neoplasms"),
    (240, 279, "C03", "Endocrine, nutritional, metabolic and immunity"),
    (280, 289, "C04", "Diseases of the blood"),
    (290, 319, "C05", "Mental disorders"),
    (320, 389, "C06", "Nervous system and sense organs"),
    (390, 459, "C07", "Circulatory system"),
    (460, 519, "C08", "Respiratory system"),
    (520, 579, "C09", "Digestive system"),
    (580, 629, "C10", "Genitourinary system"),
    (630, 679, "C11", "Pregnancy and childbirth"),
    (680, 709, "C12", "Skin and subcutaneous tissue"),
    (710, 739, "C13", "Musculoskeletal and connective tissue"),
    (740, 759, "C14", "Congenital anomalies"),
    (760, 779, "C15", "Perinatal conditions"),
    (780, 799, "C16", "Symptoms, signs and ill-defined conditions"),
    (800, 999, "C17", "Injury and poisoning"),
]
CHAPTER_V = ("CV", "Supplementary: factors influencing health status")
CHAPTER_E = ("CE", "Supplementary: external causes of injury")

CHAPTER_NAMES = {c: name for _, _, c, name in ICD9_CHAPTERS}
CHAPTER_NAMES[CHAPTER_V[0]] = CHAPTER_V[1]
CHAPTER_NAMES[CHAPTER_E[0]] = CHAPTER_E[1]


def normalise_code(code) -> str | None:
    """Return a clean ICD-9 code string, or None for missing values."""
    if code is None:
        return None
    if isinstance(code, float) and math.isnan(code):
        return None
    text = str(code).strip()
    if not text or text == "?":
        return None
    return text


def chapter_of(code: str) -> str:
    """Map an ICD-9 code to its chapter id."""
    head = code[0].upper()
    if head == "V":
        return CHAPTER_V[0]
    if head == "E":
        return CHAPTER_E[0]
    try:
        value = int(float(code.split(".")[0]))
    except ValueError:
        return ROOT
    for low, high, chapter, _ in ICD9_CHAPTERS:
        if low <= value <= high:
            return chapter
    return ROOT


def category_of(code: str) -> str:
    """Three-character ICD-9 category (``250.83`` -> ``250``)."""
    return code.split(".")[0]


def subcategory_of(code: str) -> str:
    """Four-character ICD-9 subcategory (``250.83`` -> ``250.8``)."""
    if "." not in code:
        return code
    stem, digits = code.split(".", 1)
    return f"{stem}.{digits[:1]}" if digits else stem


def ancestors(code: str) -> list[str]:
    """
    Ordered ancestor chain of a code, most specific first, ending at the
    root.  Level prefixes keep concepts from different levels distinct;
    a level is skipped when it would repeat the text of the next, more
    general level (``428`` has no subcategory, ``250.8`` has no finer
    code).
    """
    levels = [("L4", code), ("L3", subcategory_of(code)),
              ("L2", category_of(code))]
    chain = []
    for i, (level, text) in enumerate(levels):
        if i + 1 < len(levels) and levels[i + 1][1] == text:
            continue
        chain.append(f"{level}:{text}")
    chain.extend([f"L1:{chapter_of(code)}", ROOT])
    return chain


# ──────────────────────────────────────────────
# Anti-diabetic drugs present in the dataset
# ──────────────────────────────────────────────

DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

DRUG_CLASSES = {
    "metformin": ["biguanide"],
    "repaglinide": ["meglitinide"],
    "nateglinide": ["meglitinide"],
    "chlorpropamide": ["sulfonylurea"],
    "glimepiride": ["sulfonylurea"],
    "acetohexamide": ["sulfonylurea"],
    "glipizide": ["sulfonylurea"],
    "glyburide": ["sulfonylurea"],
    "tolbutamide": ["sulfonylurea"],
    "tolazamide": ["sulfonylurea"],
    "pioglitazone": ["thiazolidinedione"],
    "rosiglitazone": ["thiazolidinedione"],
    "troglitazone": ["thiazolidinedione"],
    "acarbose": ["alpha_glucosidase_inhibitor"],
    "miglitol": ["alpha_glucosidase_inhibitor"],
    "insulin": ["insulin"],
    "glyburide-metformin": ["sulfonylurea", "biguanide"],
    "glipizide-metformin": ["sulfonylurea", "biguanide"],
    "glimepiride-pioglitazone": ["sulfonylurea", "thiazolidinedione"],
    "metformin-rosiglitazone": ["biguanide", "thiazolidinedione"],
    "metformin-pioglitazone": ["biguanide", "thiazolidinedione"],
    # the two columns below are never prescribed in the released data
    "examide": ["other"],
    "citoglipton": ["other"],
}

DRUG_STATES = ["No", "Steady", "Up", "Down"]
