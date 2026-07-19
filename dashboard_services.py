from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import subprocess
import sys
import unicodedata
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import lasio
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
LOCAL_WORKSPACE_ROOT = Path(
    os.environ.get("ZONA_ROKAN_WORKSPACE_ROOT", r"D:\HALLIBURTON CONSULTING")
)
PACKAGED_DATA_ROOT = APP_DIR / "data"
PACKAGED_SOURCE_ROOT = PACKAGED_DATA_ROOT / "controlled_sources"
WORKSPACE_ROOT = LOCAL_WORKSPACE_ROOT if LOCAL_WORKSPACE_ROOT.exists() else APP_DIR
LOCAL_DATA_ROOT = LOCAL_WORKSPACE_ROOT / "Data Nations" / "Zona Rokan"
DATA_ROOT = LOCAL_DATA_ROOT if LOCAL_DATA_ROOT.exists() else PACKAGED_DATA_ROOT / "zona_rokan"


def _local_or_packaged(local_path: Path, packaged_name: str) -> Path:
    """Prefer the live Windows source, with a repository copy for cloud hosting."""
    return local_path if local_path.exists() else PACKAGED_SOURCE_ROOT / packaged_name


DATA_AVAILABILITY_WORKBOOK = _local_or_packaged(
    LOCAL_WORKSPACE_ROOT / "Data Nations" / "001_Rokan Block Data Availability Final.xlsx",
    "001_Rokan Block Data Availability Final.xlsx",
)
SCREENING_WORKBOOK = _local_or_packaged(
    LOCAL_WORKSPACE_ROOT / "Data Nations" / "002_Well Screening Matrix Final.xlsx",
    "002_Well Screening Matrix Final.xlsx",
)
SCHEMATIC_WORKBOOK = _local_or_packaged(
    LOCAL_WORKSPACE_ROOT / "Data Nations" / "well schematic Rokan.xlsx",
    "well schematic Rokan.xlsx",
)
SCHEMATIC_CACHE_ROOT = APP_DIR / "data" / "schematic_workbook_cache"
SCHEMATIC_CACHE_INDEX = SCHEMATIC_CACHE_ROOT / "index.json"
PERFORATION_WORKBOOK = _local_or_packaged(
    LOCAL_WORKSPACE_ROOT
    / "Data Nations"
    / "Perforation History + NZBP interval review.xlsx",
    "Perforation History + NZBP interval review.xlsx",
)
ONLINE_MODE = DATA_AVAILABILITY_WORKBOOK.parent == PACKAGED_SOURCE_ROOT
# Kept for the optional legacy generators, but dashboard results are read from the
# three controlled workbooks above.
SCHEMATIC_ROOT = WORKSPACE_ROOT / "Data Nations" / "Final things" / "Zona_Rokan_Well_Schematics"
PERFORATION_ROOT = WORKSPACE_ROOT / "outputs" / "zona_rokan_perforation_history_detailed_all"
DATA_REQUEST_WORKBOOK = (
    WORKSPACE_ROOT / "Data Nations" / "Final things" / "Zona Rokan_Data Request.xlsx"
)
DASH_CACHE_ROOT = LOCAL_WORKSPACE_ROOT / "Coding" / "rokan_dash_dashboard_collab" / "data" / ".cache"
if not DASH_CACHE_ROOT.exists():
    DASH_CACHE_ROOT = PACKAGED_DATA_ROOT / "cache"
WELLS_CACHE = DASH_CACHE_ROOT / "zona_rokan_wells.pkl"
RAW_CACHE = DASH_CACHE_ROOT / "zona_rokan_raw.pkl"
DIGITIZER_SCRIPT = WORKSPACE_ROOT / "Coding" / "digitize_zona_rokan_well_logs.py"
TRAINING_SCRIPT = WORKSPACE_ROOT / "Coding" / "train_minas_log_digitizer.py"
TRAINING_CONFIG = WORKSPACE_ROOT / "Coding" / "minas_log_digitizer_training.json"
MINAS_TRAINING_ROOT = (
    WORKSPACE_ROOT / "outputs" / "zona_rokan_log_digitization" / "minas_training"
)
TRAINING_REPORT = (
    MINAS_TRAINING_ROOT / "training_report.json"
)
SCHEMATIC_SCRIPT = WORKSPACE_ROOT / "Coding" / "build_zona_rokan_well_schematics.py"
PERFORATION_SCRIPT = WORKSPACE_ROOT / "Coding" / "Perforation History.py"
PROFILE_REGISTRY = APP_DIR / "data" / "digitizer_profiles.json"
LEGACY_DIGITIZATION_OUTPUT = (
    WORKSPACE_ROOT / "outputs" / "zona_rokan_log_digitization" / "dashboard_runs"
)
DIGITIZATION_OUTPUT = MINAS_TRAINING_ROOT / "dashboard_runs"
FORMATION_EVALUATION_OUTPUT = (
    WORKSPACE_ROOT / "outputs" / "zona_rokan_formation_evaluation"
)


def normalize_well(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


WELL_PREFIXES = {
    "BALAMSE": "BLSE",
    "BALAM": "BLSE",
    "BLSE": "BLSE",
    "BANGKO": "BNKO",
    "BNKO": "BNKO",
    "BEKASAPSO": "BESO",
    "BEKASAP": "BESO",
    "BESO": "BESO",
    "BENAR": "BENA",
    "BENA": "BENA",
    "CANDI": "CAND",
    "CAND": "CAND",
    "DURI": "DURI",
    "FAJAR": "FAJA",
    "FAJA": "FAJA",
    "GULAMO": "GULA",
    "GULA": "GULA",
    "INTAN": "INTA",
    "INTA": "INTA",
    "JORANG": "JORA",
    "JORA": "JORA",
    "KELOK": "KELO",
    "KELO": "KELO",
    "KERANG": "KERA",
    "KERA": "KERA",
    "KOPAR": "KOPA",
    "KOPA": "KOPA",
    "LIBOSE": "LISE",
    "LISE": "LISE",
    "LIBO": "LIBO",
    "MINAS": "MINA",
    "MINA": "MINA",
    "OKI": "OKI",
    "PAGER": "PAGE",
    "PAGE": "PAGE",
    "PEMATANGBOW": "PEBO",
    "PEBO": "PEBO",
    "PEMBURU": "PEMB",
    "PEMB": "PEMB",
    "PENASA": "PENA",
    "PENA": "PENA",
    "PETANI": "PETA",
    "PETA": "PETA",
    "PETAPAHAN": "PETP",
    "PETP": "PETP",
    "PUNCAK": "PUNC",
    "PUNC": "PUNC",
    "PUNGUT": "PUNG",
    "PUNG": "PUNG",
    "SINGA": "SING",
    "SING": "SING",
}

CANONICAL_FIELD_NAMES = {
    "BENA": "Benar",
    "BESO": "Bekasap South",
    "BLSE": "Balam South East",
    "BNKO": "Bangko",
    "CAND": "Candi",
    "DURI": "Duri",
    "FAJA": "Fajar",
    "GULA": "Gulamo",
    "INTA": "Intan",
    "JORA": "Jorang",
    "KELO": "Kelok",
    "KERA": "Kerang",
    "KOPA": "Kopar",
    "LIBO": "Libo",
    "LISE": "Libo South East",
    "MINA": "Minas",
    "OKI": "Oki",
    "PAGE": "Pager",
    "PEBO": "Pematang Bow",
    "PEMB": "Pemburu",
    "PENA": "Penasa",
    "PETA": "Petani",
    "PETP": "Petapahan",
    "PUNC": "Puncak",
    "PUNG": "Pungut",
    "SING": "Singa",
}


def well_signature(value: object, field: object = "") -> tuple[str, int] | None:
    """Return a field-prefix/number identity shared by folder and workbook names."""
    normalized = normalize_well(value)
    number_match = re.search(r"\d+", normalized)
    if number_match is None:
        return None
    letters = normalized[: number_match.start()]
    normalized_field = normalize_well(field)
    prefix = WELL_PREFIXES.get(letters) or WELL_PREFIXES.get(normalized_field)
    if prefix is None:
        for candidate, mapped in sorted(
            WELL_PREFIXES.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if letters.startswith(candidate) or normalized_field.startswith(candidate):
                prefix = mapped
                break
    if prefix is None:
        return None
    return prefix, int(number_match.group())


def well_matches(
    source_well: object,
    canonical_well: object,
    source_field: object = "",
) -> bool:
    source_signature = well_signature(source_well, source_field)
    return source_signature is not None and source_signature == well_signature(canonical_well)


def source_file_token(path: Path) -> tuple[str, int, int]:
    if not path.exists():
        return str(path), -1, -1
    stat = path.stat()
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_display_text(value: object) -> str:
    """Remove OCR artifacts while preserving useful engineering punctuation."""
    text = clean_text(value)
    replacements = {
        "\ufffd": "",
        "\u00a0": " ",
        "\u00b0": " deg ",
        "\u00b7": " ",
        "\u2022": " ",
        "\u00ab": "",
        "\u00bb": "",
        "\u00a9": "",
        "\u00ae": "",
        "\u2122": "",
        "\u20ac": "",
        "\u00a3": "",
        "\u00a5": "",
        "\u00a2": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([|;,:])(?:\s*\1)+", r"\1", text)
    text = re.sub(r"[-_=]{4,}", " ", text)
    return text.strip(" |;")


def clean_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype.kind in "OUS":
            output[column] = output[column].map(
                lambda value: clean_display_text(value) if pd.notna(value) else value
            )
    return output


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def read_csv_if_available(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def read_pickle_if_available(path: Path) -> pd.DataFrame:
    """Read an optional dashboard cache without making startup depend on it."""
    dataframe_type = getattr(pd, "DataFrame", None)
    if dataframe_type is None:
        raise RuntimeError(
            "The Pandas installation is incomplete. Run this dashboard with "
            r"C:\Users\HP\.venvs\zona_rokan_well_log_analyst\Scripts\python.exe."
        )
    if not path.exists():
        return dataframe_type()
    pickle_reader = getattr(pd, "read_pickle", None)
    if not callable(pickle_reader):
        return dataframe_type()
    try:
        frame = pickle_reader(path)
    except (
        AttributeError,
        EOFError,
        OSError,
        ValueError,
        ImportError,
        ModuleNotFoundError,
        pickle.UnpicklingError,
    ):
        return dataframe_type()
    return frame if isinstance(frame, dataframe_type) else dataframe_type()


def load_well_cache() -> pd.DataFrame:
    return read_pickle_if_available(WELLS_CACHE)


def load_raw_cache() -> pd.DataFrame:
    return read_pickle_if_available(RAW_CACHE)


def load_portable_inventory_rows() -> pd.DataFrame:
    """Build the canonical selector inventory when the live Zona Rokan tree is absent."""
    register, _, _, _ = load_screening_matrix_tables()
    if register.empty:
        return pd.DataFrame(columns=["Field", "Well"])
    rows = register[["Field", "Well Name"]].rename(columns={"Well Name": "Well"}).copy()
    signatures = rows.apply(
        lambda row: well_signature(row.get("Well"), row.get("Field")),
        axis=1,
    )
    rows = rows.loc[
        signatures.map(lambda value: value != ("BNKO", 14))
    ].copy()
    signatures = signatures.loc[rows.index]
    rows = rows.loc[rows["Well"].map(normalize_well).ne("")]
    signatures = signatures.loc[rows.index]
    rows["Field"] = signatures.map(
        lambda value: CANONICAL_FIELD_NAMES.get(value[0], "Zona Rokan")
        if value is not None
        else "Zona Rokan"
    )
    rows["Well"] = rows["Well"].map(clean_text)
    return (
        rows.drop_duplicates(subset=["Field", "Well"])
        .sort_values(["Field", "Well"], kind="stable")
        .reset_index(drop=True)
    )


def _parse_depth_interval(value: object) -> tuple[float, float] | None:
    text = clean_text(value).replace(",", "")
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:-|\u2013|\u2014|TO)\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    first, second = float(match.group(1)), float(match.group(2))
    return min(first, second), max(first, second)


@lru_cache(maxsize=3)
def _load_data_availability_cached(
    token: tuple[str, int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(token[0])
    if token[1] < 0:
        return pd.DataFrame(), pd.DataFrame()
    try:
        definitions = pd.read_excel(path, sheet_name="Data_Package_Index").dropna(
            how="all"
        )
        availability = pd.read_excel(
            path,
            sheet_name="Data_Availability_Matrix",
            header=1,
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    definitions = definitions.rename(
        columns={
            "REQ Code": "Request Code",
            "Data Package Name": "Data Package",
        }
    )
    unnamed = [column for column in availability.columns if str(column).startswith("Unnamed:")]
    availability = availability.drop(columns=unnamed, errors="ignore")
    if "Well Name" in availability.columns:
        availability = availability.loc[
            availability["Well Name"].map(normalize_well).ne("")
        ].copy()
    availability.reset_index(drop=True, inplace=True)
    return definitions, availability


def load_data_availability_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions, availability = _load_data_availability_cached(
        source_file_token(DATA_AVAILABILITY_WORKBOOK)
    )
    return definitions.copy(), availability.copy()


SCREENING_REGISTER_COLUMNS = (
    "Source No",
    "Well Name",
    "Operator",
    "X",
    "Y",
    "Field",
    "Reservoir / Zone",
    "Perforation Sand",
    "Developed Producing Reserve (MBO)",
    "Developed Unproducing NZBP Reserve (MBO)",
    "Existing + NZBP Reserve (MBO)",
    "Initial PI",
    "Last PI",
    "Well Status",
    "Trajectory",
    "Last Production Date",
    "Idle Years",
    "Last Oil Rate (BOPD)",
    "Oil Rate Remarks",
    "Last Water Cut (%)",
    "Water Cut Remarks",
    "Completion Type",
    "Artificial Lift",
    "Integrity Status",
    "Shutdown Reason",
    "Well Classification",
    "Location Access",
    "Existing Facility",
    "Oil Gain Score",
    "Reserve Score",
    "Reservoir Pressure Score",
    "Water Cut Score",
    "Integrity Score",
    "Intervention Cost Score",
    "Access Score",
    "Workover Complexity Score",
    "PI Score",
    "Data Completeness Score",
    "Artificial Lift Score",
    "Compartment Score",
    "Weighted Score",
    "HSE KO",
    "Integrity KO",
    "Facility KO",
    "Tier",
    "Recommendation",
    "Screening Narrative",
    "Unused AV",
    "Unused AW",
    "Reservoir Aspect",
    "WBI Aspect",
)

SCREENING_SCORE_COLUMNS = (
    "Oil Gain Score",
    "Reserve Score",
    "Reservoir Pressure Score",
    "Water Cut Score",
    "Integrity Score",
    "Intervention Cost Score",
    "Access Score",
    "Workover Complexity Score",
    "PI Score",
    "Data Completeness Score",
    "Artificial Lift Score",
    "Compartment Score",
)


def _empty_screening_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@lru_cache(maxsize=3)
def _load_screening_matrix_cached(
    token: tuple[str, int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the saved Excel results without recalculating the source workbook."""
    path = Path(token[0])
    if token[1] < 0:
        return _empty_screening_tables()
    try:
        raw_register = pd.read_excel(
            path,
            sheet_name="01_WellRegister and Scoring",
            header=None,
        )
        tiers = pd.read_excel(
            path,
            sheet_name="02_Tier Definitions & KPIs",
            header=5,
            usecols="B:F",
        )
        rubric = pd.read_excel(
            path,
            sheet_name="03_Scoring Rubric",
            header=4,
            usecols="B:H",
        )
        weights = pd.read_excel(
            path,
            sheet_name="04_Parameter Weights",
            header=4,
            usecols="B:H",
        )
    except Exception:
        return _empty_screening_tables()

    column_count = len(SCREENING_REGISTER_COLUMNS)
    raw_register = raw_register.reindex(columns=range(column_count))
    register = raw_register.iloc[8:, :column_count].copy()
    register.columns = list(SCREENING_REGISTER_COLUMNS)
    register["Source Row"] = register.index + 1
    register = register.loc[register["Well Name"].notna()].copy()
    register = register.loc[
        register["Well Name"].astype(str).str.strip().ne("")
    ].copy()
    register.drop(columns=["Unused AV", "Unused AW"], errors="ignore", inplace=True)
    numeric_columns = [
        "Source No",
        "X",
        "Y",
        "Developed Producing Reserve (MBO)",
        "Developed Unproducing NZBP Reserve (MBO)",
        "Existing + NZBP Reserve (MBO)",
        "Initial PI",
        "Last PI",
        "Idle Years",
        "Last Oil Rate (BOPD)",
        "Last Water Cut (%)",
        *SCREENING_SCORE_COLUMNS,
        "Weighted Score",
        "Reservoir Aspect",
        "WBI Aspect",
    ]
    for column in numeric_columns:
        register[column] = pd.to_numeric(register[column], errors="coerce")
    for column in register.select_dtypes(include="object"):
        register[column] = register[column].map(clean_text)
    register["Workbook"] = str(path)
    register["Sheet"] = "01_WellRegister and Scoring"
    register.reset_index(drop=True, inplace=True)

    tiers = tiers.iloc[:, :5].copy()
    tiers.columns = [
        "Tier",
        "Label",
        "Score Range",
        "Typical Operations",
        "Characteristics",
    ]
    tiers = tiers.loc[tiers["Tier"].notna()].copy()
    tiers.reset_index(drop=True, inplace=True)

    rubric = rubric.iloc[:, :7].copy()
    rubric.columns = [
        "Parameter #",
        "Parameter",
        "1 - Poor",
        "2 - Weak",
        "3 - Moderate",
        "4 - Good",
        "5 - Excellent",
    ]
    rubric["Parameter #"] = pd.to_numeric(rubric["Parameter #"], errors="coerce")
    rubric = rubric.loc[rubric["Parameter #"].notna()].copy()
    rubric["Parameter #"] = rubric["Parameter #"].astype(int)
    rubric.reset_index(drop=True, inplace=True)

    weights = weights.iloc[:, :7].copy()
    weights.columns = [
        "Parameter #",
        "Parameter",
        "Dimension",
        "Document Weight",
        "Weight",
        "Knockout",
        "Rationale",
    ]
    weights["Parameter #"] = pd.to_numeric(weights["Parameter #"], errors="coerce")
    weights["Weight"] = pd.to_numeric(weights["Weight"], errors="coerce")
    weights = weights.loc[weights["Parameter #"].notna()].copy()
    weights["Parameter #"] = weights["Parameter #"].astype(int)
    weights.reset_index(drop=True, inplace=True)
    return register, tiers, rubric, weights


def load_screening_matrix_tables(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    register, tiers, rubric, weights = _load_screening_matrix_cached(
        source_file_token(SCREENING_WORKBOOK)
    )
    return register.copy(), tiers.copy(), rubric.copy(), weights.copy()


def selected_screening_row(well: str) -> dict[str, Any]:
    register, _, _, _ = load_screening_matrix_tables()
    target = well_signature(well)
    if register.empty or target is None:
        return {}
    matches = register.loc[
        register.apply(
            lambda row: well_signature(row.get("Well Name"), row.get("Field"))
            == target,
            axis=1,
        )
    ]
    return {} if matches.empty else matches.iloc[0].to_dict()


@lru_cache(maxsize=3)
def _load_legacy_data_request_cached(token: tuple[str, int, int]) -> pd.DataFrame:
    path = Path(token[0])
    if token[1] < 0:
        return pd.DataFrame()
    try:
        matrix = pd.read_excel(
            path,
            sheet_name="A.Data Available&B.Data Request",
            header=None,
        )
    except Exception:
        return pd.DataFrame()
    section_rows = matrix.index[
        matrix.iloc[:, 0].fillna("").astype(str).str.startswith("SECTION ")
    ].tolist()
    if len(section_rows) < 2:
        return pd.DataFrame()
    request_header = section_rows[1] + 1
    request = matrix.iloc[request_header + 1 :].copy()
    request.columns = matrix.iloc[request_header].tolist()
    request.dropna(how="all", inplace=True)
    request.reset_index(drop=True, inplace=True)
    return request


def load_data_request_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the controlled availability result plus the legacy request actions."""
    definitions, availability = load_data_availability_tables()
    request = _load_legacy_data_request_cached(source_file_token(DATA_REQUEST_WORKBOOK))
    return definitions, availability, request.copy()


@lru_cache(maxsize=3)
def _load_perforation_workbook_cached(token: tuple[str, int, int]) -> pd.DataFrame:
    path = Path(token[0])
    if token[1] < 0:
        return pd.DataFrame()
    try:
        frame = pd.read_excel(
            path,
            sheet_name="Existing perfo + NZBP",
            header=1,
        )
    except Exception:
        return pd.DataFrame()
    frame = frame.drop(
        columns=[column for column in frame.columns if str(column).startswith("Unnamed:")],
        errors="ignore",
    )
    explicit_well = (
        frame.get("Well", pd.Series("", index=frame.index, dtype=object))
        .map(clean_text)
        .replace("", np.nan)
    )
    frame["Source Well Row"] = explicit_well.notna()
    for column in ("Well", "Well Type"):
        if column in frame.columns:
            cleaned = frame[column].map(clean_text).replace("", np.nan)
            frame[column] = cleaned.ffill()
    if "Well" not in frame.columns:
        return pd.DataFrame()
    frame = frame.loc[frame["Well"].map(normalize_well).ne("")].copy()
    if "Sand Interval" in frame.columns:
        has_interval = frame["Sand Interval"].map(clean_text).ne("")
        frame = frame.loc[has_interval | frame["Source Well Row"]].copy()
    frame["Source Row"] = frame.index + 3
    parsed = frame.get("Sand Interval", pd.Series(index=frame.index, dtype=object)).map(
        _parse_depth_interval
    )
    frame["Perforation Top (ft)"] = parsed.map(
        lambda value: value[0] if value is not None else np.nan
    )
    frame["Perforation Base (ft)"] = parsed.map(
        lambda value: value[1] if value is not None else np.nan
    )
    source_interval = frame.get(
        "Sand Interval", pd.Series(index=frame.index, dtype=object)
    ).map(clean_text)
    reversed_source = source_interval.map(
        lambda value: (
            (lambda match: bool(match and float(match.group(1)) > float(match.group(2))))(
                re.search(r"(\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)", value)
            )
        )
    )
    source_status = frame.get("Status", pd.Series(index=frame.index, dtype=object)).map(
        clean_text
    )
    status_map = {
        "OPEN": "OPEN",
        "CLOSED": "CLOSE",
        "CLOSE": "CLOSE",
        "NZBP": "NZBP",
    }
    frame["Action Status"] = source_status.str.upper().map(status_map).fillna(
        source_status.str.upper()
    )
    frame["QA Status"] = np.where(
        parsed.isna(),
        np.where(
            source_interval.ne(""),
            "SOURCE CONTEXT - NOT AN INTERVAL",
            "NO INTERVAL LISTED IN WORKBOOK",
        ),
        np.where(reversed_source, "REVIEW: REVERSED SOURCE INTERVAL", "OK"),
    )
    frame["Event Date"] = pd.to_datetime(frame.get("Date"), errors="coerce")
    frame["Top Formation"] = frame.get("Formation", "")
    frame["Depth (Feet)"] = source_interval
    frame["Source File"] = path.name
    frame["Source Path"] = str(path)
    frame["Source Sheet"] = "Existing perfo + NZBP"
    frame["Evidence"] = frame.apply(
        lambda row: (
            f"{clean_text(row.get('Formation'))} | {clean_text(row.get('Sand Interval'))} | "
            f"{clean_text(row.get('Status'))}"
        ).strip(" |"),
        axis=1,
    )
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_perforation_history() -> pd.DataFrame:
    return _load_perforation_workbook_cached(
        source_file_token(PERFORATION_WORKBOOK)
    ).copy()


def load_perforation_coverage() -> pd.DataFrame:
    frame = load_perforation_history()
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("Well", sort=False):
        actions = group["Action Status"].fillna("").astype(str)
        interval_count = int(
            group[["Perforation Top (ft)", "Perforation Base (ft)"]]
            .notna()
            .all(axis=1)
            .sum()
        )
        rows.append(
            {
                "Well": well,
                "Perforation Events": interval_count,
                "Source Rows": len(group),
                "Open Intervals": int(actions.eq("OPEN").sum()),
                "Closed Intervals": int(actions.eq("CLOSE").sum()),
                "NZBP Intervals": int(actions.eq("NZBP").sum()),
                "QA Review": int(
                    group["QA Status"].fillna("").astype(str).str.startswith("REVIEW").sum()
                ),
                "Coverage Status": (
                    "WORKBOOK INTERVALS AVAILABLE"
                    if interval_count
                    else "WELL LISTED - NO INTERVAL"
                ),
                "Diagram Available": "No",
            }
        )
    return pd.DataFrame(rows)


def _sheet_orientation(value: object) -> str:
    normalized = normalize_well(value)
    match = re.search(r"([VDH])(?:1)?$", normalized)
    return match.group(1) if match else ""


def _parse_schematic_sheet(worksheet: Any) -> dict[str, Any]:
    rows = list(worksheet.iter_rows(values_only=True))
    header_index = -1
    columns: dict[str, int] = {}
    for row_index, row in enumerate(rows[:15]):
        labels = {clean_text(value).upper(): index for index, value in enumerate(row)}
        if "WELL DIAGRAM" in labels and "CASING" in labels:
            header_index = row_index
            columns = labels
            break
    if header_index < 0:
        return {
            "sheet_name": worksheet.title,
            "orientation": _sheet_orientation(worksheet.title),
            "image_bytes": b"",
            "image_format": "png",
            "casing": pd.DataFrame(),
            "intervals": pd.DataFrame(),
        }

    formation_column = columns.get("CASING SHOE")
    interval_column = columns.get("WELL DIAGRAM")
    hole_column = columns.get("HOLE DETAILS")
    casing_column = columns.get("CASING")
    md_column = columns.get("CASING DEPTH (MD)")
    tvd_column = columns.get("CASING DEPTH (TVD)")
    status_column = hole_column - 1 if hole_column is not None and hole_column > 0 else None

    def value_at(row: tuple[Any, ...], column: int | None) -> Any:
        return row[column] if column is not None and column < len(row) else None

    casing_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for source_row, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        casing_text = clean_text(value_at(row, casing_column))
        hole_text = clean_text(value_at(row, hole_column))
        md_value = pd.to_numeric(value_at(row, md_column), errors="coerce")
        tvd_value = pd.to_numeric(value_at(row, tvd_column), errors="coerce")
        if casing_text or hole_text or pd.notna(md_value) or pd.notna(tvd_value):
            casing_rows.append(
                {
                    "Hole Details": hole_text,
                    "Casing": casing_text,
                    "Top (ft)": 0.0,
                    "Base (ft)": md_value,
                    "Casing Depth (MD)": md_value,
                    "Casing Depth (TVD)": tvd_value,
                    "Source Row": source_row,
                }
            )
        interval_text = clean_text(value_at(row, interval_column))
        depth_interval = _parse_depth_interval(interval_text)
        if depth_interval is not None:
            formation = clean_text(value_at(row, formation_column))
            status = clean_text(value_at(row, status_column))
            interval_rows.append(
                {
                    "Formation": formation,
                    "Sand": formation,
                    "Interval": interval_text,
                    "Top (ft)": depth_interval[0],
                    "Base (ft)": depth_interval[1],
                    "Status": status,
                    "Kegiatan": "Perforation interval",
                    "Temuan": f"{formation} {status}".strip(),
                    "Depth (feet)": interval_text,
                    "Source Row": source_row,
                }
            )

    image_bytes = b""
    image_format = "png"
    images = list(getattr(worksheet, "_images", []))
    if images:
        largest = max(
            images,
            key=lambda image: float(getattr(image, "width", 0))
            * float(getattr(image, "height", 0)),
        )
        try:
            image_bytes = largest._data()
            image_format = str(getattr(largest, "format", "png") or "png").lower()
        except Exception:
            image_bytes = b""
    return {
        "sheet_name": worksheet.title,
        "orientation": _sheet_orientation(worksheet.title),
        "image_bytes": image_bytes,
        "image_format": image_format,
        "casing": pd.DataFrame(casing_rows),
        "intervals": pd.DataFrame(interval_rows),
    }


@lru_cache(maxsize=2)
def _load_schematic_workbook_cached(
    token: tuple[str, int, int],
) -> dict[str, dict[str, Any]]:
    path = Path(token[0])
    if token[1] < 0:
        return {}
    try:
        from openpyxl import load_workbook

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workbook = load_workbook(
                path,
                read_only=False,
                data_only=True,
                keep_links=False,
            )
    except Exception:
        return {}
    output: dict[str, dict[str, Any]] = {}
    template_names = {"VERTICAL", "DIRECTIONAL", "HORIZONTAL"}
    try:
        for worksheet in workbook.worksheets:
            if clean_text(worksheet.title).upper() in template_names:
                continue
            parsed = _parse_schematic_sheet(worksheet)
            parsed["source_path"] = str(path)
            output[worksheet.title] = parsed
    finally:
        workbook.close()
    return output


@lru_cache(maxsize=4)
def _load_schematic_cache_index_cached(
    source_token: tuple[str, int, int],
    index_token: tuple[str, int, int],
) -> dict[str, dict[str, Any]]:
    if source_token[1] < 0 or index_token[1] < 0:
        return {}
    try:
        document = json.loads(Path(index_token[0]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not ONLINE_MODE:
        if int(document.get("source_size", -1)) != source_token[1]:
            return {}
        expected_hash = clean_text(document.get("source_sha256")).lower()
        if expected_hash:
            try:
                digest = hashlib.sha256(Path(source_token[0]).read_bytes()).hexdigest()
            except OSError:
                return {}
            if digest != expected_hash:
                return {}
        elif int(document.get("source_mtime_ns", -1)) != source_token[2]:
            # Backward compatibility for caches generated before stable hashes were added.
            return {}
    # In cloud mode, the workbook and cache are immutable files from one private
    # Git commit. That commit is the integrity boundary; checkout layers may rewrite
    # ZIP container metadata and therefore cannot be compared with local fingerprints.
    return {
        clean_text(entry.get("sheet_name")): entry
        for entry in document.get("entries", [])
        if clean_text(entry.get("sheet_name"))
    }


def load_schematic_workbook() -> dict[str, dict[str, Any]]:
    """Return the lightweight, source-fingerprinted worksheet catalog."""
    return _load_schematic_cache_index_cached(
        source_file_token(SCHEMATIC_WORKBOOK),
        source_file_token(SCHEMATIC_CACHE_INDEX),
    )


def _select_schematic_entry(well: object) -> dict[str, Any]:
    signature = well_signature(well)
    candidates = [
        entry
        for entry in load_schematic_workbook().values()
        if well_signature(entry.get("sheet_name")) == signature
    ]
    if not candidates:
        return {}
    expected = _sheet_orientation(well) or "V"
    return next(
        (entry for entry in candidates if entry.get("orientation") == expected),
        candidates[0],
    )


@lru_cache(maxsize=16)
def _load_schematic_payload_cached(
    source_token: tuple[str, int, int],
    data_token: tuple[str, int, int],
    image_token: tuple[str, int, int],
) -> dict[str, Any]:
    if source_token[1] < 0 or data_token[1] < 0:
        return {}
    try:
        payload = json.loads(Path(data_token[0]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    image_bytes = b""
    if image_token[1] >= 0:
        try:
            image_bytes = Path(image_token[0]).read_bytes()
        except OSError:
            image_bytes = b""
    return {
        "sheet_name": payload.get("sheet_name", ""),
        "orientation": payload.get("orientation", ""),
        "image_format": payload.get("image_format", ""),
        "image_bytes": image_bytes,
        "casing": pd.DataFrame(payload.get("casing", [])),
        "intervals": pd.DataFrame(payload.get("intervals", [])),
        "source_path": source_token[0],
    }


def _select_schematic_payload(well: object) -> dict[str, Any]:
    entry = _select_schematic_entry(well)
    if entry:
        data_path = SCHEMATIC_CACHE_ROOT / clean_text(entry.get("data_file"))
        image_name = clean_text(entry.get("image_file"))
        image_path = SCHEMATIC_CACHE_ROOT / image_name if image_name else Path("__missing__")
        return _load_schematic_payload_cached(
            source_file_token(SCHEMATIC_WORKBOOK),
            source_file_token(data_path),
            source_file_token(image_path),
        )
    return {}


def load_schematic_summary() -> pd.DataFrame:
    rows = []
    for entry in load_schematic_workbook().values():
        rows.append(
            {
                "Well": entry.get("sheet_name", ""),
                "Sheet": entry.get("sheet_name", ""),
                "Orientation": entry.get("orientation", ""),
                "Events": int(entry.get("interval_count", 0)),
                "Casing Strings": int(entry.get("casing_count", 0)),
                "Sand Intervals": int(entry.get("interval_count", 0)),
                "Image Available": bool(entry.get("image_available")),
                "Workbook": str(SCHEMATIC_WORKBOOK),
            }
        )
    return pd.DataFrame(rows)


def selected_data_request(
    well: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definitions, availability, request = load_data_request_tables()
    key = normalize_well(well)

    def selected(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "Well Name" not in frame.columns:
            return pd.DataFrame()
        source_fields = frame.get(
            "Field", pd.Series("", index=frame.index, dtype=object)
        )
        mask = [
            well_matches(source_well, well, source_field)
            for source_well, source_field in zip(frame["Well Name"], source_fields)
        ]
        return frame.loc[mask].copy()

    return definitions.copy(), selected(availability), selected(request)


def load_training_report() -> list[dict[str, Any]]:
    if not TRAINING_REPORT.exists():
        return []
    try:
        value = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _best_cached_well_rows(wells: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if wells.empty:
        return {}
    frame = wells.copy()
    key_source = frame.get("Well Folder", frame.get("Well Name", pd.Series(dtype=object)))
    frame["__normalized_well"] = key_source.map(normalize_well)
    completeness = pd.to_numeric(
        frame.get("Data Completeness (%)", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(0)
    frame["__completeness"] = completeness
    frame = frame.sort_values("__completeness", ascending=False)
    return {
        key: group.iloc[0].drop(labels=["__normalized_well", "__completeness"]).to_dict()
        for key, group in frame.groupby("__normalized_well", sort=False)
        if key
    }


def _indexed_rows(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "Well" not in frame.columns:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = normalize_well(row.get("Well"))
        if key:
            output[key] = row.to_dict()
    return output


def build_master_table(inventory: Iterable[Any]) -> pd.DataFrame:
    cached_wells = load_well_cache()
    if cached_wells.empty:
        screening_register, _, _, _ = load_screening_matrix_tables()
        cached_wells = screening_register.rename(columns={"Well Status": "Status"})
    cache_rows = _best_cached_well_rows(cached_wells)
    _, availability, _ = load_data_request_tables()
    availability_rows = (
        {
            well_signature(row.get("Well Name"), row.get("Field")): row.to_dict()
            for _, row in availability.iterrows()
            if well_signature(row.get("Well Name"), row.get("Field")) is not None
        }
        if not availability.empty
        else {}
    )
    rows: list[dict[str, Any]] = []

    for record in inventory:
        key = normalize_well(getattr(record, "well", ""))
        cached = cache_rows.get(key, {})
        schematic = selected_schematic_row(getattr(record, "well", ""))
        perforation_history = selected_perforation_history(
            getattr(record, "well", "")
        )
        data_status = availability_rows.get(
            well_signature(getattr(record, "well", "")), {}
        )
        specialized_perforation_count = (
            int(
                perforation_history[
                    ["Perforation Top (ft)", "Perforation Base (ft)"]
                ]
                .notna()
                .all(axis=1)
                .sum()
            )
            if not perforation_history.empty
            else 0
        )
        qa_review = (
            int(
                perforation_history["QA Status"]
                .fillna("")
                .astype(str)
                .str.startswith("REVIEW")
                .sum()
            )
            if not perforation_history.empty and "QA Status" in perforation_history
            else 0
        )
        request_values = [
            clean_text(value)
            for column, value in data_status.items()
            if str(column).startswith("REQ-")
        ]
        normalized_statuses = [value.upper() for value in request_values]
        available_count = sum(value == "AVAILABLE" for value in normalized_statuses)
        partial_count = sum(
            value == "PARTIAL: NOT COMPREHENSIVE" for value in normalized_statuses
        )
        missing_count = sum(value == "MISSING" for value in normalized_statuses)
        scanned_files = getattr(record, "scanned_files", ())
        row = dict(cached)
        row.update(
            {
                "Field": getattr(record, "field", ""),
                "Well": getattr(record, "well", ""),
                "Normalized Well": key,
                "LAS Files": len(getattr(record, "las_files", ())),
                "Scanned Logs": len(scanned_files),
                "Production Files": len(getattr(record, "production_files", ())),
                "Schematic Available": bool(schematic.get("Sheet")),
                "Schematic Events": pd.to_numeric(schematic.get("Events"), errors="coerce"),
                "Casing Strings": pd.to_numeric(schematic.get("Casing Strings"), errors="coerce"),
                "Sand Intervals": pd.to_numeric(schematic.get("Sand Intervals"), errors="coerce"),
                "Schematic Detail Available": bool(schematic.get("Sheet")),
                "Perforation Events": int(specialized_perforation_count),
                "Detailed History Available": not perforation_history.empty,
                "Detailed History Events": int(specialized_perforation_count),
                "Perforation QA Review": qa_review,
                "Perforation Diagram": specialized_perforation_count > 0,
                "Data Request Available": available_count,
                "Data Request Partial": partial_count,
                "Data Request Missing": missing_count,
            }
        )
        completeness = pd.to_numeric(row.get("Data Completeness (%)"), errors="coerce")
        status_total = available_count + partial_count + missing_count
        if pd.isna(completeness) and status_total:
            row["Data Completeness (%)"] = round(
                100.0 * (available_count + 0.5 * partial_count) / status_total,
                1,
            )
        rows.append(row)

    master = pd.DataFrame(rows)
    if master.empty:
        return master
    for column in (
        "Last Oil Rate (BOPD)",
        "Last Water Cut (%)",
        "Data Completeness (%)",
        "Source File Count",
        "Document Count",
        "LAS Files",
        "Scanned Logs",
        "Production Files",
        "Schematic Events",
        "Casing Strings",
        "Sand Intervals",
        "Perforation Events",
        "Detailed History Events",
        "Perforation QA Review",
        "Data Request Available",
        "Data Request Partial",
        "Data Request Missing",
    ):
        if column not in master.columns:
            master[column] = np.nan
        master[column] = pd.to_numeric(master[column], errors="coerce")
    return apply_screening_scores(master)


def _contains(value: object, *terms: str) -> bool:
    text = clean_text(value).lower()
    return any(term.lower() in text for term in terms)


def apply_screening_scores(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return master
    output = master.copy()
    technical_scores: list[float] = []
    uplift_scores: list[float] = []
    complexity_scores: list[float] = []
    readiness_scores: list[float] = []
    gaps: list[str] = []

    for _, row in output.iterrows():
        integrity = row.get("Integrity Status")
        completion = row.get("Completion Type")
        facility = row.get("Existing Facility")
        access = row.get("Location Access")
        status = row.get("Status")
        lift = row.get("Artificial Lift")
        oil = pd.to_numeric(row.get("Last Oil Rate (BOPD)"), errors="coerce")
        water_cut = pd.to_numeric(row.get("Last Water Cut (%)"), errors="coerce")
        completeness = pd.to_numeric(row.get("Data Completeness (%)"), errors="coerce")
        oil = 0.0 if pd.isna(oil) else max(0.0, float(oil))
        water_cut = 100.0 if pd.isna(water_cut) else clamp(float(water_cut), 0, 100)
        completeness = 0.0 if pd.isna(completeness) else clamp(float(completeness))

        technical = 35.0
        if _contains(integrity, "good", "acceptable", "intact"):
            technical += 25
        elif _contains(integrity, "critical", "failed", "leak", "corrosion"):
            technical -= 25
        elif clean_text(integrity):
            technical += 8
        if bool(row.get("Schematic Available")):
            technical += 12
        if float(row.get("Perforation Events") or 0) > 0:
            technical += 8
        if clean_text(completion):
            technical += 8
        if clean_text(facility):
            technical += 7
        if float(row.get("LAS Files") or 0) > 0:
            technical += 5

        uplift = 15.0
        uplift += min(35.0, math.log1p(oil) / math.log(1001) * 35.0)
        uplift += (100.0 - water_cut) * 0.25
        if _contains(status, "idle", "shut", "closed", "plug", "pop", "inactive"):
            uplift += 15
        if clean_text(row.get("Reservoir / Zone")):
            uplift += 5
        if float(row.get("LAS Files") or 0) > 0:
            uplift += 5

        complexity = 25.0
        if _contains(integrity, "critical", "failed", "leak", "corrosion"):
            complexity += 30
        elif not clean_text(integrity):
            complexity += 12
        if _contains(access, "remote", "swamp", "marsh", "difficult"):
            complexity += 20
        if not clean_text(facility) or _contains(facility, "none", "n/a", "standalone"):
            complexity += 15
        if not bool(row.get("Schematic Available")):
            complexity += 12
        if not clean_text(completion):
            complexity += 10
        if _contains(lift, "esp", "gas lift", "srp"):
            complexity += 5

        data_flags = [
            completeness >= 60,
            bool(row.get("Schematic Available")),
            float(row.get("Production Files") or 0) > 0,
            float(row.get("LAS Files") or 0) > 0,
            clean_text(integrity) != "",
            clean_text(completion) != "",
            clean_text(facility) != "",
            float(row.get("Perforation Events") or 0) > 0,
        ]
        readiness = 100.0 * sum(data_flags) / len(data_flags)
        missing: list[str] = []
        if float(row.get("LAS Files") or 0) <= 0:
            missing.append("LAS")
        if not bool(row.get("Schematic Available")):
            missing.append("schematic")
        if float(row.get("Perforation Events") or 0) <= 0:
            missing.append("perforation history")
        if not clean_text(integrity):
            missing.append("integrity")
        if not clean_text(completion):
            missing.append("completion")
        if not clean_text(facility):
            missing.append("facility")

        technical_scores.append(clamp(technical))
        uplift_scores.append(clamp(uplift))
        complexity_scores.append(clamp(complexity))
        readiness_scores.append(clamp(readiness))
        gaps.append(", ".join(missing) if missing else "No critical gap flagged")

    output["Technical Viability"] = technical_scores
    output["Production Uplift Potential"] = uplift_scores
    output["Execution Complexity"] = complexity_scores
    output["Phase 2 Readiness"] = readiness_scores
    output["Critical Data Gaps"] = gaps
    return rank_candidates(output, 0.4, 0.4, 0.2)


def rank_candidates(
    frame: pd.DataFrame,
    technical_weight: float,
    uplift_weight: float,
    execution_weight: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    total = technical_weight + uplift_weight + execution_weight
    if total <= 0:
        technical_weight, uplift_weight, execution_weight, total = 0.4, 0.4, 0.2, 1.0
    score = (
        output["Technical Viability"] * technical_weight
        + output["Production Uplift Potential"] * uplift_weight
        + (100.0 - output["Execution Complexity"]) * execution_weight
    ) / total
    output["Opportunity Score"] = score.round(1)
    output["Priority Tier"] = pd.cut(
        output["Opportunity Score"],
        bins=[-np.inf, 49.999, 69.999, np.inf],
        labels=["Tier 3 - Deferred", "Tier 2 - Engineering", "Tier 1 - Quick Win"],
    ).astype(str)
    output["Phase 2 Candidate"] = np.where(
        (output["Phase 2 Readiness"] >= 62.5)
        & (output["Technical Viability"] >= 50)
        & (output["Opportunity Score"] >= 50),
        "READY FOR REVIEW",
        "DATA / ENGINEERING HOLD",
    )
    return output.sort_values(
        ["Opportunity Score", "Phase 2 Readiness"], ascending=False
    ).reset_index(drop=True)


def selected_well_evidence(well: str) -> pd.DataFrame:
    raw = load_raw_cache()
    if raw.empty:
        return raw
    key_column = raw.get("Well Key", raw.get("Well Folder", pd.Series(dtype=object)))
    mask = key_column.map(normalize_well) == normalize_well(well)
    output = raw.loc[mask].copy()
    for column in ("Source Path", "Folder Path"):
        if column not in output.columns:
            continue

        def current_workspace_path(value: object) -> object:
            text = clean_text(value)
            marker = r"\Data Nations\Zona Rokan"
            position = text.lower().find(marker.lower())
            if position < 0:
                return value
            relative = text[position + len(marker) :].lstrip("\\/")
            candidate = DATA_ROOT / Path(relative)
            return str(candidate) if candidate.exists() else value

        output[column] = output[column].map(current_workspace_path)
    return output


def selected_perforation_history(well: str) -> pd.DataFrame:
    frame = load_perforation_history()
    if frame.empty:
        return frame
    return frame.loc[frame["Well"].map(lambda value: well_matches(value, well))].copy()


def selected_schematic_row(well: str) -> dict[str, Any]:
    entry = _select_schematic_entry(well)
    if not entry:
        return {}
    return {
        "Well": well,
        "Sheet": entry.get("sheet_name", ""),
        "Orientation": entry.get("orientation", ""),
        "Workbook": str(SCHEMATIC_WORKBOOK),
        "Image Available": bool(entry.get("image_available")),
        "Events": int(entry.get("interval_count", 0)),
        "Casing Strings": int(entry.get("casing_count", 0)),
        "Sand Intervals": int(entry.get("interval_count", 0)),
    }


def selected_schematic_workbook(well: str) -> dict[str, Any]:
    payload = _select_schematic_payload(well)
    if not payload:
        return {}
    return {
        **payload,
        "casing": payload.get("casing", pd.DataFrame()).copy(),
        "intervals": payload.get("intervals", pd.DataFrame()).copy(),
    }


def selected_schematic_artifacts(well: str) -> dict[str, Path]:
    # Retained as a compatibility shim for external callers. Workbook-backed
    # schematic data is exposed through selected_schematic_workbook().
    return {}


def selected_schematic_detail(well: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = selected_schematic_workbook(well)
    if not payload:
        return pd.DataFrame(), pd.DataFrame()
    intervals = payload.get("intervals", pd.DataFrame()).copy()
    audit = pd.DataFrame(
        [
            {
                "Source Workbook": Path(payload.get("source_path", "")).name,
                "Source Path": payload.get("source_path", ""),
                "Source Sheet": payload.get("sheet_name", ""),
                "Orientation": payload.get("orientation", ""),
                "Casing Rows": len(payload.get("casing", pd.DataFrame())),
                "Interval Rows": len(intervals),
                "Embedded Image": bool(payload.get("image_bytes")),
            }
        ]
    )
    return intervals, audit


def selected_perforation_diagram(well: str) -> Path | None:
    return None


def perforation_related_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    searchable = pd.Series("", index=events.index, dtype=object)
    for column in ("Kegiatan", "Temuan", "Depth (feet)"):
        if column in events.columns:
            searchable = searchable.str.cat(events[column].fillna("").astype(str), sep=" ")
    pattern = (
        r"perfor|perf\b|reperf|squeeze|isolation|shut.?off|plug|cement|"
        r"completion|workover|swab|well test"
    )
    return events.loc[searchable.str.contains(pattern, case=False, regex=True)].copy()


WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2)


def utm_zone_from_longitude(longitude: float) -> int:
    return int((longitude + 180) // 6) + 1


def latlon_to_utm(
    latitude: float,
    longitude: float,
    zone: int | None = None,
) -> tuple[float, float, str, int]:
    """Convert WGS84 decimal degrees to standard UTM X/Y."""
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude/longitude is outside the valid range.")
    zone = zone or utm_zone_from_longitude(longitude)
    hemisphere = "N" if latitude >= 0 else "S"
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    lon_origin = math.radians((zone - 1) * 6 - 180 + 3)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = WGS84_EP2 * cos_lat * cos_lat
    a = cos_lat * (lon_rad - lon_origin)
    m = WGS84_A * (
        (1 - WGS84_E2 / 4 - 3 * WGS84_E2**2 / 64 - 5 * WGS84_E2**3 / 256)
        * lat_rad
        - (3 * WGS84_E2 / 8 + 3 * WGS84_E2**2 / 32 + 45 * WGS84_E2**3 / 1024)
        * math.sin(2 * lat_rad)
        + (15 * WGS84_E2**2 / 256 + 45 * WGS84_E2**3 / 1024)
        * math.sin(4 * lat_rad)
        - (35 * WGS84_E2**3 / 3072) * math.sin(6 * lat_rad)
    )
    easting = 500000.0 + 0.9996 * n * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * WGS84_EP2) * a**5 / 120
    )
    northing = 0.9996 * (
        m
        + n
        * tan_lat
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * WGS84_EP2) * a**6 / 720
        )
    )
    if hemisphere == "S":
        northing += 10000000.0
    epsg = (32600 if hemisphere == "N" else 32700) + zone
    return easting, northing, f"{zone}{hemisphere}", epsg


def _dms_to_decimal(
    degrees: str,
    minutes: str | None,
    seconds: str | None,
    hemisphere: str | None,
) -> float:
    value = abs(float(degrees))
    value += float(minutes or 0) / 60.0
    value += float(seconds or 0) / 3600.0
    if (hemisphere or "").upper() in {"S", "W"} or float(degrees) < 0:
        value *= -1
    return value


def _coordinate_from_text(text: str, label: str) -> float | None:
    label_pattern = "LAT(?:ITUDE)?" if label == "lat" else "LON(?:GITUDE)?"
    dms = re.search(
        rf"(?i){label_pattern}\s*[:=]?\s*(-?\d{{1,3}})"
        rf"(?:\s*[oO\u00b0\u00ba]\s*|\s+)(\d{{1,2}})?"
        rf"(?:\s*['\u2032]\s*)?(\d{{1,2}}(?:\.\d+)?)?"
        rf"(?:\s*[\"\u2033])?\s*([NSEW])?",
        text,
    )
    if dms and (dms.group(2) or dms.group(3)):
        return _dms_to_decimal(*dms.groups())
    decimal = re.search(
        rf"(?i){label_pattern}\s*[:=]?\s*(-?\d{{1,3}}(?:\.\d+)?)\s*([NSEW])?",
        text,
    )
    if not decimal:
        return None
    value = float(decimal.group(1))
    if (decimal.group(2) or "").upper() in {"S", "W"}:
        value = -abs(value)
    return value


def _xy_from_text(text: str, axis: str) -> float | None:
    if axis == "x":
        patterns = (
            r"(?i)EASTING\s*(?:X)?\s*[:=]?\s*(-?\d{4,9}(?:\.\d+)?)",
            r"(?i)\bE\s*[:=]\s*(-?\d{4,9}(?:\.\d+)?)",
        )
    else:
        patterns = (
            r"(?i)NORTHING\s*(?:Y)?\s*[:=]?\s*(-?\d{4,9}(?:\.\d+)?)",
            r"(?i)\bN\s*[:=]\s*(-?\d{4,9}(?:\.\d+)?)",
        )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


@lru_cache(maxsize=256)
def las_coordinate_rows(path_texts: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path_text in path_texts:
        path = Path(path_text)
        if not path.exists():
            continue
        try:
            las = lasio.read(
                str(path),
                ignore_header_errors=True,
                mnemonic_case="upper",
            )
        except Exception:
            continue
        header_text = " | ".join(
            f"{item.mnemonic}: {item.value} {item.descr}" for item in las.well
        )
        latitude = _coordinate_from_text(header_text, "lat")
        longitude = _coordinate_from_text(header_text, "lon")
        source_x = _xy_from_text(header_text, "x")
        source_y = _xy_from_text(header_text, "y")
        if latitude is None and longitude is None and source_x is None and source_y is None:
            continue
        converted_x = converted_y = np.nan
        zone = ""
        epsg: int | float = np.nan
        delta = np.nan
        if latitude is not None and longitude is not None:
            try:
                converted_x, converted_y, zone, epsg = latlon_to_utm(
                    latitude, longitude
                )
            except ValueError:
                pass
        if (
            source_x is not None
            and source_y is not None
            and np.isfinite(converted_x)
            and np.isfinite(converted_y)
        ):
            delta = math.hypot(source_x - converted_x, source_y - converted_y)
        if np.isfinite(delta) and delta <= 250:
            qc = "SOURCE X/Y AGREES WITH WGS84 UTM"
        elif np.isfinite(delta):
            qc = "SOURCE X/Y IS LOCAL GRID OR DIFFERENT CRS"
        elif latitude is not None and longitude is not None:
            qc = "WGS84 UTM CONVERTED FROM LAT/LONG"
        else:
            qc = "SOURCE GRID ONLY - CRS/TIE POINTS REQUIRED"
        rows.append(
            {
                "Source File": path.name,
                "Latitude": latitude,
                "Longitude": longitude,
                "Source X / Easting": source_x,
                "Source Y / Northing": source_y,
                "Converted UTM X": round(converted_x, 3)
                if np.isfinite(converted_x)
                else np.nan,
                "Converted UTM Y": round(converted_y, 3)
                if np.isfinite(converted_y)
                else np.nan,
                "UTM Zone": zone,
                "EPSG": epsg,
                "Source-vs-UTM delta (m)": round(delta, 1)
                if np.isfinite(delta)
                else np.nan,
                "Coordinate QC": qc,
                "Source Path": str(path),
            }
        )
    return pd.DataFrame(rows)


def selected_coordinate_evidence(record: Any) -> pd.DataFrame:
    # Binary DLIS/LIS logs now share the petrophysics inventory. Coordinate
    # extraction remains LAS-header based, so avoid handing binary logs to lasio.
    paths = tuple(
        str(path)
        for path in getattr(record, "las_files", ())
        if Path(path).suffix.lower() == ".las"
    )
    return las_coordinate_rows(paths).copy()


def trajectory_source_files(record: Any) -> list[Path]:
    path = Path(getattr(record, "path", "__missing__"))
    if not path.exists():
        return []
    directional = re.compile(
        r"(?i)(?:DIRECTIONAL|DEVIATION|TRAJECTORY|INCLINATION|AZIMUTH|"
        r"(?:^|[_ -])SURVEY(?:[_ .-]|$)|MAIN.?TVD|FINAL.?SURVEY)"
    )
    excluded = re.compile(
        r"(?i)(?:TEMPERATURE|PRESSURE|SPINNER|DIPMETER|PRODUCTION.?LOG|PLT)"
    )
    supported = {".csv", ".txt", ".xlsx", ".xls", ".pds", ".las"}
    files = {
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in supported
        and directional.search(item.name)
        and not excluded.search(item.name)
    }
    return sorted(
        files,
        key=lambda item: (
            0 if item.suffix.lower() == ".csv" else 1,
            0 if "survey" in item.name.lower() else 1,
            item.name.lower(),
        ),
    )


def _survey_column(
    columns: Iterable[object],
    candidates: tuple[str, ...],
) -> object | None:
    normalized = {
        re.sub(r"[^A-Z0-9]+", "", str(column).upper()): column
        for column in columns
    }
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _normalize_survey_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    md = _survey_column(
        frame.columns, ("MD", "MEASUREDDEPTH", "DEPTH", "MEASUREDDEPTHFT")
    )
    inc = _survey_column(
        frame.columns, ("INC", "INCLINATION", "INCL", "DEVIATION", "INCLANGLE")
    )
    azi = _survey_column(
        frame.columns, ("AZI", "AZIMUTH", "DIRECTION", "AZIMUTHANGLE")
    )
    if md is None or inc is None or azi is None:
        return pd.DataFrame()
    output = pd.DataFrame(
        {
            "MD": pd.to_numeric(frame[md], errors="coerce"),
            "Inclination": pd.to_numeric(frame[inc], errors="coerce"),
            "Azimuth": pd.to_numeric(frame[azi], errors="coerce"),
        }
    )
    optional = {
        "TVD": ("TVD", "TRUEVERTICALDEPTH", "VERTICALDEPTH"),
        "North Departure": ("NS", "NORTHSOUTH", "NORTHDEPARTURE", "DISPLNS"),
        "East Departure": ("EW", "EASTWEST", "EASTDEPARTURE", "DISPLEW"),
        "DLS (deg/100ft)": (
            "DLS",
            "DLSDEG100FT",
            "DOGLEGSEVERITY",
            "DOGLEG",
        ),
    }
    for target, candidates in optional.items():
        source = _survey_column(frame.columns, candidates)
        if source is not None:
            output[target] = pd.to_numeric(frame[source], errors="coerce")
    output = output.dropna(subset=["MD", "Inclination", "Azimuth"])
    output = output.loc[
        output["MD"].ge(0)
        & output["Inclination"].between(0, 180)
        & output["Azimuth"].between(-360, 360)
    ]
    output["Azimuth"] %= 360.0
    output = output.drop_duplicates("MD", keep="last").sort_values("MD")
    if {"North Departure", "East Departure"}.issubset(output.columns):
        output["Horizontal Departure"] = np.hypot(
            output["North Departure"], output["East Departure"]
        )
    return output.reset_index(drop=True)


def _parse_embedded_pds_survey(path: Path) -> pd.DataFrame:
    text = path.read_bytes().decode("latin1", errors="ignore")
    pattern = re.compile(
        r"(?<![\d.])(\d{1,3})\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)"
    )
    rows = []
    for values in pattern.findall(text):
        numeric = [float(value) for value in values]
        sequence, md, inc, azi, _, tvd, _, north, east, _, _, dls = numeric
        if (
            sequence < 1
            or md < 0
            or not 0 <= inc <= 180
            or not -360 <= azi <= 360
            or tvd < -1000
            or tvd > md + 1000
        ):
            continue
        rows.append(
            {
                "MD": md,
                "Inclination": inc,
                "Azimuth": azi % 360.0,
                "TVD": tvd,
                "North Departure": north,
                "East Departure": east,
                "DLS (deg/100ft)": dls,
            }
        )
    return _normalize_survey_table(pd.DataFrame(rows))


@lru_cache(maxsize=256)
def parse_trajectory_source(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    try:
        if suffix in {".pds"}:
            return _parse_embedded_pds_survey(path)
        if suffix in {".xlsx", ".xls"}:
            return _normalize_survey_table(pd.read_excel(path))
        if suffix == ".csv":
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            header_index = next(
                (
                    index
                    for index, line in enumerate(lines[:100])
                    if re.search(r"(?i)(?:^|,)MD(?:,|$)", line)
                    and re.search(r"(?i)(?:^|,)INC(?:,|$)", line)
                    and re.search(r"(?i)(?:^|,)AZI(?:,|$)", line)
                ),
                0,
            )
            return _normalize_survey_table(
                pd.read_csv(
                    path,
                    skiprows=header_index,
                    low_memory=False,
                    encoding="latin1",
                )
            )
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            rows = [
                tuple(float(value) for value in match)
                for match in re.findall(
                    r"(?m)^\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$",
                    text,
                )
            ]
            return _normalize_survey_table(
                pd.DataFrame(rows, columns=["MD", "Inclination", "Azimuth"])
            )
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return pd.DataFrame()
    return pd.DataFrame()


def discover_trajectory_survey(record: Any) -> tuple[pd.DataFrame, Path | None]:
    candidates: list[tuple[tuple[float, float, float], pd.DataFrame, Path]] = []
    for path in trajectory_source_files(record):
        if path.suffix.lower() == ".las":
            continue
        table = parse_trajectory_source(str(path)).copy()
        if len(table) < 2:
            continue
        span = float(table["MD"].max() - table["MD"].min())
        reported_tvd = float(table.get("TVD", pd.Series(dtype=float)).notna().sum())
        score = (1.0 if reported_tvd else 0.0, span, float(len(table)))
        candidates.append((score, table, path))
    if not candidates:
        return pd.DataFrame(), None
    _, table, path = max(candidates, key=lambda item: item[0])
    return table, path


def selected_trajectory_evidence(record: Any) -> pd.DataFrame:
    patterns = re.compile(
        r"(?i)(?:\bTVD\b|MAIN.?TVD|HLS|HORIZONTAL|DEVIATION|DIRECTIONAL|"
        r"INCLINATION|AZIMUTH|TRAJECTORY)"
    )
    files = {
        path
        for path in (
            *getattr(record, "las_files", ()),
            *getattr(record, "scanned_files", ()),
            *trajectory_source_files(record),
        )
        if patterns.search(path.name)
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: item.name.lower()):
        terms = [term.upper() for term in patterns.findall(path.name)]
        survey = parse_trajectory_source(str(path))
        evidence_type = "Logging filename indicator"
        if len(survey) >= 2:
            evidence_type = "Numeric directional survey"
        elif re.search(r"(?i)HORIZONTAL", path.name):
            evidence_type = "Horizontal logging evidence"
        elif re.search(r"(?i)DEVIATION|DIRECTIONAL|INCLINATION|AZIMUTH|TRAJECTORY", path.name):
            evidence_type = "Directional survey indicator"
        elif re.search(r"(?i)MAIN.?TVD|\bTVD\b", path.name):
            evidence_type = "TVD-indexed log evidence"
        elif re.search(r"(?i)HLS", path.name):
            evidence_type = "HLS service-code indicator"
        if evidence_type == "Numeric directional survey":
            engineering_note = (
                f"Parsed {len(survey)} MD/inclination/azimuth stations from source."
            )
        elif evidence_type == "HLS service-code indicator":
            engineering_note = (
                "HLS alone is not horizontal-well proof; inclination/azimuth is required."
            )
        elif evidence_type == "Horizontal logging evidence":
            engineering_note = (
                "Explicit horizontal wording found; confirm with inclination/azimuth survey."
            )
        elif evidence_type == "Directional survey indicator":
            engineering_note = "Directional-survey indicator found in filename."
        else:
            engineering_note = (
                "TVD evidence supports depth-reference review but not inclination/azimuth."
            )
        rows.append(
            {
                "Evidence Level": evidence_type,
                "File": path.name,
                "Format": path.suffix.upper().lstrip("."),
                "Matched Indicator": ", ".join(dict.fromkeys(terms)),
                "Survey Stations": len(survey),
                "Maximum Inclination": (
                    round(float(survey["Inclination"].max()), 2)
                    if not survey.empty
                    else np.nan
                ),
                "Reported TVD": (
                    "Yes"
                    if "TVD" in survey and survey["TVD"].notna().any()
                    else "No"
                ),
                "Source Path": str(path),
                "Engineering Note": engineering_note,
            }
        )
    if not rows and normalize_well(getattr(record, "well", "")).endswith("D1"):
        rows.append(
            {
                "Evidence Level": "Directional naming indicator only",
                "File": "",
                "Format": "",
                "Matched Indicator": "D1 well suffix",
                "Source Path": str(getattr(record, "path", "")),
                "Engineering Note": (
                    "The well name suggests a directional completion, but no TVD, "
                    "inclination, or azimuth file was found."
                ),
            }
        )
    return pd.DataFrame(rows)


def supporting_files(record: Any, category: str) -> pd.DataFrame:
    patterns = {
        "core": re.compile(
            r"(?i)\bXRD\b|PETROGRAPH|\bRCAL\b|\bSCAL\b|SIDEWALL CORE|"
            r"\bSWC\b|CORE|THIN SECTION|\bSEM\b"
        ),
        "fluid": re.compile(
            r"(?i)SALINITY|WATER ANALYSIS|PICKETT|FORMATION WATER|\bPVT\b|"
            r"FLUID ANALYSIS"
        ),
        "mudlog": re.compile(
            r"(?i)MUD.?LOG|GAS CHROMAT|HYDROCARBON SHOW|FLUORESCENCE"
        ),
    }
    pattern = patterns.get(category)
    if pattern is None:
        return pd.DataFrame()
    path = Path(getattr(record, "path", "__missing__"))
    if not path.exists():
        return pd.DataFrame()
    rows = [
        {
            "File": item.name,
            "Format": item.suffix.upper().lstrip("."),
            "Source Path": str(item),
        }
        for item in path.rglob("*")
        if item.is_file() and pattern.search(item.name)
    ]
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def collect_scanned_logs(well_path: Path) -> list[Path]:
    supported = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    roots = [
        well_path / "1_Well Log Scanned",
        well_path / well_path.name / "1_Well Log Scanned",
        well_path / "5_Well_Log_Raw",
        well_path / well_path.name / "5_Well_Log_Raw",
    ]
    files = {
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    }
    return sorted(files, key=lambda path: path.name.lower())


def collect_digitized_las(field: str, well: str) -> list[Path]:
    normalized_field = normalize_well(field)
    normalized_well = normalize_well(well)
    roots = (
        DIGITIZATION_OUTPUT / normalized_field / normalized_well,
        LEGACY_DIGITIZATION_OUTPUT / normalized_field / normalized_well,
        MINAS_TRAINING_ROOT,
    )
    files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.las"):
            if root == MINAS_TRAINING_ROOT and normalized_well not in normalize_well(
                str(path.relative_to(root))
            ):
                continue
            files.add(path)
    ordered = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)
    unique: list[Path] = []
    seen: set[tuple[str, int]] = set()
    for path in ordered:
        key = (path.name.lower(), path.stat().st_size)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def collect_digitizer_artifacts(field: str, well: str) -> pd.DataFrame:
    normalized_field = normalize_well(field)
    normalized_well = normalize_well(well)
    roots = (
        DIGITIZATION_OUTPUT / normalized_field / normalized_well,
        LEGACY_DIGITIZATION_OUTPUT / normalized_field / normalized_well,
        MINAS_TRAINING_ROOT,
    )
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            if root == MINAS_TRAINING_ROOT and normalized_well not in normalize_well(
                relative
            ):
                continue
            rows.append(
                {
                    "File": path.name,
                    "Type": path.suffix.upper().lstrip("."),
                    "Size (KB)": round(path.stat().st_size / 1024.0, 1),
                    "Modified": pd.Timestamp(path.stat().st_mtime, unit="s"),
                    "Result Root": str(root),
                    "Path": str(path),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .drop_duplicates("Path")
        .sort_values("Modified", ascending=False)
        .drop_duplicates(["File", "Size (KB)"], keep="first")
    )


@lru_cache(maxsize=1)
def load_digitizer_module():
    spec = importlib.util.spec_from_file_location("zona_rokan_digitizer", DIGITIZER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load digitizer module from {DIGITIZER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_profile_registry() -> dict[str, Any]:
    if not PROFILE_REGISTRY.exists():
        return {}
    try:
        value = json.loads(PROFILE_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def available_digitizer_profiles() -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if TRAINING_CONFIG.exists():
        try:
            payload = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        for job in payload.get("jobs", []):
            source = Path(job.get("input", ""))
            well = normalize_well(source.parts[-3] if len(source.parts) >= 3 else job.get("id"))
            calibration = job.get("calibration", {})
            if well and calibration:
                profiles[f"TRAINING::{well}::{job.get('id', well)}"] = {
                    "well": well,
                    "label": f"{job.get('id', well)} ({job.get('calibration_status', 'UNKNOWN')})",
                    "status": str(job.get("calibration_status", "UNKNOWN")).upper(),
                    "source": "MINAS training config",
                    "calibration": calibration,
                }
    for key, profile in _read_profile_registry().items():
        if isinstance(profile, dict) and isinstance(profile.get("calibration"), dict):
            profiles[f"CUSTOM::{key}"] = {
                **profile,
                "well": normalize_well(profile.get("well", key)),
                "label": profile.get("label", key),
                "status": profile.get("status", "READY"),
                "source": "Dashboard profile registry",
            }
    return profiles


def save_digitizer_profile(
    field: str,
    well: str,
    curve: str,
    calibration: dict[str, Any],
) -> Path:
    PROFILE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry = _read_profile_registry()
    key = f"{normalize_well(field)}::{normalize_well(well)}::{normalize_well(curve)}"
    registry[key] = {
        "field": field,
        "well": well,
        "curve": curve,
        "label": f"{field}/{well} {curve}",
        "status": "READY",
        "calibration": calibration,
    }
    temporary = PROFILE_REGISTRY.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    temporary.replace(PROFILE_REGISTRY)
    return PROFILE_REGISTRY


def run_python_tool(script: Path, arguments: list[str], timeout: int = 900) -> dict[str, Any]:
    command = [sys.executable, str(script), *arguments]
    result = subprocess.run(
        command,
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _subtract_interval(
    intervals: list[tuple[float, float]], close_top: float, close_base: float
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for top, base in intervals:
        if close_base <= top or close_top >= base:
            output.append((top, base))
            continue
        if close_top > top:
            output.append((top, min(close_top, base)))
        if close_base < base:
            output.append((max(close_base, top), base))
    return [(top, base) for top, base in output if base - top > 0.01]


def active_perforation_intervals(history: pd.DataFrame) -> list[tuple[float, float]]:
    if history.empty:
        return []
    frame = history.copy()
    frame["__date"] = pd.to_datetime(frame.get("Event Date"), errors="coerce", dayfirst=True)
    frame["__top"] = pd.to_numeric(frame.get("Perforation Top (ft)"), errors="coerce")
    frame["__base"] = pd.to_numeric(frame.get("Perforation Base (ft)"), errors="coerce")
    frame = frame.dropna(subset=["__top", "__base"]).sort_values("__date")
    source_sheets = frame.get(
        "Source Sheet", pd.Series("", index=frame.index, dtype=object)
    ).fillna("").astype(str)
    if source_sheets.eq("Existing perfo + NZBP").any():
        open_rows = frame.loc[
            frame["Action Status"].fillna("").astype(str).str.upper().eq("OPEN")
        ]
        return sorted(
            {
                tuple(sorted((float(row["__top"]), float(row["__base"]))))
                for _, row in open_rows.iterrows()
            }
        )
    active: list[tuple[float, float]] = []
    for _, row in frame.iterrows():
        top, base = sorted((float(row["__top"]), float(row["__base"])))
        action = clean_text(row.get("Action Status")).upper()
        if action == "CLOSE":
            active = _subtract_interval(active, top, base)
        elif action == "OPEN":
            active.append((top, base))
    return sorted(active)


def interval_overlap(
    top: float, base: float, intervals: Iterable[tuple[float, float]]
) -> float:
    thickness = max(base - top, 1e-9)
    overlap = sum(max(0.0, min(base, other_base) - max(top, other_top)) for other_top, other_base in intervals)
    return clamp(100.0 * overlap / thickness, 0, 100)


def find_nzbp_candidates(
    work: pd.DataFrame,
    reservoir_flag: pd.Series,
    active_perforations: list[tuple[float, float]],
    minimum_thickness: float,
    has_perforation_history: bool,
) -> pd.DataFrame:
    columns = [
        "Top MD",
        "Base MD",
        "Gross Thickness",
        "Average Vsh",
        "Average PhiE",
        "Average Sw",
        "Average Resistivity",
        "Average Permeability (mD)",
        "Perforation Overlap (%)",
        "NZBP Status",
        "Screening Score",
    ]
    if work.empty:
        return pd.DataFrame(columns=columns)

    depths = pd.to_numeric(pd.Index(work.index), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(depths)
    frame = work.iloc[np.where(valid)[0]].copy()
    depths = depths[valid]
    flags = reservoir_flag.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    if not flags.any():
        return pd.DataFrame(columns=columns)
    steps = np.diff(depths)
    finite_steps = np.abs(steps[np.isfinite(steps) & (steps != 0)])
    sample_step = float(np.median(finite_steps)) if finite_steps.size else 0.5
    gap_limit = max(sample_step * 1.75, sample_step + 1e-6)

    groups: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for index, is_reservoir in enumerate(flags):
        continuous = (
            previous is not None
            and abs(depths[index] - depths[previous]) <= gap_limit
        )
        if is_reservoir and (start is None or continuous):
            start = index if start is None else start
        elif is_reservoir:
            groups.append((start, previous))  # type: ignore[arg-type]
            start = index
        elif start is not None:
            groups.append((start, previous))  # type: ignore[arg-type]
            start = None
        previous = index
    if start is not None and previous is not None:
        groups.append((start, previous))

    candidates: list[dict[str, Any]] = []
    for start_index, end_index in groups:
        top, base = sorted((float(depths[start_index]), float(depths[end_index])))
        thickness = base - top + sample_step
        if thickness < minimum_thickness:
            continue
        interval = frame.iloc[start_index : end_index + 1]
        vsh = float(interval["VSH"].mean()) if "VSH" in interval else np.nan
        phi = float(interval["POROSITY"].mean()) if "POROSITY" in interval else np.nan
        sw = float(interval["SW"].mean()) if "SW" in interval else np.nan
        resistivity = float(interval["RES"].mean()) if "RES" in interval else np.nan
        permeability = (
            float(interval["PERM_MD"].mean()) if "PERM_MD" in interval else np.nan
        )
        overlap = interval_overlap(top, base, active_perforations)
        if not has_perforation_history:
            nzbp_status = "UNCONFIRMED - NO PERFORATION HISTORY"
        elif overlap <= 1.0:
            nzbp_status = "NZBP SCREENING CANDIDATE"
        elif overlap < 50.0:
            nzbp_status = "PARTIALLY BEHIND PIPE"
        else:
            nzbp_status = "CURRENT / HISTORICALLY OPEN"

        quality = 0.0
        weight = 0.0
        if np.isfinite(vsh):
            quality += clamp((1.0 - vsh) * 100) * 0.25
            weight += 0.25
        if np.isfinite(phi):
            quality += clamp(phi / 0.25 * 100) * 0.30
            weight += 0.30
        if np.isfinite(sw):
            quality += clamp((1.0 - sw) * 100) * 0.25
            weight += 0.25
        if np.isfinite(resistivity):
            quality += clamp(math.log1p(max(resistivity, 0)) / math.log(101) * 100) * 0.10
            weight += 0.10
        quality += clamp(thickness / 20.0 * 100) * 0.10
        weight += 0.10
        score = quality / max(weight, 1e-9)
        if overlap > 0:
            score *= max(0.2, 1.0 - overlap / 100.0)

        candidates.append(
            {
                "Top MD": round(top, 2),
                "Base MD": round(base, 2),
                "Gross Thickness": round(thickness, 2),
                "Average Vsh": round(vsh, 3) if np.isfinite(vsh) else np.nan,
                "Average PhiE": round(phi, 3) if np.isfinite(phi) else np.nan,
                "Average Sw": round(sw, 3) if np.isfinite(sw) else np.nan,
                "Average Resistivity": round(resistivity, 2)
                if np.isfinite(resistivity)
                else np.nan,
                "Average Permeability (mD)": round(permeability, 2)
                if np.isfinite(permeability)
                else np.nan,
                "Perforation Overlap (%)": round(overlap, 1),
                "NZBP Status": nzbp_status,
                "Screening Score": round(score, 1),
            }
        )
    return pd.DataFrame(candidates, columns=columns).sort_values(
        ["Screening Score", "Gross Thickness"], ascending=False
    )
