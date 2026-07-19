from __future__ import annotations

import io
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import lasio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_DATA_ROOT = Path(
    r"D:\HALLIBURTON CONSULTING\Data Nations\Zona Rokan"
)
NULL_VALUES = (-999.25, -9999.0, -999.0, -99999.0)
SUPPORTED_LOG_EXTENSIONS = {".las", ".dlis", ".lis"}
DEPTH_ALIASES = ("DEPT", "DEPTH", "TDEP", "MD", "MEASURED_DEPTH", "MDEPTH", "TVD", "TVDSS")
BINARY_PARSE_TIMEOUT_SECONDS = 60

# Aliases are ordered by preference and expanded for the Zona Rokan LAS files.
CURVE_ALIASES = {
    "Gamma Ray": ("GR", "HGR", "GR_EDTC", "CGR", "SGR"),
    "Deep Resistivity": (
        "DRES",
        "RT",
        "RDEP",
        "LLD",
        "ILD",
        "RLLL",
        "RLLD",
        "RLL",
        "HRD",
        "AT90",
        "RTCH",
        "RTCH_R",
    ),
    "Shallow Resistivity": ("SRES", "RXO", "MSFL", "LLS", "ILM", "AT10"),
    "Density": ("RHOB", "DEN", "ZDEN", "RHOZ"),
    "Density Correction": ("DRHO", "HDRA", "RHOCOR"),
    "Neutron": ("NPHI", "HCNL", "TNPH", "NEU", "NPHI_LS"),
    "Caliper": ("CALI", "CALX", "CALS", "HCAL"),
    "Bit Size": ("BIT", "BS", "BSZ"),
    "PEF": ("PE", "PEF", "PEFZ"),
    "Compressional Sonic": ("DTC_FINAL", "DTC", "DT", "AC", "DTCO"),
    "Shear Sonic": ("DTS_FINAL", "DTS", "DTSM", "ACS", "DT_S"),
    "Interpreted Porosity": ("PHIE", "PHIT", "PHIT_E", "PHID", "POR"),
    "Interpreted Vsh": ("VSH", "VSH_GR", "VCL", "VCLGR"),
    "Water Saturation": ("SWE", "SW", "SWIRR", "SWC"),
    "Permeability": ("PERM", "K", "KLOGH"),
    "Total Gas": ("TGAS", "GAS", "TOTALGAS", "C1C5", "GTOT"),
}


@dataclass(frozen=True)
class WellRecord:
    field: str
    well: str
    path: Path
    las_files: tuple[Path, ...]
    scanned_files: tuple[Path, ...]
    production_files: tuple[Path, ...]


@dataclass
class LoadedLog:
    """A common depth-indexed representation for LAS, DLIS, and LIS inputs."""

    data: pd.DataFrame
    source_type: str
    depth_mnemonic: str
    depth_unit: str
    well: str = ""
    field: str = ""
    curve_units: dict[str, str] | None = None

    def curve_unit(self, mnemonic: str | None) -> str:
        if not mnemonic or not self.curve_units:
            return ""
        return str(self.curve_units.get(str(mnemonic).upper(), ""))


def normalize_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def safe_header_value(las: lasio.LASFile, mnemonic: str, fallback: str = "") -> str:
    try:
        value = las.well[mnemonic].value
    except (KeyError, AttributeError, TypeError):
        return fallback
    text = str(value).replace("\x00", "").strip()
    return text or fallback


def _record_inventory_issue(
    issues: list[str], path: Path, action: str, error: OSError
) -> None:
    """Store a concise, deduplicated warning for an unreadable data path."""
    winerror = getattr(error, "winerror", None)
    error_number = f"WinError {winerror}" if winerror else error.__class__.__name__
    detail = error.strerror or str(error)
    message = f"{action}: {path} ({error_number}: {detail})"
    if message not in issues:
        issues.append(message)


def _safe_is_directory(path: Path, issues: list[str], action: str) -> bool:
    try:
        return path.is_dir()
    except OSError as error:
        _record_inventory_issue(issues, path, action, error)
        return False


def _safe_directories(path: Path, issues: list[str]) -> tuple[Path, ...]:
    """List direct child directories without allowing a damaged path to abort a scan."""
    directories: list[Path] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        directories.append(Path(entry.path))
                except OSError as error:
                    _record_inventory_issue(issues, Path(entry.path), "Inspecting directory", error)
    except OSError as error:
        _record_inventory_issue(issues, path, "Listing directories", error)
    return tuple(sorted(directories, key=lambda item: item.name.lower()))


def _safe_files(
    path: Path,
    issues: list[str],
    suffixes: set[str],
    name_contains: str | None = None,
) -> tuple[Path, ...]:
    """Return matching direct child files while gracefully skipping unreadable folders."""
    files: list[Path] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if not entry.is_file():
                        continue
                except OSError as error:
                    _record_inventory_issue(issues, Path(entry.path), "Inspecting file", error)
                    continue

                candidate = Path(entry.path)
                if candidate.suffix.lower() not in suffixes:
                    continue
                if name_contains and name_contains not in candidate.name.lower():
                    continue
                files.append(candidate)
    except OSError as error:
        _record_inventory_issue(issues, path, "Listing files", error)
    return tuple(files)


def _safe_file_size(path: Path, issues: list[str]) -> int:
    try:
        return path.stat().st_size
    except OSError as error:
        _record_inventory_issue(issues, path, "Reading file metadata", error)
        return -1


@st.cache_data(show_spinner=False)
def _scan_inventory(root_text: str) -> tuple[tuple[WellRecord, ...], tuple[str, ...]]:
    root = Path(root_text)
    issues: list[str] = []
    if not _safe_is_directory(root, issues, "Accessing data root"):
        return (), tuple(issues)

    records: list[WellRecord] = []
    for field_dir in _safe_directories(root, issues):
        if field_dir.name.startswith(".") or field_dir.name.upper() == "HP":
            continue
        for well_dir in _safe_directories(field_dir, issues):
            # Two Zona Rokan wells have a duplicated well-name directory.
            candidate_roots = (well_dir, well_dir / well_dir.name)
            log_roots = tuple(
                candidate_root / "5_Well_Log_Raw"
                for candidate_root in candidate_roots
                if _safe_is_directory(
                    candidate_root / "5_Well_Log_Raw", issues, "Accessing log folder"
                )
            )
            production_roots = tuple(
                candidate_root / "6_Prod"
                for candidate_root in candidate_roots
                if _safe_is_directory(
                    candidate_root / "6_Prod", issues, "Accessing production folder"
                )
            )
            scanned_roots = tuple(
                candidate_root / "1_Well Log Scanned"
                for candidate_root in candidate_roots
                if _safe_is_directory(
                    candidate_root / "1_Well Log Scanned", issues, "Accessing scanned-log folder"
                )
            )
            las_files = tuple(
                sorted(
                    {
                        p
                        for log_root in log_roots
                        for p in _safe_files(log_root, issues, SUPPORTED_LOG_EXTENSIONS)
                    },
                    key=lambda p: (_safe_file_size(p, issues), p.name.lower()),
                    reverse=True,
                )
            )
            scanned_files = tuple(
                sorted(
                    {
                        p
                        for scanned_root in scanned_roots
                        for p in _safe_files(
                            scanned_root,
                            issues,
                            {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"},
                        )
                    },
                    key=lambda p: p.name.lower(),
                )
            )
            production_files = tuple(
                sorted(
                    {
                        p
                        for production_root in production_roots
                        for p in _safe_files(
                            production_root, issues, {".xlsx"}, name_contains="prod"
                        )
                    }
                )
            )
            records.append(
                WellRecord(
                    field=field_dir.name,
                    well=well_dir.name,
                    path=well_dir,
                    las_files=las_files,
                    scanned_files=scanned_files,
                    production_files=production_files,
                )
            )
    return tuple(records), tuple(issues)


def scan_inventory(root_text: str) -> list[WellRecord]:
    """Return the usable inventory, ignoring unreadable source directories."""
    records, _ = _scan_inventory(root_text)
    return list(records)


def inventory_scan_issues(root_text: str) -> list[str]:
    """Return non-fatal source-data access warnings from the latest inventory scan."""
    _, issues = _scan_inventory(root_text)
    return list(issues)


def read_las(source: Path | bytes) -> lasio.LASFile:
    if isinstance(source, Path):
        return lasio.read(
            str(source),
            ignore_header_errors=True,
            mnemonic_case="upper",
        )

    text = source.decode("utf-8", errors="ignore")
    return lasio.read(
        io.StringIO(text),
        ignore_header_errors=True,
        mnemonic_case="upper",
    )


def prepare_dataframe(las: lasio.LASFile) -> pd.DataFrame:
    df = las.df().copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.loc[~pd.isna(df.index)]
    df.index.name = "DEPTH"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace(NULL_VALUES, np.nan, inplace=True)
    return df


def _prepare_binary_dataframe(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize a scalar DLIS/LIS table with a depth column into dashboard form."""
    frame = table.copy()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    normalized = {normalize_name(column): column for column in frame.columns}
    depth_column = next(
        (normalized[normalize_name(alias)] for alias in DEPTH_ALIASES if normalize_name(alias) in normalized),
        None,
    )
    if depth_column is None:
        available = ", ".join(frame.columns[:15]) or "none"
        raise ValueError(f"no recognised depth channel; found: {available}")
    frame = frame.set_index(depth_column)
    frame.index = pd.to_numeric(frame.index, errors="coerce")
    frame = frame.loc[frame.index.notna()].copy()
    frame.index.name = "DEPTH"
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.replace(NULL_VALUES, np.nan, inplace=True)
    frame = frame.groupby(level=0).median().sort_index()
    return frame, str(depth_column)


def _binary_curve_score(frame: pd.DataFrame) -> tuple[int, int, int]:
    mapping = auto_curve_map(frame)
    critical_roles = ("Gamma Ray", "Deep Resistivity", "Density", "Neutron")
    critical = sum(int(mapping.get(role) is not None) for role in critical_roles)
    available = sum(int(value is not None) for value in mapping.values())
    samples = sum(
        int(frame[column].notna().sum())
        for column in mapping.values()
        if column is not None
    )
    return critical, available, samples


def _write_binary_payload(payload: bytes, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="zona_rokan_log_", suffix=suffix, delete=False)
    try:
        handle.write(payload)
        return Path(handle.name)
    finally:
        handle.close()


def _read_binary_log_payload(payload: bytes, file_type: str) -> tuple[pd.DataFrame, str, str]:
    """Read DLIS/LIS in a timed worker subprocess so malformed files cannot hang Streamlit."""
    if len(payload) < 512:
        raise ValueError(f"{file_type.upper()} file is too small to contain a valid indexed log")
    temporary_path = _write_binary_payload(payload, f".{file_type}")
    result_handle = tempfile.NamedTemporaryFile(prefix="zona_rokan_log_result_", suffix=".pkl", delete=False)
    result_path = Path(result_handle.name)
    result_handle.close()
    worker_script = Path(__file__).with_name("wireline_binary_parser.py")
    try:
        try:
            completed = subprocess.run(
                [sys.executable, str(worker_script), file_type, str(temporary_path), str(result_path)],
                capture_output=True,
                text=True,
                timeout=BINARY_PARSE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError(
                f"{file_type.upper()} indexing exceeded {BINARY_PARSE_TIMEOUT_SECONDS} seconds; the file may be corrupt"
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown parser error").strip().splitlines()[-1]
            raise ValueError(f"{file_type.upper()} parser failed: {detail}")
        result = pd.read_pickle(result_path)
        candidates: list[tuple[str, pd.DataFrame, str]] = []
        failures: list[str] = []
        for label, table in result:
            try:
                frame, depth_mnemonic = _prepare_binary_dataframe(table)
                if not frame.empty:
                    candidates.append((label, frame, depth_mnemonic))
            except Exception as error:
                failures.append(f"{label}: {error}")
        if not candidates:
            detail = f" First issue: {failures[0]}" if failures else ""
            raise ValueError(f"no readable depth-indexed {file_type.upper()} frame/logset was found.{detail}")
        label, frame, depth_mnemonic = max(candidates, key=lambda candidate: _binary_curve_score(candidate[1]))
        return frame, depth_mnemonic, label
    finally:
        temporary_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def read_log(source: Path | bytes, source_name: str = "") -> LoadedLog:
    """Load a LAS, DLIS, or LIS file into a common data/metadata container."""
    name = source.name if isinstance(source, Path) else source_name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_LOG_EXTENSIONS:
        raise ValueError(f"Unsupported log format: {suffix or 'no extension'}")
    if suffix == ".las":
        las = read_las(source)
        units = {str(curve.mnemonic).upper(): str(curve.unit or "").strip() for curve in las.curves}
        depth_mnemonic = str(las.curves[0].mnemonic).upper() if las.curves else "DEPTH"
        return LoadedLog(
            data=prepare_dataframe(las),
            source_type="LAS",
            depth_mnemonic=depth_mnemonic,
            depth_unit=units.get(depth_mnemonic, ""),
            well=safe_header_value(las, "WELL", safe_header_value(las, "UWI", "")),
            field=safe_header_value(las, "FLD", ""),
            curve_units=units,
        )
    payload = source.read_bytes() if isinstance(source, Path) else source
    file_type = suffix.removeprefix(".")
    frame, depth_mnemonic, frame_label = _read_binary_log_payload(payload, file_type)
    return LoadedLog(
        data=frame,
        source_type=f"{file_type.upper()} ({frame_label})",
        depth_mnemonic=depth_mnemonic,
        depth_unit="",
        curve_units={},
    )


@st.cache_data(show_spinner=False)
def load_log_path(path_text: str) -> LoadedLog:
    return read_log(Path(path_text))


def curve_unit(las: lasio.LASFile, mnemonic: str | None) -> str:
    if not mnemonic:
        return ""
    try:
        return str(las.curves[mnemonic].unit).strip()
    except (KeyError, AttributeError, TypeError):
        return ""


def best_curve(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    by_normalized = {normalize_name(c): c for c in df.columns}
    candidates: list[str] = []
    for alias in aliases:
        exact = by_normalized.get(normalize_name(alias))
        if exact is not None:
            candidates.append(exact)
    if not candidates:
        return None
    return max(candidates, key=lambda c: int(df[c].notna().sum()))


def auto_curve_map(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        role: best_curve(df, aliases)
        for role, aliases in CURVE_ALIASES.items()
    }


def normalize_neutron(series: pd.Series, unit: str) -> pd.Series:
    result = series.astype(float).copy()
    finite = result.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return result
    unit_upper = unit.upper()
    if "V/V" in unit_upper or finite.quantile(0.95) <= 1.5:
        result *= 100.0
    return result


def normalize_fraction(series: pd.Series) -> pd.Series:
    result = series.astype(float).copy()
    finite = result.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty and finite.quantile(0.95) > 1.5:
        result /= 100.0
    return result


MINERAL_DENSITIES = {
    "quartz": 2.65,
    "sandstone": 2.65,
    "feldspar": 2.62,
    "calcite": 2.71,
    "limestone": 2.71,
    "dolomite": 2.87,
    "anhydrite": 2.98,
    "clay": 2.58,
    "illite": 2.58,
    "kaolinite": 2.60,
    "smectite": 2.45,
    "mica": 2.83,
    "siderite": 3.96,
    "pyrite": 5.00,
}


def read_table_upload(uploaded_file) -> pd.DataFrame:
    """Read an uploaded core, XRD, or evidence table without modifying the source."""
    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.getvalue()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(payload))
    try:
        return pd.read_csv(BytesIO(payload))
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(payload), sep=r"\s+", engine="python")


def xrd_matrix_density(xrd: pd.DataFrame | None) -> tuple[float | None, pd.DataFrame]:
    """Calculate a mineral-weighted matrix density from a simple XRD table."""
    columns = ["Mineral", "Fraction (%)", "Density (g/cc)"]
    if xrd is None or xrd.empty:
        return None, pd.DataFrame(columns=columns)
    text_columns = xrd.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = xrd.select_dtypes(include=np.number).columns.tolist()
    if not text_columns or not numeric_columns:
        return None, pd.DataFrame(columns=columns)
    normalized = {normalize_name(column): column for column in numeric_columns}
    fraction_column = next(
        (
            normalized[key]
            for key in ("PERCENT", "PCT", "FRACTION", "WT", "VOLUME")
            if key in normalized
        ),
        numeric_columns[0],
    )
    fractions = pd.to_numeric(xrd[fraction_column], errors="coerce")
    if fractions.max(skipna=True) <= 1.2:
        fractions *= 100.0
    rows: list[dict[str, object]] = []
    for mineral, fraction in zip(xrd[text_columns[0]].astype(str), fractions):
        key = mineral.lower().strip()
        density = next((value for name, value in MINERAL_DENSITIES.items() if name in key), np.nan)
        rows.append({"Mineral": mineral, "Fraction (%)": fraction, "Density (g/cc)": density})
    table = pd.DataFrame(rows).dropna(subset=["Fraction (%)"])
    usable = table.dropna(subset=["Density (g/cc)"])
    if usable.empty or usable["Fraction (%)"].sum() <= 0:
        return None, table
    density = float(np.average(usable["Density (g/cc)"], weights=usable["Fraction (%)"]))
    return density, table


def validate_core_measurements(core: pd.DataFrame | None, interval: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Depth-match uploaded core porosity/permeability against interpreted logs."""
    if core is None or core.empty:
        return pd.DataFrame(), {}
    normalized = {normalize_name(column): column for column in core.columns}
    depth_column = next(
        (normalized[key] for key in ("MD", "DEPTH", "DEPT", "COREDEPTH") if key in normalized),
        None,
    )
    phi_column = next(
        (normalized[key] for key in ("PHI", "POROSITY", "COREPHI", "PHIT", "PHIE") if key in normalized),
        None,
    )
    permeability_column = next(
        (normalized[key] for key in ("K", "PERM", "PERMEABILITY", "COREK", "KCORE") if key in normalized),
        None,
    )
    if depth_column is None:
        return pd.DataFrame(), {}
    result = pd.DataFrame({"MD": pd.to_numeric(core[depth_column], errors="coerce")}).dropna()
    if phi_column is not None:
        result["CORE_PHI"] = pd.to_numeric(core.loc[result.index, phi_column], errors="coerce")
        if result["CORE_PHI"].median(skipna=True) > 1.2:
            result["CORE_PHI"] /= 100.0
    if permeability_column is not None:
        result["CORE_K"] = pd.to_numeric(core.loc[result.index, permeability_column], errors="coerce")
    for source_column, target_column in (("PHIE", "PHIE"), ("PERM_MD", "PERM_MD"), ("SW", "SW")):
        if source_column not in interval:
            continue
        source = interval[source_column].dropna().sort_index()
        if len(source) >= 2:
            result[target_column] = np.interp(result["MD"], source.index.to_numpy(float), source.to_numpy(float), left=np.nan, right=np.nan)
    metrics: dict[str, float | int] = {}
    if {"CORE_PHI", "PHIE"}.issubset(result.columns):
        pair = result.dropna(subset=["CORE_PHI", "PHIE"])
        if len(pair) >= 2:
            residual = pair["PHIE"] - pair["CORE_PHI"]
            metrics["Porosity samples"] = int(len(pair))
            metrics["Porosity MAE (p.u.)"] = float(residual.abs().mean())
            metrics["Porosity bias (p.u.)"] = float(residual.mean())
    if {"CORE_K", "PERM_MD"}.issubset(result.columns):
        pair = result[(result["CORE_K"] > 0) & (result["PERM_MD"] > 0)].dropna(subset=["CORE_K", "PERM_MD"])
        if len(pair) >= 2:
            metrics["Permeability samples"] = int(len(pair))
            metrics["log10 K MAE"] = float(np.abs(np.log10(pair["PERM_MD"]) - np.log10(pair["CORE_K"])).mean())
    return result, metrics


def dataframe_xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31], index=name == "Computed logs")
    return buffer.getvalue()


def calculate_vsh(
    gr: pd.Series,
    clean_gr: float,
    shale_gr: float,
    method: str,
) -> pd.Series:
    denominator = shale_gr - clean_gr
    if abs(denominator) < 1e-9:
        return pd.Series(np.nan, index=gr.index)
    igr = ((gr - clean_gr) / denominator).clip(0.0, 1.0)
    if method == "Linear":
        vsh = igr
    elif method == "Larionov Tertiary":
        vsh = 0.083 * (2 ** (3.7 * igr) - 1)
    elif method == "Larionov Older Rocks":
        vsh = 0.33 * (2 ** (2 * igr) - 1)
    elif method == "Clavier":
        vsh = 1.7 - np.sqrt(3.38 - (igr + 0.7) ** 2)
    else:
        vsh = igr / (3.0 - 2.0 * igr)
    return pd.Series(vsh, index=gr.index).clip(0.0, 1.0)


def density_porosity(
    density: pd.Series,
    matrix_density: float,
    fluid_density: float,
) -> pd.Series:
    denominator = matrix_density - fluid_density
    if abs(denominator) < 1e-9:
        return pd.Series(np.nan, index=density.index)
    return ((matrix_density - density) / denominator).clip(-0.1, 0.6)


def sonic_porosity(
    dt: pd.Series,
    matrix_dt: float,
    fluid_dt: float,
) -> pd.Series:
    denominator = fluid_dt - matrix_dt
    if abs(denominator) < 1e-9:
        return pd.Series(np.nan, index=dt.index)
    return ((dt - matrix_dt) / denominator).clip(-0.1, 0.6)


def archie_water_saturation(
    porosity: pd.Series,
    resistivity: pd.Series,
    rw: float,
    a: float,
    m: float,
    n: float,
) -> pd.Series:
    phi = porosity.where(porosity > 0)
    rt = resistivity.where(resistivity > 0)
    sw = ((a * rw) / (rt * phi.pow(m))).pow(1.0 / n)
    return sw.clip(0.0, 1.0)


def estimate_gr_depth_shift(
    reference: pd.Series,
    moving: pd.Series,
    max_shift: float = 50.0,
    shift_step: float = 0.5,
) -> tuple[float, float]:
    """Estimate the moving-run depth correction that maximizes GR correlation."""
    ref = reference.dropna().sort_index()
    mov = moving.dropna().sort_index()
    if len(ref) < 20 or len(mov) < 20 or shift_step <= 0:
        return 0.0, np.nan
    overlap_top = max(float(ref.index.min()), float(mov.index.min()) - max_shift)
    overlap_base = min(float(ref.index.max()), float(mov.index.max()) + max_shift)
    if overlap_base <= overlap_top:
        return 0.0, np.nan
    sample = max(
        estimate_sample_thickness(ref.index),
        estimate_sample_thickness(mov.index),
        shift_step,
    )
    grid = np.arange(overlap_top, overlap_base + sample * 0.5, sample)
    ref_values = np.interp(
        grid,
        ref.index.to_numpy(dtype=float),
        ref.to_numpy(dtype=float),
        left=np.nan,
        right=np.nan,
    )
    best_shift = 0.0
    best_correlation = -np.inf
    shifts = np.arange(-max_shift, max_shift + shift_step * 0.5, shift_step)
    for shift in shifts:
        moving_values = np.interp(
            grid - shift,
            mov.index.to_numpy(dtype=float),
            mov.to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(ref_values) & np.isfinite(moving_values)
        if valid.sum() < 20:
            continue
        correlation = np.corrcoef(ref_values[valid], moving_values[valid])[0, 1]
        if np.isfinite(correlation) and correlation > best_correlation:
            best_correlation = float(correlation)
            best_shift = float(shift)
    return best_shift, best_correlation if np.isfinite(best_correlation) else np.nan


def merge_shifted_runs(
    reference: pd.DataFrame,
    moving_runs: list[tuple[pd.DataFrame, float]],
) -> pd.DataFrame:
    """Splice runs on a common depth index after applying GR-derived shifts."""
    output = reference.copy()
    for frame, shift in moving_runs:
        shifted = frame.copy()
        shifted.index = pd.Index(
            pd.to_numeric(shifted.index, errors="coerce") + shift,
            name=reference.index.name,
        )
        shifted = shifted.loc[~pd.isna(shifted.index)]
        shifted = shifted[~shifted.index.duplicated(keep="first")].sort_index()
        output = output.combine_first(shifted)
    return output.sort_index()


def condition_log_curves(
    frame: pd.DataFrame,
    caliper: pd.Series,
    bit_size: pd.Series,
    density_correction: pd.Series,
    washout_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove physically unrealistic values and produce borehole QC flags."""
    output = frame.copy()
    ranges = {
        "GR": (0.0, 350.0),
        "RES": (0.001, 200000.0),
        "RHOB": (1.0, 3.5),
        "NPHI_FRAC": (-0.15, 0.80),
        "CALI": (2.0, 40.0),
        "DTC": (30.0, 350.0),
        "PEF": (0.0, 20.0),
    }
    unrealistic = pd.Series(False, index=output.index)
    for column, (minimum, maximum) in ranges.items():
        if column not in output:
            continue
        invalid = output[column].notna() & ~output[column].between(minimum, maximum)
        unrealistic |= invalid
        output.loc[invalid, column] = np.nan
    badhole = pd.Series(False, index=output.index)
    if caliper.notna().any() and bit_size.notna().any():
        badhole |= caliper > bit_size + washout_tolerance
    if density_correction.notna().any():
        badhole |= density_correction.abs() > 0.15
    qc = pd.DataFrame(
        {
            "BADHOLE_FLAG": badhole.fillna(False),
            "UNREALISTIC_REMOVED": unrealistic.fillna(False),
        },
        index=output.index,
    )
    return output, qc


def modified_simandoux_water_saturation(
    porosity: pd.Series,
    resistivity: pd.Series,
    vshale: pd.Series,
    rw: pd.Series | float,
    rsh: float,
    a: float,
    m: float,
    n: float,
) -> pd.Series:
    """
    Solve 1/Rt = Sw^n*Phi^m/(a*Rw) + Vsh*Sw/Rsh.

    This Bardon-Pied/modified-Simandoux form uses effective porosity and is
    solved numerically so saturation exponent n is not restricted to 2.
    """
    index = porosity.index
    phi = porosity.astype(float).reindex(index)
    rt = resistivity.astype(float).reindex(index)
    shale = vshale.astype(float).reindex(index).clip(0.0, 1.0)
    rw_series = (
        pd.Series(float(rw), index=index)
        if np.isscalar(rw)
        else pd.Series(rw, index=index, dtype=float)
    )
    values = np.full(len(index), np.nan)
    for position, (phi_value, rt_value, shale_value, rw_value) in enumerate(
        zip(phi, rt, shale, rw_series)
    ):
        if (
            not np.isfinite(phi_value)
            or not np.isfinite(rt_value)
            or not np.isfinite(shale_value)
            or not np.isfinite(rw_value)
            or phi_value <= 0
            or rt_value <= 0
            or rw_value <= 0
            or rsh <= 0
            or a <= 0
            or m <= 0
            or n <= 0
        ):
            continue
        archie_term = phi_value**m / (a * rw_value)
        shale_term = shale_value / rsh
        target = 1.0 / rt_value

        def residual(sw_value: float) -> float:
            return archie_term * sw_value**n + shale_term * sw_value - target

        if residual(1.0) <= 0:
            values[position] = 1.0
            continue
        low, high = 0.0, 1.0
        for _ in range(35):
            midpoint = (low + high) / 2.0
            if residual(midpoint) > 0:
                high = midpoint
            else:
                low = midpoint
        values[position] = (low + high) / 2.0
    return pd.Series(values, index=index).clip(0.0, 1.0)


def coates_timur_permeability(
    effective_porosity: pd.Series,
    bound_water: pd.Series,
    coefficient: float = 10000.0,
    porosity_exponent: float = 4.0,
    fluid_ratio_exponent: float = 2.0,
    irreducible_saturation: float = 0.25,
) -> pd.Series:
    """
    Configurable Timur-Coates proxy: k=A*PhiE^B*(FFI/BVI)^C.

    Without an NMR T2 distribution, log-derived bound water is a proxy and the
    result must be calibrated against core permeability.
    """
    phi = effective_porosity.clip(lower=0.0)
    bvi_floor = phi * max(0.01, min(float(irreducible_saturation), 0.95))
    bvi = pd.concat([bound_water.clip(lower=0.0), bvi_floor], axis=1).max(axis=1)
    bvi = bvi.clip(lower=0.001)
    ffi = (phi - bvi).clip(lower=0.001)
    permeability = (
        coefficient
        * phi.pow(porosity_exponent)
        * (ffi / bvi).pow(fluid_ratio_exponent)
    )
    return permeability.replace([np.inf, -np.inf], np.nan).clip(0.0, 1_000_000.0)


def infer_matrix_from_pef(
    pef: pd.Series,
    gamma_ray: pd.Series,
) -> tuple[str, float, str]:
    """Infer a provisional principal matrix from clean-interval PEF response."""
    clean = pef.loc[
        pef.notna()
        & (
            gamma_ray <= gamma_ray.quantile(0.35)
            if gamma_ray.notna().any()
            else True
        )
    ]
    if len(clean) < 10:
        return "Quartz / sandstone", 2.65, "Default sandstone; insufficient clean PEF"
    median_pef = float(clean.median())
    minerals = {
        "Quartz / sandstone": (1.81, 2.65),
        "Dolomite": (3.14, 2.87),
        "Calcite / limestone": (5.08, 2.71),
    }
    model, (_, density) = min(
        minerals.items(), key=lambda item: abs(item[1][0] - median_pef)
    )
    return model, density, f"Clean-interval median PEF={median_pef:.2f}"


def estimate_pickett_rw(
    porosity: pd.Series,
    resistivity: pd.Series,
    vshale: pd.Series,
    a: float = 1.0,
    m: float = 2.0,
    fallback: float = 0.20,
) -> tuple[float, int]:
    """Estimate a screening Rw from clean, lower-resistivity Pickett points."""
    frame = pd.DataFrame(
        {"PHIE": porosity, "RT": resistivity, "VSH": vshale}
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna()
    frame = frame.loc[
        frame["PHIE"].between(0.04, 0.45)
        & frame["RT"].gt(0)
        & frame["VSH"].le(0.35)
    ]
    if len(frame) < 20:
        return float(fallback), len(frame)
    wet_limit = frame["RT"].quantile(0.40)
    wet = frame.loc[frame["RT"] <= wet_limit]
    apparent_rw = wet["RT"] * wet["PHIE"].pow(m) / max(a, 1e-9)
    apparent_rw = apparent_rw.replace([np.inf, -np.inf], np.nan).dropna()
    apparent_rw = apparent_rw.loc[apparent_rw.between(0.005, 5.0)]
    if len(apparent_rw) < 10:
        return float(fallback), len(apparent_rw)
    return float(np.clip(apparent_rw.median(), 0.01, 2.0)), len(apparent_rw)


def autonomous_cutoffs(
    interval: pd.DataFrame,
    validation_mask: pd.Series | None = None,
) -> dict[str, float | str | int]:
    """Derive conservative cutoffs from gas/perforation-validated samples."""
    defaults: dict[str, float | str | int] = {
        "VSH": 0.40,
        "PHIE": 0.10,
        "SW": 0.60,
        "PERM_MD": 1.0,
        "Validation Samples": 0,
        "Basis": "Zona Rokan conservative defaults",
    }
    required = {"VSH", "PHIE", "SW", "PERM_MD"}
    if not required.issubset(interval.columns):
        return defaults
    valid = pd.Series(False, index=interval.index)
    bases: list[str] = []
    if validation_mask is not None:
        valid |= validation_mask.reindex(interval.index).fillna(False)
        bases.append("completion/test interval")
    if "TOTAL_GAS" in interval and interval["TOTAL_GAS"].notna().sum() >= 20:
        gas = interval["TOTAL_GAS"]
        valid |= gas >= gas.quantile(0.75)
        bases.append("upper-quartile total gas")
    candidates = interval.loc[valid, list(required)].dropna()
    if len(candidates) < 10:
        candidates = interval[list(required)].dropna()
        candidates = candidates.loc[
            (candidates["VSH"] <= 0.60)
            & (candidates["PHIE"] >= 0.05)
            & (candidates["PERM_MD"] > 0)
        ]
        bases.append("log-response fallback")
    if len(candidates) < 10:
        return defaults
    return {
        "VSH": float(np.clip(candidates["VSH"].quantile(0.80), 0.25, 0.78)),
        "PHIE": float(np.clip(candidates["PHIE"].quantile(0.20), 0.05, 0.20)),
        "SW": float(np.clip(candidates["SW"].quantile(0.80), 0.45, 0.85)),
        "PERM_MD": float(
            np.clip(candidates["PERM_MD"].quantile(0.20), 0.1, 100.0)
        ),
        "Validation Samples": int(len(candidates)),
        "Basis": ", ".join(dict.fromkeys(bases)) or "log-response distribution",
    }


def minimum_curvature_survey(survey: pd.DataFrame) -> pd.DataFrame:
    """Calculate TVD, north/east departure, closure, and dogleg severity."""
    normalized = {normalize_name(column): column for column in survey.columns}
    md_column = next(
        (normalized[key] for key in ("MD", "MEASUREDDEPTH", "DEPTH") if key in normalized),
        None,
    )
    inc_column = next(
        (normalized[key] for key in ("INC", "INCLINATION", "DEVIATION") if key in normalized),
        None,
    )
    azi_column = next(
        (normalized[key] for key in ("AZI", "AZIMUTH", "DIRECTION") if key in normalized),
        None,
    )
    if md_column is None or inc_column is None or azi_column is None:
        raise ValueError("Deviation survey requires MD, inclination, and azimuth columns.")
    work = survey[[md_column, inc_column, azi_column]].copy()
    work.columns = ["MD", "Inclination", "Azimuth"]
    work = work.apply(pd.to_numeric, errors="coerce").dropna().sort_values("MD")
    work = work.drop_duplicates("MD", keep="last").reset_index(drop=True)
    if work.empty:
        raise ValueError("Deviation survey has no valid stations.")

    initial_md = float(work.loc[0, "MD"])
    initial_inc = math.radians(float(work.loc[0, "Inclination"]))
    initial_azi = math.radians(float(work.loc[0, "Azimuth"]))
    tvd = [initial_md * math.cos(initial_inc)]
    north = [initial_md * math.sin(initial_inc) * math.cos(initial_azi)]
    east = [initial_md * math.sin(initial_inc) * math.sin(initial_azi)]
    dogleg = [0.0]
    dls = [0.0]
    for index in range(1, len(work)):
        md_delta = float(work.loc[index, "MD"] - work.loc[index - 1, "MD"])
        inc1, inc2 = np.radians(
            [work.loc[index - 1, "Inclination"], work.loc[index, "Inclination"]]
        )
        azi1, azi2 = np.radians(
            [work.loc[index - 1, "Azimuth"], work.loc[index, "Azimuth"]]
        )
        cosine = np.clip(
            np.cos(inc1) * np.cos(inc2)
            + np.sin(inc1) * np.sin(inc2) * np.cos(azi2 - azi1),
            -1.0,
            1.0,
        )
        beta = float(np.arccos(cosine))
        ratio = 1.0 if abs(beta) < 1e-12 else 2.0 / beta * np.tan(beta / 2.0)
        tvd.append(
            tvd[-1] + md_delta / 2.0 * (np.cos(inc1) + np.cos(inc2)) * ratio
        )
        north.append(
            north[-1]
            + md_delta
            / 2.0
            * (np.sin(inc1) * np.cos(azi1) + np.sin(inc2) * np.cos(azi2))
            * ratio
        )
        east.append(
            east[-1]
            + md_delta
            / 2.0
            * (np.sin(inc1) * np.sin(azi1) + np.sin(inc2) * np.sin(azi2))
            * ratio
        )
        dogleg.append(np.degrees(beta))
        dls.append(np.degrees(beta) / md_delta * 100.0 if md_delta > 0 else 0.0)
    work["TVD"] = tvd
    work["North Departure"] = north
    work["East Departure"] = east
    work["Horizontal Departure"] = np.hypot(work["North Departure"], work["East Departure"])
    work["Dogleg (deg)"] = dogleg
    work["DLS (deg/100ft)"] = dls
    return work


def estimate_sample_thickness(depth: pd.Index) -> float:
    values = pd.Series(depth.astype(float)).sort_values()
    diffs = values.diff().abs()
    positive = diffs[diffs > 0]
    return float(positive.median()) if not positive.empty else 0.0


def lump_net_intervals(
    interval: pd.DataFrame,
    net_flag: pd.Series,
    minimum_thickness: float = 1.0,
) -> pd.DataFrame:
    """Group contiguous net samples into engineering-ready pay summaries."""
    columns = [
        "Top MD",
        "Base MD",
        "Top TVD",
        "Base TVD",
        "Gross Thickness",
        "Average Vsh",
        "Average PhiE",
        "Average PhiT",
        "Average Sw",
        "Average Perm (mD)",
        "Average Total Gas",
        "Samples",
    ]
    if interval.empty:
        return pd.DataFrame(columns=columns)
    depths = pd.to_numeric(pd.Index(interval.index), errors="coerce").to_numpy(float)
    flags = net_flag.reindex(interval.index).fillna(False).to_numpy(bool)
    step = estimate_sample_thickness(interval.index)
    gap_limit = max(step * 1.75, step + 1e-6)
    groups: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for position, flag in enumerate(flags):
        continuous = (
            previous is not None
            and abs(depths[position] - depths[previous]) <= gap_limit
        )
        if flag and start is None:
            start = position
        elif flag and not continuous:
            groups.append((start, previous))  # type: ignore[arg-type]
            start = position
        elif not flag and start is not None:
            groups.append((start, previous))  # type: ignore[arg-type]
            start = None
        previous = position
    if start is not None and previous is not None:
        groups.append((start, previous))
    rows: list[dict[str, float | int]] = []
    for first, last in groups:
        top, base = sorted((float(depths[first]), float(depths[last])))
        gross = base - top + step
        if gross < minimum_thickness:
            continue
        segment = interval.iloc[first : last + 1]
        tvd = segment.get("TVD", pd.Series(np.nan, index=segment.index))
        rows.append(
            {
                "Top MD": top,
                "Base MD": base,
                "Top TVD": float(tvd.dropna().iloc[0]) if tvd.notna().any() else np.nan,
                "Base TVD": float(tvd.dropna().iloc[-1]) if tvd.notna().any() else np.nan,
                "Gross Thickness": gross,
                "Average Vsh": float(segment["VSH"].mean()),
                "Average PhiE": float(segment["PHIE"].mean()),
                "Average PhiT": float(segment["PHIT"].mean()),
                "Average Sw": float(segment["SW"].mean()),
                "Average Perm (mD)": float(segment["PERM_MD"].mean()),
                "Average Total Gas": (
                    float(segment["TOTAL_GAS"].mean())
                    if "TOTAL_GAS" in segment
                    else np.nan
                ),
                "Samples": int(len(segment)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def load_tops(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        source = pd.read_excel(uploaded_file)
    else:
        uploaded_file.seek(0)
        try:
            source = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            source = pd.read_csv(uploaded_file, sep=r"\s+", engine="python")

    source.columns = [str(c).strip() for c in source.columns]
    normalized = {normalize_name(c): c for c in source.columns}
    name_column = next(
        (
            normalized[key]
            for key in ("SURFACENAME", "FORMATION", "TOP", "NAME", "HORIZON")
            if key in normalized
        ),
        None,
    )
    depth_column = next(
        (
            normalized[key]
            for key in ("MD", "DEPTH", "TOPMD", "MEASUREDDEPTH")
            if key in normalized
        ),
        None,
    )
    if name_column is None or depth_column is None:
        raise ValueError(
            "Formation tops need a name column and an MD/depth column."
        )

    tops = source[[name_column, depth_column]].copy()
    tops.columns = ["Formation", "Depth"]
    tops["Formation"] = tops["Formation"].astype(str).str.strip()
    tops["Depth"] = pd.to_numeric(tops["Depth"], errors="coerce")
    return tops.dropna().drop_duplicates().sort_values("Depth")


def add_tops(ax, tops: pd.DataFrame, minimum: float, maximum: float) -> None:
    for row in tops.itertuples(index=False):
        if minimum <= row.Depth <= maximum:
            ax.axhline(row.Depth, color="#8B0000", lw=0.7, ls="--", alpha=0.75)
            ax.text(
                0.02,
                row.Depth,
                str(row.Formation),
                transform=ax.get_yaxis_transform(),
                fontsize=7,
                color="#8B0000",
                va="bottom",
            )


def make_log_figure(
    interval: pd.DataFrame,
    tops: pd.DataFrame,
    depth_unit: str,
) -> plt.Figure:
    track_specs = [
        ("GR", "Gamma Ray", "#2E8B57", False),
        ("RES", "Resistivity", "#B22222", True),
        ("POROSITY", "Porosity", "#1F77B4", False),
        ("VSH", "Vsh", "#8C564B", False),
        ("SW", "Water Sat.", "#17BECF", False),
        ("CALI", "Caliper", "#9467BD", False),
    ]
    available = [spec for spec in track_specs if interval[spec[0]].notna().any()]
    if not available:
        raise ValueError("No plottable standard curves were found.")

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(2.5 * len(available), 10),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    top_depth = float(interval.index.min())
    base_depth = float(interval.index.max())

    for ax, (column, label, color, log_scale) in zip(axes, available):
        values = interval[column]
        ax.plot(values, interval.index, color=color, lw=0.8)
        if log_scale:
            positive = values[values > 0]
            if not positive.empty:
                ax.set_xscale("log")
        if column in {"POROSITY", "VSH", "SW"}:
            ax.set_xlim(1.0, 0.0) if column == "POROSITY" else ax.set_xlim(0.0, 1.0)
        ax.set_title(label, color=color, fontsize=10, fontweight="bold")
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        add_tops(ax, tops, top_depth, base_depth)

    axes[0].set_ylabel(f"Measured Depth ({depth_unit or 'LAS units'})")
    axes[0].set_ylim(base_depth, top_depth)
    fig.suptitle("Zona Rokan Well Log Interpretation", fontsize=14, fontweight="bold")
    return fig


def _evaluation_depth(interval: pd.DataFrame) -> tuple[pd.Series, str]:
    if "TVD" in interval and interval["TVD"].notna().mean() >= 0.80:
        return interval["TVD"].astype(float), "TVD"
    return pd.Series(interval.index.astype(float), index=interval.index), "MD"


def make_formation_evaluation_figure(
    interval: pd.DataFrame,
    cutoffs: dict[str, float | str | int],
    clean_gr: float,
    shale_gr: float,
) -> plt.Figure:
    """Create a compact multi-track formation-evaluation panel."""
    depth, depth_name = _evaluation_depth(interval)
    valid = depth.notna()
    data = interval.loc[valid].copy()
    depth = depth.loc[valid]
    if data.empty:
        raise ValueError("No samples are available for formation-evaluation plotting.")
    fig, axes = plt.subplots(
        1,
        8,
        figsize=(19, 11),
        sharey=True,
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.35, 0.9, 1.15, 1.0, 1.0, 0.55]},
    )
    top, base = float(depth.min()), float(depth.max())

    ax = axes[0]
    if data["GR"].notna().any():
        ax.plot(data["GR"], depth, color="#2e8b57", lw=0.7)
        ax.axvline(clean_gr, color="#d7191c", ls="--", lw=0.8)
        ax.axvline(shale_gr, color="#2166ac", ls="--", lw=0.8)
        if "COAL_FLAG" in data:
            ax.fill_betweenx(
                depth, 0, data["COAL_FLAG"].astype(float) * max(shale_gr, 1),
                color="#d7191c", alpha=0.18
            )
    ax.set_title("GR / References", fontsize=9, fontweight="bold")
    ax.set_xlabel("API")

    ax = axes[1]
    positive_res = data["RES"].where(data["RES"] > 0)
    ax.plot(positive_res, depth, color="#1b7837", lw=0.7)
    if positive_res.notna().any():
        ax.set_xscale("log")
    ax.set_title("Deep Resistivity", fontsize=9, fontweight="bold")
    ax.set_xlabel("ohm.m")

    ax = axes[2]
    ax.plot(data["RHOB"], depth, color="#d73027", lw=0.7, label="RHOB")
    neutron = data["NPHI_FRAC"]
    neutron_density_scale = 2.95 - neutron * 2.0
    ax.plot(neutron_density_scale, depth, color="#2166ac", lw=0.7, label="NPHI")
    ax.set_xlim(3.0, 1.8)
    ax.set_title("Density / Neutron", fontsize=9, fontweight="bold")
    ax.set_xlabel("g/cc + NPHI")
    ax.legend(fontsize=6, loc="lower center")

    ax = axes[3]
    ax.plot(data["VSH"], depth, color="#7f6000", lw=0.7)
    ax.fill_betweenx(depth, 0, data["VSH"], color="#8c8c62", alpha=0.55)
    ax.axvline(float(cutoffs["VSH"]), color="#d7191c", lw=1)
    ax.set_xlim(0, 1)
    ax.set_title("Vshale", fontsize=9, fontweight="bold")

    ax = axes[4]
    ax.plot(data["PHIT"], depth, color="#00a6a6", lw=0.8, label="PHIT")
    ax.plot(data["PHIE"], depth, color="#0057b8", lw=0.8, label="PHIE")
    ax.fill_betweenx(
        depth,
        data["PHIE"],
        data["PHIT"],
        color="#70d6d6",
        alpha=0.45,
        label="Bound water",
    )
    ax.axvline(float(cutoffs["PHIE"]), color="#d7191c", lw=1)
    ax.set_xlim(0.45, 0)
    ax.set_title("Porosity", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="lower center")

    ax = axes[5]
    positive_perm = data["PERM_MD"].where(data["PERM_MD"] > 0)
    ax.plot(positive_perm, depth, color="#1f78b4", lw=0.75)
    if positive_perm.notna().any():
        ax.set_xscale("log")
    ax.axvline(float(cutoffs["PERM_MD"]), color="#d7191c", lw=1)
    ax.set_title("Permeability", fontsize=9, fontweight="bold")
    ax.set_xlabel("mD")

    ax = axes[6]
    ax.plot(data["SW"], depth, color="#0057b8", lw=0.8)
    ax.fill_betweenx(depth, data["SW"], 1.0, color="#00c8ff", alpha=0.55)
    ax.axvline(float(cutoffs["SW"]), color="#d7191c", lw=1)
    ax.set_xlim(1, 0)
    ax.set_title("Water Saturation", fontsize=9, fontweight="bold")

    ax = axes[7]
    net = data.get("RESERVOIR_FLAG", pd.Series(False, index=data.index)).astype(float)
    ax.fill_betweenx(depth, 0, net, color="#00a651", alpha=0.9)
    if "TOTAL_GAS" in data and data["TOTAL_GAS"].notna().any():
        gas = data["TOTAL_GAS"].clip(lower=0)
        gas = gas / max(float(gas.quantile(0.98)), 1e-9)
        ax.plot(gas.clip(0, 1), depth, color="#d7191c", lw=0.65)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Net / Gas", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.set_ylim(base, top)
        ax.grid(True, which="both", alpha=0.18)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel(f"{depth_name} ({interval.attrs.get('depth_unit', 'LAS units')})")
    fig.suptitle(
        "Autonomous Formation Evaluation - Zona Rokan",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def make_validation_dashboard(interval: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    clean = interval.replace([np.inf, -np.inf], np.nan)

    data = clean[["GR", "VSH", "NPHI_FRAC", "PHID"]].dropna()
    if not data.empty:
        separation = data["NPHI_FRAC"] - data["PHID"]
        points = axes[0, 0].scatter(
            data["GR"], data["VSH"], c=separation, cmap="coolwarm", s=8, alpha=0.65
        )
        fig.colorbar(points, ax=axes[0, 0], label="N-D separation")
    axes[0, 0].set_title("Vsh Validation")
    axes[0, 0].set_xlabel("Gamma Ray")
    axes[0, 0].set_ylabel("Vsh")

    data = clean[["PHIE", "PHIT", "VSH"]].dropna()
    if not data.empty:
        points = axes[0, 1].scatter(
            data["PHIE"], data["PHIT"], c=data["VSH"], cmap="YlOrBr", s=8, alpha=0.65
        )
        limit = max(float(data[["PHIE", "PHIT"]].max().max()), 0.1)
        axes[0, 1].plot([0, limit], [0, limit], color="#333333", ls="--")
        fig.colorbar(points, ax=axes[0, 1], label="Vsh")
    axes[0, 1].set_title("Porosity / Bound-Water Validation")
    axes[0, 1].set_xlabel("PHIE")
    axes[0, 1].set_ylabel("PHIT")

    data = clean[["PHIE", "PERM_MD", "VSH"]].dropna()
    data = data.loc[data["PERM_MD"] > 0]
    if not data.empty:
        points = axes[1, 0].scatter(
            data["PHIE"], data["PERM_MD"], c=data["VSH"], cmap="YlOrBr", s=8, alpha=0.65
        )
        axes[1, 0].set_yscale("log")
        fig.colorbar(points, ax=axes[1, 0], label="Vsh")
    axes[1, 0].set_title("Coates-Timur Permeability")
    axes[1, 0].set_xlabel("PHIE")
    axes[1, 0].set_ylabel("Permeability (mD)")

    saturation_columns = ["SW", "RES", "VSH"]
    if "TOTAL_GAS" in clean:
        saturation_columns.append("TOTAL_GAS")
    data = clean[saturation_columns].dropna(subset=["SW", "RES", "VSH"])
    data = data.loc[data["RES"] > 0]
    if not data.empty:
        color = (
            data["TOTAL_GAS"]
            if "TOTAL_GAS" in data and data["TOTAL_GAS"].notna().any()
            else data["VSH"]
        )
        points = axes[1, 1].scatter(
            data["SW"], data["RES"], c=color, cmap="turbo", s=8, alpha=0.65
        )
        axes[1, 1].set_yscale("log")
        fig.colorbar(
            points,
            ax=axes[1, 1],
            label="Total gas" if "TOTAL_GAS" in data else "Vsh",
        )
    axes[1, 1].set_title("Modified Simandoux Validation")
    axes[1, 1].set_xlabel("Sw")
    axes[1, 1].set_ylabel("Deep Resistivity")

    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.22)
    fig.suptitle("Petrophysical Validation Dashboard", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def make_cutoff_analysis_figure(
    interval: pd.DataFrame,
    cutoffs: dict[str, float | str | int],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    data = interval.replace([np.inf, -np.inf], np.nan)
    pairs = [
        ("VSH", "PERM_MD", "Vsh", float(cutoffs["VSH"]), "vertical"),
        ("PHIE", "PERM_MD", "PHIE", float(cutoffs["PHIE"]), "vertical"),
        (
            "SW",
            "TOTAL_GAS" if data.get("TOTAL_GAS", pd.Series(dtype=float)).notna().any() else "PERM_MD",
            "Sw",
            float(cutoffs["SW"]),
            "vertical",
        ),
    ]
    for ax, (x_name, y_name, label, cutoff, _) in zip(axes, pairs):
        subset_columns = list(dict.fromkeys((x_name, y_name, "VSH")))
        subset = data[subset_columns].dropna()
        if not subset.empty:
            color = subset["VSH"]
            points = ax.scatter(
                subset[x_name],
                subset[y_name],
                c=color,
                cmap="turbo",
                s=7,
                alpha=0.55,
            )
            fig.colorbar(points, ax=ax, label="Vsh")
            if y_name == "PERM_MD" and (subset[y_name] > 0).any():
                ax.set_yscale("log")
        ax.axvline(cutoff, color="#d7191c", lw=1.5)
        if y_name == "PERM_MD":
            ax.axhline(float(cutoffs["PERM_MD"]), color="#d7191c", lw=1.0, ls="--")
        ax.set_xlabel(label)
        ax.set_ylabel("Total Gas" if y_name == "TOTAL_GAS" else "Permeability (mD)")
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("Cutoff Analysis and Validation", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def make_crossplot(interval: pd.DataFrame) -> plt.Figure | None:
    data = interval[["NPHI_FRAC", "RHOB", "VSH"]].dropna()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    points = ax.scatter(
        data["NPHI_FRAC"] * 100,
        data["RHOB"],
        c=data["VSH"],
        cmap="YlOrBr",
        s=10,
        alpha=0.75,
        vmin=0,
        vmax=1,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Neutron Porosity (pu)")
    ax.set_ylabel("Bulk Density (g/cc)")
    ax.set_title("Neutron-Density Crossplot")
    ax.grid(alpha=0.25)
    fig.colorbar(points, ax=ax, label="Vsh")
    fig.tight_layout()
    return fig


def make_pickett_plot(
    interval: pd.DataFrame,
    a: float,
    m: float,
) -> plt.Figure | None:
    required = ["PHIE", "RES", "VSH", "RW"]
    if any(column not in interval for column in required):
        return None
    data = interval[required].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[(data["PHIE"] > 0) & (data["RES"] > 0) & (data["RW"] > 0)]
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    points = ax.scatter(
        data["PHIE"],
        data["RES"],
        c=data["VSH"],
        cmap="YlOrBr",
        s=10,
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    phi_line = np.logspace(
        np.log10(max(0.01, data["PHIE"].quantile(0.02))),
        np.log10(min(0.6, data["PHIE"].quantile(0.98))),
        80,
    )
    rw_reference = float(data["RW"].median())
    rt_water = a * rw_reference / np.power(phi_line, m)
    ax.plot(phi_line, rt_water, color="#0b3d91", lw=1.5, label=f"Sw=1, Rw={rw_reference:.3g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective Porosity")
    ax.set_ylabel("Deep Resistivity (ohm.m)")
    ax.set_title("Pickett Plot Screening")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.colorbar(points, ax=ax, label="Vsh")
    fig.tight_layout()
    return fig


def make_nzbp_schematic(
    top_depth: float,
    base_depth: float,
    candidates: pd.DataFrame,
    active_perforations: list[tuple[float, float]],
    casing: pd.DataFrame | None = None,
    sands: pd.DataFrame | None = None,
) -> plt.Figure:
    """Draw a compact wellbore-style NZBP screening diagram."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(-2.2, 2.8)
    ax.set_ylim(base_depth, top_depth)
    ax.set_ylabel("Measured Depth")
    ax.set_xticks([])
    ax.set_title("NZBP Wellbore Screening Diagram", fontweight="bold")
    ax.axvspan(-0.24, 0.24, color="#d8e4ef", ec="#28384a", lw=2, label="Wellbore")

    if casing is not None and not casing.empty:
        for _, row in casing.iterrows():
            casing_top = pd.to_numeric(row.get("Top (ft)", row.get("Top")), errors="coerce")
            casing_base = pd.to_numeric(
                row.get("Base (ft)", row.get("Base", row.get("Depth"))),
                errors="coerce",
            )
            if pd.isna(casing_base):
                continue
            casing_top = top_depth if pd.isna(casing_top) else float(casing_top)
            ax.plot(
                [-0.48, -0.48],
                [max(top_depth, casing_top), min(base_depth, float(casing_base))],
                color="#51606f",
                lw=2,
            )
            ax.plot(
                [0.48, 0.48],
                [max(top_depth, casing_top), min(base_depth, float(casing_base))],
                color="#51606f",
                lw=2,
            )

    if sands is not None and not sands.empty:
        for _, row in sands.iterrows():
            sand_top = pd.to_numeric(row.get("Top (ft)", row.get("Top")), errors="coerce")
            sand_base = pd.to_numeric(
                row.get("Base (ft)", row.get("Base")), errors="coerce"
            )
            if pd.isna(sand_top):
                continue
            sand_base = sand_top if pd.isna(sand_base) else sand_base
            label = str(row.get("Sand", row.get("Name", row.get("Label", "Sand"))))
            ax.axhspan(
                max(top_depth, float(sand_top)),
                min(base_depth, float(sand_base)),
                xmin=0.63,
                xmax=0.96,
                color="#78b7d0",
                alpha=0.22,
            )
            ax.text(1.05, float(sand_top), label, fontsize=7, va="bottom")

    for perf_top, perf_base in active_perforations:
        if perf_base < top_depth or perf_top > base_depth:
            continue
        ax.axhspan(
            max(top_depth, perf_top),
            min(base_depth, perf_base),
            xmin=0.38,
            xmax=0.48,
            color="#33b66f",
            alpha=0.9,
        )
        ax.text(-0.72, (perf_top + perf_base) / 2, "OPEN", fontsize=7, ha="right")

    if not candidates.empty:
        color_map = {
            "NZBP SCREENING CANDIDATE": "#f0b429",
            "PARTIALLY BEHIND PIPE": "#f48c46",
            "CURRENT / HISTORICALLY OPEN": "#43d17b",
            "UNCONFIRMED - NO PERFORATION HISTORY": "#a4adba",
        }
        for _, row in candidates.iterrows():
            interval_top = float(row["Top MD"])
            interval_base = float(row["Base MD"])
            status = str(row["NZBP Status"])
            color = color_map.get(status, "#a4adba")
            ax.axhspan(
                interval_top,
                interval_base,
                xmin=0.48,
                xmax=0.62,
                color=color,
                alpha=0.85,
            )
            ax.text(
                0.38,
                (interval_top + interval_base) / 2,
                f"{status}\nScore {row.get('Screening Score', 'N/A')}",
                fontsize=7,
                va="center",
            )
    ax.text(-1.85, top_depth, "Active perforations", color="#24844f", fontsize=9)
    ax.text(0.35, top_depth, "Reservoir/NZBP intervals", color="#996b00", fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return fig


def make_trajectory_figure(survey: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(survey["Horizontal Departure"], survey["TVD"], color="#0b77a5", lw=2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Horizontal Departure")
    axes[0].set_ylabel("TVD")
    axes[0].set_title("Vertical Section")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        survey["East Departure"],
        survey["North Departure"],
        color="#b46c00",
        lw=2,
    )
    axes[1].scatter(
        survey["East Departure"].iloc[-1],
        survey["North Departure"].iloc[-1],
        color="#f0b429",
        s=45,
    )
    axes[1].set_xlabel("East Departure")
    axes[1].set_ylabel("North Departure")
    axes[1].set_title("Plan View")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Minimum-Curvature Well Trajectory", fontweight="bold")
    fig.tight_layout()
    return fig


def build_pdf(
    title: str,
    summary: pd.DataFrame,
    log_figure: plt.Figure,
    crossplot: plt.Figure | None,
) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        cover = plt.figure(figsize=(8.27, 11.69))
        cover_ax = cover.add_subplot(111)
        cover_ax.axis("off")
        cover.text(0.08, 0.92, title, fontsize=20, fontweight="bold", color="#0B3D91")
        cover.text(0.08, 0.87, "Zona Rokan petrophysical screening report", fontsize=12)
        y = 0.80
        for row in summary.itertuples(index=False):
            cover.text(0.10, y, f"{row.Metric}: {row.Value}", fontsize=11)
            y -= 0.045
        cover.text(
            0.08,
            0.08,
            "Screening output only. Validate cutoffs, environmental corrections, "
            "mineral model, Rw, and formation parameters before reservoir decisions.",
            fontsize=9,
            wrap=True,
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        pdf.savefig(log_figure, bbox_inches="tight")
        if crossplot is not None:
            pdf.savefig(crossplot, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def format_number(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def render_app() -> None:
    st.set_page_config(
        page_title="Zona Rokan Well Log Analyst",
        page_icon=None,
        layout="wide",
    )
    st.title("Zona Rokan Well Log Analyst")
    st.caption(
        "Local LAS browser and petrophysical screening for the canonical "
        "Zona Rokan well inventory."
    )

    with st.sidebar:
        st.header("Data source")
        root_text = st.text_input("Zona Rokan data root", str(DEFAULT_DATA_ROOT))
        source_mode = st.radio("LAS source", ("Zona Rokan inventory", "Upload LAS"))

    inventory = scan_inventory(root_text)
    scan_issues = inventory_scan_issues(root_text)
    selected_record: WellRecord | None = None
    selected_path: Path | None = None
    uploaded = None

    if source_mode == "Zona Rokan inventory":
        if scan_issues:
            st.warning(
                f"Skipped {len(scan_issues)} unreadable data path(s). "
                "Affected files are excluded from the inventory."
            )
            with st.expander("Data access warnings", expanded=False):
                st.code("\n".join(scan_issues), language=None)
        if not inventory:
            st.error("No field/well folders were found under the selected data root.")
            st.stop()
        fields = sorted({record.field for record in inventory})
        with st.sidebar:
            selected_field = st.selectbox("Field", fields)
            field_records = [r for r in inventory if r.field == selected_field]
            selected_well = st.selectbox(
                "Well",
                [r.well for r in field_records],
                format_func=lambda well: (
                    f"{well} "
                    f"({len(next(r for r in field_records if r.well == well).las_files)} LAS)"
                ),
            )
            selected_record = next(r for r in field_records if r.well == selected_well)
            if not selected_record.las_files:
                st.warning("This canonical well has no LAS file. Select another well or upload LAS.")
                st.stop()
            selected_path = st.selectbox(
                "LAS file",
                selected_record.las_files,
                format_func=lambda p: f"{p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)",
            )
    else:
        with st.sidebar:
            uploaded = st.file_uploader("Upload LAS file", type=["las"])
        if uploaded is None:
            st.info("Upload a LAS file to start the analysis.")
            st.stop()

    try:
        with st.spinner("Reading LAS data..."):
            las = read_las(selected_path if selected_path is not None else uploaded.getvalue())
            raw_df = prepare_dataframe(las)
    except Exception as exc:
        st.error(f"LAS loading failed: {exc}")
        st.stop()

    if raw_df.empty:
        st.error("The selected LAS file contains no readable curve data.")
        st.stop()

    auto_map = auto_curve_map(raw_df)
    well_name = (
        selected_record.well
        if selected_record
        else safe_header_value(las, "UWI", safe_header_value(las, "WELL", uploaded.name))
    )
    field_name = (
        selected_record.field
        if selected_record
        else safe_header_value(las, "FLD", "Zona Rokan")
    )
    depth_unit = curve_unit(las, str(las.index[0])) or curve_unit(las, "DEPT")

    st.subheader(f"{field_name} / {well_name}")
    metrics = st.columns(5)
    metrics[0].metric("Depth samples", f"{len(raw_df):,}")
    metrics[1].metric("Curves", len(raw_df.columns))
    metrics[2].metric("Top depth", format_number(float(raw_df.index.min())))
    metrics[3].metric("Base depth", format_number(float(raw_df.index.max())))
    metrics[4].metric("LAS files for well", len(selected_record.las_files) if selected_record else 1)

    st.sidebar.header("Curve mapping")
    options = ["-- Not available --", *raw_df.columns.tolist()]
    mapping: dict[str, str | None] = {}
    for role in CURVE_ALIASES:
        suggested = auto_map[role]
        index = options.index(suggested) if suggested in options else 0
        selected = st.sidebar.selectbox(role, options, index=index, key=f"curve_{role}")
        mapping[role] = None if selected == options[0] else selected

    def series(role: str) -> pd.Series:
        mnemonic = mapping[role]
        if mnemonic is None:
            return pd.Series(np.nan, index=raw_df.index, dtype=float)
        return raw_df[mnemonic].astype(float)

    gr = series("Gamma Ray")
    resistivity = series("Deep Resistivity")
    density = series("Density")
    neutron = normalize_neutron(
        series("Neutron"),
        curve_unit(las, mapping["Neutron"]),
    )
    neutron_fraction = neutron / 100.0
    caliper = series("Caliper")
    dtc = series("Compressional Sonic")

    with st.sidebar:
        st.header("Interpretation settings")
        finite_gr = gr.dropna()
        default_clean = float(finite_gr.quantile(0.05)) if not finite_gr.empty else 25.0
        default_shale = float(finite_gr.quantile(0.95)) if not finite_gr.empty else 125.0
        clean_gr = st.number_input("Clean GR", value=default_clean)
        shale_gr = st.number_input("Shale GR", value=default_shale)
        vsh_method = st.selectbox(
            "Vsh method",
            (
                "Larionov Tertiary",
                "Linear",
                "Larionov Older Rocks",
                "Clavier",
                "Steiber",
            ),
        )
        matrix_density = st.number_input("Matrix density (g/cc)", value=2.65, step=0.01)
        fluid_density = st.number_input("Fluid density (g/cc)", value=1.00, step=0.01)
        matrix_dt = st.number_input("Matrix DT (us/ft)", value=55.5, step=0.5)
        fluid_dt = st.number_input("Fluid DT (us/ft)", value=189.0, step=1.0)
        vsh_cutoff = st.slider("Maximum Vsh", 0.0, 1.0, 0.40, 0.05)
        phi_cutoff = st.slider("Minimum effective porosity", 0.0, 0.40, 0.10, 0.01)
        use_sw_cutoff = st.checkbox("Apply water saturation cutoff", value=False)
        sw_cutoff = st.slider("Maximum Sw", 0.0, 1.0, 0.60, 0.05)

    calculated_vsh = calculate_vsh(gr, clean_gr, shale_gr, vsh_method)
    interpreted_vsh = normalize_fraction(series("Interpreted Vsh"))
    vsh = interpreted_vsh.combine_first(calculated_vsh)

    phi_density = density_porosity(density, matrix_density, fluid_density)
    phi_sonic = sonic_porosity(dtc, matrix_dt, fluid_dt)
    interpreted_phi = normalize_fraction(series("Interpreted Porosity"))
    phi_total = interpreted_phi.combine_first(
        pd.concat([phi_density, neutron_fraction], axis=1).mean(axis=1)
    ).combine_first(phi_sonic)
    phi_effective = (phi_total * (1.0 - vsh)).clip(0.0, 0.6)

    with st.sidebar.expander("Archie saturation", expanded=False):
        rw = st.number_input("Rw (ohm.m)", min_value=0.001, value=0.20, step=0.01)
        archie_a = st.number_input("Archie a", min_value=0.1, value=1.0, step=0.1)
        archie_m = st.number_input("Archie m", min_value=0.1, value=2.0, step=0.1)
        archie_n = st.number_input("Archie n", min_value=0.1, value=2.0, step=0.1)

    interpreted_sw = normalize_fraction(series("Water Saturation"))
    calculated_sw = archie_water_saturation(
        phi_effective, resistivity, rw, archie_a, archie_m, archie_n
    )
    sw = interpreted_sw.combine_first(calculated_sw)

    work = pd.DataFrame(
        {
            "GR": gr,
            "RES": resistivity,
            "RHOB": density,
            "NPHI_PU": neutron,
            "NPHI_FRAC": neutron_fraction,
            "CALI": caliper,
            "DTC": dtc,
            "VSH": vsh,
            "POROSITY": phi_effective,
            "SW": sw,
        },
        index=raw_df.index,
    )

    top_default = float(work.index.min())
    base_default = float(work.index.max())
    st.sidebar.header("Depth interval")
    top_depth = st.sidebar.number_input("Top MD", value=top_default)
    base_depth = st.sidebar.number_input("Base MD", value=base_default)
    if top_depth > base_depth:
        top_depth, base_depth = base_depth, top_depth
    interval = work.loc[(work.index >= top_depth) & (work.index <= base_depth)].copy()
    if interval.empty:
        st.error("The selected depth interval contains no samples.")
        st.stop()

    tops = pd.DataFrame(columns=["Formation", "Depth"])
    tops_upload = st.sidebar.file_uploader(
        "Optional formation tops",
        type=["xlsx", "xls", "csv", "txt", "dat"],
    )
    if tops_upload is not None:
        try:
            tops = load_tops(tops_upload)
        except Exception as exc:
            st.sidebar.error(f"Tops loading failed: {exc}")

    reservoir_flag = (interval["VSH"] <= vsh_cutoff) & (
        interval["POROSITY"] >= phi_cutoff
    )
    if use_sw_cutoff:
        reservoir_flag &= interval["SW"] <= sw_cutoff
    interval["RESERVOIR_FLAG"] = reservoir_flag.fillna(False)

    sample_thickness = estimate_sample_thickness(interval.index)
    gross = max(float(interval.index.max() - interval.index.min()), 0.0)
    net = float(interval["RESERVOIR_FLAG"].sum()) * sample_thickness
    ntg = net / gross if gross > 0 else np.nan
    reservoir_rows = interval.loc[interval["RESERVOIR_FLAG"]]

    summary = pd.DataFrame(
        [
            ("Field", field_name),
            ("Well", well_name),
            ("Interval top", format_number(top_depth)),
            ("Interval base", format_number(base_depth)),
            ("Gross interval", format_number(gross)),
            ("Estimated net reservoir", format_number(net)),
            ("Net-to-gross", format_number(ntg, 3)),
            ("Average Vsh", format_number(reservoir_rows["VSH"].mean(), 3)),
            (
                "Average effective porosity",
                format_number(reservoir_rows["POROSITY"].mean(), 3),
            ),
            ("Average Sw", format_number(reservoir_rows["SW"].mean(), 3)),
        ],
        columns=["Metric", "Value"],
    )

    tabs = st.tabs(("Log display", "Petrophysics", "Data and export", "Inventory QA"))
    with tabs[0]:
        try:
            log_figure = make_log_figure(interval, tops, depth_unit)
            st.pyplot(log_figure, width="stretch")
        except ValueError as exc:
            st.warning(str(exc))
            log_figure = None

    with tabs[1]:
        result_metrics = st.columns(4)
        result_metrics[0].metric("Gross interval", format_number(gross))
        result_metrics[1].metric("Net reservoir", format_number(net))
        result_metrics[2].metric("Net-to-gross", format_number(ntg, 3))
        result_metrics[3].metric(
            "Average PhiE",
            format_number(reservoir_rows["POROSITY"].mean(), 3),
        )
        left, right = st.columns(2)
        with left:
            st.dataframe(summary, hide_index=True, width="stretch")
        with right:
            crossplot = make_crossplot(interval)
            if crossplot is not None:
                st.pyplot(crossplot, width="stretch")
            else:
                st.info("Density and neutron curves are required for the crossplot.")

        if series("Permeability").notna().any():
            st.caption("An interpreted permeability curve is present in the LAS file.")

    with tabs[2]:
        export = interval.copy()
        export.index.name = f"DEPTH_{depth_unit or 'LAS'}"
        st.dataframe(export.head(500), width="stretch")
        csv_bytes = export.to_csv().encode("utf-8")
        st.download_button(
            "Download interpreted interval CSV",
            data=csv_bytes,
            file_name=f"{normalize_name(well_name)}_interpreted_interval.csv",
            mime="text/csv",
        )
        if log_figure is not None:
            report_bytes = build_pdf(
                f"{field_name} / {well_name}",
                summary,
                log_figure,
                make_crossplot(interval),
            )
            st.download_button(
                "Download PDF screening report",
                data=report_bytes,
                file_name=f"{normalize_name(well_name)}_well_log_report.pdf",
                mime="application/pdf",
            )
        st.caption(
            "Calculated properties are screening estimates. Existing interpreted "
            "LAS curves are used first when available."
        )

    with tabs[3]:
        total_las = sum(len(r.las_files) for r in inventory)
        wells_with_las = sum(bool(r.las_files) for r in inventory)
        qa_metrics = st.columns(4)
        qa_metrics[0].metric("Canonical well folders", len(inventory))
        qa_metrics[1].metric("Wells with LAS", wells_with_las)
        qa_metrics[2].metric("LAS files", total_las)
        qa_metrics[3].metric("Wells without LAS", len(inventory) - wells_with_las)
        qa_table = pd.DataFrame(
            {
                "Field": [r.field for r in inventory],
                "Well": [r.well for r in inventory],
                "LAS files": [len(r.las_files) for r in inventory],
                "Production files": [len(r.production_files) for r in inventory],
                "Normalized well ID": [normalize_name(r.well) for r in inventory],
            }
        )
        st.dataframe(qa_table, hide_index=True, width="stretch")

    with st.expander("Detected LAS curves"):
        curve_table = pd.DataFrame(
            [
                {
                    "Mnemonic": curve.mnemonic,
                    "Unit": curve.unit,
                    "Description": curve.descr,
                    "Non-null samples": int(raw_df.get(curve.mnemonic, pd.Series()).notna().sum()),
                }
                for curve in las.curves
                if curve.mnemonic in raw_df.columns
            ]
        )
        st.dataframe(curve_table, hide_index=True, width="stretch")


if __name__ == "__main__":
    render_app()
