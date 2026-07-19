from __future__ import annotations

import importlib
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import dashboard_services as services
import well_log_core as logs


EXPECTED_CLOUD_DEPLOYMENT_VERSION = "2026-07-19.1"
if (
    getattr(services, "CLOUD_DEPLOYMENT_VERSION", "")
    != EXPECTED_CLOUD_DEPLOYMENT_VERSION
):
    services = importlib.reload(services)


st.set_page_config(
    page_title="Zona Rokan Integrated Well Intelligence",
    page_icon=":material/oil_barrel:",
    layout="wide",
    initial_sidebar_state="expanded",
)


ALL_PAGES = (
    "Executive overview",
    "Well screening matrix",
    "Well 360",
    "Data availability",
    "Formation evaluation",
    "NZBP screening",
    "Scanned log digitizer",
    "Well schematic",
    "Perforation and NZBP",
    "Phase 2/3 readiness",
)
PAGES = tuple(
    page
    for page in ALL_PAGES
    if not (services.ONLINE_MODE and page == "Scanned log digitizer")
)


@st.cache_data(show_spinner=False)
def load_master(
    root_text: str,
    source_tokens: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    del source_tokens  # Included in the cache key so controlled workbook edits refresh.
    inventory = logs.scan_inventory(root_text)
    if not inventory and services.ONLINE_MODE:
        inventory = load_portable_inventory()
    master = services.build_master_table(inventory)
    if "Detailed History Available" not in master.columns:
        master["Detailed History Available"] = master.get(
            "Schematic Available", pd.Series(False, index=master.index)
        ).fillna(False)
    return master


@st.cache_data(show_spinner=False)
def load_raw_evidence(well: str) -> pd.DataFrame:
    return services.selected_well_evidence(well)


@st.cache_data(show_spinner=False)
def load_portable_inventory() -> list[logs.WellRecord]:
    rows = services.load_portable_inventory_rows()
    return [
        logs.WellRecord(
            field=str(row["Field"]),
            well=str(row["Well"]),
            path=services.DATA_ROOT / str(row["Field"]) / str(row["Well"]),
            las_files=(),
            scanned_files=(),
            production_files=(),
        )
        for _, row in rows.iterrows()
    ]


def load_perforation_history(well: str) -> pd.DataFrame:
    return services.selected_perforation_history(well)


@st.cache_data(show_spinner=False)
def load_log_path(path_text: str) -> pd.DataFrame:
    return logs.load_log_path(path_text).data


def resolve_tvd_from_zona_rokan(
    selected_record: logs.WellRecord,
    source_depth_mnemonic: str,
    raw: pd.DataFrame,
    selected_path: Path | None,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    depth = raw.index.to_numpy(dtype=float)
    source_curve = str(source_depth_mnemonic).upper()
    source_name = selected_path.name.upper() if selected_path is not None else ""
    direct_tvd = "TVD" in source_curve or bool(
        re.search(r"(?i)(?:MAIN.?TVD|[_ -]TVD(?:[_ .-]|$))", source_name)
    )
    if direct_tvd:
        return (
            pd.Series(depth, index=raw.index),
            pd.DataFrame(),
            {
                "basis": "DIRECT TVD INDEX FROM ZONA ROKAN LAS",
                "source": str(selected_path or "Uploaded LAS"),
                "coverage": 1.0,
                "quality": "SOURCE TVD",
            },
        )

    survey, survey_path = services.discover_trajectory_survey(selected_record)
    trajectory = pd.DataFrame()
    if not survey.empty:
        try:
            calculated = logs.minimum_curvature_survey(survey)
            trajectory = calculated.copy()
            for column in (
                "TVD",
                "North Departure",
                "East Departure",
                "Horizontal Departure",
                "DLS (deg/100ft)",
            ):
                if column in survey and survey[column].notna().any():
                    trajectory[column] = survey.set_index("MD")[column].reindex(
                        trajectory["MD"]
                    ).to_numpy()
            if {
                "North Departure",
                "East Departure",
            }.issubset(trajectory.columns):
                trajectory["Horizontal Departure"] = np.hypot(
                    trajectory["North Departure"], trajectory["East Departure"]
                )
            survey_top = float(trajectory["MD"].min())
            survey_base = float(trajectory["MD"].max())
            in_range = (depth >= survey_top) & (depth <= survey_base)
            tvd_values = np.full(len(depth), np.nan)
            tvd_values[in_range] = np.interp(
                depth[in_range],
                trajectory["MD"],
                trajectory["TVD"],
            )
            coverage = float(in_range.mean())
            return (
                pd.Series(tvd_values, index=raw.index),
                trajectory,
                {
                    "basis": "REPORTED / MINIMUM-CURVATURE TVD FROM ZONA ROKAN SURVEY",
                    "source": str(survey_path),
                    "coverage": coverage,
                    "quality": (
                        "FULL SURVEY COVERAGE"
                        if coverage >= 0.95
                        else "PARTIAL SURVEY COVERAGE"
                    ),
                },
            )
        except ValueError:
            trajectory = pd.DataFrame()

    directional_name = services.normalize_well(selected_record.well).endswith(
        ("D1", "H1")
    )
    if not directional_name:
        return (
            pd.Series(depth, index=raw.index),
            trajectory,
            {
                "basis": "VERTICAL-WELL ASSUMPTION: TVD = MD",
                "source": "No directional indicator found in Zona Rokan files",
                "coverage": 1.0,
                "quality": "ASSUMPTION",
            },
        )
    return (
        pd.Series(np.nan, index=raw.index),
        trajectory,
        {
            "basis": "TVD UNAVAILABLE FOR DIRECTIONAL/HORIZONTAL WELL",
            "source": "No parseable TVD index or directional survey found",
            "coverage": 0.0,
            "quality": "DATA GAP",
        },
    )


def hero(title: str, subtitle: str, eyebrow: str = "Zona Rokan Well Intelligence") -> None:
    with st.container(border=True):
        st.caption(f":material/oil_barrel: {eyebrow}")
        st.title(title)
        st.write(subtitle)


def source_caption(path: Path, sheet: str | None = None) -> None:
    if not path.exists():
        st.caption(f":material/error: Source unavailable: {path.name}")
        return
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y, %H:%M")
    sheet_text = f" · Sheet: {sheet}" if sheet else ""
    st.caption(
        f":material/database: Controlled source: {path.name}{sheet_text} · "
        f"Updated {modified} · {path.stat().st_size / (1024 * 1024):.1f} MB"
    )


def make_workbook_schematic_figure(
    well: str,
    orientation: str,
    casing: pd.DataFrame,
    intervals: pd.DataFrame,
) -> plt.Figure:
    casing_depths = pd.to_numeric(
        casing.get("Casing Depth (MD)", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    interval_bases = pd.to_numeric(
        intervals.get("Base (ft)", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    all_depths = pd.concat([casing_depths, interval_bases], ignore_index=True)
    total_depth = float(all_depths.max()) if not all_depths.empty else 1000.0
    total_depth = max(total_depth, 100.0)

    fig, ax = plt.subplots(figsize=(8, 11), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_ylim(total_depth * 1.04, -total_depth * 0.04)
    ax.set_xlim(-2.4, 3.6)
    ax.set_xticks([])
    ax.set_ylabel("Measured depth (ft)", color="#CBD5E1")
    ax.tick_params(axis="y", colors="#CBD5E1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.45)
    ax.axhline(0, color="#E2E8F0", linewidth=2)
    ax.text(0, -total_depth * 0.018, "WELLHEAD", ha="center", color="#F1F5F9", weight="bold")

    valid_casing = casing.copy()
    if not valid_casing.empty:
        valid_casing["__depth"] = pd.to_numeric(
            valid_casing.get("Casing Depth (MD)"), errors="coerce"
        )
        valid_casing = valid_casing.dropna(subset=["__depth"]).sort_values("__depth")
    widths = np.linspace(1.15, 0.42, max(len(valid_casing), 1))
    for index, (_, row) in enumerate(valid_casing.iterrows()):
        depth = float(row["__depth"])
        width = float(widths[min(index, len(widths) - 1)])
        color = ["#94A3B8", "#60A5FA", "#38BDF8", "#A78BFA"][index % 4]
        ax.plot([-width, -width], [0, depth], color=color, linewidth=2.2)
        ax.plot([width, width], [0, depth], color=color, linewidth=2.2)
        ax.plot([-width, width], [depth, depth], color=color, linewidth=2.2)
        label = str(row.get("Casing") or row.get("Hole Details") or "Casing").strip()
        if label:
            ax.text(
                width + 0.08,
                depth,
                f"{label} · {depth:,.0f} ft",
                color="#CBD5E1",
                fontsize=7.5,
                va="center",
            )

    status_colors = {"OPEN": "#34D399", "CLOSED": "#94A3B8", "CLOSE": "#94A3B8", "NZBP": "#FBBF24"}
    interval_frame = intervals.copy()
    if not interval_frame.empty:
        interval_frame["__top"] = pd.to_numeric(interval_frame.get("Top (ft)"), errors="coerce")
        interval_frame["__base"] = pd.to_numeric(interval_frame.get("Base (ft)"), errors="coerce")
        interval_frame = interval_frame.dropna(subset=["__top", "__base"])
    for _, row in interval_frame.iterrows():
        top, base = sorted((float(row["__top"]), float(row["__base"])))
        status = str(row.get("Status", "")).strip().upper()
        color = status_colors.get(status, "#60A5FA")
        ax.plot([-0.34, -0.05], [top, top], color=color, linewidth=4)
        ax.plot([-0.34, -0.05], [base, base], color=color, linewidth=4)
        ax.fill_betweenx([top, base], 0.05, 0.34, color=color, alpha=0.75)
        formation = str(row.get("Formation", "")).strip() or "Interval"
        ax.text(
            1.45,
            (top + base) / 2,
            f"{formation} · {top:,.0f}-{base:,.0f} ft · {status or 'UNSPECIFIED'}",
            color=color,
            fontsize=7.2,
            va="center",
        )

    orientation_label = {"D": "Directional", "H": "Horizontal", "V": "Vertical"}.get(
        orientation, "Workbook-defined"
    )
    ax.set_title(
        f"{well} · {orientation_label} completion schematic (MD)",
        color="#F1F5F9",
        weight="bold",
        pad=14,
    )
    ax.text(
        -2.35,
        total_depth * 1.02,
        "Source geometry is not inferred; casing shoes and intervals are plotted at workbook MD.",
        color="#94A3B8",
        fontsize=7.5,
    )
    fig.tight_layout()
    return fig


def perforation_interval_chart(history: pd.DataFrame) -> alt.Chart:
    frame = history.copy()
    frame["Top"] = pd.to_numeric(frame["Perforation Top (ft)"], errors="coerce")
    frame["Base"] = pd.to_numeric(frame["Perforation Base (ft)"], errors="coerce")
    frame = frame.dropna(subset=["Top", "Base"])
    frame["Formation label"] = frame.get("Formation", "Unspecified").fillna("Unspecified")
    frame["Source status"] = frame.get("Status", frame.get("Action Status", "Unspecified")).fillna("Unspecified")
    return (
        alt.Chart(frame)
        .mark_bar(size=18, cornerRadius=3)
        .encode(
            x=alt.X("Formation label:N", title="Formation", sort=None),
            y=alt.Y(
                "Top:Q",
                title="Measured depth (ft)",
                scale=alt.Scale(reverse=True),
            ),
            y2="Base:Q",
            color=alt.Color(
                "Source status:N",
                title="Workbook status",
                scale=alt.Scale(
                    domain=["Open", "Closed", "NZBP"],
                    range=["#34D399", "#94A3B8", "#FBBF24"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Well:N"),
                alt.Tooltip("Formation label:N", title="Formation"),
                alt.Tooltip("Sand:N"),
                alt.Tooltip("Top:Q", format=",.1f"),
                alt.Tooltip("Base:Q", format=",.1f"),
                alt.Tooltip("Source status:N", title="Status"),
                alt.Tooltip("Noted:N"),
            ],
        )
        .properties(height=540)
        .interactive()
    )


SCREENING_TIER_ORDER = [
    "Tier 1",
    "Tier 2",
    "Tier 3",
    "Excluded",
    "Defer / Abandon",
]
SCREENING_TIER_COLORS = [
    "#34D399",
    "#60A5FA",
    "#FBBF24",
    "#F87171",
    "#94A3B8",
]


def screening_tier_chart(matrix: pd.DataFrame) -> alt.Chart:
    counts = (
        matrix["Tier"]
        .fillna("Unspecified")
        .value_counts()
        .rename_axis("Tier")
        .reset_index(name="Wells")
    )
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Wells:Q", title="Wells"),
            y=alt.Y(
                "Tier:N",
                title=None,
                sort=SCREENING_TIER_ORDER,
            ),
            color=alt.Color(
                "Tier:N",
                legend=None,
                scale=alt.Scale(
                    domain=SCREENING_TIER_ORDER,
                    range=SCREENING_TIER_COLORS,
                ),
            ),
            tooltip=[alt.Tooltip("Tier:N"), alt.Tooltip("Wells:Q")],
        )
        .properties(height=260)
    )


def screening_score_profile_chart(row: dict[str, Any]) -> alt.Chart:
    labels = {
        "Oil Gain Score": "Oil gain",
        "Reserve Score": "Remaining reserve",
        "Reservoir Pressure Score": "Reservoir pressure",
        "Water Cut Score": "Water cut",
        "Integrity Score": "Integrity",
        "Intervention Cost Score": "Intervention cost",
        "Access Score": "Accessibility",
        "Workover Complexity Score": "Workover complexity",
        "PI Score": "Productivity index",
        "Data Completeness Score": "Data completeness",
        "Artificial Lift Score": "Artificial lift",
        "Compartment Score": "Compartmentalization",
    }
    profile = pd.DataFrame(
        [
            {
                "Parameter": label,
                "Score": pd.to_numeric(row.get(column), errors="coerce"),
                "Order": order,
            }
            for order, (column, label) in enumerate(labels.items(), start=1)
        ]
    ).dropna(subset=["Score"])
    return (
        alt.Chart(profile)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Score:Q",
                title="Workbook score (1-5)",
                scale=alt.Scale(domain=[0, 5]),
            ),
            y=alt.Y("Parameter:N", title=None, sort=alt.SortField("Order")),
            color=alt.Color(
                "Score:Q",
                legend=None,
                scale=alt.Scale(domain=[1, 5], range=["#F87171", "#34D399"]),
            ),
            tooltip=[
                alt.Tooltip("Parameter:N"),
                alt.Tooltip("Score:Q", format=".0f"),
            ],
        )
        .properties(height=430)
    )


def canonical_row(master: pd.DataFrame, well: str) -> pd.Series:
    if master.empty:
        return pd.Series(dtype=object)
    match = master.loc[master["Normalized Well"] == services.normalize_well(well)]
    return pd.Series(dtype=object) if match.empty else match.iloc[0]


def format_value(value: Any, suffix: str = "", digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{float(value):,.{digits}f}{suffix}"
    text = str(value).strip()
    return text if text else "N/A"


def integer_value(value: Any) -> int:
    parsed = pd.to_numeric(value, errors="coerce")
    return 0 if pd.isna(parsed) else int(parsed)


def safe_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    selected = [column for column in columns if column in frame.columns]
    return frame[selected].copy() if selected else pd.DataFrame()


def selected_record_controls(
    inventory: list[logs.WellRecord],
) -> tuple[logs.WellRecord, str]:
    fields = sorted({record.field for record in inventory})
    field = st.sidebar.selectbox("Field", fields, key="global_field")
    field_records = [record for record in inventory if record.field == field]
    well = st.sidebar.selectbox(
        "Canonical well",
        [record.well for record in field_records],
        key="global_well",
        format_func=lambda value: (
            f"{value} | "
            f"{len(next(record for record in field_records if record.well == value).las_files)} logs | "
            f"{len(next(record for record in field_records if record.well == value).scanned_files)} scans"
        ),
    )
    return next(record for record in field_records if record.well == well), field


def render_executive(master: pd.DataFrame, selected_record: logs.WellRecord) -> None:
    hero(
        "Integrated Well Reactivation Command Center",
        "Canonical 100-well portfolio view combining production context, data readiness, "
        "LAS availability, schematic coverage, perforation history, and a transparent "
        "three-pillar screening score.",
    )
    if master.empty:
        st.error("The well-level dashboard cache is not available.")
        return

    with st.expander("Portfolio filters and screening weights", expanded=True):
        first, second, third = st.columns([1.4, 1, 1])
        field_options = sorted(master["Field"].dropna().astype(str).unique())
        selected_fields = first.multiselect("Fields", field_options)
        tier_options = sorted(master["Priority Tier"].dropna().astype(str).unique())
        selected_tiers = second.multiselect("Priority tiers", tier_options)
        search = third.text_input("Search well/status/gap")
        weight_columns = st.columns(3)
        technical_weight = weight_columns[0].slider(
            "Technical viability weight", 0.0, 1.0, 0.4, 0.05
        )
        uplift_weight = weight_columns[1].slider(
            "Production uplift weight", 0.0, 1.0, 0.4, 0.05
        )
        execution_weight = weight_columns[2].slider(
            "Execution ease weight", 0.0, 1.0, 0.2, 0.05
        )

    ranked = services.rank_candidates(
        master, technical_weight, uplift_weight, execution_weight
    )
    filtered = ranked.copy()
    if selected_fields:
        filtered = filtered.loc[filtered["Field"].isin(selected_fields)]
    if selected_tiers:
        filtered = filtered.loc[filtered["Priority Tier"].isin(selected_tiers)]
    if search.strip():
        text_columns = [
            column
            for column in (
                "Well",
                "Field",
                "Status",
                "Shutdown Reason",
                "Critical Data Gaps",
                "Priority Tier",
            )
            if column in filtered.columns
        ]
        haystack = filtered[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered.loc[
            haystack.str.contains(re.escape(search.strip()), case=False, regex=True)
        ]

    metrics = st.columns(3)
    metrics[0].metric("Canonical wells", f"{len(filtered):,}")
    metrics[1].metric(
        "Phase 2 review-ready",
        int((filtered["Phase 2 Candidate"] == "READY FOR REVIEW").sum()),
    )
    metrics[2].metric("Wells with LAS", int((filtered["LAS Files"] > 0).sum()))
    metrics = st.columns(3)
    metrics[0].metric(
        "Schematics available", int(filtered["Schematic Available"].fillna(False).sum())
    )
    metrics[1].metric(
        "Detailed history packages",
        int(filtered["Detailed History Available"].fillna(False).sum()),
    )
    metrics[2].metric(
        "Tier 1 candidates",
        int((filtered["Priority Tier"] == "Tier 1 - Quick Win").sum()),
    )

    left, center, right = st.columns([1.1, 1, 1.35])
    with left:
        st.subheader("Priority funnel")
        tier_counts = (
            filtered["Priority Tier"].value_counts().rename_axis("Tier").to_frame("Wells")
        )
        st.bar_chart(tier_counts)
    with center:
        st.subheader("Data readiness")
        readiness = pd.DataFrame(
            {
                "Coverage": [
                    int((filtered["LAS Files"] > 0).sum()),
                    int(filtered["Schematic Available"].fillna(False).sum()),
                    int(filtered["Detailed History Available"].fillna(False).sum()),
                    int((filtered["Production Files"] > 0).sum()),
                ]
            },
            index=["LAS", "Schematic", "Detailed history", "Production"],
        )
        st.bar_chart(readiness)
    with right:
        st.subheader("Top ranked candidates")
        top = filtered.head(12).set_index("Well")
        st.bar_chart(top[["Opportunity Score"]])

    display_columns = [
        "Field",
        "Well",
        "Status",
        "Last Prod Date",
        "Last Oil Rate (BOPD)",
        "Last Water Cut (%)",
        "Technical Viability",
        "Production Uplift Potential",
        "Execution Complexity",
        "Phase 2 Readiness",
        "Opportunity Score",
        "Priority Tier",
        "Phase 2 Candidate",
        "Critical Data Gaps",
    ]
    st.subheader("Ranked well portfolio")
    st.dataframe(
        safe_columns(filtered, display_columns),
        hide_index=True,
        width="stretch",
        height=470,
        column_config={
            "Opportunity Score": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
            "Phase 2 Readiness": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
        },
    )
    st.download_button(
        "Download ranked portfolio CSV",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        "Zona_Rokan_Phase2_Screening_Portfolio.csv",
        "text/csv",
    )

    selected = canonical_row(ranked, selected_record.well)
    if not selected.empty:
        st.subheader(f"Selected well spotlight: {selected_record.field}/{selected_record.well}")
        spotlight = st.columns(5)
        spotlight[0].metric(
            "Opportunity", format_value(selected.get("Opportunity Score"), digits=1)
        )
        spotlight[1].metric(
            "Technical", format_value(selected.get("Technical Viability"), digits=1)
        )
        spotlight[2].metric(
            "Uplift", format_value(selected.get("Production Uplift Potential"), digits=1)
        )
        spotlight[3].metric(
            "Complexity", format_value(selected.get("Execution Complexity"), digits=1)
        )
        spotlight[4].metric(
            "Readiness", format_value(selected.get("Phase 2 Readiness"), "%", 1)
        )
        st.info(
            f"{selected.get('Priority Tier')} · {selected.get('Phase 2 Candidate')} · "
            f"Gaps: {selected.get('Critical Data Gaps')}",
            icon=":material/flag:",
        )
    st.caption(
        "The ranking is a screening aid based on available data, not a reserves estimate, "
        "final candidate approval, or intervention design."
    )


def render_well_360(master: pd.DataFrame, selected_record: logs.WellRecord) -> None:
    hero(
        f"{selected_record.field} / {selected_record.well}",
        "A single-well workspace for Production Technology and Completions/Workover "
        "review, preserving source evidence and data gaps.",
        "Well 360 Workspace",
    )
    row = canonical_row(master, selected_record.well)
    if row.empty:
        st.warning("No normalized well-level cache row matched this canonical well.")
        return

    metrics = st.columns(3)
    metrics[0].metric("Status", format_value(row.get("Status")))
    metrics[1].metric(
        "Last oil", format_value(row.get("Last Oil Rate (BOPD)"), " BOPD")
    )
    metrics[2].metric(
        "Last water cut", format_value(row.get("Last Water Cut (%)"), "%")
    )
    metrics = st.columns(3)
    metrics[0].metric("LAS files", integer_value(row.get("LAS Files")))
    metrics[1].metric("Scanned logs", integer_value(row.get("Scanned Logs")))
    metrics[2].metric(
        "Phase 2 readiness", format_value(row.get("Phase 2 Readiness"), "%")
    )

    production_tab, completion_tab, evidence_tab = st.tabs(
        ("Production Technology", "Completions / Workover", "Source Evidence")
    )
    evidence = load_raw_evidence(selected_record.well)
    definitions, availability_row, request_row = services.selected_data_request(
        selected_record.well
    )
    availability_long = request_row_to_long(
        availability_row, definitions, "Availability Status"
    )
    request_long = request_row_to_long(request_row, definitions, "Request Action")

    with production_tab:
        left, right = st.columns([1, 1.3])
        with left:
            st.subheader("Production and operability")
            production_summary = pd.DataFrame(
                [
                    ("Last production date", row.get("Last Prod Date")),
                    ("Artificial lift", row.get("Artificial Lift")),
                    ("Integrity status", row.get("Integrity Status")),
                    ("Shutdown reason", row.get("Shutdown Reason")),
                    ("Location access", row.get("Location Access")),
                    ("Existing facility", row.get("Existing Facility")),
                    ("Reservoir / zone", row.get("Reservoir / Zone")),
                ],
                columns=["Requirement", "Current evidence"],
            )
            st.dataframe(production_summary, hide_index=True, width="stretch")
        with right:
            st.subheader("Production/test evidence timeline")
            if evidence.empty:
                st.info("No raw normalized evidence rows were found in the dashboard cache.")
            else:
                timeline = evidence.copy()
                timeline["Date"] = pd.to_datetime(
                    timeline.get("__prod_date_dt", timeline.get("Last Prod Date")),
                    errors="coerce",
                )
                timeline["Oil BOPD"] = pd.to_numeric(
                    timeline.get(
                        "Last Oil Rate (BOPD)",
                        timeline.get("BOPD", pd.Series(index=timeline.index, dtype=float)),
                    ),
                    errors="coerce",
                )
                timeline["Water Cut %"] = pd.to_numeric(
                    timeline.get(
                        "Last Water Cut (%)",
                        timeline.get(
                            "WATER_CUT", pd.Series(index=timeline.index, dtype=float)
                        ),
                    ),
                    errors="coerce",
                )
                timeline = (
                    timeline.dropna(subset=["Date"])
                    .sort_values("Date")
                    .drop_duplicates(subset=["Date"], keep="last")
                    .set_index("Date")
                )
                chart_data = timeline[["Oil BOPD", "Water Cut %"]].dropna(how="all")
                if chart_data.empty:
                    st.info("No dated oil-rate or water-cut series is available.")
                else:
                    st.line_chart(chart_data, height=360)
        gaps = row.get("Critical Data Gaps", "N/A")
        st.warning(
            f"Production Technology data gaps: {gaps}",
            icon=":material/data_alert:",
        )
        production_codes = {
            "REQ-02",
            "REQ-03",
            "REQ-04",
            "REQ-05",
            "REQ-07",
            "REQ-09",
            "REQ-11",
            "REQ-12",
            "REQ-13",
            "REQ-14",
            "REQ-19",
        }
        role_status = availability_long.loc[
            availability_long.get("Request Code", pd.Series(dtype=str)).isin(
                production_codes
            )
        ].merge(
            request_long[["Request Code", "Request Action"]],
            on="Request Code",
            how="left",
        )
        st.subheader("Production Technologist requirement matrix")
        st.dataframe(
            role_status,
            hide_index=True,
            width="stretch",
            height=430,
        )

    with completion_tab:
        schematic = services.selected_schematic_row(selected_record.well)
        perf = load_perforation_history(selected_record.well)
        detail_events, _ = services.selected_schematic_detail(selected_record.well)
        related_detail = services.perforation_related_events(detail_events)
        completion_metrics = st.columns(5)
        completion_metrics[0].metric(
            "Schematic", "Available" if row.get("Schematic Available") else "Missing"
        )
        completion_metrics[1].metric(
            "Casing strings", integer_value(row.get("Casing Strings"))
        )
        completion_metrics[2].metric(
            "Perf/workover detail", len(related_detail)
        )
        completion_metrics[3].metric(
            "Perf QA review", integer_value(row.get("Perforation QA Review"))
        )
        completion_metrics[4].metric(
            "Production files", integer_value(row.get("Production Files"))
        )
        left, right = st.columns([1, 1.25])
        with left:
            st.subheader("Completion configuration")
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Completion type", format_value(row.get("Completion Type"))),
                        ("Artificial lift", format_value(row.get("Artificial Lift"))),
                        ("Integrity status", format_value(row.get("Integrity Status"))),
                        ("Schematic events", format_value(schematic.get("Events"))),
                        ("Sand intervals", format_value(schematic.get("Sand Intervals"))),
                        ("Casing strings", format_value(schematic.get("Casing Strings"))),
                    ],
                    columns=["Item", "Evidence"],
                ),
                hide_index=True,
                width="stretch",
            )
        with right:
            st.subheader("Perforation and isolation timeline")
            if perf.empty:
                st.info("No perforation event rows are available for this well.")
            else:
                st.dataframe(
                    safe_columns(
                        perf,
                        [
                            "Event Date",
                            "Action Status",
                            "Sand",
                            "Depth (Feet)",
                            "QA Status",
                            "Evidence",
                        ],
                    ),
                    hide_index=True,
                    width="stretch",
                    height=330,
                )
        completion_codes = {
            "REQ-01",
            "REQ-06",
            "REQ-07",
            "REQ-08",
            "REQ-12",
            "REQ-13",
            "REQ-14",
            "REQ-16",
            "REQ-18",
        }
        completion_status = availability_long.loc[
            availability_long.get("Request Code", pd.Series(dtype=str)).isin(
                completion_codes
            )
        ].merge(
            request_long[["Request Code", "Request Action"]],
            on="Request Code",
            how="left",
        )
        st.subheader("Completions / Workover requirement matrix")
        st.dataframe(
            completion_status,
            hide_index=True,
            width="stretch",
            height=430,
        )
        trajectory_evidence = services.selected_trajectory_evidence(selected_record)
        coordinate_evidence = services.selected_coordinate_evidence(selected_record)
        context = st.columns(3)
        context[0].metric("Coordinate evidence rows", len(coordinate_evidence))
        context[1].metric("Trajectory evidence rows", len(trajectory_evidence))
        context[2].metric("Active perforation intervals", len(services.active_perforation_intervals(perf)))

    with evidence_tab:
        if evidence.empty:
            st.info("No source evidence rows found.")
        else:
            evidence_columns = [
                "File Name",
                "Source Category",
                "Source Sheet",
                "Document Type",
                "Document Date",
                "Operation Date",
                "Operation Summary",
                "Integrity Status",
                "Completion Type",
                "Artificial Lift",
                "Source Path",
            ]
            st.dataframe(
                safe_columns(evidence, evidence_columns),
                hide_index=True,
                width="stretch",
                height=520,
            )


def request_row_to_long(
    row_frame: pd.DataFrame,
    definitions: pd.DataFrame,
    value_label: str,
) -> pd.DataFrame:
    output_columns = [
        "Request Code",
        "Data Package",
        value_label,
        "Data Request Detail",
        "Decision Use",
    ]
    if row_frame.empty:
        return pd.DataFrame(columns=output_columns)
    row = row_frame.iloc[0]
    definition_map = (
        definitions.set_index("Request Code").to_dict("index")
        if not definitions.empty and "Request Code" in definitions.columns
        else {}
    )
    rows: list[dict[str, Any]] = []
    for column in row_frame.columns:
        code_match = re.match(r"(REQ-\d+)", str(column))
        if not code_match:
            continue
        code = code_match.group(1)
        detail = definition_map.get(code, {})
        rows.append(
            {
                "Request Code": code,
                "Data Package": detail.get(
                    "Data Package", str(column).split("\n", 1)[-1]
                ),
                value_label: row.get(column),
                "Data Request Detail": detail.get("Data Request Detail", ""),
                "Decision Use": detail.get("Decision Use", ""),
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def render_coordinates_and_requests(
    master: pd.DataFrame,
    selected_record: logs.WellRecord,
) -> None:
    hero(
        "Coordinates, data availability, and requests",
        "Well location evidence, the controlled 001 Rokan availability result, and the "
        "existing request-action register in one auditable workspace.",
        "Data foundation",
    )
    coordinates = services.selected_coordinate_evidence(selected_record)
    definitions, availability_row, request_row = services.selected_data_request(
        selected_record.well
    )
    _, all_availability, all_requests = services.load_data_request_tables()

    coordinate_tab, availability_tab, request_tab, definitions_tab = st.tabs(
        (
            "Well coordinates",
            "Data availability",
            "Data requests",
            "Request definitions",
        )
    )
    with coordinate_tab:
        if coordinates.empty:
            st.warning(
                "No coordinate values were found in the selected well's LAS headers. "
                "REQ-01 remains a data request; coordinates are not estimated."
            )
            default_lat, default_lon = 0.0, 101.0
        else:
            best = coordinates.sort_values(
                ["Latitude", "Longitude"], na_position="last"
            ).iloc[0]
            default_lat = (
                float(best["Latitude"]) if pd.notna(best["Latitude"]) else 0.0
            )
            default_lon = (
                float(best["Longitude"]) if pd.notna(best["Longitude"]) else 101.0
            )
            metrics = st.columns(4)
            metrics[0].metric("Latitude", format_value(best.get("Latitude"), digits=6))
            metrics[1].metric("Longitude", format_value(best.get("Longitude"), digits=6))
            metrics[2].metric(
                "Converted UTM X",
                format_value(best.get("Converted UTM X"), digits=3),
            )
            metrics[3].metric(
                "Converted UTM Y",
                format_value(best.get("Converted UTM Y"), digits=3),
            )
            st.dataframe(
                coordinates,
                hide_index=True,
                width="stretch",
                height=300,
            )
            local_grid = coordinates.loc[
                coordinates["Coordinate QC"].astype(str).str.contains(
                    "LOCAL GRID|DIFFERENT CRS", case=False, regex=True
                )
            ]
            if not local_grid.empty:
                st.warning(
                    "A source X/Y pair differs materially from standard WGS84 UTM. "
                    "The dashboard therefore preserves it as local-grid evidence and uses "
                    "the source latitude/longitude to calculate standardized UTM X/Y. "
                    "A grid definition or control-point transformation is required when "
                    "latitude/longitude is absent."
                )

        st.subheader("Coordinate conversion check")
        conversion = st.columns(3)
        latitude = conversion[0].number_input(
            "Latitude (decimal degrees)",
            min_value=-90.0,
            max_value=90.0,
            value=default_lat,
            format="%.8f",
        )
        longitude = conversion[1].number_input(
            "Longitude (decimal degrees)",
            min_value=-180.0,
            max_value=180.0,
            value=default_lon,
            format="%.8f",
        )
        automatic_zone = services.utm_zone_from_longitude(longitude)
        zone = conversion[2].number_input(
            "UTM zone",
            min_value=1,
            max_value=60,
            value=automatic_zone,
            step=1,
        )
        try:
            x_value, y_value, zone_label, epsg = services.latlon_to_utm(
                latitude, longitude, int(zone)
            )
            result = st.columns(4)
            result[0].metric("X / Easting", f"{x_value:,.3f} m")
            result[1].metric("Y / Northing", f"{y_value:,.3f} m")
            result[2].metric("Zone", zone_label)
            result[3].metric("EPSG", epsg)
        except ValueError as exc:
            st.error(str(exc))
        st.caption(
            "Conversion uses WGS84 Transverse Mercator. A local-grid X/Y value cannot "
            "be transformed reliably without its datum/projection or surveyed tie points."
        )

    with availability_tab:
        source_caption(
            services.DATA_AVAILABILITY_WORKBOOK,
            "Data_Availability_Matrix",
        )
        canonical_wells = master.get("Well", pd.Series(dtype=object)).dropna().tolist()
        matched_rows = all_availability.get(
            "Well Name", pd.Series(dtype=object)
        ).map(
            lambda workbook_well: any(
                services.well_matches(workbook_well, canonical_well)
                for canonical_well in canonical_wells
            )
        )
        request_columns = [
            column for column in all_availability.columns if str(column).startswith("REQ-")
        ]
        overview = st.columns(4)
        overview[0].metric("Workbook well rows", len(all_availability), border=True)
        overview[1].metric("Canonical wells matched", int(matched_rows.sum()), border=True)
        overview[2].metric(
            "Workbook-only rows", int((~matched_rows).sum()), border=True
        )
        overview[3].metric("Request categories", len(request_columns), border=True)
        if (~matched_rows).any():
            workbook_only = ", ".join(
                all_availability.loc[~matched_rows, "Well Name"].astype(str)
            )
            st.info(
                f"Workbook-only coverage is preserved for audit: {workbook_only}. "
                "It is not silently assigned to a different canonical folder.",
                icon=":material/info:",
            )
        selected_availability = request_row_to_long(
            availability_row, definitions, "Availability Status"
        )
        if availability_row.empty:
            st.warning(
                "No selected-well availability row was found in the controlled workbook.",
                icon=":material/data_alert:",
            )
        else:
            row = availability_row.iloc[0]
            status_values = [
                str(row[column]).strip()
                for column in availability_row.columns
                if str(column).startswith("REQ-")
            ]
            status_upper = [value.upper() for value in status_values]
            metrics = st.columns(4)
            metrics[0].metric(
                "Available", sum(value == "AVAILABLE" for value in status_upper), border=True
            )
            metrics[1].metric(
                "Partial: Not Comprehensive",
                sum(value == "PARTIAL: NOT COMPREHENSIVE" for value in status_upper),
                border=True,
            )
            metrics[2].metric(
                "Missing", sum(value == "MISSING" for value in status_upper), border=True
            )
            metrics[3].metric(
                "Other source values",
                sum(
                    value
                    not in {"AVAILABLE", "PARTIAL: NOT COMPREHENSIVE", "MISSING"}
                    for value in status_upper
                ),
                border=True,
            )
            coverage_score = {
                "AVAILABLE": 100,
                "PARTIAL: NOT COMPREHENSIVE": 50,
                "MISSING": 0,
            }
            selected_availability["Coverage score"] = selected_availability[
                "Availability Status"
            ].fillna("").astype(str).str.upper().map(coverage_score)
            st.dataframe(
                selected_availability,
                hide_index=True,
                width="stretch",
                height=520,
                column_config={
                    "Coverage score": st.column_config.ProgressColumn(
                        "Coverage", min_value=0, max_value=100, format="%d%%"
                    )
                },
            )
        with st.expander(f"All {len(all_availability)} workbook rows"):
            st.dataframe(
                all_availability,
                hide_index=True,
                width="stretch",
                height=520,
            )

    with request_tab:
        source_caption(services.DATA_REQUEST_WORKBOOK, "A.Data Available&B.Data Request")
        st.caption(
            "Request actions remain sourced from the existing request register; "
            "availability status is sourced only from 001_Rokan Block Data Availability Final.xlsx."
        )
        selected_request = request_row_to_long(
            request_row, definitions, "Request Action"
        )
        if selected_request.empty:
            st.warning("No selected-well data-request row was found in the workbook.")
        else:
            actionable = selected_request.loc[
                ~selected_request["Request Action"].fillna("").astype(str).isin(["", "—"])
            ]
            st.metric("Open request packages", len(actionable))
            st.dataframe(
                selected_request,
                hide_index=True,
                width="stretch",
                height=520,
            )
        with st.expander(f"All {len(all_requests)} data-request rows"):
            st.dataframe(
                all_requests,
                hide_index=True,
                width="stretch",
                height=520,
            )

    with definitions_tab:
        source_caption(services.DATA_AVAILABILITY_WORKBOOK, "Data_Package_Index")
        st.dataframe(
            definitions,
            hide_index=True,
            width="stretch",
            height=650,
        )
        st.download_button(
            "Download request definitions CSV",
            definitions.to_csv(index=False).encode("utf-8-sig"),
            "Zona_Rokan_Data_Request_Definitions.csv",
            "text/csv",
        )


def load_selected_log(
    selected_record: logs.WellRecord,
) -> tuple[logs.LoadedLog, pd.DataFrame, str, str, str, Path | None] | None:
    digitized = services.collect_digitized_las(selected_record.field, selected_record.well)
    source_options = (
        ("Upload log",)
        if services.ONLINE_MODE
        else ("Canonical inventory", "Dashboard digitization output", "Upload log")
    )
    source_mode = st.radio(
        "Log source",
        source_options,
        horizontal=True,
    )
    selected_path: Path | None = None
    uploaded = None
    if source_mode == "Canonical inventory":
        if not selected_record.las_files:
            st.warning("This canonical well has no LAS, DLIS, or LIS file.")
            return None
        selected_path = st.selectbox(
            "Log file",
            selected_record.las_files,
            format_func=lambda path: f"{path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)",
        )
    elif source_mode == "Dashboard digitization output":
        if not digitized:
            st.warning("No digitized LAS output exists for this selected well.")
            return None
        selected_path = st.selectbox(
            "Digitized LAS file", digitized, format_func=lambda path: path.name
        )
    else:
        uploaded = st.file_uploader("Upload log", type=["las", "dlis", "lis"])
        if uploaded is None:
            st.info("Upload a LAS, DLIS, or LIS file to begin.")
            return None
    try:
        loaded_log = logs.read_log(
            selected_path if selected_path else uploaded.getvalue(),
            "" if selected_path else uploaded.name,
        )
        raw = loaded_log.data
    except Exception as exc:
        st.error(f"Log loading failed: {exc}")
        return None
    if raw.empty:
        st.error("The selected log contains no readable curve samples.")
        return None
    well_name = (
        selected_record.well
        if selected_path
        else loaded_log.well or uploaded.name
    )
    field_name = (
        selected_record.field
        if selected_path
        else loaded_log.field or "Zona Rokan"
    )
    depth_unit = loaded_log.depth_unit
    return loaded_log, raw, field_name, well_name, depth_unit, selected_path


def render_las_and_nzbp(
    selected_record: logs.WellRecord,
    workspace: str = "formation",
) -> None:
    if workspace == "nzbp":
        hero(
            "New Zone Behind Pipe Screening",
            "Use the completed formation-evaluation curves with completion, casing, "
            "sand, and perforation evidence to identify unperforated reservoir intervals.",
            "Subsurface / Recompletion",
        )
    else:
        hero(
            "LAS and Autonomous Formation Evaluation",
            "Run source-backed conditioning, TVD resolution, Vshale, porosity, "
            "Coates-Timur permeability, Modified Simandoux saturation, validation, "
            "cutoff optimization, and pay lumping.",
            "Petrophysics",
        )
    loaded = load_selected_log(selected_record)
    if loaded is None:
        return
    loaded_log, raw, field_name, well_name, depth_unit, selected_path = loaded
    autonomous = st.toggle(
        "Autonomous Formation Evaluation",
        value=True,
        help=(
            "Automatically selects additional runs, TVD evidence, matrix defaults, "
            "Pickett Rw, validation samples, and cutoffs. Every basis remains visible."
        ),
        key=f"autonomous_fe_{workspace}_{selected_record.well}",
    )
    schematic_source = services.selected_schematic_workbook(selected_record.well)
    casing = schematic_source.get("casing", pd.DataFrame()).copy()
    sands = schematic_source.get("intervals", pd.DataFrame()).copy()

    shift_rows: list[dict[str, Any]] = []
    if selected_path is not None and len(selected_record.las_files) > 1:
        with st.expander("Data conditioning - depth shift, splice, and merge", expanded=False):
            other_runs = [
                path for path in selected_record.las_files if path != selected_path
            ]
            selected_runs = st.multiselect(
                "Additional LAS runs to align and splice",
                other_runs,
                default=other_runs if autonomous else [],
                format_func=lambda path: path.name,
            )
            shift_controls = st.columns(2)
            maximum_shift = shift_controls[0].number_input(
                "Maximum GR depth shift",
                min_value=1.0,
                value=50.0,
                step=5.0,
            )
            shift_step = shift_controls[1].number_input(
                "Shift search step",
                min_value=0.1,
                value=0.5,
                step=0.1,
            )
            reference_map = logs.auto_curve_map(raw)
            reference_gr_name = reference_map.get("Gamma Ray")
            moving_runs: list[tuple[pd.DataFrame, float]] = []
            for path in selected_runs:
                try:
                    moving = load_log_path(str(path))
                    moving_map = logs.auto_curve_map(moving)
                    moving_gr_name = moving_map.get("Gamma Ray")
                    if reference_gr_name and moving_gr_name:
                        shift, correlation = logs.estimate_gr_depth_shift(
                            raw[reference_gr_name],
                            moving[moving_gr_name],
                            max_shift=float(maximum_shift),
                            shift_step=float(shift_step),
                        )
                    else:
                        shift, correlation = 0.0, np.nan
                    moving_runs.append((moving, shift))
                    shift_rows.append(
                        {
                            "Run": path.name,
                            "GR depth correction": shift,
                            "GR correlation": correlation,
                            "Samples": len(moving),
                            "Status": (
                                "ALIGNED BY GR"
                                if np.isfinite(correlation)
                                else "MERGED WITHOUT GR SHIFT"
                            ),
                        }
                    )
                except Exception as exc:
                    shift_rows.append(
                        {
                            "Run": path.name,
                            "GR depth correction": np.nan,
                            "GR correlation": np.nan,
                            "Samples": 0,
                            "Status": f"FAILED: {exc}",
                        }
                    )
            if moving_runs:
                raw = logs.merge_shifted_runs(raw, moving_runs)
            if shift_rows:
                st.dataframe(pd.DataFrame(shift_rows), hide_index=True, width="stretch")

    auto_map = logs.auto_curve_map(raw)

    summary_metrics = st.columns(6)
    summary_metrics[0].metric("Depth samples", f"{len(raw):,}")
    summary_metrics[1].metric("Curves", len(raw.columns))
    summary_metrics[2].metric("Top MD", logs.format_number(float(raw.index.min())))
    summary_metrics[3].metric("Base MD", logs.format_number(float(raw.index.max())))
    summary_metrics[4].metric("Source type", loaded_log.source_type)
    summary_metrics[5].metric("Depth unit", depth_unit or "Source units")

    with st.expander("Curve mapping", expanded=False):
        options = ["-- Not available --", *raw.columns.tolist()]
        mapping: dict[str, str | None] = {}
        columns = st.columns(4)
        for index, role in enumerate(logs.CURVE_ALIASES):
            suggested = auto_map[role]
            option_index = options.index(suggested) if suggested in options else 0
            selected = columns[index % 4].selectbox(
                role,
                options,
                index=option_index,
                key=f"las_map_{selected_record.well}_{role}",
            )
            mapping[role] = None if selected == options[0] else selected

    def series(role: str) -> pd.Series:
        mnemonic = mapping[role]
        if mnemonic is None:
            return pd.Series(np.nan, index=raw.index, dtype=float)
        return raw[mnemonic].astype(float)

    gr = series("Gamma Ray")
    resistivity = series("Deep Resistivity")
    density = series("Density")
    density_correction = series("Density Correction")
    neutron = logs.normalize_neutron(
        series("Neutron"), loaded_log.curve_unit(mapping["Neutron"])
    )
    neutron_fraction = neutron / 100.0
    caliper = series("Caliper")
    bit_size = series("Bit Size")
    dtc = series("Compressional Sonic")
    pef = series("PEF")
    total_gas = series("Total Gas")

    conditioning = st.columns(4)
    washout_tolerance = conditioning[0].number_input(
        "Caliper washout tolerance", min_value=0.0, value=1.0, step=0.25
    )
    coal_density_max = conditioning[1].number_input(
        "Coal RHOB maximum", min_value=1.0, value=2.15, step=0.05
    )
    coal_neutron_min = conditioning[2].number_input(
        "Coal NPHI minimum", min_value=0.0, max_value=0.8, value=0.25, step=0.02
    )
    shale_separation_min = conditioning[3].number_input(
        "Shale N-D separation minimum",
        min_value=0.0,
        max_value=0.8,
        value=0.12,
        step=0.02,
    )
    casing_depths = pd.Series(dtype=float)
    for column in casing.columns:
        if "bottom" in str(column).lower() or "base" in str(column).lower():
            casing_depths = pd.to_numeric(
                casing[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
                errors="coerce",
            )
            if casing_depths.notna().any():
                break
    casing_shoe_default = (
        float(casing_depths.max())
        if casing_depths.notna().any()
        else float(raw.index.min())
    )
    casing_shoe_depth = st.number_input(
        "Casing shoe / last casing depth",
        value=casing_shoe_default,
        help="Source-backed casing depth from the schematic table; edit after engineering review.",
    )

    preconditioned = pd.DataFrame(
        {
            "GR": gr,
            "RES": resistivity,
            "RHOB": density,
            "NPHI_PU": neutron,
            "NPHI_FRAC": neutron_fraction,
            "DRHO": density_correction,
            "CALI": caliper,
            "BIT": bit_size,
            "DTC": dtc,
            "PEF": pef,
            "TOTAL_GAS": total_gas,
        },
        index=raw.index,
    )
    conditioned, qc = logs.condition_log_curves(
        preconditioned,
        caliper,
        bit_size,
        density_correction,
        float(washout_tolerance),
    )
    gr = conditioned["GR"]
    resistivity = conditioned["RES"]
    density = conditioned["RHOB"]
    neutron_fraction = conditioned["NPHI_FRAC"]
    caliper = conditioned["CALI"]
    dtc = conditioned["DTC"]
    pef = conditioned["PEF"]
    total_gas = conditioned["TOTAL_GAS"]
    coal_flag = (
        (density <= coal_density_max)
        & (neutron_fraction >= coal_neutron_min)
        & (gr <= gr.quantile(0.25) if gr.notna().any() else False)
        & density.notna()
        & neutron_fraction.notna()
    )
    density_porosity_preview = logs.density_porosity(density, 2.65, 1.0)
    nd_separation = neutron_fraction - density_porosity_preview
    shale_reference_flag = (
        (nd_separation >= shale_separation_min)
        & gr.notna()
        & (gr >= gr.quantile(0.60) if gr.notna().any() else False)
    )

    settings_left, settings_center, settings_right = st.columns(3)
    finite_gr = gr.dropna()
    coal_reference = gr.loc[coal_flag].dropna()
    shale_reference = gr.loc[shale_reference_flag].dropna()
    clean_gr = settings_left.number_input(
        "Clean GR",
        value=(
            float(coal_reference.median())
            if not coal_reference.empty
            else float(finite_gr.quantile(0.05))
            if not finite_gr.empty
            else 25.0
        ),
        help="Coal-flagged intervals are used as the clean reference when available.",
    )
    shale_gr = settings_left.number_input(
        "Shale GR",
        value=(
            float(shale_reference.median())
            if not shale_reference.empty
            else float(finite_gr.quantile(0.95))
            if not finite_gr.empty
            else 125.0
        ),
        help="High neutron-density separation and high GR define the initial shale reference.",
    )
    automatic_mineral, automatic_matrix_density, mineral_basis = (
        logs.infer_matrix_from_pef(pef, gr)
    )
    xrd_upload = st.file_uploader(
        "XRD / petrography mineral table (optional)",
        type=["csv", "xlsx", "xls"],
        key=f"xrd_{workspace}_{selected_record.well}",
    )
    core_upload = st.file_uploader(
        "Core / SWC validation table (optional)",
        type=["csv", "xlsx", "xls"],
        key=f"core_{workspace}_{selected_record.well}",
    )
    xrd_source: pd.DataFrame | None = None
    core_source: pd.DataFrame | None = None
    xrd_density: float | None = None
    xrd_table = pd.DataFrame(columns=["Mineral", "Fraction (%)", "Density (g/cc)"])
    if xrd_upload is not None:
        try:
            xrd_source = logs.read_table_upload(xrd_upload)
            xrd_density, xrd_table = logs.xrd_matrix_density(xrd_source)
        except Exception as exc:
            st.warning(f"XRD/petrography table could not be read: {exc}")
    if core_upload is not None:
        try:
            core_source = logs.read_table_upload(core_upload)
        except Exception as exc:
            st.warning(f"Core/SWC table could not be read: {exc}")
    vsh_method = settings_left.selectbox(
        "Vsh method",
        (
            "Larionov Tertiary",
            "Linear",
            "Larionov Older Rocks",
            "Clavier",
            "Steiber",
        ),
    )
    mineral_options = (
        ("XRD weighted", "Quartz / sandstone", "Calcite / limestone", "Dolomite", "Manual")
        if xrd_density is not None
        else ("Quartz / sandstone", "Calcite / limestone", "Dolomite", "Manual")
    )
    mineral_model = settings_center.selectbox(
        "Principal matrix mineral",
        mineral_options,
        index=(
            mineral_options.index(
                automatic_mineral
            )
            if autonomous and automatic_mineral in mineral_options
            else 0
        ),
    )
    density_defaults = {
        "Quartz / sandstone": 2.65,
        "Calcite / limestone": 2.71,
        "Dolomite": 2.87,
        "Manual": 2.65,
        "XRD weighted": xrd_density or 2.65,
    }
    matrix_density = settings_center.number_input(
        "Matrix density (g/cc)",
        value=(
            xrd_density
            if mineral_model == "XRD weighted" and xrd_density is not None
            else automatic_matrix_density
            if autonomous and mineral_model == automatic_mineral
            else density_defaults[mineral_model]
        ),
        step=0.01,
        help="Override using XRD/petrography results when available.",
    )
    fluid_density = settings_center.number_input(
        "Fluid density (g/cc)", value=1.00, step=0.01
    )
    bound_water_factor = settings_center.number_input(
        "Shale bound-water porosity",
        min_value=0.0,
        max_value=0.6,
        value=0.25,
        step=0.01,
    )
    cutoff_mode = settings_right.selectbox(
        "Cutoff mode",
        ("Autonomous validation-driven", "Manual"),
        index=0 if autonomous else 1,
    )
    manual_vsh_cutoff = settings_right.slider(
        "Maximum Vsh", 0.0, 1.0, 0.40, 0.05
    )
    manual_phi_cutoff = settings_right.slider(
        "Minimum effective porosity", 0.0, 0.40, 0.10, 0.01
    )
    manual_sw_cutoff = settings_right.slider(
        "Maximum Sw", 0.0, 1.0, 0.60, 0.05
    )
    manual_permeability_cutoff = settings_right.number_input(
        "Minimum permeability (mD)", min_value=0.0, value=1.0, step=0.5
    )

    preview_vsh = logs.calculate_vsh(gr, clean_gr, shale_gr, vsh_method)
    preview_phi_density = logs.density_porosity(
        density, matrix_density, fluid_density
    )
    preview_phi_effective = (
        preview_phi_density * (1.0 - preview_vsh)
    ).clip(0.0, 0.6)
    automatic_rw, automatic_rw_samples = logs.estimate_pickett_rw(
        preview_phi_effective,
        resistivity,
        preview_vsh,
    )
    saturation = st.columns(5)
    rw = saturation[0].number_input(
        "Default Rw (ohm.m)",
        min_value=0.001,
        value=automatic_rw if autonomous else 0.20,
        help=(
            f"Autonomous Pickett estimate uses {automatic_rw_samples} clean/wet samples."
        ),
    )
    rsh = saturation[1].number_input("Rsh (ohm.m)", min_value=0.01, value=2.0)
    archie_a = saturation[2].number_input("Tortuosity a", min_value=0.1, value=1.0)
    archie_m = saturation[3].number_input("Cementation m", min_value=0.1, value=2.0)
    archie_n = saturation[4].number_input("Saturation n", min_value=0.1, value=2.0)

    rw_intervals = st.data_editor(
        pd.DataFrame(
            [
                {
                    "Top MD": float(raw.index.min()),
                    "Base MD": float(raw.index.max()),
                    "Rw (ohm.m)": rw,
                    "Basis": "Default; adjust from water analysis/Pickett plot",
                }
            ]
        ),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"rw_intervals_{selected_record.well}",
    )

    calculated_vsh = logs.calculate_vsh(gr, clean_gr, shale_gr, vsh_method)
    vsh = logs.normalize_fraction(series("Interpreted Vsh")).combine_first(
        calculated_vsh
    )
    phi_density = logs.density_porosity(density, matrix_density, fluid_density)
    interpreted_porosity = logs.normalize_fraction(series("Interpreted Porosity"))
    phi_effective = interpreted_porosity.combine_first(
        (phi_density * (1.0 - vsh)).clip(0.0, 0.6)
    )
    phi_bound_water = (
        (neutron_fraction - phi_effective)
        .clip(lower=0.0)
        .clip(upper=(vsh * bound_water_factor).clip(lower=0.0))
    )
    phi_total = (phi_effective + phi_bound_water).clip(0.0, 0.6)
    rw_series = pd.Series(float(rw), index=raw.index)
    for _, rw_row in rw_intervals.iterrows():
        interval_top = pd.to_numeric(rw_row.get("Top MD"), errors="coerce")
        interval_base = pd.to_numeric(rw_row.get("Base MD"), errors="coerce")
        interval_rw = pd.to_numeric(rw_row.get("Rw (ohm.m)"), errors="coerce")
        if pd.isna(interval_top) or pd.isna(interval_base) or pd.isna(interval_rw):
            continue
        interval_top, interval_base = sorted((float(interval_top), float(interval_base)))
        rw_series.loc[(rw_series.index >= interval_top) & (rw_series.index <= interval_base)] = float(interval_rw)
    sw = logs.normalize_fraction(series("Water Saturation")).combine_first(
        logs.modified_simandoux_water_saturation(
            phi_effective,
            resistivity,
            vsh,
            rw_series,
            float(rsh),
            float(archie_a),
            float(archie_m),
            float(archie_n),
        )
    )
    permeability_controls = st.columns(4)
    perm_coefficient = permeability_controls[0].number_input(
        "Coates-Timur coefficient A", min_value=1.0, value=10000.0
    )
    perm_phi_exp = permeability_controls[1].number_input(
        "Porosity exponent B", min_value=0.1, value=4.0, step=0.5
    )
    perm_ratio_exp = permeability_controls[2].number_input(
        "FFI/BVI exponent C", min_value=0.1, value=2.0, step=0.5
    )
    swirr_floor = permeability_controls[3].number_input(
        "Irreducible Sw floor",
        min_value=0.01,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )
    permeability = series("Permeability").combine_first(
        logs.coates_timur_permeability(
            phi_effective,
            phi_bound_water,
            float(perm_coefficient),
            float(perm_phi_exp),
            float(perm_ratio_exp),
            float(swirr_floor),
        )
    )

    tvd, trajectory, tvd_metadata = resolve_tvd_from_zona_rokan(
        selected_record,
        loaded_log.depth_mnemonic,
        raw,
        selected_path,
    )
    tvd_basis = str(tvd_metadata["basis"])
    deviation_survey = st.file_uploader(
        "Optional override survey for TVD (CSV/XLSX with MD, inclination, azimuth)",
        type=["csv", "xlsx", "xls"],
        key=f"deviation_{selected_record.well}",
    )
    if deviation_survey is not None:
        try:
            if Path(deviation_survey.name).suffix.lower() in {".xlsx", ".xls"}:
                survey_source = pd.read_excel(deviation_survey)
            else:
                survey_source = pd.read_csv(deviation_survey)
            trajectory = logs.minimum_curvature_survey(survey_source)
            tvd = pd.Series(
                np.interp(
                    raw.index.to_numpy(dtype=float),
                    trajectory["MD"],
                    trajectory["TVD"],
                ),
                index=raw.index,
            )
            tvd_basis = "MINIMUM-CURVATURE TVD FROM UPLOADED SURVEY"
            tvd_metadata = {
                "basis": tvd_basis,
                "source": deviation_survey.name,
                "coverage": 1.0,
                "quality": "USER OVERRIDE",
            }
        except Exception as exc:
            st.error(f"Deviation survey could not be calculated: {exc}")

    work = pd.DataFrame(
        {
            "TVD": tvd,
            "GR": gr,
            "RES": resistivity,
            "RHOB": density,
            "NPHI_PU": neutron,
            "NPHI_FRAC": neutron_fraction,
            "CALI": caliper,
            "DTC": dtc,
            "PEF": pef,
            "TOTAL_GAS": total_gas,
            "VSH": vsh,
            "PHID": phi_density,
            "PHIBW": phi_bound_water,
            "PHIT": phi_total,
            "PHIE": phi_effective,
            "POROSITY": phi_effective,
            "SW": sw,
            "PERM_MD": permeability,
            "RW": rw_series,
            "COAL_FLAG": coal_flag.fillna(False),
            "SHALE_REFERENCE_FLAG": shale_reference_flag.fillna(False),
            "BADHOLE_FLAG": qc["BADHOLE_FLAG"],
            "UNREALISTIC_REMOVED": qc["UNREALISTIC_REMOVED"],
            "CASING_FLAG": pd.Series(
                raw.index.to_numpy(dtype=float) <= float(casing_shoe_depth),
                index=raw.index,
            ),
        },
        index=raw.index,
    )

    interval_columns = st.columns(4)
    top_depth = interval_columns[0].number_input(
        "Analysis top MD", value=float(work.index.min())
    )
    base_depth = interval_columns[1].number_input(
        "Analysis base MD", value=float(work.index.max())
    )
    minimum_nzbp = interval_columns[2].number_input(
        "Minimum NZBP thickness", min_value=0.5, value=5.0, step=0.5
    )
    exclude_badhole = interval_columns[3].checkbox(
        "Exclude badhole", value=True
    )
    top_depth, base_depth = sorted((top_depth, base_depth))
    interval = work.loc[(work.index >= top_depth) & (work.index <= base_depth)].copy()
    if interval.empty:
        st.error("The selected interval contains no LAS samples.")
        return

    history = load_perforation_history(selected_record.well)
    active_perfs = services.active_perforation_intervals(history)
    has_perforation_intervals = (
        not history.empty
        and history[["Perforation Top (ft)", "Perforation Base (ft)"]]
        .notna()
        .all(axis=1)
        .any()
    )
    completion_validation = pd.Series(False, index=interval.index)
    for perf_top, perf_base in active_perfs:
        completion_validation |= interval.index.to_series().between(
            perf_top, perf_base
        )
    recommended_cutoffs = logs.autonomous_cutoffs(
        interval,
        completion_validation,
    )
    if cutoff_mode == "Autonomous validation-driven":
        vsh_cutoff = float(recommended_cutoffs["VSH"])
        phi_cutoff = float(recommended_cutoffs["PHIE"])
        sw_cutoff = float(recommended_cutoffs["SW"])
        permeability_cutoff = float(recommended_cutoffs["PERM_MD"])
    else:
        vsh_cutoff = float(manual_vsh_cutoff)
        phi_cutoff = float(manual_phi_cutoff)
        sw_cutoff = float(manual_sw_cutoff)
        permeability_cutoff = float(manual_permeability_cutoff)
    applied_cutoffs: dict[str, float | str | int] = {
        **recommended_cutoffs,
        "VSH": vsh_cutoff,
        "PHIE": phi_cutoff,
        "SW": sw_cutoff,
        "PERM_MD": permeability_cutoff,
        "Mode": cutoff_mode,
    }

    reservoir_flag = (interval["VSH"] <= vsh_cutoff) & (
        interval["PHIE"] >= phi_cutoff
    )
    reservoir_flag &= interval["SW"] <= sw_cutoff
    reservoir_flag &= interval["PERM_MD"] >= permeability_cutoff
    if exclude_badhole:
        reservoir_flag &= ~interval["BADHOLE_FLAG"]
    if interval["TOTAL_GAS"].notna().any():
        gas_cutoff = st.number_input(
            "Minimum total-gas validation cutoff",
            min_value=0.0,
            value=float(interval["TOTAL_GAS"].dropna().quantile(0.50)),
        )
        apply_gas_cutoff = st.checkbox("Apply total-gas cutoff", value=False)
        if apply_gas_cutoff:
            reservoir_flag &= interval["TOTAL_GAS"] >= gas_cutoff
    interval["RESERVOIR_FLAG"] = reservoir_flag.fillna(False)
    core_validation, core_metrics = logs.validate_core_measurements(core_source, interval)

    lumps = logs.lump_net_intervals(
        interval,
        interval["RESERVOIR_FLAG"],
        minimum_thickness=1.0,
    )
    candidates = services.find_nzbp_candidates(
        interval,
        interval["RESERVOIR_FLAG"],
        active_perfs,
        minimum_nzbp,
        bool(has_perforation_intervals),
    )
    sample_thickness = logs.estimate_sample_thickness(interval.index)
    gross = max(float(interval.index.max() - interval.index.min()), 0.0)
    net = float(interval["RESERVOIR_FLAG"].sum()) * sample_thickness
    ntg = net / gross if gross else np.nan

    core_files = services.supporting_files(selected_record, "core")
    fluid_files = services.supporting_files(selected_record, "fluid")
    mudlog_files = services.supporting_files(selected_record, "mudlog")
    candidate_count = int(
        (
            candidates.get("NZBP Status", pd.Series(dtype=str))
            == "NZBP SCREENING CANDIDATE"
        ).sum()
    )

    if workspace == "nzbp":
        nzbp_metrics = st.columns(5)
        nzbp_metrics[0].metric("Screening candidates", candidate_count)
        nzbp_metrics[1].metric("Known active perfs", len(active_perfs))
        nzbp_metrics[2].metric(
            "Candidate gross thickness",
            format_value(
                candidates.loc[
                    candidates.get("NZBP Status")
                    == "NZBP SCREENING CANDIDATE",
                    "Gross Thickness",
                ].sum()
                if not candidates.empty
                else 0,
                digits=1,
            ),
        )
        nzbp_metrics[3].metric(
            "Perforation evidence",
            "Available" if has_perforation_intervals else "Missing",
        )
        nzbp_metrics[4].metric("Formation-evaluation net", format_value(net, digits=1))
        if not has_perforation_intervals:
            st.warning(
                "No perforation history is available. Reservoir intervals are shown as "
                "unconfirmed and cannot be called behind-pipe candidates."
            )
        nzbp_tabs = st.tabs(
            ("Wellbore schematic", "Candidate intervals", "Interpretation basis")
        )
        with nzbp_tabs[0]:
            nzbp_figure = logs.make_nzbp_schematic(
                top_depth,
                base_depth,
                candidates,
                active_perfs,
                casing,
                sands,
            )
            st.pyplot(nzbp_figure, width="stretch")
        with nzbp_tabs[1]:
            st.dataframe(candidates, hide_index=True, width="stretch", height=500)
            if not candidates.empty:
                st.download_button(
                    "Download NZBP screening CSV",
                    candidates.to_csv(index=False).encode("utf-8-sig"),
                    f"{services.normalize_well(well_name)}_NZBP_screening.csv",
                    "text/csv",
                )
        with nzbp_tabs[2]:
            interpretation_basis = pd.DataFrame(
                [
                    ("Formation-evaluation basis", cutoff_mode),
                    ("Vsh cutoff", vsh_cutoff),
                    ("PhiE cutoff", phi_cutoff),
                    ("Sw cutoff", sw_cutoff),
                    ("Permeability cutoff (mD)", permeability_cutoff),
                    ("TVD basis", tvd_basis),
                    (
                        "Perforation intervals",
                        int(
                            history[["Perforation Top (ft)", "Perforation Base (ft)"]]
                            .notna()
                            .all(axis=1)
                            .sum()
                        )
                        if not history.empty
                        else 0,
                    ),
                    ("Casing strings", len(casing)),
                    ("Sand intervals", len(sands)),
                ],
                columns=["Input", "Value"],
            )
            interpretation_basis["Value"] = interpretation_basis["Value"].map(str)
            st.dataframe(
                interpretation_basis,
                hide_index=True,
                width="stretch",
            )
        st.caption(
            "NZBP is intentionally separated from petrophysical interpretation. It uses "
            "the formation-evaluation result as one input, then applies completion and "
            "perforation overlap logic. Engineering confirmation remains mandatory."
        )
        return

    interval.attrs["depth_unit"] = depth_unit or "LAS units"
    tabs = st.tabs(
        (
            "1. Data conditioning",
            "2-6. Formation evaluation",
            "Validation",
            "TVD / trajectory",
            "7. Cutoff and lumping",
            "Results and export",
        )
    )
    with tabs[0]:
        conditioning_metrics = st.columns(5)
        conditioning_metrics[0].metric("Merged runs", 1 + len(shift_rows))
        conditioning_metrics[1].metric(
            "Unrealistic samples removed", int(interval["UNREALISTIC_REMOVED"].sum())
        )
        conditioning_metrics[2].metric(
            "Badhole samples", int(interval["BADHOLE_FLAG"].sum())
        )
        conditioning_metrics[3].metric("Coal reference samples", int(coal_flag.sum()))
        conditioning_metrics[4].metric(
            "Shale reference samples", int(shale_reference_flag.sum())
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ("Depth matching", "Gamma-ray cross-correlation", len(shift_rows)),
                    ("Splicing", "Reference-first combine across aligned runs", len(raw)),
                    ("Badhole", "CALI > bit size + tolerance or |DRHO| > 0.15", int(qc["BADHOLE_FLAG"].sum())),
                    ("Coal", "Low RHOB and high NPHI configurable flag", int(coal_flag.sum())),
                    (
                        "Casing shoe",
                        f"Cased interval flagged above {casing_shoe_depth:g}",
                        int(interval["CASING_FLAG"].sum()),
                    ),
                    ("Unrealistic values", "Physical-range filters applied before calculation", int(qc["UNREALISTIC_REMOVED"].sum())),
                ],
                columns=["Conditioning Step", "Method", "Result"],
            ),
            hide_index=True,
            width="stretch",
        )
        if shift_rows:
            st.dataframe(pd.DataFrame(shift_rows), hide_index=True, width="stretch")

    with tabs[1]:
        try:
            figure = logs.make_formation_evaluation_figure(
                interval,
                applied_cutoffs,
                float(clean_gr),
                float(shale_gr),
            )
            st.pyplot(figure, width="stretch")
        except ValueError as exc:
            st.warning(str(exc))
            figure = None
        metrics = st.columns(6)
        metrics[0].metric("Gross interval", format_value(gross, digits=1))
        metrics[1].metric("Net reservoir", format_value(net, digits=1))
        metrics[2].metric("Net-to-gross", format_value(ntg, digits=3))
        metrics[3].metric(
            "Average PhiE",
            format_value(interval.loc[interval["RESERVOIR_FLAG"], "PHIE"].mean(), digits=3),
        )
        metrics[4].metric(
            "Average Sw",
            format_value(interval.loc[interval["RESERVOIR_FLAG"], "SW"].mean(), digits=3),
        )
        metrics[5].metric(
            "Average perm",
            format_value(interval.loc[interval["RESERVOIR_FLAG"], "PERM_MD"].mean(), " mD", 1),
        )
        st.info(
            f"Autonomous mineral basis: {mineral_model}, matrix density "
            f"{matrix_density:.2f} g/cc. {mineral_basis} Autonomous Rw basis: "
            f"{rw:.4f} ohm.m from {automatic_rw_samples} Pickett screening samples.",
            icon=":material/science:",
        )

    with tabs[2]:
        st.pyplot(logs.make_validation_dashboard(interval), width="stretch")
        crossplot = logs.make_crossplot(interval)
        pickett = logs.make_pickett_plot(interval, float(archie_a), float(archie_m))
        plot_columns = st.columns(2)
        with plot_columns[0]:
            if crossplot is not None:
                st.pyplot(crossplot, width="stretch")
            else:
                st.info("Density and neutron curves are required for the crossplot.")
        with plot_columns[1]:
            if pickett is not None:
                st.pyplot(pickett, width="stretch")
            else:
                st.info("Effective porosity and deep resistivity are required for Pickett screening.")
        correlations = []
        for validation_curve in ("RES", "DTC", "PEF"):
            valid = interval[["VSH", validation_curve]].dropna()
            correlations.append(
                (
                    validation_curve,
                    valid["VSH"].corr(valid[validation_curve])
                    if len(valid) >= 5
                    else np.nan,
                    len(valid),
                )
            )
        st.subheader("Validation evidence")
        validation_columns = st.columns(3)
        validation_columns[0].metric("Core/XRD/petrography files", len(core_files))
        validation_columns[1].metric("Fluid/Rw evidence files", len(fluid_files))
        validation_columns[2].metric("Mudlog/show files", len(mudlog_files))
        if not xrd_table.empty:
            st.subheader("Uploaded XRD / petrography interpretation")
            xrd_display, xrd_metric = st.columns([1.3, 0.7])
            with xrd_display:
                st.dataframe(xrd_table.round(3), hide_index=True, width="stretch")
            with xrd_metric:
                st.metric(
                    "XRD weighted matrix density",
                    f"{xrd_density:.3f} g/cc" if xrd_density is not None else "Unavailable",
                )
        if core_metrics:
            st.subheader("Uploaded core / SWC validation")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": list(core_metrics),
                        "Value": [
                            f"{value:.4g}" if isinstance(value, float) else value
                            for value in core_metrics.values()
                        ],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        if not core_validation.empty:
            st.dataframe(core_validation.round(4), hide_index=True, width="stretch")
        st.dataframe(
            pd.DataFrame(
                correlations,
                columns=["Vsh validation curve", "Correlation", "Paired samples"],
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("Core, fluid, and mudlog source files"):
            st.write("Core / mineralogy")
            st.dataframe(core_files, hide_index=True, width="stretch")
            st.write("Fluid / Rw")
            st.dataframe(fluid_files, hide_index=True, width="stretch")
            st.write("Mudlog / hydrocarbon shows")
            st.dataframe(mudlog_files, hide_index=True, width="stretch")
        st.caption(
            "Coates-Timur permeability uses log-derived bound water as a proxy when NMR "
            "is absent. Calibrate against core permeability. Modified Simandoux uses "
            "effective porosity, interval Rw, Rsh, and editable a/m/n parameters."
        )

    with tabs[3]:
        st.info(
            f"TVD basis: {tvd_basis} · Source: {tvd_metadata.get('source')} · "
            f"Coverage: {float(tvd_metadata.get('coverage', 0)):.1%} "
            f"({tvd_metadata.get('quality')})",
            icon=":material/straighten:",
        )
        trajectory_evidence = services.selected_trajectory_evidence(selected_record)
        st.dataframe(
            trajectory_evidence,
            hide_index=True,
            width="stretch",
            height=260,
        )
        if not trajectory.empty:
            trajectory_metrics = st.columns(4)
            trajectory_metrics[0].metric(
                "Maximum inclination",
                format_value(trajectory["Inclination"].max(), " deg", 1),
            )
            trajectory_metrics[1].metric(
                "Maximum departure",
                format_value(
                    trajectory.get(
                        "Horizontal Departure", pd.Series(dtype=float)
                    ).max(),
                    digits=1,
                ),
            )
            trajectory_metrics[2].metric(
                "Maximum DLS",
                format_value(
                    trajectory.get(
                        "DLS (deg/100ft)", pd.Series(dtype=float)
                    ).max(),
                    " deg/100ft",
                    2,
                ),
            )
            trajectory_metrics[3].metric(
                "Bottom-hole TVD", format_value(trajectory["TVD"].iloc[-1], digits=1)
            )
            st.pyplot(logs.make_trajectory_figure(trajectory), width="stretch")
            st.dataframe(trajectory, hide_index=True, width="stretch", height=360)
        else:
            st.info(
                "No numeric inclination/azimuth survey was parsed. A TVD-indexed LAS can "
                "still provide TVD depth, but it does not prove well trajectory."
            )

    with tabs[4]:
        cutoff_summary = pd.DataFrame(
            [
                ("Vsh maximum", vsh_cutoff, recommended_cutoffs["VSH"]),
                ("PhiE minimum", phi_cutoff, recommended_cutoffs["PHIE"]),
                ("Sw maximum", sw_cutoff, recommended_cutoffs["SW"]),
                (
                    "Permeability minimum (mD)",
                    permeability_cutoff,
                    recommended_cutoffs["PERM_MD"],
                ),
            ],
            columns=["Cutoff", "Applied", "Autonomous Recommendation"],
        )
        st.dataframe(cutoff_summary, hide_index=True, width="stretch")
        st.caption(
            f"Recommendation basis: {recommended_cutoffs['Basis']} "
            f"({recommended_cutoffs['Validation Samples']} samples)."
        )
        st.pyplot(
            logs.make_cutoff_analysis_figure(interval, applied_cutoffs),
            width="stretch",
        )
        lump_metrics = st.columns(4)
        lump_metrics[0].metric("Pay lumps", len(lumps))
        lump_metrics[1].metric("Net reservoir", format_value(net, digits=1))
        lump_metrics[2].metric("Net-to-gross", format_value(ntg, digits=3))
        lump_metrics[3].metric(
            "Gas validation",
            "Available" if interval["TOTAL_GAS"].notna().any() else "Not available",
        )
        st.dataframe(lumps, hide_index=True, width="stretch", height=420)
        st.subheader("Test / show evidence")
        evidence = st.data_editor(
            pd.DataFrame(
                columns=[
                    "Top MD",
                    "Base MD",
                    "Evidence type",
                    "Result / hydrocarbon show",
                    "Total gas ratio",
                    "Source / remarks",
                ]
            ),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"petro_evidence_{selected_record.well}",
        )
        st.caption(
            "This evidence is documented separately from automatic cutoffs; confirm test, show, and pressure evidence before an operational net-pay decision."
        )

    with tabs[5]:
        export = interval.copy()
        export.index.name = f"DEPTH_{depth_unit or 'LAS'}"
        st.dataframe(export.head(800), width="stretch")
        st.download_button(
            "Download interpreted interval CSV",
            export.to_csv().encode("utf-8"),
            f"{services.normalize_well(well_name)}_interpreted_interval.csv",
            "text/csv",
        )
        st.download_button(
            "Download pay lumps CSV",
            lumps.to_csv(index=False).encode("utf-8-sig"),
            f"{services.normalize_well(well_name)}_pay_lumps.csv",
            "text/csv",
        )
        equation_audit = pd.DataFrame(
            [
                ("Vsh", vsh_method, f"GRclean={clean_gr:.2f}; GRshale={shale_gr:.2f}"),
                ("Porosity", "Density + neutron-constrained bound water", f"rho_ma={matrix_density:.3f}; rho_fl={fluid_density:.3f}"),
                ("Permeability", "Timur-Coates proxy", f"A={perm_coefficient:g}; B={perm_phi_exp:.2f}; C={perm_ratio_exp:.2f}; Swirr={swirr_floor:.2f}"),
                ("Water saturation", "Modified Simandoux", f"Rw={rw:.4f}; Rsh={rsh:.2f}; a={archie_a:.2f}; m={archie_m:.2f}; n={archie_n:.2f}"),
                ("Net pay", "Vsh + PhiE + Sw + permeability + environmental QC", f"Vsh<={vsh_cutoff:.2f}; PhiE>={phi_cutoff:.2f}; Sw<={sw_cutoff:.2f}; K>={permeability_cutoff:g}"),
            ],
            columns=["Calculation", "Method", "Inputs"],
        )
        workbook = logs.dataframe_xlsx(
            {
                "Computed logs": export,
                "Net lumps": lumps,
                "Run alignment": pd.DataFrame(shift_rows),
                "Water zones": rw_intervals,
                "Evidence": evidence,
                "Core validation": core_validation,
                "XRD mineralogy": xrd_table,
                "Equation audit": equation_audit,
            }
        )
        st.download_button(
            "Download auditable formation-evaluation workbook",
            workbook,
            f"{services.normalize_well(well_name)}_formation_evaluation.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption(
            f"Persistent result root: {services.FORMATION_EVALUATION_OUTPUT}"
        )
        if st.button(
            "Save autonomous formation-evaluation package",
            type="primary",
            key=f"save_fe_{selected_record.well}",
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (
                services.FORMATION_EVALUATION_OUTPUT
                / services.normalize_well(field_name)
                / services.normalize_well(well_name)
                / timestamp
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            export.to_csv(
                output_dir / f"{services.normalize_well(well_name)}_interpreted.csv"
            )
            lumps.to_csv(
                output_dir / f"{services.normalize_well(well_name)}_pay_lumps.csv",
                index=False,
            )
            cutoff_summary.to_csv(
                output_dir / f"{services.normalize_well(well_name)}_cutoffs.csv",
                index=False,
            )
            (output_dir / "evaluation_basis.json").write_text(
                json.dumps(
                    {
                        "field": field_name,
                        "well": well_name,
                        "source_log": str(selected_path or "uploaded"),
                        "tvd": tvd_metadata,
                        "matrix_basis": mineral_basis,
                        "matrix_density": matrix_density,
                        "rw": rw,
                        "cutoffs": applied_cutoffs,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            if figure is not None:
                figure.savefig(
                    output_dir
                    / f"{services.normalize_well(well_name)}_formation_evaluation.png",
                    dpi=180,
                    bbox_inches="tight",
                )
            st.success(f"Saved formation-evaluation package: {output_dir}")
        if figure is not None:
            summary = pd.DataFrame(
                [
                    ("Field", field_name),
                    ("Well", well_name),
                    ("Interval top", top_depth),
                    ("Interval base", base_depth),
                    ("Gross interval", gross),
                    ("Net reservoir", net),
                    ("Net-to-gross", ntg),
                    ("TVD basis", tvd_basis),
                    ("TVD source", tvd_metadata.get("source")),
                    ("Pay lumps", len(lumps)),
                    ("Modified Simandoux Rsh", rsh),
                    ("Core validation files", len(core_files)),
                ],
                columns=["Metric", "Value"],
            )
            report = logs.build_pdf(
                f"{field_name} / {well_name}",
                summary,
                figure,
                logs.make_crossplot(interval),
            )
            st.download_button(
                "Download formation-evaluation PDF",
                report,
                f"{services.normalize_well(well_name)}_formation_evaluation.pdf",
                "application/pdf",
            )


def uploaded_scan_path(uploaded, selected_record: logs.WellRecord) -> Path:
    upload_root = (
        services.DIGITIZATION_OUTPUT
        / services.normalize_well(selected_record.field)
        / services.normalize_well(selected_record.well)
        / "uploads"
    )
    upload_root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", uploaded.name)
    path = upload_root / safe_name
    path.write_bytes(uploaded.getvalue())
    return path


def calibration_value(
    calibration: dict[str, Any], key: str, fallback: Any
) -> Any:
    value = calibration.get(key, fallback)
    return fallback if value is None else value


def render_digitizer(selected_record: logs.WellRecord) -> None:
    hero(
        "Scanned Well Log Digitizer",
        "Run the existing curve-tracing engine from the dashboard. Calibration is stored "
        "per field, well, and curve so each well can be adjusted independently.",
        "Digitization and Training",
    )
    st.info(
        f"Persistent digitization root: {services.MINAS_TRAINING_ROOT}. "
        f"New dashboard runs are stored under {services.DIGITIZATION_OUTPUT}.",
        icon=":material/folder_data:",
    )
    report = services.load_training_report()
    if report:
        with st.expander("MINAS training diagnostics", expanded=False):
            st.dataframe(
                pd.DataFrame(report).drop(columns=["outputs", "evaluation"], errors="ignore"),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Current evidence: MINA00526_GR completed with image QA REVIEW and "
                "evaluation grade IMPROVE; the other five configured MINAS jobs remain skipped."
            )

    saved_artifacts = services.collect_digitizer_artifacts(
        selected_record.field,
        selected_record.well,
    )
    with st.expander(
        f"Saved digitization results ({len(saved_artifacts)} files)",
        expanded=not saved_artifacts.empty,
    ):
        if saved_artifacts.empty:
            st.info("No saved digitization result is available for this well.")
        else:
            st.dataframe(
                saved_artifacts[
                    ["File", "Type", "Size (KB)", "Modified", "Result Root", "Path"]
                ],
                hide_index=True,
                width="stretch",
                height=320,
            )
            preview_rows = saved_artifacts.loc[
                saved_artifacts["File"].str.contains(
                    r"(?i)(?:QA|preview).*\.(?:png|jpg|jpeg)$", regex=True
                )
            ]
            if not preview_rows.empty:
                preview_path = Path(preview_rows.iloc[0]["Path"])
                if preview_path.exists():
                    st.image(
                        str(preview_path),
                        caption=f"Latest saved QA/preview: {preview_path.name}",
                        width="stretch",
                    )

    existing_scans = services.collect_scanned_logs(selected_record.path)
    input_mode = st.radio(
        "Scanned-log source", ("Canonical well files", "Upload scan"), horizontal=True
    )
    uploaded = None
    source_path: Path | None = None
    if input_mode == "Canonical well files":
        if not existing_scans:
            st.warning("No supported scanned-log file was found for this well.")
            return
        source_path = st.selectbox(
            "Scanned log",
            existing_scans,
            format_func=lambda path: f"{path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)",
        )
    else:
        uploaded = st.file_uploader(
            "Upload PDF/TIFF/PNG/JPEG", type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"]
        )
        if uploaded is None:
            st.info("Upload a scan to continue.")
            return
        source_path = uploaded_scan_path(uploaded, selected_record)

    profiles = services.available_digitizer_profiles()
    exact_profiles = {
        key: value
        for key, value in profiles.items()
        if value.get("well") == services.normalize_well(selected_record.well)
    }
    profile_options = ["Manual calibration"]
    profile_lookup: dict[str, str] = {}
    for key, profile in exact_profiles.items():
        label = f"{profile['label']} | {profile['source']}"
        profile_options.append(label)
        profile_lookup[label] = key
    allow_inherited = st.checkbox(
        "Show other-well profiles as starting points",
        value=False,
        help="Inherited geometry is never treated as trained for the selected well.",
    )
    if allow_inherited:
        for key, profile in profiles.items():
            if key in exact_profiles:
                continue
            label = f"INHERITED START: {profile['label']} | {profile['source']}"
            profile_options.append(label)
            profile_lookup[label] = key
    profile_label = st.selectbox("Calibration profile", profile_options)
    chosen_profile = profiles.get(profile_lookup.get(profile_label, ""), {})
    base = dict(chosen_profile.get("calibration", {}))
    inherited = profile_label.startswith("INHERITED")
    if inherited:
        st.warning(
            "This profile only pre-fills the form. Recalibrate track bounds and depth axes "
            "for this well before running."
        )

    page_settings = st.columns(3)
    page = page_settings[0].number_input("Page/frame", min_value=1, value=1, step=1)
    dpi = page_settings[1].number_input("Run DPI", min_value=50, max_value=400, value=150)
    curve = page_settings[2].text_input(
        "Curve mnemonic", value=str(calibration_value(base, "curve", "GR"))
    ).upper()

    with st.form("digitizer_calibration_form"):
        axis_columns = st.columns(4)
        unit = axis_columns[0].text_input(
            "Curve unit", value=str(calibration_value(base, "unit", "GAPI"))
        )
        depth_unit = axis_columns[1].text_input(
            "Depth unit", value=str(calibration_value(base, "depth_unit", "F"))
        )
        depth_top = axis_columns[2].number_input(
            "Depth top", value=float(calibration_value(base, "depth_top", 0.0))
        )
        depth_bottom = axis_columns[3].number_input(
            "Depth bottom", value=float(calibration_value(base, "depth_bottom", 3000.0))
        )
        crop_columns = st.columns(4)
        x_left = crop_columns[0].number_input(
            "Track left (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=float(calibration_value(base, "x_left", 0.05)),
            format="%.5f",
        )
        x_right = crop_columns[1].number_input(
            "Track right (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=float(calibration_value(base, "x_right", 0.35)),
            format="%.5f",
        )
        y_top = crop_columns[2].number_input(
            "Track top (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=float(calibration_value(base, "y_top", 0.08)),
            format="%.5f",
        )
        y_bottom = crop_columns[3].number_input(
            "Track bottom (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=float(calibration_value(base, "y_bottom", 0.98)),
            format="%.5f",
        )
        value_columns = st.columns(5)
        value_min = value_columns[0].number_input(
            "Value minimum", value=float(calibration_value(base, "value_min", 0.0))
        )
        value_max = value_columns[1].number_input(
            "Value maximum", value=float(calibration_value(base, "value_max", 150.0))
        )
        depth_step = value_columns[2].number_input(
            "Depth step",
            min_value=0.01,
            value=float(calibration_value(base, "depth_step", 0.5)),
        )
        scale = value_columns[3].selectbox(
            "X scale",
            ("linear", "log"),
            index=0 if calibration_value(base, "scale", "linear") == "linear" else 1,
        )
        mode = value_columns[4].selectbox(
            "Trace mode",
            ("dark", "color"),
            index=0 if calibration_value(base, "mode", "dark") == "dark" else 1,
        )
        trace_columns = st.columns(4)
        threshold = trace_columns[0].number_input(
            "Dark threshold", value=float(calibration_value(base, "threshold", 145.0))
        )
        target_rgb = trace_columns[1].text_input(
            "Target RGB", value=",".join(map(str, calibration_value(base, "target_rgb", [0, 0, 0])))
        )
        max_jump = trace_columns[2].number_input(
            "Maximum pixel jump",
            min_value=1,
            value=int(calibration_value(base, "max_jump", 30)),
        )
        smooth_rows = trace_columns[3].number_input(
            "Smoothing rows",
            min_value=1,
            value=int(calibration_value(base, "smooth_rows", 9)),
        )
        save_profile = st.form_submit_button("Save calibration for this well")

    try:
        rgb = [int(value.strip()) for value in target_rgb.split(",")]
        if len(rgb) != 3:
            raise ValueError
    except ValueError:
        st.error("Target RGB must contain three comma-separated integers.")
        return

    calibration_mapping = {
        **base,
        "curve": curve,
        "unit": unit,
        "depth_unit": depth_unit,
        "depth_top": depth_top,
        "depth_bottom": depth_bottom,
        "depth_step": depth_step,
        "x_left": x_left,
        "x_right": x_right,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "value_min": value_min,
        "value_max": value_max,
        "scale": scale,
        "mode": mode,
        "threshold": threshold,
        "target_rgb": rgb,
        "max_jump": int(max_jump),
        "smooth_rows": int(smooth_rows),
    }
    if save_profile:
        try:
            digitizer = services.load_digitizer_module()
            digitizer.calibration_from_mapping(calibration_mapping)
            path = services.save_digitizer_profile(
                selected_record.field,
                selected_record.well,
                curve,
                calibration_mapping,
            )
            st.success(f"Saved well-specific calibration: {path}")
        except Exception as exc:
            st.error(f"Calibration could not be saved: {exc}")

    action_columns = st.columns(2)
    if action_columns[0].button("Preview page and calibrated track", width="stretch"):
        try:
            digitizer = services.load_digitizer_module()
            preview = digitizer.load_raster(source_path, int(page), min(int(dpi), 60))
            image = Image.fromarray(preview)
            draw = ImageDraw.Draw(image)
            width, height = image.size
            draw.rectangle(
                (
                    int(x_left * (width - 1)),
                    int(y_top * (height - 1)),
                    int(x_right * (width - 1)),
                    int(y_bottom * (height - 1)),
                ),
                outline=(240, 180, 41),
                width=max(2, width // 500),
            )
            st.image(image, caption="Calibrated track preview", width="stretch")
        except Exception as exc:
            st.error(f"Preview failed: {exc}")

    if action_columns[1].button(
        "Run digitization", type="primary", width="stretch"
    ):
        try:
            digitizer = services.load_digitizer_module()
            calibration = digitizer.calibration_from_mapping(calibration_mapping)
            output_dir = (
                services.DIGITIZATION_OUTPUT
                / services.normalize_well(selected_record.field)
                / services.normalize_well(selected_record.well)
                / f"{services.normalize_well(selected_record.well)}_{curve}"
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with st.spinner("Tracing the calibrated curve and building QA outputs..."):
                result = digitizer.digitize_one(
                    input_path=source_path,
                    output_dir=output_dir,
                    calibration=calibration,
                    page=int(page),
                    dpi=int(dpi),
                    zona_root=services.DATA_ROOT,
                    output_stem=f"{services.normalize_well(selected_record.well)}_{curve}_{timestamp}",
                )
            st.session_state["digitizer_result"] = result
            st.success(f"Digitization package saved under: {output_dir}")
        except Exception as exc:
            st.error(f"Digitization failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("digitizer_result")
    if result and services.normalize_well(result.get("well")) == services.normalize_well(
        selected_record.well
    ):
        st.subheader("Latest digitization result")
        metrics = st.columns(5)
        metrics[0].metric("QA status", result.get("qa_status", "N/A"))
        metrics[1].metric(
            "Trace coverage", format_value(result.get("trace_coverage", 0) * 100, "%")
        )
        metrics[2].metric(
            "Valid output", format_value(result.get("valid_output_fraction", 0) * 100, "%")
        )
        metrics[3].metric(
            "Mean confidence", format_value(result.get("mean_output_confidence", 0), digits=3)
        )
        metrics[4].metric("Samples", result.get("samples", 0))
        outputs = result.get("outputs", {})
        qa_path = Path(outputs.get("qa_overlay", ""))
        csv_path = Path(outputs.get("csv", ""))
        las_path = Path(outputs.get("las", ""))
        metadata_path = Path(outputs.get("metadata", ""))
        if qa_path.exists():
            st.image(str(qa_path), caption="Digitizer QA overlay", width="stretch")
        if csv_path.exists():
            st.dataframe(pd.read_csv(csv_path).head(800), width="stretch")
        downloads = st.columns(3)
        if csv_path.exists():
            downloads[0].download_button(
                "Download CSV", csv_path.read_bytes(), csv_path.name, "text/csv"
            )
        if las_path.exists():
            downloads[1].download_button(
                "Download LAS", las_path.read_bytes(), las_path.name, "text/plain"
            )
        if metadata_path.exists():
            downloads[2].download_button(
                "Download metadata",
                metadata_path.read_bytes(),
                metadata_path.name,
                "application/json",
            )


def render_schematic(selected_record: logs.WellRecord) -> None:
    hero(
        "Well schematic and completion diagram",
        "A clean measured-depth schematic rebuilt directly from the selected worksheet, "
        "with casing, completion intervals, embedded source imagery, and trajectory proof.",
        "Completions / workover",
    )
    source_caption(services.SCHEMATIC_WORKBOOK)
    payload = services.selected_schematic_workbook(selected_record.well)
    row = services.selected_schematic_row(selected_record.well)
    if not row or not payload:
        summary = services.load_schematic_summary()
        st.warning(
            f"No worksheet in well schematic Rokan.xlsx matches {selected_record.well}. "
            "The dashboard will not substitute another well's diagram.",
            icon=":material/data_alert:",
        )
        coverage = st.columns(3)
        coverage[0].metric("Workbook well sheets", len(summary), border=True)
        coverage[1].metric(
            "Sheets with embedded image",
            int(summary.get("Image Available", pd.Series(dtype=bool)).fillna(False).sum()),
            border=True,
        )
        coverage[2].metric("Selected-well match", "Missing", border=True)
        return

    casing = payload.get("casing", pd.DataFrame())
    intervals = payload.get("intervals", pd.DataFrame())
    source_caption(
        services.SCHEMATIC_WORKBOOK,
        str(payload.get("sheet_name", "")),
    )
    metrics = st.columns(4)
    metrics[0].metric("Source sheet", str(payload.get("sheet_name", "")), border=True)
    metrics[1].metric(
        "Orientation",
        {"D": "Directional", "H": "Horizontal", "V": "Vertical"}.get(
            str(payload.get("orientation", "")), "Workbook-defined"
        ),
        border=True,
    )
    metrics[2].metric("Casing rows", len(casing), border=True)
    metrics[3].metric("Completion intervals", len(intervals), border=True)

    diagram_column, image_column = st.columns([1.65, 1])
    with diagram_column:
        with st.container(border=True):
            st.subheader("Measured-depth completion schematic")
            figure = make_workbook_schematic_figure(
                selected_record.well,
                str(payload.get("orientation", "")),
                casing,
                intervals,
            )
            st.pyplot(figure, width="stretch")
            plt.close(figure)
    with image_column:
        with st.container(border=True):
            st.subheader("Embedded workbook image")
            image_bytes = payload.get("image_bytes", b"")
            if image_bytes:
                st.image(
                    image_bytes,
                    caption=(
                        f"Largest embedded image from {payload.get('sheet_name')} "
                        f"({payload.get('image_format', 'image').upper()})"
                    ),
                    width="stretch",
                )
            else:
                st.info("No supported embedded image was found on this worksheet.")
            st.caption(
                "The engineering schematic is plotted from workbook casing depths and "
                "completion intervals. The embedded source image is shown separately so "
                "worksheet graphics are not mistaken for depth-scaled geometry."
            )

    trajectory_evidence = services.selected_trajectory_evidence(selected_record)
    if not trajectory_evidence.empty:
        evidence_level = trajectory_evidence.iloc[0].get("Evidence Level", "")
        st.info(
            f"Trajectory evidence: {evidence_level}. Open the trajectory evidence tab "
            "for source proof and minimum-curvature survey plotting.",
            icon=":material/route:",
        )

    tabs = st.tabs(
        (
            "Casing details",
            "Completion intervals",
            "Source audit",
            "Trajectory evidence",
        )
    )
    with tabs[0]:
        if casing.empty:
            st.info("No casing row was parsed from the selected worksheet.")
        else:
            st.dataframe(
                services.clean_display_frame(casing),
                hide_index=True,
                width="stretch",
                height=470,
            )
            st.download_button(
                "Download casing details",
                casing.to_csv(index=False).encode("utf-8-sig"),
                f"{selected_record.well}_Workbook_Casing.csv",
                "text/csv",
                icon=":material/download:",
            )
    with tabs[1]:
        if intervals.empty:
            st.info("No completion interval was parsed from the selected worksheet.")
        else:
            st.dataframe(
                services.clean_display_frame(intervals),
                hide_index=True,
                width="stretch",
                height=470,
            )
            st.download_button(
                "Download completion intervals",
                intervals.to_csv(index=False).encode("utf-8-sig"),
                f"{selected_record.well}_Workbook_Completion_Intervals.csv",
                "text/csv",
                icon=":material/download:",
            )
    with tabs[2]:
        _, audit = services.selected_schematic_detail(selected_record.well)
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption(
            "Worksheet names are reconciled to canonical folders by field prefix and "
            "numeric well identity. Directional/horizontal suffixes are used to resolve "
            "duplicate sheet variants."
        )
    with tabs[3]:
        if trajectory_evidence.empty:
            st.info(
                "No TVD, HLS, deviation, inclination, azimuth, or trajectory-named "
                "source file was found for this well."
            )
        else:
            st.dataframe(
                trajectory_evidence,
                hide_index=True,
                width="stretch",
                height=320,
            )
        survey_upload = st.file_uploader(
            "Load deviation survey proof (CSV/XLSX: MD, inclination, azimuth)",
            type=["csv", "xlsx", "xls"],
            key=f"schematic_survey_{selected_record.well}",
        )
        if survey_upload is not None:
            try:
                source = (
                    pd.read_excel(survey_upload)
                    if Path(survey_upload.name).suffix.lower() in {".xlsx", ".xls"}
                    else pd.read_csv(survey_upload)
                )
                survey = logs.minimum_curvature_survey(source)
                metrics = st.columns(4)
                metrics[0].metric(
                    "Maximum inclination",
                    format_value(survey["Inclination"].max(), " deg", 1),
                )
                metrics[1].metric(
                    "Maximum departure",
                    format_value(survey["Horizontal Departure"].max(), digits=1),
                )
                metrics[2].metric(
                    "Bottom-hole TVD",
                    format_value(survey["TVD"].iloc[-1], digits=1),
                )
                metrics[3].metric(
                    "Maximum DLS",
                    format_value(survey["DLS (deg/100ft)"].max(), " deg/100ft", 2),
                )
                st.pyplot(logs.make_trajectory_figure(survey), width="stretch")
                st.dataframe(survey, hide_index=True, width="stretch", height=380)
            except Exception as exc:
                st.error(f"Deviation survey could not be calculated: {exc}")


def render_perforation(selected_record: logs.WellRecord) -> None:
    hero(
        "Perforation history and NZBP interval review",
        "Workbook-controlled OPEN, CLOSED, and NZBP intervals with formation context, "
        "source-row traceability, active completion status, and depth visualization.",
        "Completions / workover",
    )
    source_caption(
        services.PERFORATION_WORKBOOK,
        "Existing perfo + NZBP",
    )
    interval_history = load_perforation_history(selected_record.well)
    coverage = services.load_perforation_coverage()
    coverage_row = coverage.loc[
        coverage.get("Well", pd.Series(dtype=object)).map(
            lambda value: services.well_matches(value, selected_record.well)
        )
    ]

    if interval_history.empty:
        status = (
            coverage_row.iloc[0].get("Coverage Status", "NO RECORD")
            if not coverage_row.empty
            else "NO COVERAGE ROW"
        )
        st.warning(
            f"No interval row matches {selected_record.well} in the controlled workbook. "
            f"Coverage status: {status}.",
            icon=":material/data_alert:",
        )
        source_wells = coverage.get("Well", pd.Series(dtype=object)).nunique()
        summary = st.columns(3)
        summary[0].metric("Workbook wells", source_wells, border=True)
        summary[1].metric(
            "Workbook intervals",
            int(pd.to_numeric(coverage.get("Perforation Events"), errors="coerce").sum()),
            border=True,
        )
        summary[2].metric("Selected-well intervals", 0, border=True)
        return

    valid_interval_mask = interval_history[
        ["Perforation Top (ft)", "Perforation Base (ft)"]
    ].notna().all(axis=1)
    if not valid_interval_mask.any():
        st.info(
            f"{selected_record.well} is listed in the workbook, but no perforation or "
            "NZBP interval is populated for that well.",
            icon=":material/info:",
        )
        summary = st.columns(3)
        summary[0].metric("Workbook source rows", len(interval_history), border=True)
        summary[1].metric("Parsed intervals", 0, border=True)
        summary[2].metric("Active intervals", 0, border=True)
        st.dataframe(
            services.clean_display_frame(interval_history),
            hide_index=True,
            width="stretch",
        )
        return

    active = services.active_perforation_intervals(interval_history)
    actions_upper = interval_history["Action Status"].fillna("").astype(str).str.upper()
    qa_values = interval_history.get(
        "QA Status", pd.Series("", index=interval_history.index)
    ).fillna("").astype(str).str.upper()
    qa_review = qa_values.str.startswith("REVIEW")
    net_pay = pd.to_numeric(
        interval_history.get("Net Pay (MD-ft)", pd.Series(dtype=float)), errors="coerce"
    ).sum()
    metrics = st.columns(5)
    metrics[0].metric("Interval records", int(valid_interval_mask.sum()), border=True)
    metrics[1].metric("Open", int(actions_upper.eq("OPEN").sum()), border=True)
    metrics[2].metric("Closed", int(actions_upper.eq("CLOSE").sum()), border=True)
    metrics[3].metric("NZBP", int(actions_upper.eq("NZBP").sum()), border=True)
    metrics[4].metric("Net pay (MD-ft)", f"{net_pay:,.1f}", border=True)
    if qa_review.any():
        st.warning(
            f"{int(qa_review.sum())} source interval(s) require QA review. "
            "Original text and source-row numbers are preserved below.",
            icon=":material/rule:",
        )
    if active:
        st.success(
            "Current OPEN intervals: "
            + ", ".join(f"{top:g}-{base:g} ft" for top, base in active),
            icon=":material/check_circle:",
        )

    chart_column, status_column = st.columns([1.55, 1])
    with chart_column:
        with st.container(border=True):
            st.subheader("Depth interval view")
            st.altair_chart(perforation_interval_chart(interval_history))
    with status_column:
        with st.container(border=True):
            st.subheader("Current completion status")
            status_summary = (
                interval_history.get("Status", pd.Series(dtype=object))
                .fillna("Unspecified")
                .astype(str)
                .value_counts()
                .rename_axis("Workbook status")
                .reset_index(name="Intervals")
            )
            st.dataframe(status_summary, hide_index=True, width="stretch")
            if active:
                active_table = pd.DataFrame(
                    [
                        {
                            "Active Top (ft)": top,
                            "Active Base (ft)": base,
                            "Thickness (ft)": base - top,
                        }
                        for top, base in active
                    ]
                )
                st.dataframe(active_table, hide_index=True, width="stretch", height=280)
            else:
                st.info("No interval is marked Open in the workbook.")

    history_tab, open_tab, nzbp_tab, source_tab = st.tabs(
        (
            "All interval records",
            "Open intervals",
            "NZBP intervals",
            "Source evidence and QA",
        )
    )
    with history_tab:
        display = safe_columns(
            services.clean_display_frame(interval_history),
            [
                "Well",
                "Well Type",
                "Formation",
                "Sand",
                "Sand Interval",
                "Net Pay (MD-ft)",
                "Net Pay (TVD-ft)",
                "Status",
                "Priority Status",
                "%WC",
                "Date",
                "Noted",
                "Recommendation Solution",
                "QA Status",
            ],
        )
        st.dataframe(display, hide_index=True, width="stretch", height=540)
        st.download_button(
            "Download selected-well interval review",
            interval_history.to_csv(index=False).encode("utf-8-sig"),
            f"{selected_record.well}_Perforation_NZBP_Interval_Review.csv",
            "text/csv",
            icon=":material/download:",
        )
    with open_tab:
        open_rows = interval_history.loc[actions_upper.eq("OPEN")]
        if open_rows.empty:
            st.info("No source row is marked Open for the selected well.")
        else:
            st.dataframe(
                services.clean_display_frame(open_rows),
                hide_index=True,
                width="stretch",
                height=500,
            )
        st.caption(
            "The source workbook is an interval-state review. Rows marked Open are used "
            "as the active-perforation input for NZBP overlap screening."
        )
    with nzbp_tab:
        nzbp_rows = interval_history.loc[actions_upper.eq("NZBP")]
        if nzbp_rows.empty:
            st.info("No source row is marked NZBP for the selected well.")
        else:
            st.dataframe(
                services.clean_display_frame(nzbp_rows),
                hide_index=True,
                width="stretch",
                height=500,
            )
    with source_tab:
        st.dataframe(
            safe_columns(
                services.clean_display_frame(interval_history),
                [
                    "Event Date",
                    "Action Status",
                    "Top Formation",
                    "Sand",
                    "Perforation Top (ft)",
                    "Perforation Base (ft)",
                    "QA Status",
                    "Source Row",
                    "Source File",
                    "Source Path",
                    "Source Sheet",
                    "Evidence",
                ],
            ),
            hide_index=True,
            width="stretch",
            height=540,
        )


def requirement_status(condition: bool, evidence: str) -> tuple[str, str]:
    return ("AVAILABLE" if condition else "GAP", evidence)


def render_screening_matrix(
    master: pd.DataFrame,
    selected_record: logs.WellRecord,
) -> None:
    hero(
        "Well screening matrix",
        "The controlled 101-well weighted screening result, including tier placement, "
        "knockout gates, the selected-well score profile, source recommendations, and "
        "the governing rubric and parameter weights.",
        "Well prioritization",
    )
    source_caption(
        services.SCREENING_WORKBOOK,
        "01_WellRegister and Scoring",
    )
    register, tiers, rubric, weights = services.load_screening_matrix_tables()
    if register.empty:
        st.error(
            "The controlled well-screening workbook is unavailable or could not be read.",
            icon=":material/error:",
        )
        return

    canonical_lookup: dict[tuple[str, int], tuple[str, str]] = {}
    for _, master_row in master.iterrows():
        signature = services.well_signature(
            master_row.get("Well"),
            master_row.get("Field"),
        )
        if signature is not None:
            canonical_lookup[signature] = (
                str(master_row.get("Well", "")),
                str(master_row.get("Field", "")),
            )

    matrix = register.copy()
    canonical_wells: list[str] = []
    canonical_fields: list[str] = []
    for _, source_row in matrix.iterrows():
        signature = services.well_signature(
            source_row.get("Well Name"),
            source_row.get("Field"),
        )
        match = canonical_lookup.get(signature)
        canonical_wells.append(match[0] if match else "")
        canonical_fields.append(match[1] if match else "Workbook only")
    matrix["Canonical Well"] = canonical_wells
    matrix["Canonical Field"] = canonical_fields
    matrix["Coverage"] = np.where(
        matrix["Canonical Well"].astype(str).str.strip().ne(""),
        "Canonical match",
        "Workbook only",
    )
    matrix["Display Well"] = np.where(
        matrix["Canonical Well"].astype(str).str.strip().ne(""),
        matrix["Canonical Well"],
        matrix["Well Name"],
    )
    ko_columns = ["HSE KO", "Integrity KO", "Facility KO"]
    matrix["Knockout Status"] = np.where(
        matrix[ko_columns]
        .fillna("")
        .astype(str)
        .apply(lambda column: column.str.upper().eq("YES"))
        .any(axis=1),
        "Any knockout",
        "No knockout",
    )
    matrix["Weighted Score"] = pd.to_numeric(
        matrix["Weighted Score"], errors="coerce"
    )

    workbook_only = matrix.loc[matrix["Coverage"].eq("Workbook only"), "Well Name"]
    with st.container(horizontal=True):
        st.metric("Workbook wells", len(matrix), border=True)
        st.metric(
            "Canonical matches",
            int(matrix["Coverage"].eq("Canonical match").sum()),
            border=True,
        )
        st.metric(
            "Tier 2",
            int(matrix["Tier"].eq("Tier 2").sum()),
            border=True,
        )
        st.metric(
            "Tier 3",
            int(matrix["Tier"].eq("Tier 3").sum()),
            border=True,
        )
        st.metric(
            "Excluded / defer",
            int(matrix["Tier"].isin(["Excluded", "Defer / Abandon"]).sum()),
            border=True,
        )

    if not workbook_only.empty:
        st.info(
            "Workbook-only coverage is preserved for audit: "
            f"{', '.join(workbook_only.astype(str))}. It is not silently assigned to a "
            "different canonical folder.",
            icon=":material/info:",
        )
    st.caption(
        "Workbook Weighted Score is the saved source result. It is intentionally kept "
        "separate from the dashboard's independent three-pillar Opportunity Score."
    )

    tier_values = set(matrix["Tier"].dropna().astype(str))
    tier_options = [tier for tier in SCREENING_TIER_ORDER if tier in tier_values]
    tier_options.extend(sorted(tier_values.difference(tier_options)))
    field_options = sorted(matrix["Canonical Field"].dropna().astype(str).unique())
    with st.expander("Matrix filters", expanded=True):
        filter_columns = st.columns((1.15, 1.4, 1.35, 1.0))
        selected_tiers = filter_columns[0].multiselect(
            "Screening tiers",
            tier_options,
            default=tier_options,
            key="screening_matrix_tiers",
        )
        selected_fields = filter_columns[1].multiselect(
            "Canonical fields (blank = all)",
            field_options,
            default=[],
            key="screening_matrix_fields",
        )
        query = filter_columns[2].text_input(
            "Search well, field, or recommendation",
            key="screening_matrix_search",
        )
        knockout_filter = filter_columns[3].selectbox(
            "Knockout status",
            ["All", "No knockout", "Any knockout"],
            key="screening_matrix_knockout",
        )

    filtered = matrix.loc[matrix["Tier"].isin(selected_tiers)].copy()
    if selected_fields:
        filtered = filtered.loc[filtered["Canonical Field"].isin(selected_fields)]
    if knockout_filter != "All":
        filtered = filtered.loc[filtered["Knockout Status"].eq(knockout_filter)]
    if query.strip():
        search_columns = [
            "Well Name",
            "Canonical Well",
            "Field",
            "Canonical Field",
            "Recommendation",
            "Tier",
        ]
        searchable = filtered[search_columns].fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered.loc[
            searchable.str.contains(query.strip(), case=False, regex=False)
        ]

    chart_left, chart_right = st.columns(2)
    with chart_left:
        with st.container(border=True):
            st.subheader("Tier distribution")
            if filtered.empty:
                st.info("No wells match the current filters.")
            else:
                st.altair_chart(screening_tier_chart(filtered))
    with chart_right:
        with st.container(border=True):
            st.subheader("Highest workbook scores")
            if filtered.empty:
                st.info("No wells match the current filters.")
            else:
                top_scores = (
                    filtered.dropna(subset=["Weighted Score"])
                    .nlargest(20, "Weighted Score")
                    .sort_values("Weighted Score")
                )
                score_chart = (
                    alt.Chart(top_scores)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X(
                            "Weighted Score:Q",
                            title="Weighted score (0-100)",
                            scale=alt.Scale(domain=[0, 100]),
                        ),
                        y=alt.Y("Display Well:N", title=None, sort=None),
                        color=alt.Color(
                            "Tier:N",
                            title="Tier",
                            scale=alt.Scale(
                                domain=SCREENING_TIER_ORDER,
                                range=SCREENING_TIER_COLORS,
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Display Well:N", title="Well"),
                            alt.Tooltip("Weighted Score:Q", format=".1f"),
                            alt.Tooltip("Tier:N"),
                            alt.Tooltip("Recommendation:N"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(score_chart)

    portfolio_tab, selected_tab, tiers_tab, rubric_tab, weights_tab = st.tabs(
        (
            "Portfolio matrix",
            "Selected well",
            "Tier definitions",
            "Scoring rubric",
            "Parameter weights",
        )
    )
    with portfolio_tab:
        st.subheader(f"Screening portfolio ({len(filtered)} wells)")
        portfolio_columns = [
            "Canonical Field",
            "Canonical Well",
            "Well Name",
            "Weighted Score",
            "Tier",
            "Knockout Status",
            "Existing + NZBP Reserve (MBO)",
            "Last Oil Rate (BOPD)",
            "Last Water Cut (%)",
            "Idle Years",
            "Trajectory",
            "Recommendation",
            "Coverage",
        ]
        portfolio = filtered[portfolio_columns].rename(
            columns={"Well Name": "Workbook Well"}
        )
        st.dataframe(
            portfolio,
            hide_index=True,
            width="stretch",
            height=520,
            column_config={
                "Canonical Field": st.column_config.TextColumn(pinned=True),
                "Canonical Well": st.column_config.TextColumn(pinned=True),
                "Weighted Score": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                ),
                "Existing + NZBP Reserve (MBO)": st.column_config.NumberColumn(
                    format="%.1f"
                ),
                "Last Oil Rate (BOPD)": st.column_config.NumberColumn(format="%.1f"),
                "Last Water Cut (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Idle Years": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.download_button(
            ":material/download: Download filtered screening matrix",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            "Zona_Rokan_Well_Screening_Matrix_filtered.csv",
            "text/csv",
            key="download_screening_matrix_filtered",
        )
        with st.expander("All controlled workbook columns", expanded=False):
            st.dataframe(filtered, hide_index=True, width="stretch", height=560)

    with selected_tab:
        source_row = services.selected_screening_row(selected_record.well)
        if not source_row:
            st.warning(
                f"No screening row was matched to {selected_record.well}.",
                icon=":material/data_alert:",
            )
        else:
            source_caption(
                services.SCREENING_WORKBOOK,
                "01_WellRegister and Scoring",
            )
            score = pd.to_numeric(source_row.get("Weighted Score"), errors="coerce")
            total_reserve = pd.to_numeric(
                source_row.get("Existing + NZBP Reserve (MBO)"), errors="coerce"
            )
            ko_values = {
                column: str(source_row.get(column, "")).strip()
                for column in ko_columns
            }
            has_knockout = any(value.upper() == "YES" for value in ko_values.values())
            with st.container(horizontal=True):
                st.metric(
                    "Weighted score",
                    format_value(score, digits=1),
                    border=True,
                )
                st.metric("Tier", format_value(source_row.get("Tier")), border=True)
                st.metric(
                    "Existing + NZBP reserve",
                    format_value(total_reserve, " MBO", digits=1),
                    border=True,
                )
                st.metric(
                    "Last oil rate",
                    format_value(source_row.get("Last Oil Rate (BOPD)"), " BOPD", digits=1),
                    border=True,
                )
                st.metric(
                    "Last water cut",
                    format_value(source_row.get("Last Water Cut (%)"), "%", digits=1),
                    border=True,
                )
            if has_knockout:
                st.error(
                    "Workbook knockout gate active: "
                    + ", ".join(
                        column.replace(" KO", "")
                        for column, value in ko_values.items()
                        if value.upper() == "YES"
                    ),
                    icon=":material/block:",
                )
            else:
                st.success(
                    "No HSE, integrity, or facility knockout is active in the saved workbook result.",
                    icon=":material/check_circle:",
                )
            st.info(
                format_value(source_row.get("Recommendation")),
                icon=":material/flag:",
            )

            profile_left, context_right = st.columns((1.0, 1.15))
            with profile_left:
                with st.container(border=True):
                    st.subheader("Twelve-parameter score profile")
                    st.altair_chart(screening_score_profile_chart(source_row))
            with context_right:
                with st.container(border=True):
                    st.subheader("Selected-well context")
                    context = pd.DataFrame(
                        [
                            ("Workbook well", source_row.get("Well Name")),
                            ("Workbook field", source_row.get("Field")),
                            ("Reservoir / zone", source_row.get("Reservoir / Zone")),
                            ("Perforation sand", source_row.get("Perforation Sand")),
                            ("Trajectory", source_row.get("Trajectory")),
                            ("Completion type", source_row.get("Completion Type")),
                            ("Artificial lift", source_row.get("Artificial Lift")),
                            ("Integrity status", source_row.get("Integrity Status")),
                            ("Shutdown reason", source_row.get("Shutdown Reason")),
                            ("Existing facility", source_row.get("Existing Facility")),
                        ],
                        columns=["Screening input", "Saved workbook evidence"],
                    )
                    st.dataframe(
                        context,
                        hide_index=True,
                        width="stretch",
                        height=430,
                    )
            with st.container(border=True):
                st.subheader("Screening narrative")
                st.write(format_value(source_row.get("Screening Narrative")))
                st.caption(
                    f"Source row {integer_value(source_row.get('Source Row'))} · "
                    "Saved Excel values are displayed without recalculating the workbook."
                )

    with tiers_tab:
        source_caption(services.SCREENING_WORKBOOK, "02_Tier Definitions & KPIs")
        st.dataframe(tiers, hide_index=True, width="stretch", height=430)

    with rubric_tab:
        source_caption(services.SCREENING_WORKBOOK, "03_Scoring Rubric")
        st.dataframe(
            rubric,
            hide_index=True,
            width="stretch",
            height=520,
            column_config={
                "Parameter #": st.column_config.NumberColumn(format="%d", pinned=True),
                "Parameter": st.column_config.TextColumn(pinned=True),
            },
        )

    with weights_tab:
        source_caption(services.SCREENING_WORKBOOK, "04_Parameter Weights")
        weight_frame = weights.copy()
        weight_frame["Weight (%)"] = pd.to_numeric(
            weight_frame["Weight"], errors="coerce"
        ) * 100
        with st.container(horizontal=True):
            st.metric(
                "Total parameter weight",
                f"{weight_frame['Weight (%)'].sum():.0f}%",
                border=True,
            )
            st.metric(
                "Scoring parameters",
                len(weight_frame),
                border=True,
            )
            st.metric(
                "Knockout parameters",
                int(weight_frame["Knockout"].astype(str).str.upper().eq("YES").sum()),
                border=True,
            )
        weight_chart = (
            alt.Chart(weight_frame)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Weight (%):Q", title="Weight (%)"),
                y=alt.Y("Parameter:N", title=None, sort="-x"),
                color=alt.Color("Dimension:N", title="Dimension"),
                tooltip=[
                    alt.Tooltip("Parameter:N"),
                    alt.Tooltip("Dimension:N"),
                    alt.Tooltip("Weight (%):Q", format=".0f"),
                    alt.Tooltip("Knockout:N"),
                ],
            )
            .properties(height=430)
        )
        st.altair_chart(weight_chart)
        st.dataframe(
            weight_frame.drop(columns=["Weight"]),
            hide_index=True,
            width="stretch",
            height=460,
            column_config={
                "Parameter #": st.column_config.NumberColumn(format="%d", pinned=True),
                "Parameter": st.column_config.TextColumn(pinned=True),
                "Weight (%)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                ),
            },
        )


def render_readiness(master: pd.DataFrame, selected_record: logs.WellRecord) -> None:
    hero(
        "Phase 2/3 Discipline Readiness",
        "Scope-aligned readiness for Production Technology and Completions/Workover, "
        "including well screening, bypassed-zone evaluation, intervention feasibility, "
        "engineering planning, and execution preparation.",
        "Scope of Work Alignment",
    )
    row = canonical_row(master, selected_record.well)
    if row.empty:
        st.warning("No master record matched this well.")
        return
    history = load_perforation_history(selected_record.well)
    schematic = services.selected_schematic_row(selected_record.well)
    detailed_events, _ = services.selected_schematic_detail(selected_record.well)
    related_detail = services.perforation_related_events(detailed_events)
    evidence = load_raw_evidence(selected_record.well)

    production_requirements = [
        (
            "Production history and latest test",
            *requirement_status(
                integer_value(row.get("Production Files")) > 0,
                f"{integer_value(row.get('Production Files'))} production workbook(s)",
            ),
        ),
        (
            "Artificial lift configuration",
            *requirement_status(
                format_value(row.get("Artificial Lift")) != "N/A",
                format_value(row.get("Artificial Lift")),
            ),
        ),
        (
            "Well integrity status",
            *requirement_status(
                format_value(row.get("Integrity Status")) != "N/A",
                format_value(row.get("Integrity Status")),
            ),
        ),
        (
            "Shutdown / impairment reason",
            *requirement_status(
                format_value(row.get("Shutdown Reason")) != "N/A",
                format_value(row.get("Shutdown Reason")),
            ),
        ),
        (
            "Open/cased-hole logs for bypassed zone review",
            *requirement_status(
                integer_value(row.get("LAS Files")) > 0,
                f"{integer_value(row.get('LAS Files'))} LAS file(s)",
            ),
        ),
        (
            "Facility and access context",
            *requirement_status(
                format_value(row.get("Existing Facility")) != "N/A",
                f"{format_value(row.get('Existing Facility'))}; {format_value(row.get('Location Access'))}",
            ),
        ),
    ]
    completion_requirements = [
        (
            "Wellbore schematic",
            *requirement_status(
                bool(row.get("Schematic Available")),
                f"{schematic.get('Workbook', '')} | sheet {schematic.get('Sheet', '')}",
            ),
        ),
        (
            "Tubing/casing/completion configuration",
            *requirement_status(
                integer_value(row.get("Casing Strings")) > 0
                or format_value(row.get("Completion Type")) != "N/A",
                f"{integer_value(row.get('Casing Strings'))} casing string(s); "
                f"{format_value(row.get('Completion Type'))}",
            ),
        ),
        (
            "Perforation and isolation history",
            *requirement_status(
                not related_detail.empty or not history.empty,
                f"{len(related_detail)} detailed event(s); "
                f"{len(history)} OPEN/CLOSE status event(s)",
            ),
        ),
        (
            "Workover/intervention evidence",
            *requirement_status(
                not evidence.empty
                and evidence.get("Operation Summary", pd.Series(dtype=object))
                .fillna("")
                .astype(str)
                .str.len()
                .gt(0)
                .any(),
                f"{len(evidence)} normalized evidence row(s)",
            ),
        ),
        (
            "Integrity documentation",
            *requirement_status(
                format_value(row.get("Integrity Status")) != "N/A",
                format_value(row.get("Integrity Status")),
            ),
        ),
        (
            "Operational feasibility context",
            *requirement_status(
                format_value(row.get("Location Access")) != "N/A"
                and format_value(row.get("Existing Facility")) != "N/A",
                f"{format_value(row.get('Location Access'))}; "
                f"{format_value(row.get('Existing Facility'))}",
            ),
        ),
    ]

    metrics = st.columns(5)
    metrics[0].metric(
        "Phase 2 readiness", format_value(row.get("Phase 2 Readiness"), "%")
    )
    metrics[1].metric(
        "Technical viability", format_value(row.get("Technical Viability"))
    )
    metrics[2].metric(
        "Uplift potential", format_value(row.get("Production Uplift Potential"))
    )
    metrics[3].metric(
        "Execution complexity", format_value(row.get("Execution Complexity"))
    )
    metrics[4].metric("Screening tier", format_value(row.get("Priority Tier")))

    left, right = st.columns(2)
    with left:
        st.subheader("Production Technologist")
        st.dataframe(
            pd.DataFrame(
                production_requirements,
                columns=["Scope requirement", "Status", "Current evidence"],
            ),
            hide_index=True,
            width="stretch",
            height=360,
        )
    with right:
        st.subheader("Completions / Workover Engineer")
        st.dataframe(
            pd.DataFrame(
                completion_requirements,
                columns=["Scope requirement", "Status", "Current evidence"],
            ),
            hide_index=True,
            width="stretch",
            height=360,
        )

    st.subheader("Phase progression")
    progression = pd.DataFrame(
        [
            (
                "Phase 2 - Candidate screening",
                "Integrity review, production-system readiness, subsurface access, "
                "offset/perforation context, petrophysical bypassed-zone review, and ranking.",
                row.get("Phase 2 Candidate"),
            ),
            (
                "Phase 2 - Engineering pre-screen",
                "CT reachability, zonal isolation feasibility, fluid compatibility, "
                "cement/integrity confidence, and intervention category.",
                "ENGINEERING INPUT REQUIRED",
            ),
            (
                "Phase 3 - Design engineering",
                "Per-well objective, methodology, tool string, pressure/fluid models, "
                "execution sequence, HSE controls, costs, and uplift forecast.",
                "AFTER CANDIDATE APPROVAL",
            ),
            (
                "Phase 3 - Execution readiness",
                "Rig/rigless plan, equipment, personnel, consumables, PSL services, "
                "schedule, logistics, QA/QC, SIMOPS, and risk assessment.",
                "AFTER DESIGN ACCEPTANCE",
            ),
        ],
        columns=["Stage", "Dashboard decision support", "Current gate"],
    )
    st.dataframe(progression, hide_index=True, width="stretch")
    st.warning(
        f"Current critical gaps: {row.get('Critical Data Gaps')}",
        icon=":material/data_alert:",
    )
    st.caption(
        "Scope alignment is based on SECTION_III_ScopeOfWork.pdf, especially the "
        "Production Technologist and Completions/Workover roles plus the well-screening, "
        "design-engineering, and execution-preparation workflows."
    )


def main() -> None:
    with st.sidebar:
        st.markdown("## :material/oil_barrel: Zona Rokan")
        st.caption("Integrated Phase 2/3 well intelligence")
        if services.ONLINE_MODE:
            root_text = str(services.DATA_ROOT)
            st.caption(":material/cloud_done: Secure cloud data package")
        else:
            root_text = st.text_input("Canonical data root", str(logs.DEFAULT_DATA_ROOT))
        page = st.radio("Workspace", PAGES)
        st.caption("Controlled workbook sources")
        for label, path in (
            ("Availability", services.DATA_AVAILABILITY_WORKBOOK),
            ("Screening matrix", services.SCREENING_WORKBOOK),
            ("Schematic", services.SCHEMATIC_WORKBOOK),
            ("Perforation / NZBP", services.PERFORATION_WORKBOOK),
        ):
            icon = ":material/check_circle:" if path.exists() else ":material/error:"
            st.caption(f"{icon} {label}")
    inventory = logs.scan_inventory(root_text)
    scan_issues = logs.inventory_scan_issues(root_text) if not services.ONLINE_MODE else []
    if not inventory and services.ONLINE_MODE:
        inventory = load_portable_inventory()
    if scan_issues:
        st.warning(
            f"Skipped {len(scan_issues)} unreadable data path(s). The dashboard is still "
            "available; affected files are excluded from the inventory."
        )
        with st.expander("Data access warnings", expanded=False):
            st.code("\n".join(scan_issues), language=None)
    if not inventory:
        st.error("No canonical field/well inventory was found under the selected root.")
        return
    selected_record, _ = selected_record_controls(inventory)
    with st.sidebar:
        st.metric("Canonical wells", len(inventory), border=True)
        st.metric("Selected logs", len(selected_record.las_files), border=True)
        st.metric("Selected scans", len(selected_record.scanned_files), border=True)
        if services.ONLINE_MODE:
            st.caption(
                "Cloud mode uses packaged, read-only controlled sources. Uploaded "
                "session files are temporary and are not written back to the source workbooks."
            )
        else:
            st.caption(
                "All generated results remain local under the existing outputs and "
                "Data Nations/Final things folders."
            )

    source_tokens = tuple(
        services.source_file_token(path)
        for path in (
            services.DATA_AVAILABILITY_WORKBOOK,
            services.SCREENING_WORKBOOK,
            services.SCHEMATIC_WORKBOOK,
            services.PERFORATION_WORKBOOK,
        )
    )
    source_tokens = (
        *source_tokens,
        (services.CLOUD_DEPLOYMENT_VERSION, 0, 0),
    )
    master = load_master(root_text, source_tokens)
    if page == "Executive overview":
        render_executive(master, selected_record)
    elif page == "Well screening matrix":
        render_screening_matrix(master, selected_record)
    elif page == "Well 360":
        render_well_360(master, selected_record)
    elif page == "Data availability":
        render_coordinates_and_requests(master, selected_record)
    elif page == "Formation evaluation":
        render_las_and_nzbp(selected_record, workspace="formation")
    elif page == "NZBP screening":
        render_las_and_nzbp(selected_record, workspace="nzbp")
    elif page == "Scanned log digitizer":
        render_digitizer(selected_record)
    elif page == "Well schematic":
        render_schematic(selected_record)
    elif page == "Perforation and NZBP":
        render_perforation(selected_record)
    else:
        render_readiness(master, selected_record)


if __name__ == "__main__":
    main()
