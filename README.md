# Zona Rokan Integrated Well Intelligence Dashboard

This Streamlit application integrates the existing Zona Rokan analysis and
generation workflows into one local dashboard for Production Technology and
Completions/Workover review.

The same codebase supports a private Streamlit Community Cloud deployment. It
uses the live `D:` sources on the workstation and automatically falls back to
the packaged, read-only controlled workbooks when hosted on Linux. The cloud
version excludes the scanned-log digitizer because that workflow depends on
local scripts and source folders; formation evaluation remains available for
user-uploaded LAS, DLIS, and LIS files.

## Integrated workspaces

- Executive command center with canonical well ranking and Phase 2 readiness.
- Well 360 workspace for production, completion, integrity, and source evidence.
- LAS/DLIS/LIS petrophysics plus New Zone Behind Pipe screening against perforation history, including optional XRD matrix-density, core/SWC validation, evidence capture, and auditable workbook export.
- Scanned-log digitization using `digitize_zona_rokan_well_logs.py`.
- Per-well calibration profiles and visibility of MINAS training diagnostics.
- Controlled data availability from `001_Rokan Block Data Availability Final.xlsx`.
- Controlled 101-well weighted screening matrix, tier definitions, scoring rubric,
  and parameter weights from `002_Well Screening Matrix Final.xlsx`.
- Workbook-derived well schematics from `well schematic Rokan.xlsx`, with a
  source-fingerprinted per-sheet cache for fast startup.
- Perforation history and NZBP interval review from
  `Perforation History + NZBP interval review.xlsx`.
- Phase 2/3 scope alignment for Production Technologist and
  Completions/Workover Engineer activities.

## Run

Open PowerShell in this directory:

```powershell
C:\Users\HP\.venvs\zona_rokan_streamlit_312\Scripts\python.exe -m streamlit run app.py
```

Use the Python 3.12 environment on `C:` shown above. It is the verified local
deployment environment and passes `python -m pip check` with no broken
requirements. The legacy project environments on `D:` reference Python
installations that are no longer available.

The included `run_dashboard.ps1` starts the app on `http://127.0.0.1:8501`.
If the schematic workbook changes, refresh its cache before launch:

```powershell
C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe build_schematic_workbook_cache.py
```

The default data root is:

```text
D:\HALLIBURTON CONSULTING\Data Nations\Zona Rokan
```

It can be changed from the application sidebar.

## Private cloud deployment

Deploy `app.py` with Python 3.12 from a private GitHub repository. Keep the
Streamlit app private and add approved viewers by email. The deployment package
includes the four controlled workbooks under `data/controlled_sources/` and the
source-fingerprinted schematic cache under `data/schematic_workbook_cache/`.
Never commit `.streamlit/secrets.toml`.

## Important interpretation note

The petrophysical, NZBP, ranking, and readiness outputs are screening aids.
Confirm depth matching, environmental corrections, formation properties,
pressure support, completion state, cement isolation, integrity, economics,
and intervention feasibility before approving a candidate or job design.
