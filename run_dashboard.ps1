$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\HP\.venvs\zona_rokan_streamlit_312\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Dashboard runtime not found: $Python"
}

Set-Location -LiteralPath $ProjectRoot
& $Python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
