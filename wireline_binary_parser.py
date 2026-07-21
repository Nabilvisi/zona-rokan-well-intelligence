from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from dlisio import dlis, lis


def scalar_frame(records: Any) -> pd.DataFrame:
    """Convert a structured curve array to scalar columns only."""
    names: Iterable[str] = getattr(getattr(records, "dtype", None), "names", None) or ()
    columns: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(records[name])
        if values.ndim == 1:
            columns[str(name)] = values
        elif values.ndim == 2 and values.shape[1] == 1:
            columns[str(name)] = values[:, 0]
    return pd.DataFrame(columns)


def parse_dlis(path: Path) -> list[tuple[str, pd.DataFrame]]:
    output: list[tuple[str, pd.DataFrame]] = []
    with dlis.load(str(path)) as logical_files:
        for file_number, logical_file in enumerate(logical_files, start=1):
            for frame_number, frame in enumerate(logical_file.frames, start=1):
                try:
                    table = scalar_frame(frame.curves())
                except Exception:
                    continue
                if not table.empty:
                    label = getattr(frame, "name", "") or f"file{file_number}_frame{frame_number}"
                    output.append((str(label), table))
    return output


def parse_lis(path: Path) -> list[tuple[str, pd.DataFrame]]:
    output: list[tuple[str, pd.DataFrame]] = []
    with lis.load(str(path)) as logical_files:
        for file_number, logical_file in enumerate(logical_files, start=1):
            specifications = logical_file.data_format_specs()
            for spec_number, specification in enumerate(specifications, start=1):
                try:
                    table = scalar_frame(lis.curves(logical_file, specification))
                except Exception:
                    continue
                if not table.empty:
                    index_name = getattr(specification, "index_mnem", "") or "INDEX"
                    output.append(
                        (f"file{file_number}_logset{spec_number}_{index_name}", table)
                    )
    return output


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: wireline_binary_parser.py <dlis|lis> <input> <output>")
    file_type, source_text, output_text = sys.argv[1:]
    source = Path(source_text)
    if file_type == "dlis":
        result = parse_dlis(source)
    elif file_type == "lis":
        result = parse_lis(source)
    else:
        raise ValueError(f"unsupported binary log type: {file_type}")
    with Path(output_text).open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
