from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile


CONFIG = {
    "input_dir": Path(r"D:\浅层礁灰岩数据集\Final_Result_Sorted_w256_s32"),
    "output_dir": Path(r"F:\aaa浅层礁灰岩npy转tiff文件（xy230到2840）"),
    "glob_pattern": "*.npy",
    "start_index": 230,
    "end_index": 2840,
}


NAME_PATTERN = re.compile(r"^(.*)_z(\d+)_y(\d+)_x(\d+)\.npy$", re.IGNORECASE)


def natural_npy_key(path: Path) -> tuple:
    name = path.name
    match = NAME_PATTERN.match(name)
    if match:
        prefix, z_str, y_str, x_str = match.groups()
        return (prefix, int(z_str), int(y_str), int(x_str))

    numbers = [int(x) for x in re.findall(r"\d+", name)]
    return (name, *numbers)


def collect_npy_files(input_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(glob_pattern), key=natural_npy_key)
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {input_dir}")
    return files


def select_index_range(files: list[Path], start_index: int, end_index: int) -> list[Path]:
    if start_index < 1 or end_index < 1:
        raise ValueError("start_index and end_index must be >= 1")
    if end_index < start_index:
        raise ValueError("end_index must be >= start_index")

    total = len(files)
    start_pos = start_index - 1
    end_pos = min(end_index, total)

    if start_pos >= total:
        raise IndexError(f"start_index={start_index} exceeds file count {total}")

    return files[start_pos:end_pos]


def export_npy_files_to_tiff(selected_files: list[Path], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total = len(selected_files)
    for index, npy_path in enumerate(selected_files, start=1):
        data = np.load(npy_path)
        tiff_path = output_dir / f"{npy_path.stem}.tiff"
        tifffile.imwrite(tiff_path, data, imagej=True)
        success_count += 1

        if index <= 3 or index % 100 == 0 or index == total:
            print(
                f"[{index}/{total}] {npy_path.name} -> {tiff_path.name} "
                f"| shape={data.shape} dtype={data.dtype}"
            )

    return success_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export a slice of npy volumes to tiff.")
    parser.add_argument("--input-dir", type=Path, default=CONFIG["input_dir"])
    parser.add_argument("--output-dir", type=Path, default=CONFIG["output_dir"])
    parser.add_argument("--glob-pattern", default=CONFIG["glob_pattern"])
    parser.add_argument("--start-index", type=int, default=CONFIG["start_index"])
    parser.add_argument("--end-index", type=int, default=CONFIG["end_index"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_files = collect_npy_files(input_dir, args.glob_pattern)
    selected_files = select_index_range(all_files, args.start_index, args.end_index)

    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Total files : {len(all_files)}")
    print(f"Range       : {args.start_index}..{min(args.end_index, len(all_files))}")
    print(f"To export   : {len(selected_files)}")

    if selected_files:
        print(f"First file  : {selected_files[0].name}")
        print(f"Last file   : {selected_files[-1].name}")

    count = export_npy_files_to_tiff(selected_files, output_dir)
    print(f"Done. Exported {count} npy files to tiff.")


if __name__ == "__main__":
    main()
