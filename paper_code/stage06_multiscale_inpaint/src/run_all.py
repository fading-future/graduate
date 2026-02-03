import os
import argparse
import subprocess


def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", type=str, required=True)
    parser.add_argument("--skip_coarse", action="store_true")
    parser.add_argument("--skip_refine", action="store_true")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--precompute_coarse", action="store_true")
    parser.add_argument("--coarse_cache_dir", type=str, default="")
    args = parser.parse_args()

    if not args.skip_coarse:
        run("python -m src.train_coarse")

    if args.precompute_coarse:
        if not args.coarse_cache_dir:
            raise ValueError("--coarse_cache_dir is required when --precompute_coarse is set")
        run(f"python -m src.precompute_coarse --out_dir {args.coarse_cache_dir}")

    if not args.skip_refine:
        run("python -m src.train_refine")

    if not args.skip_infer:
        run(f"python -m src.infer --raw_file \"{args.raw_file}\"")


if __name__ == "__main__":
    main()
