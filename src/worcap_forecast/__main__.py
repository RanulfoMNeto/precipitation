"""CLI for source acquisition, chronological training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data as dataset
from .io import read, validate_submission
from .runtime import sha256, source_fingerprint


def main():
    settings = read(Path(__file__).resolve().parents[2] / "SETTINGS.json")
    parser = argparse.ArgumentParser(prog="worcap-forecast")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Acquire the source files")
    download.add_argument("--data", type=Path, default=Path(settings["data_dir"]))
    download.add_argument("--competition-dir", type=Path, default=Path(settings["competition_dir"]))
    download.add_argument("--cache", type=Path, default=Path(settings["source_cache_dir"]))
    download.add_argument("--stage", choices=("all", "competition", "seasonal", "atmosphere", "gefs", "finalize"), default="all")
    prepared = sub.add_parser(
        "prepare", help="Rebuild every OOF ancestor in time order"
    )
    prepared.add_argument("--data", type=Path, default=Path(settings["data_dir"]))
    prepared.add_argument("--work", type=Path, default=Path(settings["work_dir"]))
    fitted = sub.add_parser("train", help="Fit the final temporal and GEFS models")
    fitted.add_argument("--data", type=Path, default=Path(settings["data_dir"]))
    fitted.add_argument("--work", type=Path, default=Path(settings["work_dir"]))
    inference = sub.add_parser(
        "predict", help="Run all five fitted models on verified inputs"
    )
    inference.add_argument("--data", type=Path, default=Path(settings["data_dir"]))
    inference.add_argument("--work", type=Path, default=Path(settings["work_dir"]))
    inference.add_argument("--output", type=Path, default=Path(settings["submission_path"]))
    check = sub.add_parser(
        "verify", help="Check hashes, cutoff, availability and submission"
    )
    check.add_argument("--data", type=Path, default=Path(settings["data_dir"]))
    check.add_argument("--work", type=Path, default=Path(settings["work_dir"]))
    check.add_argument("--submission", type=Path)
    check.add_argument("--competition-dir", type=Path)
    check.add_argument("--cache", type=Path, help="Rehash separately retained source GRIBs")
    args = parser.parse_args()
    if args.command == "download":
        from .acquire import run

        result = run(args.data, args.cache, args.competition_dir, args.stage)
        print(json.dumps({"stage": args.stage, "files": len(result.get("files", {})), "manifest": sha256(args.data / "manifest.json") if (args.data / "manifest.json").is_file() else None}, indent=2))
    elif args.command == "prepare":
        from .pipeline import prepare

        report = dataset.availability_report(args.data)
        prepare(args.data, args.work)
        from .runtime import atomic_json

        atomic_json(args.work / "availability.json", report)
    elif args.command == "train":
        from .pipeline import train

        train(args.data, args.work)
    elif args.command == "predict":
        from .pipeline import predict

        predict(args.data, args.work, args.output)
    else:
        result = dataset.verify(args.data)
        report = dataset.availability_report(args.data)
        prepared = args.work / "prepared.json"
        if prepared.is_file():
            receipt = read(prepared)
            distribution = receipt.get("distribution_receipt")
            if distribution and sha256(args.work / "distribution.json") != distribution:
                raise ValueError("Changed distribution provenance")
            if receipt["data_manifest"] != sha256(args.data / "manifest.json"):
                raise ValueError("Prepared data do not match")
            if receipt["source_files"] != source_fingerprint():
                raise ValueError("OOF source files have changed")
            for name, digest in receipt["context_files"].items():
                if sha256(args.work / "context" / name) != digest:
                    raise ValueError(f"Changed OOF context: {name}")
            if len(receipt["context_files"]) != 192 or list(
                (args.work / "context").glob("202[34]-*.npy")
            ):
                raise ValueError("Invalid OOF coverage or cached test contexts")
            for year in range(1999, 2024, 2):
                folder = args.work / "ancestors" / str(year)
                fold = read(folder / "result.json")
                if fold["training_end"] != f"{year - 1}-12":
                    raise ValueError(f"Chronological fold cutoff changed: {year}")
                if year == 2023 and (
                    fold.get("test_predictions_cached") is not False
                    or list(folder.glob("202[34]-*.npy"))
                    or (folder / "unet.npy").exists()
                ):
                    raise ValueError("Prepared directory contains test predictions")
                for name, digest in fold["files"].items():
                    if sha256(folder / name) != digest:
                        raise ValueError(f"Changed chronological fold: {year}/{name}")
        originals_checked = (
            dataset.verify_originals(args.data, args.competition_dir)
            if args.competition_dir
            else 0
        )
        sources_checked = 0
        if args.cache:
            from .acquire import verify_sources

            sources_checked = verify_sources(args.data, args.cache)
        model_names = [f"tabular_{year}" for year in range(2007, 2024, 2)]
        model_names.extend(("temporal_42", "temporal_2026", "gefs"))
        trained = args.work / "trained.json"
        if trained.is_file() and (
            not prepared.is_file()
            or read(trained)["data_manifest"] != sha256(args.data / "manifest.json")
            or read(trained)["source_files"] != source_fingerprint()
        ):
            raise ValueError("Trained models do not match prepared inputs")
        for name in model_names:
            folder = args.work / "models" / name
            if (folder / "result.json").is_file() or trained.is_file():
                from .pipeline import done

                if not done(folder):
                    raise ValueError(f"Changed model fit: {name}")
        if args.submission:
            if not trained.is_file():
                raise ValueError("Submission cannot be verified without the five model fits")
            receipt = read(args.submission.with_suffix(".json"))
            if (
                receipt["sha256"] != sha256(args.submission)
                or receipt["generated_by"] != "fresh_model_inference"
                or receipt["rows"] != 1_885_464
                or receipt["training_end"] != "2022-12"
                or receipt["data_manifest"] != sha256(args.data / "manifest.json")
                or receipt["source_files"] != source_fingerprint()
            ):
                raise ValueError("Submission receipt mismatch")
            required_work = {"binding.json", "prepared.json", "trained.json", "availability.json"}
            required_models = {
                "ancestors/2023/result.json",
                "models/tabular_2023/result.json",
                "models/temporal_42/result.json",
                "models/temporal_2026/result.json",
                "models/gefs/result.json",
            }
            if (
                set(receipt["work_files"]) != required_work
                or set(receipt["model_receipts"]) != required_models
            ):
                raise ValueError("Incomplete submission model receipt")
            for name, digest in {**receipt["work_files"], **receipt["model_receipts"]}.items():
                if sha256(args.work / name) != digest:
                    raise ValueError(f"Submission model identity mismatch: {name}")
            validate_submission(
                args.data / "sample_submission.csv", args.submission, 1_885_464
            )
        print(
            json.dumps(
                {
                    "files": len(result["files"]),
                    "availability": report,
                    "prepared": prepared.is_file(),
                    "submission_checked": bool(args.submission),
                    "competition_originals_checked": originals_checked,
                    "source_files_checked": sources_checked,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
