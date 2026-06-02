"""
run_pipeline.py
---------------
End-to-end orchestrator for the Sleman research-grade data pipeline.

Usage:
    python src/run_pipeline.py [--skip-gee] [--skip-osm] [--force]

Steps
-----
1. Fetch GEE features   -> data/interim/features_gee.csv
2. Fetch OSM POI counts -> data/interim/features_osm.csv
3. Merge & validate     -> data/raw/final_dataset.csv
"""
import argparse
import sys
from pathlib import Path

try:
    from . import config
    from data_acquisition.gee import GEEFetcher
    from data_acquisition.osm import OSMFetcher
    from utils import (
        LOG,
        enforce_schema,
        load_csv,
        merge_feature_csvs,
        save_csv,
    )
except ImportError:  # pragma: no cover
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from . import config
    from data_acquisition.gee import GEEFetcher
    from data_acquisition.osm import OSMFetcher
    from utils import (
        LOG,
        enforce_schema,
        load_csv,
        merge_feature_csvs,
        save_csv,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sleman Grid Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-gee",
        action="store_true",
        help="Skip the GEE (remote-sensing) acquisition step.",
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip the OSM (POI) acquisition step.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run fetchers even if their output CSVs already exist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.FINAL_OUTPUT,
        help="Path for the final merged dataset.",
    )
    return parser.parse_args(argv)


def run_gee(force: bool = False) -> Path:
    """Run the GEE fetcher if needed."""
    if config.GEE_OUTPUT.exists() and not force:
        LOG.info("GEE output already exists: %s (use --force to overwrite)", config.GEE_OUTPUT)
        return config.GEE_OUTPUT
    fetcher = GEEFetcher()
    fetcher.run(out_path=config.GEE_OUTPUT)
    return config.GEE_OUTPUT


def run_osm(force: bool = False) -> Path:
    """Run the OSM fetcher if needed."""
    if config.OSM_OUTPUT.exists() and not force:
        LOG.info("OSM output already exists: %s (use --force to overwrite)", config.OSM_OUTPUT)
        return config.OSM_OUTPUT
    fetcher = OSMFetcher()
    fetcher.run(out_path=config.OSM_OUTPUT)
    return config.OSM_OUTPUT


def merge_and_validate(output_path: Path) -> None:
    """Merge intermediate CSVs, enforce schema, and write the final dataset."""
    intermediates = [config.GEE_OUTPUT, config.OSM_OUTPUT]
    for p in intermediates:
        if not p.exists():
            raise FileNotFoundError(
                f"Intermediate file missing: {p}. "
                "Run the pipeline without --skip-* flags first."
            )

    LOG.info("Merging intermediate features...")
    merged = merge_feature_csvs(intermediates)
    merged = enforce_schema(merged, config.FINAL_SCHEMA)

    # Basic sanity checks
    n_rows = len(merged)
    n_cols = len(merged.columns)
    missing_grid = merged[config.GRID_ID_COL].isna().sum()
    LOG.info("Final dataset: %d rows x %d cols | missing grid_id: %d", n_rows, n_cols, missing_grid)

    if missing_grid:
        raise ValueError(f"Final dataset contains {missing_grid} missing grid_id rows.")

    save_csv(merged, output_path)
    LOG.info("Pipeline finished successfully -> %s", output_path)


def main(argv=None):
    args = parse_args(argv)

    LOG.info("=" * 60)
    LOG.info("Sleman Data Pipeline -- starting")
    LOG.info("=" * 60)

    # Step 1 & 2: Fetchers
    if not args.skip_gee:
        run_gee(force=args.force)
    if not args.skip_osm:
        run_osm(force=args.force)

    # Step 3: Merge
    merge_and_validate(args.output)

    # Quick preview
    df = load_csv(args.output)
    print("\n--- Final Dataset Preview ---")
    print(df.head(10).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"Saved to: {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
