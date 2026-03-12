from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

# Local Windows default, overridden in Docker by DATASET_ROOT=/data
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", r"D:\DATASET"))

# Keep runs on the mounted project folder
RUNS_DIR = Path(os.getenv("RUNS_DIR", str(PROJECT_ROOT / "runs")))