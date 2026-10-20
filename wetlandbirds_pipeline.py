"""
Visual WetlandBirds pipeline (v3).

Two tasks on the Visual WetlandBirds dataset (Rodriguez-Juan et al., 2025):

  Phase 1  species / behavior classification
           frozen pretrained backbones (4 CNNs + DINOv2 ViT-S/14), embeddings
           cached to disk, logistic-regression / MLP heads, StratifiedGroupKFold
           grouped by bird or video, one locked test fold, McNemar + Holm + 95% CI.
           Behavior clips use 4 frames pooled as [mean, std, last - first].

  Phase 2  species detection with YOLOv9 (ultralytics), same family as the paper
           2a: config tuning (imgsz) with grouped k-fold on train/val videos only;
               test videos are never exported or evaluated here.
           2b: retrain the winning config and evaluate once per seed on the test
               videos; mean +/- std over seeds is the number compared to the paper.
           Frames are sampled with stride 10 per video to match the paper's
           downsampling.

Expected files under --data-root:
    bounding_boxes.csv, crops.csv, species_ID.csv, behaviors_ID.csv, splits.json,
    and a videos/ folder (or videos.zip)

Examples:
    python wetlandbirds_pipeline.py --data-root data                 # full run
    python wetlandbirds_pipeline.py --data-root data --no-detection  # classification only
    python wetlandbirds_pipeline.py --data-root data --skip-dev-tuning
    python wetlandbirds_pipeline.py --data-root data --resume        # reuse cached runs
    python wetlandbirds_pipeline.py --data-root data --max-frames 200 --dev-epochs 2 --final-epochs 2 --seeds 42
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import itertools
import json
import logging
import random
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("wetlandbirds_v3")

# ============================================================================
# REPRODUCIBILITY + DEVICE
# ============================================================================
_DEVICE = None


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic cuDNN so the multi-seed report is reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    import torch
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        # re-enable the cuDNN autotuner for the long YOLO runs
        torch.backends.cudnn.benchmark = True
        logger.info("GPU detected: %s -- running on CUDA (device 0)", torch.cuda.get_device_name(0))
    else:
        _DEVICE = torch.device("cpu")
        logger.warning("No CUDA GPU detected, running on CPU. Classification is fine on CPU; "
                        "YOLO training will be very slow.")
    return _DEVICE


# ============================================================================
# PATHS
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")
VIDEO_DIR_CANDIDATE_NAMES = ["videos", "Videos", "videos_extracted"]
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(output_dir / "run_log.txt", encoding="utf-8")],
    )


# ============================================================================
# CONFIG
# ============================================================================
BACKBONE_REGISTRY = {
    # "hub": DINOv2 via torch.hub; "torchvision": ImageNet CNN without its classifier
    "dinov2_vits14": {"kind": "hub", "repo": "facebookresearch/dinov2", "hub_name": "dinov2_vits14",
                       "embed_dim": 384, "input_size": 224},
    "resnet18": {"kind": "torchvision", "embed_dim": 512, "input_size": 224},
    "mobilenet_v2": {"kind": "torchvision", "embed_dim": 1280, "input_size": 224},
    "efficientnet_b0": {"kind": "torchvision", "embed_dim": 1280, "input_size": 224},
    "densenet121": {"kind": "torchvision", "embed_dim": 1024, "input_size": 224},
}


@dataclasses.dataclass
class DataConfig:
    root: str = str(DEFAULT_DATA_ROOT)
    videos_subdir: str = "videos"

    @property
    def root_path(self) -> Path:
        return Path(self.root)


@dataclasses.dataclass
class SamplingConfig:
    crops_per_class: int = 60
    frames_per_clip: int = 4           # frames pooled per behavior clip


@dataclasses.dataclass
class ClassificationConfig:
    backbones: list = dataclasses.field(default_factory=lambda: ["dinov2_vits14", "resnet18",
                                                                    "mobilenet_v2", "efficientnet_b0",
                                                                    "densenet121"])
    heads: list = dataclasses.field(default_factory=lambda: ["logreg", "mlp"])
    tasks: list = dataclasses.field(default_factory=lambda: ["species", "behavior"])
    outer_test_fraction_folds: int = 5   # 1/5 of the groups held out as the locked test fold
    inner_k_folds: int = 5
    embed_batch_size: int = 16


@dataclasses.dataclass
class DetectionConfig:
    enabled: bool = True
    yolo_model: str = "yolov9c.pt"     # or yolov9e.pt
    imgsz: int = 640
    batch_size: int = 16
    lr0: float = 0.01                  # same as the paper
    conf_threshold: float = 0.001      # keep low so the mAP sweep is not truncated
    iou_nms: float = 0.7
    device: str = "0"                  # first CUDA GPU; switched to "cpu" in main() if none

    # frame sampling
    sampling_mode: str = "stride"      # "stride" = every Nth annotated frame (paper), "random" = per-video cap
    frame_stride: int = 10             # paper downsamples by 10
    frames_per_video: int = 20         # random mode only
    max_frames: Optional[int] = None   # optional global cap

    # phase 2a: tuning on train/val videos only
    dev_cv_folds: int = 3
    dev_epochs: int = 30
    dev_configs: list = dataclasses.field(default_factory=lambda: [{"imgsz": 640}, {"imgsz": 960}])
    tuning_metric: str = "mAP50_95"

    # phase 2b: final report on the test videos, averaged over seeds
    final_epochs: int = 50             # same as the paper
    final_seeds: list = dataclasses.field(default_factory=lambda: [42, 43, 44])


@dataclasses.dataclass
class StatsConfig:
    alpha: float = 0.05
    mcnemar_exact_threshold: int = 25


@dataclasses.dataclass
class Config:
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    seed: int = 42
    resume: bool = False
    use_official_split: bool = True
    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
    classification: ClassificationConfig = dataclasses.field(default_factory=ClassificationConfig)
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    stats: StatsConfig = dataclasses.field(default_factory=StatsConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def results_dir(self) -> Path:
        return self.output_path / "model_results"

    @property
    def crops_dir(self) -> Path:
        return self.output_path / "sample_crops"

    @property
    def embed_cache_dir(self) -> Path:
        return self.output_path / "embedding_cache"

    @property
    def ckpt_dir(self) -> Path:
        return self.output_path / "checkpoints"

    def make_dirs(self) -> None:
        for d in (self.output_path, self.results_dir, self.crops_dir, self.embed_cache_dir,
                  self.ckpt_dir, self.crops_dir / "species", self.crops_dir / "behavior",
                  self.crops_dir / "species_detection"):
            d.mkdir(parents=True, exist_ok=True)


def apply_cli_overrides(cfg: Config, args) -> Config:
    if args.data_root is not None:
        cfg.data.root = args.data_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.tasks is not None:
        cfg.classification.tasks = args.tasks
    if args.backbones is not None:
        cfg.classification.backbones = args.backbones
    if args.no_detection:
        cfg.detection.enabled = False
    if args.ignore_official_split:
        cfg.use_official_split = False
    if args.yolo_model is not None:
        cfg.detection.yolo_model = args.yolo_model
    if args.dev_epochs is not None:
        cfg.detection.dev_epochs = args.dev_epochs
    if args.final_epochs is not None:
        cfg.detection.final_epochs = args.final_epochs
    if args.dev_folds is not None:
        cfg.detection.dev_cv_folds = args.dev_folds
    if args.seeds is not None:
        cfg.detection.final_seeds = args.seeds
    if args.max_frames is not None:
        cfg.detection.max_frames = args.max_frames
    if args.sampling_mode is not None:
        cfg.detection.sampling_mode = args.sampling_mode
    if args.skip_dev_tuning:
        cfg.detection.dev_configs = cfg.detection.dev_configs[:1]
    cfg.resume = bool(args.resume)
    return cfg


# ============================================================================
# DATA LOADING
# ============================================================================
def find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None


def safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("file not found: %s", path)
        return None
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep, **kwargs)
            if df.shape[1] > 1:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            continue
    logger.warning("could not parse %s with any common separator", path)
    return None


def build_id_map(df: Optional[pd.DataFrame], id_candidates: list, name_candidates: list) -> dict:
    if df is None:
        return {}
    id_col = find_col(df, id_candidates)
    name_col = find_col(df, name_candidates)
    if id_col is None or name_col is None:
        if df.shape[1] >= 2:
            id_col, name_col = df.columns[0], df.columns[1]
        else:
            return {}
    return dict(zip(df[id_col], df[name_col]))


def parse_bbox_cell(cell) -> list:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple)):
        parsed = cell
    else:
        s = str(cell).strip()
        if not s or s.lower() in ("nan", "none", "[]", "{}"):
            return []
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", s)]
            return [{"bird_id": None, "box": tuple(nums[i:i + 4])}
                    for i in range(0, len(nums) - 3, 4)]

    boxes = []

    def handle_dict(d):
        keys_lower = {str(k).lower(): k for k in d.keys()}

        def getkey(cands):
            for c in cands:
                if c in keys_lower:
                    return d[keys_lower[c]]
            return None

        x1 = getkey(["x1", "xmin", "x_min", "left"])
        y1 = getkey(["y1", "ymin", "y_min", "top"])
        x2 = getkey(["x2", "xmax", "x_max", "right"])
        y2 = getkey(["y2", "ymax", "y_max", "bottom"])
        bird_id = getkey(["bird_id", "id", "subject_id", "subject"])
        if None not in (x1, y1, x2, y2):
            try:
                boxes.append({"bird_id": bird_id, "box": (float(x1), float(y1), float(x2), float(y2))})
            except (TypeError, ValueError):
                pass

    def handle_list_of_nums(lst):
        nums = [v for v in lst if isinstance(v, (int, float))]
        if len(nums) >= 4:
            boxes.append({"bird_id": None, "box": tuple(float(v) for v in nums[:4])})

    if isinstance(parsed, dict):
        handle_dict(parsed)
    elif isinstance(parsed, (list, tuple)) and len(parsed) > 0:
        if isinstance(parsed[0], dict):
            for item in parsed:
                if isinstance(item, dict):
                    handle_dict(item)
        elif isinstance(parsed[0], (list, tuple)):
            for item in parsed:
                handle_list_of_nums(list(item))
        else:
            handle_list_of_nums(list(parsed))
    return boxes


def detect_bbox_columns(bbox_df: pd.DataFrame) -> dict:
    cols = {
        "species_id": find_col(bbox_df, ["species_id", "specie_id"]),
        "species": find_col(bbox_df, ["species", "specie", "species_name"]),
        "video": find_col(bbox_df, ["video_name", "video_id", "video"]),
        "frame": find_col(bbox_df, ["frame", "frame_id", "frame_num", "frame_number"]),
        "boxes": find_col(bbox_df, ["bounding_boxes", "bounding_box", "bbox", "boxes"]),
        "bird_id": find_col(bbox_df, ["bird_id", "subject_id", "individual_id"]),
    }
    missing = [k for k, v in cols.items() if v is None and k != "bird_id"]
    if missing:
        logger.warning("could not auto-detect these bounding_boxes.csv columns: %s. Available: %s",
                        missing, list(bbox_df.columns))
    return cols


def detect_crops_columns(crops_df: pd.DataFrame) -> dict:
    cols = {
        "video": find_col(crops_df, ["video_name", "video_id", "video"]),
        "bird_id": find_col(crops_df, ["bird_id", "subject_id", "individual_id"]),
        "species_id": find_col(crops_df, ["species_id", "specie_id"]),
        "action_id": find_col(crops_df, ["action_id", "behavior_id", "behaviour_id", "activity_id"]),
        "start": find_col(crops_df, ["start_frame", "frame_start", "start"]),
        "end": find_col(crops_df, ["end_frame", "frame_end", "end"]),
    }
    missing = [k for k, v in cols.items() if v is None and k != "bird_id"]
    if missing:
        logger.warning("could not auto-detect these crops.csv columns: %s. Available: %s",
                        missing, list(crops_df.columns))
    return cols


def load_dataset(data_root: Path) -> dict:
    files = {
        "behaviors": data_root / "behaviors_ID.csv",
        "species": data_root / "species_ID.csv",
        "bboxes": data_root / "bounding_boxes.csv",
        "crops": data_root / "crops.csv",
        "splits": data_root / "splits.json",
    }
    behaviors_raw = safe_read_csv(files["behaviors"])
    species_raw = safe_read_csv(files["species"])
    bbox_df = safe_read_csv(files["bboxes"])
    crops_df = safe_read_csv(files["crops"])

    splits = None
    if files["splits"].exists():
        try:
            splits = json.loads(files["splits"].read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("could not read splits.json: %s", e)

    species_map = build_id_map(species_raw, ["id", "species_id", "class_id", "specie_id"],
                                ["species", "name", "species_name", "specie", "class_name", "common_name"])
    behavior_map = build_id_map(behaviors_raw, ["id", "behavior_id", "behaviour_id", "activity_id",
                                                  "action_id", "class_id"],
                                 ["behavior", "behaviour", "name", "activity", "action", "label", "class_name"])

    data = {
        "bbox_df": bbox_df, "crops_df": crops_df,
        "species_map": species_map, "behavior_map": behavior_map, "splits": splits,
        "bbox_cols": detect_bbox_columns(bbox_df) if bbox_df is not None else None,
        "crops_cols": detect_crops_columns(crops_df) if crops_df is not None else None,
    }
    logger.info("species map entries: %d, behavior map entries: %d", len(species_map), len(behavior_map))
    if bbox_df is not None:
        logger.info("bounding_boxes.csv: %d rows", bbox_df.shape[0])
    if crops_df is not None:
        logger.info("crops.csv: %d rows", crops_df.shape[0])

    # Some rows hold several boxes under one species/behavior label. Crops use
    # biggest_box(), which can pick the wrong bird in that case, so warn.
    if bbox_df is not None and data["bbox_cols"]["boxes"]:
        sample = bbox_df[data["bbox_cols"]["boxes"]].dropna().sample(
            min(500, bbox_df[data["bbox_cols"]["boxes"]].notna().sum()), random_state=0)
        multi = sum(1 for c in sample if len(parse_bbox_cell(c)) > 1)
        if multi:
            logger.warning(
                "%d/%d sampled bounding_boxes.csv rows contain MULTIPLE boxes under one row. "
                "If a row's 'species'/label applies to only one specific bird, crops built from "
                "biggest_box() on a multi-box row risk a wrong species/behavior label. Manually "
                "inspect a few multi-box rows before trusting classification crop labels.",
                multi, len(sample))
    return data


def _normalize_video_key(name) -> str:
    """Lowercased file stem, so names with/without extension compare equal."""
    return Path(str(name)).stem.strip().lower()


def load_official_splits(splits_raw) -> Optional[dict]:
    """Parse splits.json into {'train': set, 'val': set, 'test': set} of video keys.

    Handles three layouts: flat {train/val/test: [...]}, the same nested one
    level down, and a per-video {name: split} map. Returns None if nothing
    matches; the caller then falls back to a derived grouped split.

    Known issue: the released file uses train_set/val_set/test_set, which the
    aliases below do not include yet, so the run reported in the internship
    report used the derived split.
    """
    if not splits_raw:
        return None

    def as_video_set(value) -> set:
        """Accepts a list of names, a list of dicts, or a dict keyed by name."""
        keys = set()
        if isinstance(value, dict):
            candidates = list(value.keys())
        elif isinstance(value, list):
            candidates = value
        else:
            return keys
        for item in candidates:
            if isinstance(item, str):
                keys.add(_normalize_video_key(item))
            elif isinstance(item, dict):
                for k in ("video", "video_name", "video_id", "name", "file", "filename"):
                    if k in item:
                        keys.add(_normalize_video_key(item[k]))
                        break
        return keys

    split_key_aliases = {"train": ["train", "training"], "val": ["val", "validation", "valid"],
                          "test": ["test", "testing"]}

    # Shape A: {"train": [...], "val": [...], "test": [...]}  (flat, top-level)
    if isinstance(splits_raw, dict) and any(
            any(alias in splits_raw for alias in aliases) for aliases in split_key_aliases.values()):
        result = {}
        for canon, aliases in split_key_aliases.items():
            for alias in aliases:
                if alias in splits_raw:
                    result[canon] = as_video_set(splits_raw[alias])
                    break
        if all(k in result and result[k] for k in ("train", "test")):
            logger.info("splits.json: parsed flat {train/val/test: [...]} shape -- "
                        "%d train / %d val / %d test videos",
                        len(result.get("train", [])), len(result.get("val", [])), len(result.get("test", [])))
            return result

    # Shape B: nested one level, e.g. {"species": {"train": [...], ...}, "behavior": {...}}
    # or {"splits": {"train": [...], ...}} -- try any nested dict that itself
    # matches Shape A.
    if isinstance(splits_raw, dict):
        for _, sub in splits_raw.items():
            if isinstance(sub, dict) and any(
                    any(alias in sub for alias in aliases) for aliases in split_key_aliases.values()):
                result = {}
                for canon, aliases in split_key_aliases.items():
                    for alias in aliases:
                        if alias in sub:
                            result[canon] = as_video_set(sub[alias])
                            break
                if all(k in result and result[k] for k in ("train", "test")):
                    logger.info("splits.json: parsed nested {...: {train/val/test: [...]}} shape -- "
                                "%d train / %d val / %d test videos",
                                len(result.get("train", [])), len(result.get("val", [])),
                                len(result.get("test", [])))
                    return result

    # Shape C: {"video_name": "train"/"val"/"test", ...}  (per-video label map)
    if isinstance(splits_raw, dict):
        values_lower = {str(v).lower() for v in splits_raw.values() if isinstance(v, (str,))}
        if values_lower and values_lower.issubset({"train", "val", "validation", "test", "training", "testing"}):
            result = {"train": set(), "val": set(), "test": set()}
            for video_name, split_label in splits_raw.items():
                sl = str(split_label).lower()
                canon = "train" if sl in ("train", "training") else ("test" if sl in ("test", "testing") else "val")
                result[canon].add(_normalize_video_key(video_name))
            if result["train"] and result["test"]:
                logger.info("splits.json: parsed per-video label-map shape -- "
                            "%d train / %d val / %d test videos",
                            len(result["train"]), len(result["val"]), len(result["test"]))
                return result

    logger.warning("splits.json layout not recognized (top-level keys: %s); falling back to a "
                    "derived grouped split. Results are not directly comparable to the paper.",
                    list(splits_raw.keys()) if isinstance(splits_raw, dict) else type(splits_raw))
    return None


# ============================================================================
# VIDEO INDEX
# ============================================================================
def build_video_index(videos_dir: Path, videos_zip: Path, extract_dir: Path) -> dict:
    index = {}
    if videos_dir.is_dir():
        for f in videos_dir.rglob("*"):
            if f.suffix.lower() in VIDEO_EXTS:
                index[f.name] = str(f)
    if not index and videos_zip.exists():
        logger.info("extracting videos.zip (first run only)...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(videos_zip, "r") as zf:
            zf.extractall(extract_dir)
        for f in Path(extract_dir).rglob("*"):
            if f.suffix.lower() in VIDEO_EXTS:
                index[f.name] = str(f)
    if index:
        logger.info("video index ready: %d files", len(index))
    return index


def resolve_data_video_index(data_root: Path, output_dir: Path) -> dict:
    for name in VIDEO_DIR_CANDIDATE_NAMES:
        candidate = data_root / name
        if candidate.is_dir():
            idx = build_video_index(candidate, data_root / "videos.zip", output_dir / "videos_extracted")
            if idx:
                return idx
    return build_video_index(data_root / "videos", data_root / "videos.zip", output_dir / "videos_extracted")


def resolve_video_path(video_name: str, video_index: dict):
    vn = Path(str(video_name)).name.strip()
    if vn in video_index:
        return video_index[vn]
    vn_lower = vn.lower()
    for name, path in video_index.items():
        if name.lower() == vn_lower:
            return path
    if not Path(vn).suffix:
        for ext in VIDEO_EXTS:
            target = vn_lower + ext
            for name, path in video_index.items():
                if name.lower() == target:
                    return path
    for name, path in video_index.items():
        nl = name.lower()
        if vn_lower in nl or nl in vn_lower:
            return path
    return None


def _read_and_crop(cv2, cap_cache, vpath, frame_idx, box):
    if vpath not in cap_cache:
        cap_cache[vpath] = cv2.VideoCapture(vpath)
    cap = cap_cache[vpath]
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(int(x1), w_img)), max(0, min(int(x2), w_img))))
    y1, y2 = sorted((max(0, min(int(y1), h_img)), max(0, min(int(y2), h_img))))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return frame[y1:y2, x1:x2]


def _safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def biggest_box(boxes: list):
    best, best_area = None, -1.0
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        area = abs((x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area, best = area, b
    return best


def stratified_sample(df: pd.DataFrame, group_col: str, n_per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(group_col, sort=False):
        n = min(len(g), n_per_group)
        if n > 0:
            parts.append(g.sample(n, random_state=seed))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=False)


# ============================================================================
# SAMPLE BUILDERS (each sample keeps video_name and bird_id for grouping)
# ============================================================================
def build_species_crops(data: dict, video_index: dict, out_dir: Path, per_class: int, seed: int) -> list:
    bbox_df, bcols = data["bbox_df"], data["bbox_cols"]
    if bbox_df is None or not bcols or not all([bcols["species"], bcols["video"], bcols["frame"], bcols["boxes"]]):
        logger.warning("cannot build species crops: required columns missing")
        return []
    keep_cols = [bcols["species"], bcols["video"], bcols["frame"], bcols["boxes"]]
    if bcols.get("bird_id"):
        keep_cols.append(bcols["bird_id"])
    work = bbox_df[keep_cols].dropna(subset=[bcols["species"], bcols["video"], bcols["frame"]])
    sampled = stratified_sample(work, bcols["species"], per_class * 3, seed)

    if not video_index:
        logger.warning("no video files available, cannot build species crops")
        return []
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, per_class_count, cap_cache = [], defaultdict(int), {}
    n_video_miss = 0

    for _, r in sampled.iterrows():
        label = r[bcols["species"]]
        if per_class_count[label] >= per_class:
            continue
        box = biggest_box(parse_bbox_cell(r[bcols["boxes"]]))
        if box is None:
            continue
        vpath = resolve_video_path(r[bcols["video"]], video_index)
        if vpath is None:
            n_video_miss += 1
            continue
        crop = _read_and_crop(cv2, cap_cache, vpath, int(r[bcols["frame"]]), box["box"])
        if crop is None:
            continue
        fpath = out_dir / f"{_safe_name(label)}_{len(samples)}.jpg"
        cv2.imwrite(str(fpath), crop)
        samples.append({
            "image_path": str(fpath), "label": label, "video_name": str(r[bcols["video"]]),
            "bird_id": str(r[bcols["bird_id"]]) if bcols.get("bird_id") and pd.notna(r.get(bcols["bird_id"])) else None,
        })
        per_class_count[label] += 1

    for cap in cap_cache.values():
        cap.release()
    logger.info("built %d species crops across %d classes%s", len(samples), len(per_class_count),
                f" ({n_video_miss} rows skipped: video not found)" if n_video_miss else "")
    return samples


def build_behavior_clip_samples(data: dict, video_index: dict, out_dir: Path, per_class: int,
                                 frames_per_clip: int, seed: int) -> list:
    """Sample up to `frames_per_clip` frames spread over each clip's
    [start, end] range. Returns one dict per clip with its frame paths."""
    crops_df, ccols = data["crops_df"], data["crops_cols"]
    bbox_df, bcols = data["bbox_df"], data["bbox_cols"]
    if crops_df is None or ccols is None or bbox_df is None or bcols is None:
        logger.warning("cannot build behavior clips: required files/columns missing")
        return []

    if "behavior_name" not in crops_df.columns:
        crops_df = crops_df.copy()
        crops_df["behavior_name"] = (
            crops_df[ccols["action_id"]].map(data["behavior_map"]).fillna(crops_df[ccols["action_id"]].astype(str))
        )
    sampled = stratified_sample(crops_df, "behavior_name", per_class * 3, seed)

    bbox_index = {}
    for video_name, frame, boxcell in zip(bbox_df[bcols["video"]], bbox_df[bcols["frame"]], bbox_df[bcols["boxes"]]):
        bbox_index.setdefault(video_name, {})[int(frame)] = boxcell

    if not video_index:
        logger.warning("no video files available, cannot build behavior clips")
        return []
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    clips, per_class_count, cap_cache = [], defaultdict(int), {}
    n_video_miss = n_no_frame_match = 0

    for clip_i, (_, r) in enumerate(sampled.iterrows()):
        label = r["behavior_name"]
        if per_class_count[label] >= per_class:
            continue
        video_name = r[ccols["video"]]
        start_f, end_f = int(r[ccols["start"]]), int(r[ccols["end"]])
        frames_for_video = bbox_index.get(video_name)
        if not frames_for_video:
            n_no_frame_match += 1
            continue

        # spread targets over the clip and snap each to the nearest annotated frame (<= 60 frames away)
        targets = np.linspace(start_f, end_f, num=min(frames_per_clip, max(1, end_f - start_f + 1)))
        chosen_frames = []
        for t in targets:
            candidate = min(frames_for_video, key=lambda f: abs(f - t))
            if abs(candidate - t) <= 60 and candidate not in chosen_frames:
                chosen_frames.append(candidate)
        if not chosen_frames:
            n_no_frame_match += 1
            continue

        vpath = resolve_video_path(video_name, video_index)
        if vpath is None:
            n_video_miss += 1
            continue

        frame_paths = []
        for fi in chosen_frames:
            box = biggest_box(parse_bbox_cell(frames_for_video[fi]))
            if box is None:
                continue
            crop = _read_and_crop(cv2, cap_cache, vpath, fi, box["box"])
            if crop is None:
                continue
            fpath = out_dir / f"{_safe_name(label)}_{clip_i}_f{fi}.jpg"
            cv2.imwrite(str(fpath), crop)
            frame_paths.append(str(fpath))
        if not frame_paths:
            continue

        clips.append({"clip_id": clip_i, "label": label, "video_name": str(video_name),
                       "frame_paths": frame_paths})
        per_class_count[label] += 1

    for cap in cap_cache.values():
        cap.release()
    logger.info("built %d behavior clips (avg %.1f frames/clip) across %d classes%s", len(clips),
                np.mean([len(c["frame_paths"]) for c in clips]) if clips else 0.0, len(per_class_count),
                f" ({n_video_miss} video misses, {n_no_frame_match} no nearby frame)"
                if (n_video_miss or n_no_frame_match) else "")
    return clips


def build_species_detection_samples(data: dict, video_index: dict, out_dir: Path,
                                     det_cfg, seed: int):
    """One sample per (video, frame) with every box and species label in it.

    sampling_mode "stride": every `frame_stride`-th annotated frame per video
    (the paper's downsampling); "random": up to `frames_per_video` random
    frames per video. `max_frames` caps the total in both modes.
    """
    bbox_df, bcols = data["bbox_df"], data["bbox_cols"]
    if bbox_df is None or not bcols or not all([bcols["species"], bcols["video"], bcols["frame"], bcols["boxes"]]):
        logger.warning("cannot build detection samples: required columns missing")
        return [], {}
    work = bbox_df[[bcols["video"], bcols["frame"], bcols["species"], bcols["boxes"]]].dropna(
        subset=[bcols["video"], bcols["frame"], bcols["species"]])
    work = work.assign(_group_key=list(zip(work[bcols["video"]], work[bcols["frame"]])))

    groups_per_video = defaultdict(list)
    for key, _ in work.groupby("_group_key", sort=False):
        groups_per_video[key[0]].append(key)

    rng = random.Random(seed)
    selected = []
    if det_cfg.sampling_mode == "stride":
        for video, keys in groups_per_video.items():
            keys_sorted = sorted(keys, key=lambda k: k[1])
            selected.extend(keys_sorted[::max(1, det_cfg.frame_stride)])
        rng.shuffle(selected)
        if det_cfg.max_frames:
            selected = selected[:det_cfg.max_frames]
        logger.info("sampled %d frames across %d videos (stride=%d, matching the paper's downsampling "
                    "approach%s)", len(selected), len(groups_per_video), det_cfg.frame_stride,
                    f", capped at {det_cfg.max_frames} total" if det_cfg.max_frames else ", uncapped")
    else:
        for keys in groups_per_video.values():
            keys = list(keys)
            rng.shuffle(keys)
            selected.extend(keys[:det_cfg.frames_per_video])
        rng.shuffle(selected)
        if det_cfg.max_frames:
            selected = selected[:det_cfg.max_frames]
        logger.info("sampled %d frames across %d videos (random mode, cap: %d/video, %s total)",
                    len(selected), len(groups_per_video), det_cfg.frames_per_video,
                    det_cfg.max_frames if det_cfg.max_frames else "uncapped")
    selected_set = set(selected)

    if not video_index:
        logger.warning("no video files available, cannot build detection frames")
        return [], {}
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = work[work["_group_key"].isin(selected_set)].groupby("_group_key", sort=False)

    samples, label_set, cap_cache = [], set(), {}
    n_video_miss = n_no_valid_box = 0
    for (video_name, frame_idx), rows in grouped:
        vpath = resolve_video_path(video_name, video_index)
        if vpath is None:
            n_video_miss += 1
            continue
        try:
            if vpath not in cap_cache:
                cap_cache[vpath] = cv2.VideoCapture(vpath)
            cap = cap_cache[vpath]
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_idx) - 1, 0))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h_img, w_img = frame.shape[:2]
            boxes, labels = [], []
            for _, r in rows.iterrows():
                species_label = r[bcols["species"]]
                for parsed in parse_bbox_cell(r[bcols["boxes"]]):
                    x1, y1, x2, y2 = parsed["box"]
                    x1, x2 = sorted((max(0, min(int(x1), w_img)), max(0, min(int(x2), w_img))))
                    y1, y2 = sorted((max(0, min(int(y1), h_img)), max(0, min(int(y2), h_img))))
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    labels.append(species_label)
            if not boxes:
                n_no_valid_box += 1
                continue
            fname = f"{_safe_name(video_name)}_f{int(frame_idx)}.jpg"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), frame)
            samples.append({"image_path": str(fpath), "video_name": str(video_name), "boxes": boxes, "labels": labels})
            label_set.update(labels)
        except Exception:
            continue
    for cap in cap_cache.values():
        cap.release()

    label_to_idx = {name: i for i, name in enumerate(sorted(label_set))}
    n_boxes = sum(len(s["boxes"]) for s in samples)
    logger.info("built %d detection frames, %d bird instances, %d species%s%s",
                len(samples), n_boxes, len(label_to_idx),
                f", {n_video_miss} frames skipped (video not found)" if n_video_miss else "",
                f", {n_no_valid_box} frames skipped (no valid box)" if n_no_valid_box else "")
    return samples, label_to_idx


# ============================================================================
# EMBEDDING EXTRACTION (frozen backbones, cached to disk)
# ============================================================================
def load_backbone(name: str, device):
    import torch.nn as nn
    from torchvision import models

    spec = BACKBONE_REGISTRY[name]
    if spec["kind"] == "hub":
        try:
            import torch
            model = torch.hub.load(spec["repo"], spec["hub_name"])
            model.eval().to(device)
            for p in model.parameters():
                p.requires_grad_(False)
            return model, spec["embed_dim"], spec["input_size"]
        except Exception as e:
            logger.warning("could not load %s via torch.hub (%s) -- falling back to resnet18. "
                            "DINOv2 needs internet access on first use to download its weights.",
                            name, e)
            name = "resnet18"
            spec = BACKBONE_REGISTRY[name]

    builders = {"resnet18": models.resnet18, "mobilenet_v2": models.mobilenet_v2,
                "efficientnet_b0": models.efficientnet_b0, "densenet121": models.densenet121}
    weight_enums = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                    "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
                    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
                    "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1}
    try:
        net = builders[name](weights=weight_enums[name])
    except Exception:
        net = builders[name](weights=None)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    pooler = nn.AdaptiveAvgPool2d(1)
    if name == "resnet18":
        extractor = nn.Sequential(*list(net.children())[:-2], pooler, nn.Flatten())
    elif name in ("mobilenet_v2", "efficientnet_b0"):
        extractor = nn.Sequential(net.features, pooler, nn.Flatten())
    else:  # densenet121
        extractor = nn.Sequential(net.features, nn.ReLU(inplace=True), pooler, nn.Flatten())
    extractor.eval().to(device)
    return extractor, spec["embed_dim"], spec["input_size"]


@dataclasses.dataclass
class _Embedder:
    model: object
    device: object
    input_size: int
    batch_size: int = 16

    def __call__(self, image_paths: list) -> np.ndarray:
        import torch
        from torchvision import transforms
        from PIL import Image

        tf = transforms.Compose([
            transforms.Resize(int(self.input_size * 256 / 224)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        feats = []
        with torch.no_grad():
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                imgs = []
                for p in batch_paths:
                    try:
                        imgs.append(tf(Image.open(p).convert("RGB")))
                    except Exception:
                        imgs.append(torch.zeros(3, self.input_size, self.input_size))
                x = torch.stack(imgs).to(self.device)
                out = self.model(x)
                if isinstance(out, dict):  # some DINOv2 builds return a dict
                    out = out["x_norm_clstoken"] if "x_norm_clstoken" in out else next(iter(out.values()))
                feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


def _cache_key(image_paths: list) -> str:
    h = hashlib.sha1("|".join(image_paths).encode("utf-8")).hexdigest()[:16]
    return h


def get_or_build_embeddings(backbone_name: str, image_paths: list, cache_dir: Path,
                             device, batch_size: int) -> np.ndarray:
    """Cache keyed on a hash of the ordered path list, so a new sample set
    never reuses old embeddings."""
    key = _cache_key(image_paths)
    cache_file = cache_dir / f"{backbone_name}_{key}.npy"
    if cache_file.exists():
        logger.info("using cached embeddings: %s", cache_file.name)
        return np.load(cache_file)

    model, embed_dim, input_size = load_backbone(backbone_name, device)
    embedder = _Embedder(model, device, input_size, batch_size)
    t0 = time.time()
    emb = embedder(image_paths)
    logger.info("extracted %s embeddings for %d images in %.1fs (dim=%d)",
                backbone_name, len(image_paths), time.time() - t0, emb.shape[1])
    np.save(cache_file, emb)
    return emb


def pool_clip_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    """[n_frames, dim] -> [3*dim]: mean, std and last - first (a cheap motion proxy)."""
    if frame_embeddings.shape[0] == 1:
        z = np.zeros_like(frame_embeddings[0])
        return np.concatenate([frame_embeddings[0], z, z])
    mean = frame_embeddings.mean(axis=0)
    std = frame_embeddings.std(axis=0)
    delta = frame_embeddings[-1] - frame_embeddings[0]
    return np.concatenate([mean, std, delta])


# ============================================================================
# STATISTICS
# ============================================================================
def mcnemar_test(y_true, pred_a, pred_b, exact_threshold: int = 25) -> dict:
    from scipy import stats as spstats
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    correct_a, correct_b = (pred_a == y_true), (pred_b == y_true)
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n_discordant = n01 + n10
    if n_discordant == 0:
        return {"n10": n10, "n01": n01, "n_discordant": 0, "test": "no_discordant_pairs",
                "statistic": np.nan, "p_value": 1.0}
    if n_discordant < exact_threshold:
        k = min(n01, n10)
        result = spstats.binomtest(k, n_discordant, 0.5, alternative="two-sided")
        return {"n10": n10, "n01": n01, "n_discordant": n_discordant, "test": "exact_binomial",
                "statistic": float(k), "p_value": float(result.pvalue)}
    stat = (abs(n01 - n10) - 1) ** 2 / n_discordant
    p = float(1 - spstats.chi2.cdf(stat, df=1))
    return {"n10": n10, "n01": n01, "n_discordant": n_discordant, "test": "chi2_continuity_corrected",
            "statistic": float(stat), "p_value": p}


def holm_bonferroni(pvals) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def diff_ci_95(acc_a: float, acc_b: float, n: int) -> tuple:
    """Normal-approximation 95% CI on the difference of two accuracies (same n samples)."""
    diff = acc_a - acc_b
    se = np.sqrt((acc_a * (1 - acc_a) + acc_b * (1 - acc_b)) / max(n, 1))
    return diff, diff - 1.96 * se, diff + 1.96 * se


# ============================================================================
# CLASSIFICATION HEADS (trained on cached embeddings)
# ============================================================================
def build_head(name: str, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    if name == "mlp":
        return make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(128,), max_iter=800,
                                            early_stopping=True, random_state=seed))
    raise ValueError(f"unknown head '{name}'")


# ============================================================================
# PHASE 1: CLASSIFICATION (grouped CV, locked test fold)
# ============================================================================
def run_classification_task(task_name: str, entities: list, embeddings_by_backbone: dict,
                             cfg: ClassificationConfig, stats_cfg: StatsConfig,
                             results_dir: Path, seed: int, official_test_videos: Optional[set] = None) -> dict:
    """Grouped CV + locked test evaluation for one task.

    `entities`: dicts with 'label', 'video_name' and optionally 'bird_id'.
    `embeddings_by_backbone[name]`: [N, D] array aligned with `entities`.
    If `official_test_videos` is given, the test fold is exactly those videos;
    otherwise one StratifiedGroupKFold fold is held out. Inner CV groups by
    bird_id when available, else by video.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score

    labels_all = sorted(set(e["label"] for e in entities))
    label_to_idx = {l: i for i, l in enumerate(labels_all)}
    y = np.array([label_to_idx[e["label"]] for e in entities])
    groups = np.array([e.get("bird_id") or e["video_name"] for e in entities])

    if official_test_videos:
        video_names = np.array([_normalize_video_key(e["video_name"]) for e in entities])
        test_mask = np.isin(video_names, list(official_test_videos))
        dev_idx, test_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
        if len(test_idx) == 0:
            logger.warning("[%s] no sample matches the official test videos (naming mismatch?); "
                            "using a derived split instead", task_name)
            official_test_videos = None
        else:
            logger.info("[%s] official test partition: %d dev / %d test samples",
                        task_name, len(dev_idx), len(test_idx))
    if not official_test_videos:
        n_groups = len(set(groups))
        if n_groups < cfg.outer_test_fraction_folds:
            logger.warning("[%s] only %d distinct groups; skipping locked test split, using all data for CV",
                            task_name, n_groups)
            dev_idx, test_idx = np.arange(len(entities)), np.array([], dtype=int)
        else:
            outer = StratifiedGroupKFold(n_splits=cfg.outer_test_fraction_folds, shuffle=True, random_state=seed)
            dev_idx, test_idx = next(outer.split(np.zeros(len(entities)), y, groups=groups))
        logger.info("[%s] derived split: %d dev / %d locked test samples, %d groups",
                    task_name, len(dev_idx), len(test_idx), n_groups)

    min_group_count_dev = pd.Series(groups[dev_idx]).value_counts().shape[0]
    k_folds = min(cfg.inner_k_folds, min_group_count_dev)
    if k_folds < 2:
        logger.warning("[%s] not enough groups for inner CV; skipping", task_name)
        return {}

    per_fold_rows, oof_preds = [], {}
    test_rows = []

    for backbone_name, emb in embeddings_by_backbone.items():
        emb_dev, y_dev, groups_dev = emb[dev_idx], y[dev_idx], groups[dev_idx]
        for head_name in cfg.heads:
            key = f"{backbone_name}__{head_name}"
            oof = np.full(len(dev_idx), -1, dtype=int)
            inner = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            for fold_idx, (tr, va) in enumerate(inner.split(emb_dev, y_dev, groups=groups_dev)):
                head = build_head(head_name, seed)
                head.fit(emb_dev[tr], y_dev[tr])
                preds = head.predict(emb_dev[va])
                oof[va] = preds
                acc = accuracy_score(y_dev[va], preds)
                f1 = f1_score(y_dev[va], preds, average="macro", zero_division=0)
                per_fold_rows.append({"task": task_name, "backbone": backbone_name, "head": head_name,
                                       "fold": fold_idx, "val_accuracy": round(float(acc), 4),
                                       "val_macro_f1": round(float(f1), 4), "n_val": len(va)})
            oof_preds[key] = oof

            # refit on all dev data, evaluate once on the test fold
            if len(test_idx) > 0:
                head = build_head(head_name, seed)
                head.fit(emb_dev, y_dev)
                test_preds = head.predict(emb[test_idx])
                test_rows.append({
                    "task": task_name, "backbone": backbone_name, "head": head_name,
                    "test_accuracy": round(float(accuracy_score(y[test_idx], test_preds)), 4),
                    "test_macro_f1": round(float(f1_score(y[test_idx], test_preds, average="macro",
                                                            zero_division=0)), 4),
                    "n_test": len(test_idx),
                })

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(results_dir / f"cv_results_{task_name}.csv", index=False)

    summary_rows = []
    for (backbone_name, head_name), sub in per_fold_df.groupby(["backbone", "head"]):
        accs = sub["val_accuracy"].values
        n = len(accs)
        mean_acc, std_acc = float(accs.mean()), float(accs.std(ddof=1)) if n > 1 else 0.0
        summary_rows.append({"task": task_name, "backbone": backbone_name, "head": head_name,
                              "k_folds": n, "mean_accuracy": round(mean_acc, 4),
                              "std_accuracy": round(std_acc, 4)})
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_accuracy", ascending=False)
    summary_df.to_csv(results_dir / f"cv_summary_{task_name}.csv", index=False)

    if test_rows:
        test_df = pd.DataFrame(test_rows).sort_values("test_accuracy", ascending=False)
        test_df.to_csv(results_dir / f"locked_test_{task_name}.csv", index=False)
        logger.info("[%s] best locked-test result: %s / %s -> acc=%.4f (n=%d)",
                    task_name, test_df.iloc[0]["backbone"], test_df.iloc[0]["head"],
                    test_df.iloc[0]["test_accuracy"], test_df.iloc[0]["n_test"])

    # pairwise McNemar + Holm on out-of-fold dev predictions (test fold not used)
    pair_rows = []
    combos = list(oof_preds.keys())
    for combo_a, combo_b in itertools.combinations(combos, 2):
        valid = (oof_preds[combo_a] >= 0) & (oof_preds[combo_b] >= 0)
        if valid.sum() == 0:
            continue
        y_dev = y[dev_idx]
        result = mcnemar_test(y_dev[valid], oof_preds[combo_a][valid], oof_preds[combo_b][valid],
                               exact_threshold=stats_cfg.mcnemar_exact_threshold)
        acc_a = float(accuracy_score(y_dev[valid], oof_preds[combo_a][valid]))
        acc_b = float(accuracy_score(y_dev[valid], oof_preds[combo_b][valid]))
        diff, ci_lo, ci_hi = diff_ci_95(acc_a, acc_b, int(valid.sum()))
        pair_rows.append({
            "task": task_name, "combo_a": combo_a, "combo_b": combo_b, "n_compared": int(valid.sum()),
            "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
            "acc_diff": round(diff, 4), "acc_diff_ci95_lo": round(ci_lo, 4), "acc_diff_ci95_hi": round(ci_hi, 4),
            "test_type": result["test"], "p_value": result["p_value"],
        })
    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["p_value_holm"] = holm_bonferroni(pair_df["p_value"].values)
        pair_df[f"significant_at_{stats_cfg.alpha}_holm"] = pair_df["p_value_holm"] < stats_cfg.alpha
        pair_df = pair_df.sort_values("p_value").reset_index(drop=True)
        pair_df.to_csv(results_dir / f"mcnemar_{task_name}.csv", index=False)
        n_sig = int(pair_df[f"significant_at_{stats_cfg.alpha}_holm"].sum())
        logger.info("[%s] McNemar (Holm): %d/%d pairwise comparisons significant at alpha=%s",
                     task_name, n_sig, len(pair_df), stats_cfg.alpha)

    return {"summary": summary_df, "pairwise": pair_df if not pair_df.empty else None}


# ============================================================================
# PHASE 2: DETECTION (YOLOv9 via ultralytics)
#   2a: config tuning with grouped k-fold on train/val videos only
#   2b: train the chosen config, evaluate once per seed on the test videos
# A k-fold mean and a single held-out test number are different measurements,
# so only 2b is compared with the paper. Tuning never touches the test videos.
# ============================================================================
PAPER_YOLOV9_METRICS = {"precision": 0.835, "recall": 0.759, "mAP50": 0.801, "mAP50_95": 0.556}


def box_to_yolo_line(class_idx: int, box, img_w: int, img_h: int) -> str:
    """(x1, y1, x2, y2) in pixels -> 'cls xc yc w h' normalized to [0, 1]."""
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    xc, yc, w, h = (min(max(v, 0.0), 1.0) for v in (xc, yc, w, h))
    return f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def assign_video_partition(samples: list, official_splits: Optional[dict], seed: int) -> dict:
    """{video_key: 'train'|'val'|'test'} for every video in `samples`.
    Uses splits.json when available, else a derived 70/15/15 video split."""
    all_keys = sorted({_normalize_video_key(s["video_name"]) for s in samples})

    if official_splits:
        partition = {}
        for k in all_keys:
            if k in official_splits.get("train", set()):
                partition[k] = "train"
            elif k in official_splits.get("val", set()):
                partition[k] = "val"
            elif k in official_splits.get("test", set()):
                partition[k] = "test"
        covered = len(partition)
        if covered >= max(3, int(0.5 * len(all_keys))):
            logger.info("detection: using OFFICIAL video partition -- %d/%d sampled videos matched "
                        "(train=%d, val=%d, test=%d)", covered, len(all_keys),
                        sum(v == "train" for v in partition.values()),
                        sum(v == "val" for v in partition.values()),
                        sum(v == "test" for v in partition.values()))
            # unmatched videos are left out
            return partition
        logger.warning("detection: official split matched only %d/%d videos; using a derived "
                        "70/15/15 split", covered, len(all_keys))

    rng = np.random.default_rng(seed)
    keys = np.array(all_keys)
    rng.shuffle(keys)
    n = len(keys)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    test_keys, val_keys = set(keys[:n_test]), set(keys[n_test:n_test + n_val])
    train_keys = set(keys[n_test + n_val:])
    logger.info("detection: derived fallback partition -- %d train / %d val / %d test videos",
                len(train_keys), len(val_keys), len(test_keys))
    return {**{k: "train" for k in train_keys}, **{k: "val" for k in val_keys}, **{k: "test" for k in test_keys}}


def export_yolo_dataset(samples: list, label_to_idx: dict, partition: dict, out_dir: Path,
                         splits=("train", "val", "test")) -> tuple:
    """Write an ultralytics dataset (images/, labels/, data.yaml).
    Returns (data_yaml_path, {split: n_images})."""
    from PIL import Image
    out_dir = Path(out_dir)
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {s: 0 for s in splits}
    for sample in samples:
        vkey = _normalize_video_key(sample["video_name"])
        split = partition.get(vkey)
        if split not in splits:
            continue
        src = Path(sample["image_path"])
        stem = src.stem
        dst_img = out_dir / "images" / split / f"{stem}.jpg"
        if not dst_img.exists():
            shutil.copyfile(src, dst_img)
        try:
            with Image.open(src) as im:
                img_w, img_h = im.size
        except Exception:
            continue
        lines = [box_to_yolo_line(label_to_idx[label], box, img_w, img_h)
                 for box, label in zip(sample["boxes"], sample["labels"])]
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        counts[split] += 1

    for split in splits:
        (logger.warning if counts[split] == 0 else logger.info)(
            "YOLO export: split '%s' -> %d images%s", split, counts[split],
            " (empty, check video-name matching)" if counts[split] == 0 else "")

    names = {idx: name for name, idx in label_to_idx.items()}
    data_yaml_dict = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                       "names": names}
    if "test" in splits:
        data_yaml_dict["test"] = "images/test"
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml_dict, f, sort_keys=False)
    return yaml_path, counts


def _yolo_metrics_to_dict(metrics) -> dict:
    box = metrics.box
    return {"precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50_95": float(box.map)}


def _train_and_val_yolo(data_yaml: Path, project_dir: Path, run_name: str, yolo_model: str,
                         epochs: int, imgsz: int, batch: int, seed: int, lr0: float,
                         conf: float, iou: float, device: str, val_split: str, resume: bool) -> dict:
    """One train() + val() run. Metrics are saved to <run>/final_metrics.json
    so --resume can skip runs that already finished."""
    from ultralytics import YOLO
    metrics_path = Path(project_dir) / run_name / "final_metrics.json"
    if resume and metrics_path.exists():
        logger.info("[%s] cached metrics found, skipping (delete %s to rerun)",
                    run_name, metrics_path)
        return json.loads(metrics_path.read_text())

    model = YOLO(yolo_model)
    t0 = time.time()
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, lr0=lr0,
                project=str(project_dir), name=run_name, exist_ok=True, device=(device or None),
                verbose=False, plots=False)
    train_time = time.time() - t0
    val_metrics = model.val(data=str(data_yaml), split=val_split, conf=conf, iou=iou,
                             project=str(project_dir), name=f"{run_name}_val", exist_ok=True, verbose=False)
    result = _yolo_metrics_to_dict(val_metrics)
    result["train_time_sec"] = round(train_time, 1)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result))
    logger.info("[%s] precision=%.4f recall=%.4f mAP50=%.4f mAP50-95=%.4f (%.0fs)",
                run_name, result["precision"], result["recall"], result["mAP50"], result["mAP50_95"], train_time)
    return result


def run_dev_cv_yolov9(samples: list, partition: dict, det_cfg: DetectionConfig, results_dir: Path,
                       ckpt_dir: Path, seed: int, resume: bool):
    """Phase 2a: grouped k-fold over train/val videos only (test videos excluded)."""
    from sklearn.model_selection import GroupKFold

    dev_keys = sorted([k for k, v in partition.items() if v in ("train", "val")])
    dev_samples = [s for s in samples if _normalize_video_key(s["video_name"]) in dev_keys]
    if len(dev_samples) == 0:
        logger.error("dev CV: no samples in the train/val partition, skipping tuning")
        return None, {}
    yolo_label_to_idx = {name: i for i, name in enumerate(sorted({l for s in samples for l in s["labels"]}))}

    dev_video_arr = np.array([_normalize_video_key(s["video_name"]) for s in dev_samples])
    n_unique = len(set(dev_video_arr))
    k_folds = max(2, min(det_cfg.dev_cv_folds, n_unique // 2)) if n_unique >= 4 else 0
    if k_folds == 0:
        logger.warning("dev CV: only %d dev videos, using the first config without tuning", n_unique)
        return (det_cfg.dev_configs[0] if det_cfg.dev_configs else {}), yolo_label_to_idx

    gkf = GroupKFold(n_splits=k_folds)
    fold_splits = list(gkf.split(np.zeros(len(dev_samples)), groups=dev_video_arr))

    all_rows = []
    for cfg_idx, hp in enumerate(det_cfg.dev_configs):
        tag = "_".join(f"{k}{v}" for k, v in hp.items()) or f"cfg{cfg_idx}"
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits):
            tr_keys, va_keys = set(dev_video_arr[tr_idx]), set(dev_video_arr[va_idx])
            assert tr_keys.isdisjoint(va_keys), "video present in both train and val fold"
            fold_partition = {k: "train" for k in tr_keys}
            fold_partition.update({k: "val" for k in va_keys})
            export_dir = ckpt_dir / "dev_cv" / f"{tag}_fold{fold_idx}" / "yolo_data"
            data_yaml, counts = export_yolo_dataset(dev_samples, yolo_label_to_idx, fold_partition,
                                                     export_dir, splits=("train", "val"))
            if counts["train"] == 0 or counts["val"] == 0:
                logger.warning("dev CV: skipping %s fold %d (empty train or val)", tag, fold_idx)
                continue
            metrics = _train_and_val_yolo(
                data_yaml, ckpt_dir / "dev_cv", f"{tag}_fold{fold_idx}", det_cfg.yolo_model,
                epochs=det_cfg.dev_epochs, imgsz=hp.get("imgsz", det_cfg.imgsz), batch=det_cfg.batch_size,
                seed=seed, lr0=hp.get("lr0", det_cfg.lr0), conf=det_cfg.conf_threshold, iou=det_cfg.iou_nms,
                device=det_cfg.device, val_split="val", resume=resume)
            all_rows.append({"config": tag, "fold": fold_idx, **hp, **metrics})

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir / "dev_cv_yolov9_results.csv", index=False)
    if df.empty:
        logger.error("dev CV produced no usable folds")
        return (det_cfg.dev_configs[0] if det_cfg.dev_configs else {}), yolo_label_to_idx

    summary = df.groupby("config").agg(
        mean_mAP50_95=("mAP50_95", "mean"), std_mAP50_95=("mAP50_95", "std"),
        mean_mAP50=("mAP50", "mean"), mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"), n_folds=("fold", "count")).reset_index()
    summary.to_csv(results_dir / "dev_cv_yolov9_summary.csv", index=False)

    metric_col = f"mean_{det_cfg.tuning_metric}" if f"mean_{det_cfg.tuning_metric}" in summary.columns else "mean_mAP50_95"
    best_row = summary.sort_values(metric_col, ascending=False).iloc[0]
    best_tag = best_row["config"]
    best_hp = next((hp for i, hp in enumerate(det_cfg.dev_configs)
                     if ("_".join(f"{k}{v}" for k, v in hp.items()) or f"cfg{i}") == best_tag), {})
    logger.info("dev CV winner: '%s' (%s=%.4f over %d folds)",
                best_tag, metric_col, best_row[metric_col], int(best_row["n_folds"]))
    return best_hp, yolo_label_to_idx


def run_final_report_yolov9(samples: list, partition: dict, best_hp: dict, det_cfg: DetectionConfig,
                             results_dir: Path, ckpt_dir: Path, resume: bool):
    """Phase 2b: train on train(+val), evaluate once per seed on the test videos."""
    yolo_label_to_idx = {name: i for i, name in enumerate(sorted({l for s in samples for l in s["labels"]}))}
    export_dir = ckpt_dir / "final_report" / "yolo_data"
    data_yaml, counts = export_yolo_dataset(samples, yolo_label_to_idx, partition, export_dir,
                                             splits=("train", "val", "test"))
    if counts["test"] == 0:
        logger.error("final report: test split is empty (video-name mismatch?), skipping")
        return None
    if counts["val"] == 0:
        logger.warning("final report: val split is empty; training-time validation will not be "
                        "meaningful (test metrics are unaffected)")

    rows = []
    for seed in det_cfg.final_seeds:
        metrics = _train_and_val_yolo(
            data_yaml, ckpt_dir / "final_report", f"seed{seed}", det_cfg.yolo_model,
            epochs=det_cfg.final_epochs, imgsz=(best_hp or {}).get("imgsz", det_cfg.imgsz),
            batch=det_cfg.batch_size, seed=seed, lr0=(best_hp or {}).get("lr0", det_cfg.lr0),
            conf=det_cfg.conf_threshold, iou=det_cfg.iou_nms, device=det_cfg.device,
            val_split="test", resume=resume)
        rows.append({"seed": seed, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "final_report_yolov9_per_seed.csv", index=False)
    summary = {"n_seeds": len(df)}
    for m in ("precision", "recall", "mAP50", "mAP50_95"):
        summary[f"mean_{m}"] = round(float(df[m].mean()), 4)
        summary[f"std_{m}"] = round(float(df[m].std(ddof=1)) if len(df) > 1 else 0.0, 4)
    pd.DataFrame([summary]).to_csv(results_dir / "final_report_yolov9_summary.csv", index=False)

    comparison = pd.DataFrame([
        {"source": "Paper (YOLOv9, official test split)", **PAPER_YOLOV9_METRICS},
        {"source": f"Ours (YOLOv9/{det_cfg.yolo_model} via ultralytics, {det_cfg.final_epochs} epochs, "
                    f"{summary['n_seeds']} seeds, SAME official test split)",
         "precision": summary["mean_precision"], "recall": summary["mean_recall"],
         "mAP50": summary["mean_mAP50"], "mAP50_95": summary["mean_mAP50_95"]},
    ])
    comparison.to_csv(results_dir / "final_report_vs_paper_OFFICIAL_SPLIT.csv", index=False)

    n_beats = int((df["mAP50_95"] > PAPER_YOLOV9_METRICS["mAP50_95"]).sum())
    logger.info("[final_report] mAP50-95 above the paper's value in %d/%d seeds",
                n_beats, summary["n_seeds"])
    return summary

# ============================================================================
# ORCHESTRATION
# ============================================================================
def run_pipeline(cfg: Config) -> None:
    t_start = time.time()
    cfg.make_dirs()
    set_all_seeds(cfg.seed)
    device = get_device()

    data = load_dataset(Path(cfg.data.root))
    if data["bbox_df"] is None:
        logger.error("bounding_boxes.csv could not be loaded -- nothing else can run")
        return
    video_index = resolve_data_video_index(Path(cfg.data.root), cfg.output_path)

    official_splits = None
    if cfg.use_official_split:
        official_splits = load_official_splits(data.get("splits"))
        if official_splits is None:
            logger.warning("Running without the official split; results are not directly comparable "
                            "to the paper. Use --ignore-official-split to silence this.")
    else:
        logger.info("--ignore-official-split set: using derived StratifiedGroupKFold splits throughout.")
    official_test_videos = official_splits["test"] if official_splits else None

    # ---------------- PHASE 1: classification ----------------
    logger.info("=" * 70); logger.info("PHASE 1: CLASSIFICATION (frozen embeddings, grouped CV)"); logger.info("=" * 70)

    if "species" in cfg.classification.tasks:
        species_crops = build_species_crops(data, video_index, cfg.crops_dir / "species",
                                             cfg.sampling.crops_per_class, cfg.seed)
        if species_crops:
            embeddings_by_backbone = {}
            for bb in cfg.classification.backbones:
                paths = [c["image_path"] for c in species_crops]
                embeddings_by_backbone[bb] = get_or_build_embeddings(
                    bb, paths, cfg.embed_cache_dir, device, cfg.classification.embed_batch_size)
            run_classification_task("species", species_crops, embeddings_by_backbone,
                                     cfg.classification, cfg.stats, cfg.results_dir, cfg.seed,
                                     official_test_videos=official_test_videos)

    if "behavior" in cfg.classification.tasks:
        behavior_clips = build_behavior_clip_samples(data, video_index, cfg.crops_dir / "behavior",
                                                       cfg.sampling.crops_per_class,
                                                       cfg.sampling.frames_per_clip, cfg.seed)
        if behavior_clips:
            embeddings_by_backbone = {}
            all_frame_paths = sorted({p for c in behavior_clips for p in c["frame_paths"]})
            path_to_row = {p: i for i, p in enumerate(all_frame_paths)}
            for bb in cfg.classification.backbones:
                frame_emb = get_or_build_embeddings(bb, all_frame_paths, cfg.embed_cache_dir,
                                                      device, cfg.classification.embed_batch_size)
                pooled = np.stack([
                    pool_clip_embeddings(frame_emb[[path_to_row[p] for p in c["frame_paths"]]])
                    for c in behavior_clips
                ])
                embeddings_by_backbone[bb] = pooled
            run_classification_task("behavior", behavior_clips, embeddings_by_backbone,
                                     cfg.classification, cfg.stats, cfg.results_dir, cfg.seed,
                                     official_test_videos=official_test_videos)

    # ---------------- PHASE 2: detection (YOLOv9, two-phase) ----------------
    if cfg.detection.enabled:
        logger.info("=" * 70); logger.info("PHASE 2: DETECTION (YOLOv9 via ultralytics)"); logger.info("=" * 70)
        det_samples, _unused_label_to_idx = build_species_detection_samples(
            data, video_index, cfg.crops_dir / "species_detection", cfg.detection, cfg.seed)
        if det_samples:
            partition = assign_video_partition(det_samples, official_splits, cfg.seed)
            logger.info("-" * 70); logger.info("PHASE 2a: tuning on train/val videos"); logger.info("-" * 70)
            best_hp, _ = run_dev_cv_yolov9(det_samples, partition, cfg.detection, cfg.results_dir,
                                            cfg.ckpt_dir, cfg.seed, cfg.resume)
            logger.info("-" * 70); logger.info("PHASE 2b: multi-seed evaluation on the test videos"); logger.info("-" * 70)
            run_final_report_yolov9(det_samples, partition, best_hp, cfg.detection,
                                     cfg.results_dir, cfg.ckpt_dir, cfg.resume)
    else:
        logger.info("PHASE 2 (detection) disabled")

    logger.info("DONE. Total run time: %.1f minutes.", (time.time() - t_start) / 60)
    logger.info("All results saved under: %s", cfg.output_path)


# ============================================================================
# ENTRY POINT
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Visual WetlandBirds pipeline: frozen-embedding "
                                              "classification + YOLOv9 detection")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tasks", nargs="+", default=None, choices=["species", "behavior"])
    p.add_argument("--backbones", nargs="+", default=None, choices=list(BACKBONE_REGISTRY.keys()))
    p.add_argument("--no-detection", action="store_true")
    p.add_argument("--yolo-model", default=None, help="ultralytics checkpoint name, e.g. yolov9c.pt or yolov9e.pt")
    p.add_argument("--sampling-mode", default=None, choices=["stride", "random"],
                    help="'stride' = every Nth frame (paper), 'random' = per-video random cap")
    p.add_argument("--max-frames", type=int, default=None, help="cap on total detection frames (default: none)")
    p.add_argument("--dev-folds", type=int, default=None, help="k-fold count for Phase 2a dev tuning (default 3)")
    p.add_argument("--dev-epochs", type=int, default=None, help="epochs per Phase 2a dev-CV run (default 30)")
    p.add_argument("--final-epochs", type=int, default=None, help="epochs for Phase 2b (default 50, as in the paper)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds averaged over in Phase 2b (default 42 43 44)")
    p.add_argument("--skip-dev-tuning", action="store_true",
                    help="skip Phase 2a and use the first configured imgsz")
    p.add_argument("--resume", action="store_true",
                    help="skip runs whose metrics are already cached")
    p.add_argument("--ignore-official-split", action="store_true",
                    help="use a derived 70/15/15 split instead of splits.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_cli_overrides(Config(), args)
    setup_logging(Path(cfg.output_dir))
    logger.info("=" * 70)
    logger.info("Visual WetlandBirds pipeline v3")
    logger.info("data_root   = %s", Path(cfg.data.root).resolve())
    logger.info("output_dir  = %s", Path(cfg.output_dir).resolve())
    logger.info("YOLO device = %s", cfg.detection.device)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("torch.cuda.is_available() = True  |  GPU = %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("no CUDA device, YOLO will run on CPU")
            if cfg.detection.device not in ("", "cpu"):
                cfg.detection.device = "cpu"
    except ImportError:
        logger.warning("torch not importable yet")
    logger.info("=" * 70)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
