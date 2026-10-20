# WetlandBirds Internship

Species detection, species classification and behavior recognition on the [Visual WetlandBirds](https://github.com/3dperceptionlab/Visual-WetlandBirds) dataset. The goal was to reproduce the paper's baseline and then improve on it.

Summer internship (July–August 2026), Manouba School of Engineering
**Emna Saada, Souha Hamdani**, supervised by Dr. Ali Ben Abbes and Dr. Feten Chakchouk.

Reference: Rodriguez-Juan et al., *Visual WetlandBirds Dataset: Bird Species Identification and Behavior Recognition in Videos*, Scientific Data 12:1200 (2025). The dataset has 178 videos from La Mata and El Hondo (Alicante), covering 13 species and 7 behaviors.

## Headline result: detection

We used the same detector family as the paper (YOLOv9, `yolov9c`, COCO-pretrained, via ultralytics) and the same downsampling (every 10th annotated frame).

| Metric | Paper | Ours (fold 0) | Δ |
|---|---|---|---|
| Precision | 0.835 | **0.872** | +0.037 |
| Recall | 0.759 | **0.776** | +0.017 |
| mAP50 | 0.801 | **0.833** | +0.032 |
| mAP50-95 | 0.556 | **0.626** | +0.070 (+12.6%) |

**Scope of this result.** This is the first completed tuning fold (Phase 2a, `imgsz=640`, 30 epochs, about 31 h on an RTX 4060 laptop GPU). It was evaluated on a video-grouped split we derived ourselves, not on the official test videos. The multi-seed evaluation on the official split (Phase 2b) is still to run. It is a promising result, not a final claim.

## Classification (locked test fold, v3)

| Task | Best setup | Accuracy | Macro-F1 | n |
|---|---|---|---|---|
| Species (13 classes) | EfficientNet-B0 + logistic regression | 96.95% | 0.960 | 164 |
| Behavior (7 classes) | EfficientNet-B0 + logistic regression | 43.37% | 0.448 | 83 |

- **Species** classification uses crops at known ground-truth boxes. That is an easier task than the paper's joint detection and classification, so the two numbers are not compared.
- **Behavior** is above chance (14.3%) and above the paper's S3D baseline (29%), but below its other four video models (49–56%).

## How the pipeline evolved

| Version | What changed | Why |
|---|---|---|
| v1 | 4 CNNs fine-tuned end to end; Faster R-CNN with random-init heads; CPU only (~53 h per run) | first reproduction |
| v2 | video-grouped CV + locked test fold, frozen cached embeddings, 4-frame behavior pooling, COCO-pretrained Faster R-CNN, letterbox resize, warmup + cosine LR | the audit found crop-level leakage, the official split loaded but never applied, and "best model" claims not supported by McNemar |
| v3 | YOLOv9 (the paper's detector family), stride-10 sampling, two-phase evaluation | removes the architecture confound and gives a comparable number |

The two-phase evaluation works like this:
- **Phase 2a** tunes the configuration with grouped k-fold on train/val videos only. The test videos are never touched, and an assert checks this.
- **Phase 2b** retrains the chosen configuration and evaluates it once per seed on the held-out test videos.

Only Phase 2b is meant to be compared with the paper.

## Running it

```bash
pip install -r requirements.txt

python wetlandbirds_pipeline.py --data-root data                  # full run
python wetlandbirds_pipeline.py --data-root data --no-detection   # classification only (minutes)
python wetlandbirds_pipeline.py --data-root data --skip-dev-tuning
python wetlandbirds_pipeline.py --data-root data --resume         # continue after an interruption
python wetlandbirds_pipeline.py --data-root data --max-frames 200 --dev-epochs 2 --final-epochs 2 --seeds 42   # quick check
```

`data/` should contain the files from the dataset release: `bounding_boxes.csv`, `crops.csv`, `species_ID.csv`, `behaviors_ID.csv`, `splits.json` and `videos/`. Results go to `outputs/model_results/` (CSV tables for CV, the locked test, McNemar tests and YOLO runs).

## Known issues / next steps

- `splits.json` uses the keys `train_set` / `val_set` / `test_set`, which the parser does not recognize yet. The run above therefore fell back to a derived split. Adding these aliases in `load_official_splits()` is the first fix to make.
- Phase 2a is incomplete: `imgsz=960` and folds 1–2 have not run yet. Phase 2b (3 seeds, official test videos) has not started.
- DINOv2 could not download its weights on the training machine, so it silently fell back to ResNet18. Its row in the results is not a real DINOv2 result.
- Some `bounding_boxes.csv` rows hold several boxes under one label. 82 of the 500 rows we sampled have this. Crops use the largest box, which can mislabel the bird in multi-bird frames.
- For behavior, a real temporal model (3D CNN or a video transformer head) is the natural next step after the 4-frame pooling.

## Files

```
wetlandbirds_pipeline.py   full pipeline (classification + YOLOv9 detection)
requirements.txt
```
