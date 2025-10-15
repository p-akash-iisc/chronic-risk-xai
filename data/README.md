# Data Guide

## Sources
- **Pima Indians Diabetes (UCI)** – auto-downloaded by `make data` (no data is committed).
- **Framingham Heart Study** – provide `data/raw/framingham.csv` yourself or set `dataset.source_path` in config.

## Structure
```
data/
├─ raw/        # downloads or your raw CSVs (ignored by git)
├─ processed/  # train/val/test splits after preprocessing
└─ sample/     # tiny synthetic sample for tests & demos
```

## Checksums
When we auto-download public datasets, we validate a SHA-256 checksum.

## Custom Data
1. Drop your CSV into `data/raw/`.
2. Copy a config, edit `dataset.source_path`, `dataset.target`, `features.include`.
3. Run:
   ```bash
   make data CONFIG=configs/custom.yaml
   make train CONFIG=configs/custom.yaml
   ```
