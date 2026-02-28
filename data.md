# Data Preparation

It is recommended to symlink the dataset root to `data/` under the project root directory. If your folder structure differs, you will need to modify the `data_root` path in the corresponding config files.

Data preparation scripts are located in `tools/data_prepare/`.

## Download Datasets

We recommend downloading datasets from [OpenDataLab](https://opendatalab.com/) for faster download speeds. Alternatively, refer to `tools/data_prepare/<dataset_name>/README.md` for official download links.

## Supported Datasets

| Dataset | Classes | Image Size | Annotation Format | Config |
|---|---|---|---|---|
| [DIOR](#dior) | 20 | 800x800 | VOC-style txt | `configs/_base_/datasets/dior.py` |
| [DOTA v1.0](#dota-v10) | 15 | 1024x1024 (after split) | DOTA txt | `configs/_base_/datasets/dota.py` |
| [DOTA v1.5](#dota-v15) | 16 | 1024x1024 (after split) | DOTA txt | `configs/_base_/datasets/dotav15.py` |
| [STAR](#star) | 48 | 1024x1024 (after split) | DOTA txt | `configs/_base_/datasets/star.py` |
| [FAIR1M](#fair1m) | - | 1024x1024 (after split) | DOTA txt | `configs/_base_/datasets/fair.py` |
| [HRSC](#hrsc) | 1 | - | VOC-style xml | `configs/_base_/datasets/hrsc.py` |
| [SSDD](#ssdd) | 1 | - | DOTA txt | `configs/_base_/datasets/ssdd.py` |
| [HRSID](#hrsid) | 1 | - | COCO json | `configs/_base_/datasets/hrsid.py` |
| [SRSDD](#srsdd) | 6 | - | DOTA txt | `configs/_base_/datasets/srsdd.py` |
| [RSDD](#rsdd) | 2 | - | VOC-style xml | `configs/_base_/datasets/rsdd.py` |

---

## DIOR

### Directory Structure

```
Point2RBox-v3
├── data
│   ├── dior
│   │   ├── JPEGImages-trainval/      # Train + val images
│   │   ├── JPEGImages-test/          # Test images
│   │   ├── Annotations/
│   │   │   ├── Oriented Bounding Boxes/   # Rotated box annotations
│   │   │   └── Horizontal Bounding Boxes/ # Horizontal box annotations
│   │   └── ImageSets/
│   │       └── Main/
│   │           ├── train.txt
│   │           ├── val.txt
│   │           └── test.txt
```

### Config Path

Set `data_root` to `data/dior/`.

---

## DOTA v1.0

### Raw Directory Structure

```
Point2RBox-v3
├── data
│   ├── DOTA
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
```

### Image Splitting

DOTA raw images can be very large (up to thousands of pixels). They must be split into patches before training.

**Single-scale split** (1024x1024 patches with 200-pixel overlap):

```bash
# Split trainval
python tools/data_prepare/dota/split/img_split.py --base-json \
  tools/data_prepare/dota/split/split_configs/ss_trainval.json

# Split test
python tools/data_prepare/dota/split/img_split.py --base-json \
  tools/data_prepare/dota/split/split_configs/ss_test.json
```

**Multi-scale split** (if multi-scale training is needed):

```bash
python tools/data_prepare/dota/split/img_split.py --base-json \
  tools/data_prepare/dota/split/split_configs/ms_trainval.json

python tools/data_prepare/dota/split/img_split.py --base-json \
  tools/data_prepare/dota/split/split_configs/ms_test.json
```

> **Note:** Before splitting, update the `img_dirs` and `ann_dirs` fields in the JSON config files to your actual raw data paths.

### Directory Structure After Splitting

```
Point2RBox-v3
├── data
│   ├── split_ss_dota/
│   │   ├── trainval/
│   │   │   ├── images/       # Split train+val images
│   │   │   └── annfiles/     # Annotation files (one txt per image)
│   │   └── test/
│   │       ├── images/       # Split test images
│   │       └── annfiles/
```

### Generate COCO-Format Annotations (Optional)

If you need COCO-format annotations:

```bash
python tools/data_prepare/dota/dota2coco.py \
  data/split_ss_dota/trainval/ \
  data/split_ss_dota/trainval.json

python tools/data_prepare/dota/dota2coco.py \
  data/split_ss_dota/test/ \
  data/split_ss_dota/test.json
```

### Config Path

Set `data_root` to `data/split_ss_dota/`.

---

## DOTA v1.5

Similar to DOTA v1.0, but with an additional `container-crane` class (16 classes total). The download and splitting procedures are the same; just use DOTA v1.5 data.

### Config Path

Set `data_root` to `data/split_ss_dotav15/`.

---

## STAR

### Image Splitting

STAR also requires image splitting. The procedure is similar to DOTA. Refer to the DOTA splitting scripts for guidance.

### Directory Structure After Splitting

```
Point2RBox-v3
├── data
│   ├── split_ss_star/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── annfiles/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── annfiles/
│   │   └── test/
│   │       └── images/
```

### Config Path

Set `data_root` to `data/split_ss_star/`.

---

## FAIR1M

### Image Splitting

FAIR1M requires image splitting:

```bash
# Split trainval
python tools/data_prepare/fair/split/img_split.py --base-json \
  tools/data_prepare/fair/split/split_configs/ss_trainval.json

# Split test
python tools/data_prepare/fair/split/img_split.py --base-json \
  tools/data_prepare/fair/split/split_configs/ss_test.json
```

### Config Path

Set `data_root` to `data/split_ss_fair/`.

---

## HRSC

### Directory Structure

```
Point2RBox-v3
├── data
│   ├── hrsc/
│   │   ├── FullDataSet/
│   │   │   ├── AllImages/        # All images
│   │   │   ├── Annotations/      # XML annotations
│   │   │   ├── LandMask/
│   │   │   └── Segmentations/
│   │   └── ImageSets/            # Data split files
```

### Config Path

Set `data_root` to `data/hrsc/`.

---

## SSDD

### Directory Structure

```
Point2RBox-v3
├── data
│   ├── ssdd/
│   │   ├── train/
│   │   └── test/
│   │       ├── all/              # All test images
│   │       ├── inshore/          # Inshore scenes
│   │       └── offshore/         # Offshore scenes
```

### Config Path

Set `data_root` to `data/ssdd/`.

---

## HRSID

### Directory Structure

```
Point2RBox-v3
├── data
│   ├── HRSID_JPG/
│   │   ├── JPEGImages/           # All images
│   │   └── annotations/          # COCO-format JSON annotations
```

### Config Path

Set `data_root` to `data/HRSID_JPG/`.

---

## SRSDD

### Directory Structure

```
Point2RBox-v3
├── data
│   ├── srsdd/
│   │   ├── train/
│   │   └── test/
```

### Config Path

Set `data_root` to `data/srsdd/`.

---

## RSDD

### Directory Structure

```
Point2RBox-v3
├── data
│   ├── rsdd/
│   │   ├── Annotations/          # XML annotations
│   │   ├── ImageSets/            # Data split files
│   │   ├── JPEGImages/           # Training images
│   │   └── JPEGValidation/       # Validation images
```

### Config Path

Set `data_root` to `data/rsdd/`.

---

## Using Symlinks

If your datasets are stored elsewhere (e.g., shared storage), symlinks are recommended:

```bash
# Create data directory under project root
mkdir -p data

# Create symlinks (modify paths accordingly)
ln -s /path/to/your/dior           data/dior
ln -s /path/to/your/split_ss_dota  data/split_ss_dota
ln -s /path/to/your/split_ss_star  data/split_ss_star
ln -s /path/to/your/hrsc           data/hrsc
# ... similarly for other datasets
```
