# Environment Setup

> **WARNING: Mismatched environment configurations can silently degrade model accuracy by 2~4 mAP, and the cause is extremely difficult to diagnose.** Users have already encountered reproduction failures due to inconsistent numpy, mmcv, and other package versions (see [point2rbox-v2#6](https://github.com/VisionXLab/point2rbox-v2/issues/6)). **Please carefully verify every version in the table below. Do not skip any step.**

## Requirements

- Linux (Ubuntu 18.04 / 20.04 / 22.04 recommended)
- Python 3.12
- PyTorch 2.2.0 + CUDA 12.1
- GCC 9.4+

> **Important:** The core package versions listed below have been strictly tested for compatibility. **You must match them exactly**, otherwise you may encounter incompatibility issues.

## Core Dependency Versions

| Package | Version | Notes |
|---|---|---|
| `torch` | **2.2.0+cu121** | Must use the CUDA 12.1 build |
| `torchvision` | **0.17.0** | Paired with torch 2.2.0 |
| `mmcv` | **2.2.0** | OpenMMLab computer vision library |
| `mmdet` | **3.3.0** | OpenMMLab detection framework |
| `mmengine` | **0.10.7** | OpenMMLab training engine |
| `mmrotate` | **1.0.0rc1** | Rotated object detection (this project) |
| `numpy` | **1.26.4** | Do NOT upgrade to >=2.0, causes compatibility issues |

## Other Dependencies

| Package | Version | Purpose |
|---|---|---|
| `scipy` | 1.16.3 | Scientific computing |
| `pillow` | 10.3.0 | Image I/O |
| `shapely` | 2.1.2 | Geometry computation |
| `opencv-python` | 4.11.0.86 | Image processing |
| `mobile_sam` | 1.0 | MobileSAM lightweight segmentation |
| `segment-anything` | 1.0 | SAM segmentation model |
| `timm` | 1.0.24 | PyTorch image model library |
| `pycocotools` | - | COCO evaluation |
| `matplotlib` | - | Visualization |

## Installation Steps

### Step 1: Create Conda Environment

```bash
conda create -n point2rbox-v3 python=3.12 -y
conda activate point2rbox-v3
```

### Step 2: Install PyTorch

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected output: 2.2.0+cu121 12.1 True
```

### Step 3: Install OpenMMLab Packages

The installation order matters. Please follow the exact sequence below:

```bash
# 1. Install mmengine
pip install mmengine==0.10.7

# 2. Install mmcv
pip install mmcv==2.2.0

# 3. Install mmdet
pip install mmdet==3.3.0
```

Verify OpenMMLab installation:

```bash
python -c "import mmcv; import mmdet; import mmengine; print(mmcv.__version__, mmdet.__version__, mmengine.__version__)"
# Expected output: 2.2.0 3.3.0 0.10.7
```

### Step 4: Pin numpy Version

```bash
pip install numpy==1.26.4
```

> **Note:** Installing `mmcv` or `mmdet` may automatically upgrade numpy to 2.x, which causes runtime errors. After installing all mm-series packages, **re-check and pin** the numpy version.

### Step 5: Install SAM Dependencies

```bash
pip install segment-anything timm

# MobileSAM must be installed from source
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM && pip install -e .
cd ..
```

### Step 6: Install This Project (mmrotate)

```bash
cd Point2RBox-v3
pip install -v -e .
```

### Step 7: Install Other Dependencies

```bash
pip install scipy pillow shapely opencv-python pycocotools matplotlib
```

## Verify Installation

Run the following command to verify all core package versions:

```bash
python -c "
import torch, torchvision, mmcv, mmdet, mmengine, mmrotate, numpy
print(f'torch:       {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'mmcv:        {mmcv.__version__}')
print(f'mmdet:       {mmdet.__version__}')
print(f'mmengine:    {mmengine.__version__}')
print(f'mmrotate:    {mmrotate.__version__}')
print(f'numpy:       {numpy.__version__}')
print(f'CUDA:        {torch.version.cuda}')
print(f'GPU count:   {torch.cuda.device_count()}')
"
```

Expected output:

```
torch:       2.2.0+cu121
torchvision: 0.17.0
mmcv:        2.2.0
mmdet:       3.3.0
mmengine:    0.10.7
mmrotate:    1.0.0rc1
numpy:       1.26.4
CUDA:        12.1
GPU count:   >=1
```

## FAQ

### Q: Failed to compile mmcv during installation

Make sure you are using the correct pre-built wheel URL. The `cu121/torch2.2` part must match your CUDA and PyTorch versions:

```bash
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
```

### Q: `AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0`

This is caused by **an overly strict version check in mmdet itself**, not an actual incompatibility with mmcv. mmdet 3.3.0 hardcodes `mmcv_maximum_version = '2.2.0'` (open interval, i.e., excluding 2.2.0) in its `__init__.py`, but mmcv 2.2.0 works perfectly fine in practice.

**Solution:** Locate your mmdet installation and modify the version upper bound in `mmdet/__init__.py`:

```bash
# 1. Find the mmdet installation path
python -c "import mmdet; print(mmdet.__file__)"
# Output example: /path/to/site-packages/mmdet/__init__.py

# 2. Edit that file, change:
#    mmcv_maximum_version = '2.2.0'
#    to:
#    mmcv_maximum_version = '2.3.0'
```

Or use a one-liner:

```bash
MMDET_INIT=$(python -c "import mmdet; print(mmdet.__file__)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET_INIT"
```

> **Note:** This is a known issue caused by lagging version constraints across OpenMMLab libraries. When mmdet was released, mmcv 2.2.0 did not yet exist, so the upper bound was set to `<2.2.0`. Modifying this bound does not affect functionality.

### Q: numpy gets auto-upgraded to 2.x during dependency installation, causing runtime errors

When installing mmdet, scipy, or other packages, pip may automatically upgrade numpy to 2.1.x or higher. numpy 2.x has compatibility issues with mmcv, mmrotate, and several other dependencies, leading to various `AttributeError` or `ModuleNotFoundError` errors.

**Solution:** After all packages are installed, force downgrade numpy:

```bash
pip install numpy==1.26.4
```

> **Tip:** After every `pip install`, check the numpy version (`pip show numpy`) to ensure it hasn't been accidentally upgraded.

### Q: ImportError: cannot import name 'xxx' from mmrotate

Make sure you have installed this project in development mode via `pip install -v -e .`.
