#!/bin/bash
# Point2RBox-v3 Interactive Testing Launcher

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                 Point2RBox-v3 Testing Launcher${NC}"
echo -e "${BLUE}======================================================================${NC}"

# ============= Conda Environment =============
# Override with: CONDA_ENV=your_env_name bash test.sh
CONDA_ENV="${CONDA_ENV:-point2rbox-v3}"

echo ""
echo -e "${GREEN}Activating conda environment: ${CONDA_ENV}${NC}"

# Locate conda initialization script
CONDA_SH=""
for candidate in \
    "$CONDA_PREFIX/../etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    if command -v conda &> /dev/null; then
        CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    fi
fi

if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
    echo -e "${RED}Error: Cannot find conda initialization script.${NC}"
    echo -e "${YELLOW}Please activate your conda environment manually and re-run this script.${NC}"
    exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate environment '${CONDA_ENV}'.${NC}"
    echo -e "${YELLOW}You can specify a different env name: CONDA_ENV=myenv bash test.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment '${CONDA_ENV}' activated${NC}"
echo -e "${YELLOW}Python: $(which python)${NC}"
echo ""

# ============= GPU Selection =============
echo ""
echo -e "${GREEN}Available GPUs:${NC}"
echo "======================================================================"
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{printf "GPU %s: VRAM %s/%s MB (Utilization: %s%%)\n", $1, $3-$2, $3, $4}'
echo "======================================================================"

BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    sort -t',' -k2 -rn | head -1 | cut -d',' -f1)

echo ""
echo -e "${YELLOW}Recommended GPU: ${BEST_GPU}${NC}"
read -p "Select GPU ID (press Enter for recommended): " GPU_ID
GPU_ID=${GPU_ID:-$BEST_GPU}

# ============= Config Selection =============
echo ""
echo -e "${GREEN}Available configs:${NC}"
echo "======================================================================"
echo "  1. DOTA v1.0 (default)"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py"
echo ""
echo "  2. DOTA v1.5"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-dotav1-5.py"
echo ""
echo "  3. DIOR"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-dior.py"
echo ""
echo "  4. STAR"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-star.py"
echo ""
echo "  5. Custom config path"
echo "======================================================================"

read -p "Select config (press Enter for default): " CONFIG_CHOICE
CONFIG_CHOICE=${CONFIG_CHOICE:-1}

case $CONFIG_CHOICE in
    1)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dotav1-0"
        ;;
    2)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dotav1-5.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dotav1-5"
        ;;
    3)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dior.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dior"
        ;;
    4)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-star.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-star"
        ;;
    5)
        read -p "Enter config path: " CONFIG
        DEFAULT_WORK_DIR=""
        ;;
    *)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dotav1-0"
        echo -e "${YELLOW}Invalid input, using default config.${NC}"
        ;;
esac

# ============= Checkpoint Selection =============
echo ""
echo -e "${GREEN}Checkpoint selection:${NC}"
echo "======================================================================"

if [ ! -z "$DEFAULT_WORK_DIR" ] && [ -d "$DEFAULT_WORK_DIR" ]; then
    echo -e "${YELLOW}Checkpoints found in $DEFAULT_WORK_DIR:${NC}"
    CKPTS=($(find "$DEFAULT_WORK_DIR" -name "*.pth" -type f 2>/dev/null | sort))
    if [ ${#CKPTS[@]} -gt 0 ]; then
        for i in "${!CKPTS[@]}"; do
            echo "  $((i+1)). ${CKPTS[$i]}"
        done
        echo ""
        read -p "Select checkpoint number or enter full path: " CKPT_CHOICE

        if [[ "$CKPT_CHOICE" =~ ^[0-9]+$ ]] && [ "$CKPT_CHOICE" -ge 1 ] && [ "$CKPT_CHOICE" -le ${#CKPTS[@]} ]; then
            CHECKPOINT="${CKPTS[$((CKPT_CHOICE-1))]}"
        else
            CHECKPOINT="$CKPT_CHOICE"
        fi
    else
        echo -e "${YELLOW}No checkpoint files found.${NC}"
        read -p "Enter checkpoint path: " CHECKPOINT
    fi
else
    read -p "Enter checkpoint path: " CHECKPOINT
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint file not found: $CHECKPOINT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Selected checkpoint: $CHECKPOINT${NC}"

# ============= Test Parameters =============
echo ""
echo -e "${GREEN}Test parameters (press Enter to use defaults):${NC}"
echo "======================================================================"

read -p "Work directory (default: auto): " WORK_DIR
read -p "Save visualization to directory (press Enter to skip): " SHOW_DIR
read -p "Save predictions to pickle file (press Enter to skip): " OUT_FILE

echo ""
echo "Config overrides (e.g.: model.test_cfg.nms.iou_threshold=0.1)"
read -p "cfg-options (press Enter to skip): " CFG_OPTIONS

# ============= Build Command =============
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python tools/test.py $CONFIG $CHECKPOINT"

if [ ! -z "$WORK_DIR" ]; then
    CMD="$CMD --work-dir $WORK_DIR"
fi
if [ ! -z "$SHOW_DIR" ]; then
    CMD="$CMD --show-dir $SHOW_DIR"
fi
if [ ! -z "$OUT_FILE" ]; then
    CMD="$CMD --out $OUT_FILE"
fi
if [ ! -z "$CFG_OPTIONS" ]; then
    CMD="$CMD --cfg-options $CFG_OPTIONS"
fi

# ============= Confirm & Execute =============
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Command to execute:${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

read -p "Confirm? (y/n, default: y): " CONFIRM
CONFIRM=${CONFIRM:-y}

if [ "$CONFIRM" != "y" ]; then
    echo -e "${RED}Cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}Starting testing...${NC}"
echo ""
eval $CMD
