#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by environment variables)
PROJECT_NAME="${PROJECT_NAME:-$HOME/Embedded-Projects/birds}"
BASE_MODEL="${BASE_MODEL:-$HOME/Embedded-Projects/yolov5s.pt}"
BATCH="${BATCH:-4}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"

# Ensure project directory exists
mkdir -p "$PROJECT_NAME"

python "$HOME/Embedded-Projects/yolov5/train.py" \
	--img 600 \
	--batch "$BATCH" \
	--epochs "$TRAIN_EPOCHS" \
	--data "$HOME/Embedded-Projects/data.yaml" \
	--weights "$BASE_MODEL" \
	--name 'feature_extraction' \
	--project "$PROJECT_NAME" \
	--cache \
	--freeze 12

WEIGHTS_BEST="$PROJECT_NAME/feature_extraction/weights/best.pt"

python "$HOME/Embedded-Projects/yolov5/val.py" \
	--weights "$WEIGHTS_BEST" \
	--batch "$BATCH" \
	--data "$HOME/Embedded-Projects/data.yaml" \
	--task test \
	--project "$PROJECT_NAME" \
	--name 'validation_on_test_data' \
	--augment

python "$HOME/Embedded-Projects/yolov5/detect.py" \
	--weights "$WEIGHTS_BEST" \
	--conf 0.6 \
	--source "$HOME/Embedded-Projects/data/test/images" \
	--project "$PROJECT_NAME" \
	--name 'detect_test' \
	--augment \
	--line=3