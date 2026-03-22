#!/usr/bin/env bash
set -euo pipefail

INSTA_DIR="/media/yuhao/bluessd/insta_frn/"
VGGT_DIR="/home/yuhao/git/VGGT-SLAM/"
STRIDE=10

SEQ_LST=(
    # "2026-01-31_run1"
    # "2026-01-31_run2"
    # "2026-01-31_run3"

    # "2026-03-07_run1"
    "2026-03-07_run3"
)

for seq_name in "${SEQ_LST[@]}"; do

    rm -rf "$INSTA_DIR/$seq_name/slam_map/images/cam0_pinhole_stride_${STRIDE}/"

    python ~/git/Depth-Anything-3/da3_streaming/scripts/copy_timestamped_images.py \
        "$INSTA_DIR/$seq_name/slam_map/images/cam0_pinhole/" \
        "$INSTA_DIR/$seq_name/slam_map/images/cam0_pinhole_stride_${STRIDE}/" \
        --frame-stride "${STRIDE}" --time-window-sec 100

    python "$VGGT_DIR"/main.py \
        --image_folder "$INSTA_DIR/$seq_name/slam_map/images/cam0_pinhole_stride_${STRIDE}/" \
        --output_dir "$VGGT_DIR/outputs/stride_${STRIDE}/$seq_name/" \
        --min_disparity 50 --da --submap_size 5
done
