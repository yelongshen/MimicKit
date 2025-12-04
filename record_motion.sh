#!/bin/bash
# Helper script to record motion clips as videos

# Default values
MOTION_FILE="data/motions/humanoid/humanoid_spinkick.pkl"
OUTPUT_VIDEO="output/motion_video.mp4"
NUM_ENVS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --motion)
            MOTION_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_VIDEO="$2"
            shift 2
            ;;
        --robot)
            ROBOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--motion MOTION_FILE] [--output VIDEO_PATH] [--robot humanoid|g1|go2]"
            exit 1
            ;;
    esac
done

# Determine config based on robot type
if [ -z "$ROBOT" ]; then
    ROBOT="humanoid"
fi

case $ROBOT in
    humanoid)
        CONFIG="data/envs/view_motion_humanoid_env.yaml"
        ;;
    g1)
        CONFIG="data/envs/view_motion_g1_env.yaml"
        ;;
    go2)
        CONFIG="data/envs/view_motion_go2_env.yaml"
        ;;
    *)
        echo "Unknown robot: $ROBOT"
        exit 1
        ;;
esac

echo "Recording motion: $MOTION_FILE"
echo "Output video: $OUTPUT_VIDEO"
echo "Robot: $ROBOT"
echo "Config: $CONFIG"

python mimickit/run.py \
    --mode test \
    --num_envs $NUM_ENVS \
    --env_config $CONFIG \
    --visualize true \
    --video_path $OUTPUT_VIDEO \
    --log_file output/log.txt \
    --out_model_file output/model.pt
