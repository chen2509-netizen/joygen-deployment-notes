#!/usr/bin/env bash
# run.sh — Task 1 串流化實作的統一入口
#
# 用法：
#   bash scripts/run.sh <mode> <args...>
#
# 新增 step 時直接在下面 case 加分支，不要新增 .sh 檔案。
#
#   baseline <audio> <video> <intermediate_dir> [result_dir]
#       原始兩段式流程 + 計時，供之後所有 step 對照
#
#   (step2 已被 step3 取代：joygen_stream.py 從 Step 2 一路演進到 Step 3，
#    Step 2 的「只寫 PNG」行為現在等同 step3 + 檔案 target，不再單獨提供 mode)
#
#   step3 <audio> <video> <intermediate_dir> [target] [--debug]
#       frame 即時送進常駐 ffmpeg。target 預設 udp://127.0.0.1:23000，
#       也可以給檔案路徑（例如 results/stream_step3/out.mp4）直接存檔。
#       加 --debug 會同時寫 PNG 供 diff 驗證（會拖慢，正式計時勿開）。
#
#   recv [port]
#       開 ffplay 監聽 UDP，跑 step3 之前要先開這個
#
#   diff <baseline_frames_dir> <stepN_frames_dir> [tol]
#       逐張 PNG 比對，驗證輸出跟 baseline 一致

set -euo pipefail

usage() {
    cat <<EOF
usage: bash scripts/run.sh <mode> <args...>

modes:
  baseline <audio> <video> <intermediate_dir> [result_dir]
  step3    <audio> <video> <intermediate_dir> [target] [--debug]
  recv     [port]
  diff     <baseline_frames_dir> <stepN_frames_dir> [tol]
EOF
    exit 1
}

[ $# -lt 1 ] && usage
MODE="$1"; shift

COMMON_MODEL_ARGS=(
    --unet_model_path pretrained_models/joygen
    --vae_model_path  pretrained_models/sd-vae-ft-mse
    --enable_pose_driven
    --img_size 256
    --gpu_id 0
)

case "$MODE" in
    baseline)
        [ $# -lt 3 ] && usage
        AUDIO="$1"; VIDEO="$2"; INTER="$3"
        RESULT="${4:-results/stream_baseline}"
        TAG="$(basename "${AUDIO%.*}")_$(date +%m%d_%H%M)"

        python -u -m streaming.baseline_timing \
            --audio_path "$AUDIO" --video_path "$VIDEO" --intermediate_dir "$INTER" \
            "${COMMON_MODEL_ARGS[@]}" \
            --result_dir "$RESULT" \
            --report "timing/${MODE}_${TAG}"
        ;;

    step3)
        [ $# -lt 3 ] && usage
        AUDIO="$1"; VIDEO="$2"; INTER="$3"; shift 3
        TARGET="udp://127.0.0.1:23000"
        DEBUG=""
        # 第 4 個位置參數若不是 --debug 就當成 target
        if [ $# -ge 1 ] && [ "$1" != "--debug" ]; then TARGET="$1"; shift; fi
        if [ "${1:-}" = "--debug" ]; then DEBUG="--debug"; fi

        TAG="$(basename "${AUDIO%.*}")_$(date +%m%d_%H%M)"
        echo "[run] target = $TARGET"
        [ -n "$DEBUG" ] && echo "[run] debug PNG 已開啟（會影響計時）"

        python -u -m streaming.joygen_stream \
            --audio_path "$AUDIO" --video_path "$VIDEO" --intermediate_dir "$INTER" \
            "${COMMON_MODEL_ARGS[@]}" \
            --result_dir "results/stream_step3" \
            --report "timing/step3_${TAG}" \
            --target "$TARGET" $DEBUG
        ;;

    recv)
        PORT="${1:-23000}"
        echo "[run] 監聽 udp://127.0.0.1:${PORT}，請在另一個終端機跑 step3"
        echo "[run] 注意：一定要先開這個，再啟動 step3，否則會錯過開頭"
        ffplay -fflags nobuffer -flags low_delay -framedrop \
               -i "udp://127.0.0.1:${PORT}"
        ;;

    diff)
        [ $# -lt 2 ] && usage
        if [ $# -ge 3 ]; then
            python -m streaming.diff_frames "$1" "$2" --tol "$3"
        else
            python -m streaming.diff_frames "$1" "$2"
        fi
        ;;

    *)
        echo "unknown mode: $MODE"; usage
        ;;
esac
