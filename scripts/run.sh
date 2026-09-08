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
#   step3 <audio> <video> <intermediate_dir> [target] [extra flags...]
#       frame 即時送進常駐 ffmpeg。target 預設 rtp://127.0.0.1:23000，
#       也可以給 udp://host:port 或檔案路徑（例如 out.mp4）。
#       其餘旗標原樣傳給 Python：--debug、--bitrate 4M、--gop 25 等。
#
#   recv [stream.sdp | udp://host:port]
#       RTP：先啟動 sender 產生 SDP，再跑這個
#       UDP：先跑這個再啟動 sender
#
#   diff <baseline_frames_dir> <stepN_frames_dir> [tol]
#       逐張 PNG 比對，驗證輸出跟 baseline 一致

set -euo pipefail

usage() {
    cat <<EOF
usage: bash scripts/run.sh <mode> <args...>

modes:
  baseline <audio> <video> <intermediate_dir> [result_dir]
  step3    <audio> <video> <intermediate_dir> [target] [--debug|--bitrate M|--gop N]
  recv     [stream.sdp | udp://host:port]
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
        TARGET="rtp://127.0.0.1:23000"
        EXTRA=()
        if [ $# -ge 1 ] && [[ "$1" != --* ]]; then TARGET="$1"; shift; fi
        # remaining args pass straight through (--debug, --bitrate 4M, --gop 25, ...)
        EXTRA=("$@")

        TAG="$(basename "${AUDIO%.*}")_$(date +%m%d_%H%M)"
        echo "[run] target = $TARGET"

        python -u -m streaming.joygen_stream \
            --audio_path "$AUDIO" --video_path "$VIDEO" --intermediate_dir "$INTER" \
            "${COMMON_MODEL_ARGS[@]}" \
            --result_dir "results/stream_step3" \
            --report "timing/step3_${TAG}" \
            --target "$TARGET" "${EXTRA[@]}"
        ;;

    recv)
        # RTP needs the SDP the sender writes at startup; UDP/MPEG-TS does not.
        SRC="${1:-stream.sdp}"
        if [[ "$SRC" == udp://* ]]; then
            echo "[run] listening on $SRC (start this before the sender)"
            ffplay -f mpegts -probesize 5000000 -analyzeduration 5000000 \
                   -fflags nobuffer -flags low_delay -framedrop \
                   -i "${SRC}?fifo_size=1000000&overrun_nonfatal=1"
        else
            if [ ! -f "$SRC" ]; then
                echo "[run] $SRC not found."
                echo "[run] Start the sender first — it writes the SDP on startup,"
                echo "[run] and spends ~18s preprocessing before any frame goes out,"
                echo "[run] which is plenty of time to start this receiver."
                exit 1
            fi
            echo "[run] playing $SRC (loop mode, Ctrl-C to stop)"
            while true; do
                ffplay -protocol_whitelist file,rtp,udp \
                       -reorder_queue_size 2000 \
                       -buffer_size 20000000 \
                       -max_delay 500000 \
                       -i "$SRC" 2>&1 | grep -v "non-existing PPS\|decode_slice_header"
                echo "[run] stream ended, waiting for next run..."
                sleep 1
            done
        fi
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
