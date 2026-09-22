#!/usr/bin/env bash
# run_input.sh — input streaming 的統一入口
#
# 跟 scripts/run.sh（output streaming）互不相干：兩支腳本、兩個 python 套件
# （streaming/ vs streaming_input/），不共用任何檔案。run.sh 不會被這支動到。
#
# 用法：
#   bash scripts/run_input.sh <mode> <args...>
#
#   serve <video> <intermediate_dir> <avatar_audio_key> [target] [flags...]
#       M3：常駐服務。模型與 avatar 載一次，之後由 imood voice 從 ingest
#       socket（configs/pipeline.yaml 的 joygen_input.ingest_host/port）推音訊。
#       句與句之間會送 idle frame 讓串流不中斷；輸出到檔案時加 --no_idle。
#
#       要 pose-driven 就三個一起給（M4）：
#         --enable_pose_driven --stream_motion \
#         --avatar3d_cache <notes>/cache/avatar3d/<avatar>.npz
#       服務不能讀離線的 depth 圖（那些綁定單一音檔），只能即時渲染。
#
#   cache3d <frames_dir> <n_frames> <out.npz>
#       M4：把 avatar 的 3DMM 擬合算一次存起來（MTCNN + deep3d）。之後
#       pose-driven 串流每幀只要換 exp 係數重新渲染 depth，不必再碰
#       inference_edit_expression.py 那條與音訊綁定的路徑。
#
#   depth-parity <pose_dir> <exp_npy> [cache.npz]
#       M4 關卡：用快取渲染的 depth 對照離線 _depth_edit_exp.jpg。
#
#   motion-ref <audio> <out.npy> [--seed N]
#       產生離線的表情係數當對照。務必用這支，不要直接對 JoyGen/demo 下的
#       音檔跑 audio2motion —— save_wav16k 會在音檔旁寫 _16k.wav。
#
#   motion-parity <audio> <ref.npy> [flags]
#       M4 關卡：串流 audio2motion 對照離線係數。
#
#   stream <audio> <video> <intermediate_dir> [target] [flags...]
#       M2：input streaming 主流程。target 省略時取 configs/pipeline.yaml 的
#       joygen_output.target。旗標原樣傳給 Python（--debug、--pace realtime、
#       --enable_pose_driven、--stall_at 5 --stall_ms 1500 等）。
#
#   parity [audio] [flags...]
#       M1：串流 whisper 特徵 vs 離線 audio2feat() 的數值對拍。
#       結果決定 configs/pipeline.yaml 的 whisper_* 參數。
#
#   diff <baseline_frames_dir> <stream_frames_dir> [tol]
#       逐張 PNG 比對。借用 output streaming 的 streaming.diff_frames 當外部
#       工具（純讀取、不 import，兩邊不耦合）。
#
#   recv [stream.sdp | rtp://host:port | udp://host:port]
#       收流端。RTP/UDP 要先開這個再跑 sender。
#
#   test
#       跑得動的測試全跑。沒有 CUDA 就跳過需要 GPU 的項目，不算失敗。
#
#   env
#       印出目前解析到的路徑與 python 環境，除錯用。
#
# 為什麼要 cd 進 JoyGen：utils/blending.py 在 import 時就執行
# FaceParsing()，而它用相對路徑 ./pretrained_models/face-parse-bisent/ 載權重，
# 傳絕對路徑也繞不掉。既然 cwd 在 JoyGen，所有輸出路徑一律用絕對路徑指回
# joygen-deployment-notes，否則產物會掉進 JoyGen/ 裡。

set -euo pipefail

NOTES_ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # joygen-deployment-notes/
JOYGEN_ROOT="$(cd "$NOTES_ROOT/../JoyGen" && pwd)"  # 同層的 JoyGen/

usage() {
    cat <<EOF
usage: bash scripts/run_input.sh <mode> <args...>

modes:
  serve  <video> <intermediate_dir> <avatar_audio_key> [target] [--no_idle]
  cache3d <frames_dir> <n_frames> <out.npz>
  depth-parity <pose_dir> <exp_npy> [cache.npz]
  motion-ref   <audio> <out.npy> [--seed N]
  motion-parity <audio> <ref.npy> [--control|--window ...|--hop ...]
  stream <audio> <video> <intermediate_dir> [target] [--debug|--pace realtime|...]
  parity [audio] [--chunk-ms 320] [--fps 25] [--fp32] [--window-mode anchored]
  diff   <baseline_frames_dir> <stream_frames_dir> [tol]
  recv   [stream.sdp | rtp://host:port | udp://host:port]
  test
  env
EOF
    exit 1
}

setup_env() {
    cd "$JOYGEN_ROOT"
    export PYTHONPATH="$NOTES_ROOT:$JOYGEN_ROOT:${PYTHONPATH:-}"
    # 不要在 JoyGen/ 裡留下 __pycache__ —— 這是唯一會寫進 JoyGen 的東西
    export PYTHONDONTWRITEBYTECODE=1

    if ! python -c "import torch" 2>/dev/null; then
        echo "[run_input] python 裡沒有 torch，是不是忘了 conda activate joygen ?" >&2
        exit 1
    fi
}

has_cuda() {
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

[ $# -lt 1 ] && usage
MODE="$1"; shift

case "$MODE" in
    serve)
        # 兩種形式：
        #   serve <video> <intermediate_dir> <audio_key> [target] [flags]  單一 avatar
        #   serve [flags]                                                  多 avatar，讀 config
        AVATAR_ARGS=()
        if [ $# -ge 3 ] && [[ "$1" != --* ]]; then
            AVATAR_ARGS=(--video_path "$1" --intermediate_dir "$2"
                         --intermediate_audio_key "$3")
            shift 3
        fi
        TARGET_ARG=()
        if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
            TARGET_ARG=(--target "$1"); shift
        fi

        setup_env
        if ! has_cuda; then
            echo "[run_input] 需要 CUDA，此環境沒有可用的 GPU" >&2
            exit 1
        fi

        python -u -m streaming_input.service \
            "${AVATAR_ARGS[@]}" \
            --config "$NOTES_ROOT/configs/pipeline.yaml" \
            --sdp_file "$NOTES_ROOT/stream_input.sdp" \
            "${TARGET_ARG[@]}" "$@"
        ;;

    cache3d)
        [ $# -lt 3 ] && usage
        setup_env
        python -u -m streaming_input.avatar3d \
            --frames_dir "$1" --n_frames "$2" --out "$3"
        ;;

    motion-ref)
        [ $# -lt 2 ] && usage
        setup_env
        # 一定要透過這支產生參考係數：直接對 JoyGen/demo 下的音檔跑
        # audio2motion，save_wav16k 會在那裡寫 _16k.wav（M4 紀錄第 6 節）
        python -u "$NOTES_ROOT/tests/make_motion_ref.py" "$1" --out "$2" "${@:3}"
        ;;

    motion-parity)
        [ $# -lt 2 ] && usage
        setup_env
        python -u "$NOTES_ROOT/tests/test_motion_parity.py" "$@"
        ;;

    depth-parity)
        [ $# -lt 2 ] && usage
        setup_env
        CACHE="${3:-$NOTES_ROOT/cache/avatar3d/example_5s.npz}"
        python -u "$NOTES_ROOT/tests/test_depth_cache_parity.py" \
            --pose_dir "$1" --exp_npy "$2" --cache "$CACHE"
        ;;

    stream)
        [ $# -lt 3 ] && usage
        AUDIO="$1"; VIDEO="$2"; INTER="$3"; shift 3
        TARGET_ARG=()
        if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
            TARGET_ARG=(--target "$1"); shift
        fi

        setup_env
        if ! has_cuda; then
            echo "[run_input] 需要 CUDA，此環境沒有可用的 GPU" >&2
            exit 1
        fi

        # 輸出一律絕對路徑指回 notes：cwd 在 JoyGen，相對路徑會掉進去
        python -u -m streaming_input.joygen_input_stream \
            --audio_path "$AUDIO" --video_path "$VIDEO" \
            --intermediate_dir "$INTER" \
            --config "$NOTES_ROOT/configs/pipeline.yaml" \
            --result_dir "$NOTES_ROOT/results/input_stream" \
            --sdp_file "$NOTES_ROOT/stream_input.sdp" \
            "${TARGET_ARG[@]}" "$@"
        ;;

    diff)
        [ $# -lt 2 ] && usage
        setup_env
        if [ $# -ge 3 ]; then
            python -u -m streaming.diff_frames "$1" "$2" --tol "$3"
        else
            python -u -m streaming.diff_frames "$1" "$2"
        fi
        ;;

    recv)
        SRC="${1:-$NOTES_ROOT/stream_input.sdp}"
        if [[ "$SRC" == udp://* || "$SRC" == rtp://* ]]; then
            echo "[run_input] listening on $SRC (先開這個再跑 sender)"
            ffplay -probesize 2000000 -analyzeduration 2000000 \
                   -flags low_delay -reorder_queue_size 2000 \
                   -buffer_size 20000000 -max_delay 500000 -i "$SRC"
        else
            if [ ! -f "$SRC" ]; then
                echo "[run_input] $SRC 不存在。sender 啟動時才會寫出 SDP。"
                exit 1
            fi
            ffplay -protocol_whitelist file,rtp,udp \
                   -reorder_queue_size 2000 -buffer_size 20000000 \
                   -max_delay 500000 -i "$SRC"
        fi
        ;;

    parity)
        setup_env
        if ! has_cuda; then
            echo "[run_input] 需要 CUDA，此環境沒有可用的 GPU" >&2
            exit 1
        fi
        python -u "$NOTES_ROOT/tests/test_stream_feature_parity.py" "$@"
        ;;

    test)
        setup_env
        echo "[run_input] notes  = $NOTES_ROOT"
        echo "[run_input] joygen = $JOYGEN_ROOT"
        failed=0

        # --- 不需要 GPU 的測試 ---
        for t in "$NOTES_ROOT"/tests/test_ring_buffer.py \
                 "$NOTES_ROOT"/tests/test_tts_ingest_mock.py; do
            [ -f "$t" ] || continue
            echo ""
            echo "[run_input] === $(basename "$t") ==="
            python -u "$t" || failed=1
        done

        # --- 需要 GPU 的測試 ---
        if has_cuda; then
            echo ""
            echo "[run_input] === test_stream_feature_parity.py ==="
            python -u "$NOTES_ROOT/tests/test_stream_feature_parity.py" || failed=1
        else
            echo ""
            echo "[run_input] 沒有 CUDA，跳過 GPU 測試（不算失敗）"
        fi

        echo ""
        if [ "$failed" -eq 0 ]; then
            echo "[run_input] 全部通過"
        else
            echo "[run_input] 有測試失敗" >&2
            exit 1
        fi
        ;;

    env)
        setup_env
        echo "NOTES_ROOT  = $NOTES_ROOT"
        echo "JOYGEN_ROOT = $JOYGEN_ROOT"
        echo "cwd         = $(pwd)"
        echo "PYTHONPATH  = $PYTHONPATH"
        echo "python      = $(command -v python)"
        python -c "import sys,torch; print('version    =', sys.version.split()[0]); print('torch      =', torch.__version__, 'cuda', torch.cuda.is_available())"
        echo "config      = $NOTES_ROOT/configs/pipeline.yaml"
        ;;

    *)
        echo "unknown mode: $MODE"; usage
        ;;
esac
