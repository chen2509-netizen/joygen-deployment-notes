#!/usr/bin/env bash
# batch_emotion_avatars.sh — 10 支情緒參考影片一次跑完：前處理 + 串流生成 mp4
#
# 放在 joygen-deployment-notes/scripts/ 底下，在 notes 根目錄執行：
#   conda activate joygen
#   bash scripts/batch_emotion_avatars.sh [dataset_dir] [driving_audio]
#
#   dataset_dir    預設 ~/imood_project/dataset（裡面是 Actor_07/ Actor_24/，各有 0-4.mp4）
#   driving_audio  要讓 avatar 講的話，預設 JoyGen/demo/xinwen_5s.mp3
#
# 流程（每支影片）：
#   1. 前處理：從影片抽出它自己的音軌 → audio2motion → inference_edit_expression.py
#      用影片自己的音軌，是因為 edit_expression 產生的幀數 = 表情係數長度（= 音訊長度），
#      音軌跟影片一樣長，intermediate 才剛好涵蓋整支影片、不截斷也不重複。
#      影片沒音軌的話，改用 driving_audio 截成跟影片一樣長。
#   2. 串流：run_input.sh stream，用 --intermediate_audio_key 指到步驟 1 的目錄
#      （Path A / pose_driven 關閉時只讀 _ori/_face/_box，跟音訊無關）。
#
# 已經跑過的會自動跳過，中途失敗可以直接重跑。
# 輸出：
#   results/emotion/<Actor>/edit_exp/<n>/<n>/<n>/   intermediate（給 stream / serve 用）
#   results/emotion/<Actor>/out_<n>.mp4             生成結果

set -uo pipefail

NOTES_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOYGEN_ROOT="$(cd "$NOTES_ROOT/../JoyGen" && pwd)"

DATASET="$(realpath "${1:-$HOME/imood_project/dataset}")"
DRIVE_AUDIO="$(realpath "${2:-$JOYGEN_ROOT/demo/xinwen_5s.mp3}")"
ACTORS=(Actor_07 Actor_24)
EMOTIONS=(0 1 2 3 4)

OUT_ROOT="$NOTES_ROOT/results/emotion"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

# stream 命令列要 joygen_output.mode=persistent；pipeline.yaml 平常是給網站用的
# utterance_file。複製一份暫時改成 persistent，不動到原本的設定檔。
# （config 裡的相對路徑是對 notes 根目錄解析，跟設定檔放哪無關，所以複製一份沒問題）
STREAM_CFG="$OUT_ROOT/pipeline_persistent.yaml"
sed 's/^\(  mode:\) *utterance_file/\1 persistent/' \
    "$NOTES_ROOT/configs/pipeline.yaml" > "$STREAM_CFG"
grep -q '^  mode: persistent' "$STREAM_CFG" || \
    { echo "無法把 joygen_output.mode 改成 persistent，請檢查 pipeline.yaml" >&2; exit 1; }

[ -d "$DATASET" ]     || { echo "找不到 dataset: $DATASET" >&2; exit 1; }
[ -f "$DRIVE_AUDIO" ] || { echo "找不到 driving audio: $DRIVE_AUDIO" >&2; exit 1; }
python -c "import torch" 2>/dev/null || { echo "沒有 torch，先 conda activate joygen" >&2; exit 1; }

echo "notes   = $NOTES_ROOT"
echo "joygen  = $JOYGEN_ROOT"
echo "dataset = $DATASET"
echo "audio   = $DRIVE_AUDIO"
echo

ok=(); failed=()

preprocess() {   # $1=video(abs) $2=actor_dir(abs) $3=name
    local video="$1" adir="$2" name="$3"
    local wav="$adir/audio/${name}.wav"
    local exp="${name}.npy"
    local inter="$adir/edit_exp/$name/$name/$name"

    if ls "$inter"/*_box.npy >/dev/null 2>&1; then
        echo "  [skip] 前處理已存在"
        return 0
    fi
    mkdir -p "$adir/audio" "$adir/a2m"

    # 1a. 音訊：優先用影片自己的音軌
    local dur
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video")
    if ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$video" | grep -q .; then
        ffmpeg -y -loglevel error -i "$video" -vn -ac 1 -ar 16000 "$wav" || return 1
    else
        echo "  影片沒有音軌，改用 driving audio 截成 ${dur}s"
        ffmpeg -y -loglevel error -stream_loop -1 -i "$DRIVE_AUDIO" -t "$dur" \
               -ac 1 -ar 16000 "$wav" || return 1
    fi

    # 1b. audio2motion + edit_expression（都要在 JoyGen 目錄下跑，權重是相對路徑）
    ( cd "$JOYGEN_ROOT" && \
      python inference_audio2motion.py \
          --a2m_ckpt ./pretrained_models/audio2motion/240210_real3dportrait_orig/audio2secc_vae \
          --hubert_path ./pretrained_models/audio2motion/hubert \
          --drv_aud "$wav" --seed 0 \
          --result_dir "$adir/a2m" --exp_file "$exp" && \
      python -u inference_edit_expression.py \
          --name face_recon_feat0.2_augment --epoch=20 --use_opengl False \
          --checkpoints_dir ./pretrained_models \
          --bfm_folder ./pretrained_models/BFM \
          --infer_video_path "$video" \
          --infer_exp_coeff_path "$adir/a2m/$exp" \
          --infer_result_dir "$adir/edit_exp" ) || return 1

    ls "$inter"/*_box.npy >/dev/null 2>&1 || { echo "  前處理沒有產出 $inter" >&2; return 1; }
}

for actor in "${ACTORS[@]}"; do
    adir="$OUT_ROOT/$actor"
    mkdir -p "$adir"
    for n in "${EMOTIONS[@]}"; do
        video="$DATASET/$actor/$n.mp4"
        out="$adir/out_$n.mp4"
        log="$LOG_DIR/${actor}_$n.log"
        tag="$actor/$n"
        echo "=== $tag ==="

        if [ ! -f "$video" ]; then
            echo "  找不到 $video"; failed+=("$tag(no video)"); continue
        fi

        if ! preprocess "$video" "$adir" "$n" >>"$log" 2>&1; then
            echo "  前處理失敗，看 $log"; failed+=("$tag(preprocess)"); continue
        fi
        echo "  前處理 OK"

        if [ -f "$out" ]; then
            echo "  [skip] $out 已存在"; ok+=("$tag"); continue
        fi

        if bash "$NOTES_ROOT/scripts/run_input.sh" stream \
                "$DRIVE_AUDIO" "$video" "$adir/edit_exp" "$out" \
                --intermediate_audio_key "$n" \
                --config "$STREAM_CFG" >>"$log" 2>&1 && [ -s "$out" ]; then
            echo "  → $out"; ok+=("$tag")
        else
            echo "  串流生成失敗，看 $log"; failed+=("$tag(stream)")
        fi
    done
done

echo
echo "成功 ${#ok[@]}: ${ok[*]:-}"
echo "失敗 ${#failed[@]}: ${failed[*]:-}"
[ ${#failed[@]} -eq 0 ]
