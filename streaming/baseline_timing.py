"""
baseline_timing.py — Step 1：量測現有 JoyGen 輸出流程的時間分布

行為上等同原始 inference_joygen.py，只多了計時。
目的是取得可比較的基準，並判斷「輸出端串流化」這個方向是否值得做下去。

量測五個階段：
    1_model_load    模型載入
    2_preprocess    前處理迴圈（讀圖 + VAE encode，全部做完才能開始生成）
    3_generate      生成迴圈（逐 batch decode）
    4_blend_write   貼圖迴圈（等生成全跑完才開始，逐張貼回原圖 + 寫 PNG）
    5_ffmpeg_video  一次性轉檔
    6_ffmpeg_audio  合音軌

其中 2/3/4 三段另外逐 frame 記錄（mean / stdev / 首幀）。

流程參考自 JoyGen 的 inference_joygen.py（JD.com, Apache 2.0）。
本檔不含 JoyGen 原始碼，僅 import 其既有元件後自行編排流程並加上量測。

位置：JoyGen/streaming/baseline_timing.py
執行：從 JoyGen repo 根目錄執行（見 scripts/run_baseline.sh）
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from diffusers import UNet2DConditionModel

# JoyGen 既有元件，直接沿用不重寫
from inference_joygen import (
    data_generator,
    create_mouth_mask,
    mouth_region_indices,
)
from src.audio2feature import Audio2Feature
from src.pe import PositionalEncoding
from src.modules.vae import VAE
from utils.blending import get_image

from streaming.frame_timer import FrameTimer, PhaseTimer, write_report


@torch.no_grad()
def main(args):
    phases = PhaseTimer()
    device = torch.device("cuda", args.gpu_id)

    # ---------------------------------------------------------- 1. 模型載入
    # 獨立計時、不算進 per-frame 成本。
    # 260813 那次測試每個 chunk 都重付一次這個成本，才會得到 2,253ms/frame；
    # 這裡把它分離出來，看清楚固定成本到底多大。
    with phases.phase("1_model_load"):
        audio_processor = Audio2Feature(model_path=args.whisper_model_path)
        vae = VAE(model_path=args.vae_model_path,
                  resized_img=args.img_size,
                  device=device)
        unet = UNet2DConditionModel.from_pretrained(args.unet_model_path).to(device=device)
        pe = PositionalEncoding(d_model=384)
        timesteps = torch.tensor([0], device=device)

    video_basename = os.path.basename(args.video_path).split(".")[0]
    audio_basename = os.path.basename(args.audio_path).split(".")[0]
    pose_path = os.path.join(args.intermediate_dir, video_basename,
                             audio_basename, video_basename)
    print(f"[info] pose_path = {pose_path}")
    if not os.path.exists(pose_path):
        sys.exit(f"[error] intermediate dir not found: {pose_path}\n"
                 f"        請先跑 inference_edit_expression.py 產生 pose/depth 檔案")

    output_basename = f"{video_basename}#{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename)
    os.makedirs(result_img_save_path, exist_ok=True)
    output_vid_name = os.path.join(args.result_dir, output_basename + ".mp4")

    # ------------------------------------------------------ 音訊特徵（非主軸）
    with phases.phase("0_audio_feature"):
        fps = args.fps
        whisper_feature = audio_processor.audio2feat(args.audio_path)
        whisper_chunks = audio_processor.feature2chunks(
            feature_array=whisper_feature, fps=fps)

    input_imgs_list = glob.glob(os.path.join(pose_path, "*_ori.jpg"))
    if not input_imgs_list:
        sys.exit(f"[error] no *_ori.jpg found under {pose_path}")

    indices_img = [int(os.path.basename(p).split("_")[0]) for p in input_imgs_list]
    ind_max = int(np.max(indices_img))
    n_input = len(input_imgs_list)
    print(f"[info] frames in intermediate dir: {n_input}")
    if ind_max + 1 != n_input:
        sys.exit("[error] failed to detect face in some frame of input video")

    if not args.enable_pose_driven:
        blank_depth = np.zeros((args.img_size, args.img_size, 3), np.uint8)
        latent_depth = vae.get_latents_for_nomask(blank_depth)

    # ---------------------------------------------------------- 2. 前處理
    # 這段是「第一張畫面出現之前」躲不掉的成本，對首幀延遲影響最大，
    # 所以除了階段總時間之外，也逐 frame 記錄。
    t_pre = FrameTimer("preprocess", verbose=args.verbose)
    sub_dir = os.path.dirname(input_imgs_list[0])
    ori_img_list, box_list, latent_list = [], [], []

    with phases.phase("2_preprocess"):
        t_pre.start()
        for i in tqdm(range(ind_max + 1), desc="preprocess"):
            ori_img = cv2.imread(os.path.join(sub_dir, f"{i}_ori.jpg"))
            crop_img = cv2.imread(os.path.join(sub_dir, f"{i}_face.jpg"))
            depth_img = cv2.imread(os.path.join(sub_dir, f"{i}_depth_edit_exp.jpg"))
            box = np.load(os.path.join(sub_dir, f"{i}_box.npy"))
            lmk = np.array(np.load(os.path.join(sub_dir, f"{i}_lm.npy")), np.int32)

            ori_img_list.append(ori_img)
            box_list.append(box)

            crop_img = cv2.resize(crop_img, (args.img_size, args.img_size),
                                  interpolation=cv2.INTER_LANCZOS4)
            latent, _ = vae.get_latents_for_unet(crop_img)

            if args.enable_pose_driven:
                lip_mask = create_mouth_mask(depth_img, lmk[mouth_region_indices, :])
                depth_img = depth_img * lip_mask
                depth_img = cv2.resize(depth_img, (args.img_size, args.img_size),
                                       interpolation=cv2.INTER_LANCZOS4)
                depth_img[:args.img_size // 2, ...] = 0
                latent_depth_mask = vae.get_latents_for_nomask(depth_img)
            else:
                latent_depth_mask = latent_depth

            latent_list.append(torch.cat([latent, latent_depth_mask], axis=1))
            t_pre.mark()

    # 串流輸出的畫面尺寸 = 原始影片解析度，不是 img_size
    #（img_size 只是裁切臉部的大小）。Step 3 設定 ffmpeg 會用到這個數字。
    frame_h, frame_w = ori_img_list[0].shape[:2]
    print(f"[info] 串流輸出畫面尺寸 = {frame_w}x{frame_h}"
          f"  (img_size={args.img_size} 只是臉部裁切)")

    # ---------------------------------------------------------- 3. 生成迴圈
    t_gen = FrameTimer("generate", verbose=args.verbose)
    gen = data_generator(whisper_chunks, latent_list, args.batch_size)
    n_batches = int(np.ceil(float(ind_max + 1) / args.batch_size))
    res_frame_list = []

    with phases.phase("3_generate"):
        t_gen.start()
        for whisper_batch, latent_batch in tqdm(gen, total=n_batches, desc="generate"):
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(unet.device)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.dtype)

            pred_latents = unet(latent_batch, timesteps,
                                encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
                t_gen.mark()

    num_frames = min(ind_max + 1, len(res_frame_list))
    print(f"[info] input frames: {ind_max + 1}, generated: {len(res_frame_list)}")

    # ---------------------------------------------------------- 4. 貼圖迴圈
    # 獨立的第二個 pass：要等生成全部跑完才開始。
    # Step 2 的工作就是把這段併進上面的生成迴圈。
    t_blend = FrameTimer("blend+write", verbose=args.verbose)

    with phases.phase("4_blend_write"):
        t_blend.start()
        for i in tqdm(range(num_frames), desc="blend"):
            box = [int(e) for e in box_list[i]]
            x1, y1, x2, y2 = box
            try:
                res_crop_img = cv2.resize(res_frame_list[i].astype(np.uint8),
                                          (x2 - x1, y2 - y1))
            except Exception:
                continue
            combine_img = get_image(ori_img_list[i], res_crop_img, box)
            cv2.imwrite(f"{result_img_save_path}/{i + 1}_edit.png", combine_img)
            t_blend.mark()

    # ------------------------------------------------- 5/6. 一次性 ffmpeg 收尾
    # research 指出「唯一真正卡住即時」的地方，Step 3 會換成常駐 pipe。
    # baseline 保留原行為，才知道省掉它能賺多少。
    tmp_mp4 = f"temp_{output_basename}.mp4"

    with phases.phase("5_ffmpeg_video"):
        os.system(
            f"ffmpeg -y -v fatal -r {fps} -f image2 "
            f"-i {result_img_save_path}/%d_edit.png -vcodec libx264 "
            f"-vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p "
            f"-crf 18 {tmp_mp4}")

    with phases.phase("6_ffmpeg_audio"):
        os.system(f"ffmpeg -y -v fatal -i {args.audio_path} -i {tmp_mp4} {output_vid_name}")
        if os.path.exists(tmp_mp4):
            os.remove(tmp_mp4)

    # 原版會 shutil.rmtree 掉 PNG，這裡保留：
    # Step 2 要用它比對「合併 pass 之後畫面是否仍與 baseline 一致」
    print(f"[info] result : {output_vid_name}")
    print(f"[info] frames : {result_img_save_path}  (保留供 Step 2 比對)")

    meta = {
        "mode": "baseline",
        "video": args.video_path,
        "audio": args.audio_path,
        "frames": num_frames,
        "fps": fps,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "stream_frame_size": f"{frame_w}x{frame_h}",
        "gpu_id": args.gpu_id,
    }
    write_report(args.report, phases, [t_pre, t_gen, t_blend], meta=meta, mode="baseline")

def build_parser():
    p = argparse.ArgumentParser(description="Step 1: JoyGen baseline timing")
    p.add_argument("--audio_path", type=str, required=True)
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--intermediate_dir", type=str, required=True)
    p.add_argument("--vae_model_path", type=str, default="pretrained_models/sd-vae-ft-mse")
    p.add_argument("--whisper_model_path", type=str, default="pretrained_models/whisper/tiny.pt")
    p.add_argument("--unet_model_path", type=str, default="pretrained_models/joygen")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--enable_pose_driven", action="store_true")
    p.add_argument("--result_dir", default="./results/stream_baseline")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--report", type=str, default="./timing/baseline",
                   help="報表輸出路徑前綴（會產生 .csv 與 .json）")
    p.add_argument("--verbose", action="store_true", help="逐 frame 印出時間")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(args)
    t0 = time.time()
    main(args)
    print(f"[info] wall clock total: {time.time() - t0:.1f}s")
