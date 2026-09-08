"""
joygen_stream.py — JoyGen 串流化實作

Step 2：把貼圖併進生成迴圈，透過 callback 交出每張完成的 frame。
Step 3：callback 改接常駐 ffmpeg（即時編碼送出），PNG 改成 --debug 才寫；
        並加上 warmup，避免第一次 forward 的 CUDA 初始化污染計時。

跟 baseline_timing.py 的差別：
  - baseline 有兩個 pass（生成全跑完 → 才逐張貼圖 → 全部寫 PNG → 一次性 ffmpeg）
  - 這裡是一個 pass（生成一張 → 立刻貼圖 → 立刻送出），沒有一次性 ffmpeg 收尾

位置：JoyGen/streaming/joygen_stream.py
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
from streaming.sinks import FFmpegSink, PngSink


@torch.no_grad()
def main(args):
    phases = PhaseTimer()
    device = torch.device("cuda", args.gpu_id)

    # ---------------------------------------------------------- 1. 模型載入
    with phases.phase("1_model_load"):
        audio_processor = Audio2Feature(model_path=args.whisper_model_path)
        vae = VAE(model_path=args.vae_model_path,
                  resized_img=args.img_size,
                  device=device)
        unet = UNet2DConditionModel.from_pretrained(args.unet_model_path).to(device=device)
        pe = PositionalEncoding(d_model=384)
        timesteps = torch.tensor([0], device=device)

    # ------------------------------------------------------------ 1b. Warmup
    # 第一次 forward 會觸發 CUDA kernel 編譯與記憶體配置，成本高且不穩定
    #（baseline 測試中 model_load 在 981~2737ms 之間飄，就是這類初始化雜訊）。
    # 先用假資料跑一次，把這些一次性成本吃掉，後面的計時才乾淨。
    with phases.phase("1b_warmup"):
        dummy_latent = torch.zeros(
            (1, unet.config.in_channels, args.img_size // 8, args.img_size // 8),
            device=device, dtype=unet.dtype)
        dummy_audio = torch.zeros((1, 50, 384), device=device)
        dummy_audio = pe(dummy_audio)
        _ = unet(dummy_latent, timesteps, encoder_hidden_states=dummy_audio).sample
        torch.cuda.synchronize(device)

    video_basename = os.path.basename(args.video_path).split(".")[0]
    audio_basename = os.path.basename(args.audio_path).split(".")[0]
    pose_path = os.path.join(args.intermediate_dir, video_basename,
                             audio_basename, video_basename)
    print(f"[info] pose_path = {pose_path}")
    if not os.path.exists(pose_path):
        sys.exit(f"[error] intermediate dir not found: {pose_path}")

    output_basename = f"{video_basename}#{audio_basename}"

    # ------------------------------------------------------------ 音訊特徵
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
    # 跟 baseline 完全一樣（Step 2b「前處理管線化」列為未來展望，這裡不動）
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

    # 串流畫面尺寸 = 原始影片解析度（貼回原圖後的大小），不是 img_size
    frame_h, frame_w = ori_img_list[0].shape[:2]
    print(f"[info] 串流輸出畫面尺寸 = {frame_w}x{frame_h}"
          f"  (img_size={args.img_size} 只是臉部裁切)")

    # ------------------------------------------------------------- sink 準備
    # 主要路徑：常駐 ffmpeg，frame 一產生就編碼送出
    ffmpeg_sink = FFmpegSink(width=frame_w, height=frame_h, fps=fps,
                             target=args.target,
                             preset=args.preset, tune=args.tune,
                             bitrate=args.bitrate, gop=args.gop,
                             pkt_size=args.pkt_size, sdp_path=args.sdp_file,
                             audio_path=None if args.no_audio else args.audio_path,
                             verbose=args.verbose)

    # debug 路徑：預設關閉。開啟才寫 PNG（供 diff_frames 驗證正確性用），
    # 但寫檔是阻塞 I/O，會拖慢主迴圈、污染計時，所以正式量測時不要開。
    png_sink = None
    if args.debug:
        png_dir = os.path.join(args.result_dir, output_basename)
        png_sink = PngSink(png_dir)
        print(f"[info] --debug 已開啟，同時寫 PNG 到 {png_dir}（會影響計時）")

    # ---------------------------------------------- 3. 生成 + 貼圖 + 送出
    t_frame = FrameTimer("gen+blend+send", verbose=args.verbose)
    gen = data_generator(whisper_chunks, latent_list, args.batch_size)
    n_batches = int(np.ceil(float(ind_max + 1) / args.batch_size))
    frame_idx = 0
    first_frame_ms = None
    t_start = time.perf_counter()

    with phases.phase("3_generate_blend_send"):
        t_frame.start()
        for whisper_batch, latent_batch in tqdm(gen, total=n_batches, desc="stream"):
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(unet.device)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.dtype)

            pred_latents = unet(latent_batch, timesteps,
                                encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)

            for res_frame in recon:
                if frame_idx >= ind_max + 1:
                    break

                box = [int(e) for e in box_list[frame_idx]]
                x1, y1, x2, y2 = box
                try:
                    res_crop_img = cv2.resize(res_frame.astype(np.uint8),
                                              (x2 - x1, y2 - y1))
                except Exception:
                    frame_idx += 1
                    continue
                combine_img = get_image(ori_img_list[frame_idx], res_crop_img, box)

                # 送出：這就是「畫面產生後幾毫秒內就進編碼器」的那一步
                ffmpeg_sink.write(combine_img)
                if png_sink is not None:
                    png_sink.write(combine_img, frame_idx + 1)

                if first_frame_ms is None:
                    first_frame_ms = (time.perf_counter() - t_start) * 1000.0

                frame_idx += 1
                t_frame.mark()

    ffmpeg_sink.close()
    if png_sink is not None:
        png_sink.close()

    print(f"[info] frames streamed: {frame_idx} -> {args.target}")

    meta = {
        "mode": "step3_ffmpeg_stream",
        "video": args.video_path,
        "audio": args.audio_path,
        "frames": frame_idx,
        "fps": fps,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "stream_frame_size": f"{frame_w}x{frame_h}",
        "target": args.target,
        "preset": args.preset,
        "tune": args.tune,
        "bitrate": args.bitrate,
        "gop": args.gop,
        "pkt_size": args.pkt_size,
        "audio_muxed": not args.no_audio,
        "debug_png": bool(args.debug),
        "first_frame_after_gen_start_ms": round(first_frame_ms or 0, 1),
        "gpu_id": args.gpu_id,
    }
    write_report(args.report, phases, [t_pre, t_frame], meta=meta, mode="step3")


def build_parser():
    p = argparse.ArgumentParser(description="Step 3: stream frames into a persistent ffmpeg")
    p.add_argument("--audio_path", type=str, required=True)
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--intermediate_dir", type=str, required=True)
    p.add_argument("--vae_model_path", type=str, default="pretrained_models/sd-vae-ft-mse")
    p.add_argument("--whisper_model_path", type=str, default="pretrained_models/whisper/tiny.pt")
    p.add_argument("--unet_model_path", type=str, default="pretrained_models/joygen")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--enable_pose_driven", action="store_true")
    p.add_argument("--result_dir", default="./results/stream_step3",
                   help="--debug 時 PNG 的輸出位置")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--report", type=str, default="./timing/step3")
    p.add_argument("--target", type=str, default="rtp://127.0.0.1:23000",
                   help="rtp://host:port | udp://host:port | file path")
    p.add_argument("--preset", type=str, default="ultrafast")
    p.add_argument("--tune", type=str, default="zerolatency")
    p.add_argument("--bitrate", type=str, default=None,
                   help="cap encoder bitrate, e.g. 4M; keeps keyframes small enough "
                        "for the receiver's socket buffer")
    p.add_argument("--gop", type=int, default=None,
                   help="keyframe interval in frames; lower means a receiver can "
                        "start decoding sooner")
    p.add_argument("--pkt_size", type=int, default=1200,
                   help="RTP payload size, should stay under the path MTU")
    p.add_argument("--sdp_file", type=str, default="stream.sdp",
                   help="where to write the SDP the RTP receiver needs")
    p.add_argument("--no_audio", action="store_true",
                   help="video only; by default the source audio is muxed in")
    p.add_argument("--debug", action="store_true",
                   help="同時寫 PNG 供 diff 驗證（會拖慢，正式計時勿開）")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(args)
    t0 = time.time()
    main(args)
    print(f"[info] wall clock total: {time.time() - t0:.1f}s")
