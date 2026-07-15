from __future__ import annotations

import json
import subprocess
import sys
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from tqdm.auto import tqdm


def run_json(cmd: list[str]) -> dict:
    return json.loads(subprocess.check_output(cmd, text=True))


def parse_fraction(text: str) -> Fraction:
    if not text or text == "0/0":
        return Fraction(0, 1)
    return Fraction(text)


def read_exact(stream: BinaryIO, size: int) -> bytes | None:
    """从 pipe 读取完整一帧，避免一次 read() 返回短数据。"""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise EOFError(f"收到不完整帧：还缺 {remaining} 字节")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def ffmpeg_encoder_test(options: list[str], pixel_format: str) -> tuple[bool, str]:
    """实际编码一帧，而不是只检查编码器名称。"""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1",
        "-frames:v", "1",
        "-vf", f"format={pixel_format}",
        *options,
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    return result.returncode == 0, result.stderr.strip()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法：python dpir_stream_final.py config.json")

    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    input_video = Path(config["input_video"]).resolve()
    output_video = Path(config["output_video"]).resolve()
    repo_dir = Path(config["repo_dir"]).resolve()

    if not input_video.exists():
        raise FileNotFoundError(input_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。请在 Kaggle 中选择 GPU P100。")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    arch_list = torch.cuda.get_arch_list()

    if capability == (6, 0) and "sm_60" not in arch_list:
        raise RuntimeError(
            f"当前 PyTorch 未包含 P100 所需的 sm_60。已包含架构：{arch_list}"
        )

    # ---------- 读取视频参数 ----------
    probe = run_json([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,"
        "color_space,color_transfer,color_primaries,color_range:"
        "format=duration",
        "-of", "json", str(input_video),
    ])
    stream = probe["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])

    avg_fps = parse_fraction(stream.get("avg_frame_rate", "0/0"))
    nominal_fps = parse_fraction(stream.get("r_frame_rate", "0/0"))
    fps = avg_fps if avg_fps > 0 else nominal_fps
    if fps <= 0:
        raise RuntimeError("无法从 ffprobe 解析帧率。")
    fps_text = f"{fps.numerator}/{fps.denominator}"

    duration = 0.0
    for candidate in (
        stream.get("duration"),
        probe.get("format", {}).get("duration"),
    ):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            pass

    test_mode = bool(config.get("test_mode", True))
    start_time = str(config.get("test_start", "00:00:00"))
    test_seconds = float(config.get("test_seconds", 10))
    segment_seconds = test_seconds if test_mode else duration

    nb_frames = stream.get("nb_frames")
    total_frames: int | None = None
    if not test_mode and isinstance(nb_frames, str) and nb_frames.isdigit():
        total_frames = int(nb_frames)
    elif segment_seconds > 0:
        total_frames = max(1, round(segment_seconds * float(fps)))

    if avg_fps > 0 and nominal_fps > 0 and avg_fps != nominal_fps:
        print(
            f"警告：avg_frame_rate={avg_fps}，r_frame_rate={nominal_fps}。"
            "rawvideo 管道会按平均帧率输出 CFR。"
        )

    # ---------- 加载官方 DPIR 模型 ----------
    sys.path.insert(0, str(repo_dir))
    from models.network_unet import UNetRes

    model_path = repo_dir / "model_zoo" / "drunet_deblocking_color.pth"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = UNetRes(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        bias=False,
    )

    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    torch.backends.cudnn.benchmark = True

    # 官方脚本使用 noise_level = (100 - quality_factor) / 100。
    jpeg_qf_hint = float(config.get("jpeg_qf_hint", 80))
    if not 1 <= jpeg_qf_hint <= 100:
        raise ValueError("jpeg_qf_hint 必须在 1～100。")
    deblock_level = (100.0 - jpeg_qf_hint) / 100.0

    # ---------- 自动选择 P100 可运行的 tile ----------
    force_tile = int(config.get("force_tile_size", 0))
    if force_tile:
        candidates = [force_tile]
    elif "P100" in gpu_name.upper() or capability == (6, 0):
        candidates = [640, 512, 384, 256]
    elif total_vram_gb >= 20:
        candidates = [768, 640, 512]
    elif total_vram_gb >= 12:
        candidates = [640, 512, 384]
    else:
        candidates = [384, 320, 256]

    def tile_fits(size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            x = torch.zeros(
                (1, 4, size, size),
                dtype=torch.float32,
                device=device,
            )
            with torch.inference_mode():
                y = model(x)
            del x, y
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                return False
            raise

    tile_size = next((size for size in candidates if tile_fits(size)), None)
    if tile_size is None:
        raise RuntimeError(f"候选 tile 均无法运行：{candidates}")

    configured_overlap = int(config.get("overlap", 64))
    overlap = min(configured_overlap, tile_size // 4)
    overlap = max(16, overlap)
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap 必须小于 tile_size。")

    # ---------- 真正的余弦 feather 权重 ----------
    @lru_cache(maxsize=128)
    def feather_mask(
        tile_h: int,
        tile_w: int,
        feather_top: bool,
        feather_bottom: bool,
        feather_left: bool,
        feather_right: bool,
    ) -> np.ndarray:
        wy = np.ones(tile_h, dtype=np.float32)
        wx = np.ones(tile_w, dtype=np.float32)

        def rising(n: int) -> np.ndarray:
            if n <= 1:
                return np.ones(n, dtype=np.float32)
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)
            return 0.5 - 0.5 * np.cos(np.pi * t)

        if feather_top:
            n = min(overlap, tile_h)
            wy[:n] *= rising(n)
        if feather_bottom:
            n = min(overlap, tile_h)
            wy[-n:] *= rising(n)[::-1]
        if feather_left:
            n = min(overlap, tile_w)
            wx[:n] *= rising(n)
        if feather_right:
            n = min(overlap, tile_w)
            wx[-n:] *= rising(n)[::-1]

        mask = np.outer(wy, wx)[..., None]
        return np.maximum(mask, 1e-6)

    @torch.inference_mode()
    def infer_tile(tile_u16: np.ndarray) -> np.ndarray:
        original_h, original_w = tile_u16.shape[:2]
        pad_h = (-original_h) % 8
        pad_w = (-original_w) % 8

        if pad_h or pad_w:
            pad_mode = "reflect" if min(original_h, original_w) > 1 else "edge"
            tile_u16 = np.pad(
                tile_u16,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode=pad_mode,
            )

        tile_f32 = tile_u16.astype(np.float32) / 65535.0
        image = (
            torch.from_numpy(np.ascontiguousarray(tile_f32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

        level_map = torch.full(
            (1, 1, image.shape[2], image.shape[3]),
            deblock_level,
            dtype=torch.float32,
            device=device,
        )

        prediction = model(torch.cat((image, level_map), dim=1))
        prediction = prediction[
            :, :, :original_h, :original_w
        ].clamp_(0.0, 1.0)

        return (
            prediction[0]
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )

    def process_frame(frame_u16: np.ndarray) -> np.ndarray:
        output = np.zeros((height, width, 3), dtype=np.float32)
        weights = np.zeros((height, width, 1), dtype=np.float32)

        for y0 in range(0, height, stride):
            y1 = min(y0 + tile_size, height)

            for x0 in range(0, width, stride):
                x1 = min(x0 + tile_size, width)
                tile = frame_u16[y0:y1, x0:x1]
                prediction = infer_tile(tile)

                mask = feather_mask(
                    y1 - y0,
                    x1 - x0,
                    y0 > 0,
                    y1 < height,
                    x0 > 0,
                    x1 < width,
                )

                output[y0:y1, x0:x1] += prediction * mask
                weights[y0:y1, x0:x1] += mask

        if np.any(weights <= 0):
            raise RuntimeError("tile 权重出现未覆盖像素。")

        output /= weights
        return np.rint(
            np.clip(output, 0.0, 1.0) * 65535.0
        ).astype("<u2")

    # ---------- 实际测试编码器 ----------
    cq = int(config.get("nvenc_cq", 17))
    aq_strength = int(config.get("aq_strength", 8))
    x265_crf = int(config.get("x265_crf", 16))
    prefer_nvenc = bool(config.get("prefer_nvenc", True))

    nvenc_full = [
        "-c:v", "hevc_nvenc",
        "-profile:v", "main10",
        "-preset", "p7",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(cq),
        "-b:v", "0",
        "-maxrate", "40M",
        "-bufsize", "80M",
        "-spatial-aq", "1",
        "-aq-strength", str(aq_strength),
        "-rc-lookahead", "20",
        "-multipass", "fullres",
    ]

    nvenc_simple = [
        "-c:v", "hevc_nvenc",
        "-profile:v", "main10",
        "-preset", "p7",
        "-rc", "vbr",
        "-cq", str(cq),
        "-b:v", "0",
    ]

    encoder_name = "libx265 CPU Main10"
    codec_options = [
        "-c:v", "libx265",
        "-preset", "slow",
        "-crf", str(x265_crf),
        "-profile:v", "main10",
        "-x265-params", "aq-mode=3",
    ]
    encoder_pixel_format = "yuv420p10le"

    if prefer_nvenc:
        ok, err = ffmpeg_encoder_test(nvenc_full, "p010le")

        if ok:
            encoder_name = "hevc_nvenc Main10 + Spatial AQ"
            codec_options = nvenc_full
            encoder_pixel_format = "p010le"
        else:
            ok_simple, err_simple = ffmpeg_encoder_test(
                nvenc_simple,
                "p010le",
            )

            if ok_simple:
                encoder_name = "hevc_nvenc Main10（简化参数）"
                codec_options = nvenc_simple
                encoder_pixel_format = "p010le"
                print("NVENC 完整参数测试失败，已降级：", err)
            else:
                print("NVENC Main10 不可用，回退到 CPU libx265。")
                print("完整参数错误：", err)
                print("简化参数错误：", err_simple)

    # ---------- 建立 FFmpeg 双输入管道 ----------
    # 输入0：Python 输出的增强 rawvideo
    # 输入1：原视频，用于复制音频、字幕、附件、章节和 metadata
    decoder_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
    ]

    if test_mode and start_time not in ("", "0", "00:00:00"):
        decoder_cmd += ["-ss", start_time]

    decoder_cmd += ["-i", str(input_video)]

    if test_mode and test_seconds > 0:
        decoder_cmd += ["-t", str(test_seconds)]

    decoder_cmd += [
        "-map", "0:v:0",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "pipe:1",
    ]

    encoder_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-video_size", f"{width}x{height}",
        "-framerate", fps_text,
        "-i", "pipe:0",
    ]

    if test_mode and start_time not in ("", "0", "00:00:00"):
        encoder_cmd += ["-ss", start_time]

    encoder_cmd += ["-i", str(input_video)]

    if test_mode and test_seconds > 0:
        encoder_cmd += ["-t", str(test_seconds)]

    encoder_cmd += [
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-map", "1:t?",
        "-map_metadata", "1",
        "-map_chapters", "1",

        # 16-bit RGB 管道转为 BT.709 limited-range 10-bit 4:2:0。
        "-vf",
        "scale=in_range=full:out_range=tv:"
        "out_color_matrix=bt709,"
        f"format={encoder_pixel_format}",

        *codec_options,

        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",

        "-c:a", "copy",
        "-c:s", "copy",
        "-c:t", "copy",
        str(output_video),
    ]

    print("\n=== 最终配置 ===")
    print(
        f"GPU: {gpu_name} | capability={capability} | "
        f"VRAM={total_vram_gb:.1f}GB"
    )
    print(
        f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | "
        f"arch={arch_list}"
    )
    print(
        f"视频: {width}x{height} @ {fps_text} "
        f"({float(fps):.6f} fps)"
    )
    print(
        f"DPIR: JPEG_QF_HINT={jpeg_qf_hint:g} "
        f"-> model level={deblock_level:.3f}"
    )
    print(
        f"Tile: {tile_size} | overlap={overlap} | stride={stride}"
    )
    print(f"编码器: {encoder_name}")
    print(f"输出: {output_video}")

    if test_mode:
        print(
            f"测试模式: start={start_time}, "
            f"seconds={test_seconds}"
        )

    decoder = subprocess.Popen(
        decoder_cmd,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0,
    )

    encoder = subprocess.Popen(
        encoder_cmd,
        stdin=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0,
    )

    if decoder.stdout is None or encoder.stdin is None:
        raise RuntimeError("无法建立 FFmpeg 管道。")

    frame_bytes = width * height * 3 * 2
    processed = 0
    start_clock = time.monotonic()

    bar = tqdm(
        total=total_frames,
        unit="frame",
        dynamic_ncols=True,
        smoothing=0.05,
        desc="DPIR",
    )

    try:
        while True:
            raw = read_exact(decoder.stdout, frame_bytes)

            if raw is None:
                break

            frame = np.frombuffer(
                raw,
                dtype="<u2",
            ).reshape(height, width, 3)

            enhanced = process_frame(frame)

            try:
                encoder.stdin.write(enhanced.tobytes(order="C"))
            except BrokenPipeError as exc:
                raise RuntimeError(
                    "FFmpeg 编码进程提前退出。"
                ) from exc

            processed += 1
            bar.update(1)

    except KeyboardInterrupt:
        print("\n收到中断，正在关闭子进程。")
        decoder.terminate()
        encoder.terminate()
        raise

    finally:
        bar.close()

        try:
            decoder.stdout.close()
        except Exception:
            pass

        try:
            encoder.stdin.close()
        except Exception:
            pass

    decoder_rc = decoder.wait()
    encoder_rc = encoder.wait()

    if decoder_rc != 0:
        raise RuntimeError(
            f"FFmpeg 解码失败，返回码 {decoder_rc}"
        )

    if encoder_rc != 0:
        raise RuntimeError(
            f"FFmpeg 编码失败，返回码 {encoder_rc}"
        )

    elapsed = time.monotonic() - start_clock
    print(
        f"\n完成：{processed} 帧，{elapsed / 60:.1f} 分钟，"
        f"平均 {processed / max(elapsed, 1e-9):.3f} fps"
    )
    print(output_video)


if __name__ == "__main__":
    main()
