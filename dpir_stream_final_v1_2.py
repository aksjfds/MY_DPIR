from __future__ import annotations

import inspect
import json
import queue
import subprocess
import sys
import threading
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
import torch.nn.functional as F


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



def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"

    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class RateLimitedProgress:
    """适合 Kaggle/Papermill 日志的低频进度报告器。"""

    def __init__(
        self,
        total: int | None,
        interval_seconds: float = 60.0,
    ) -> None:
        self.total = total
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        self.last_count = 0

    def update(self, count: int, force: bool = False) -> None:
        now = time.monotonic()

        should_report = (
            force
            or count == 1
            or now - self.last_report_at >= self.interval_seconds
            or (self.total is not None and count >= self.total)
        )

        if not should_report:
            return

        elapsed = max(now - self.started_at, 1e-9)
        speed = count / elapsed

        if self.total:
            percentage = min(100.0, count * 100.0 / self.total)
            remaining = max(0, self.total - count)
            eta = remaining / speed if speed > 0 else None

            message = (
                f"[进度] {count}/{self.total} "
                f"({percentage:6.2f}%) | "
                f"{speed:.3f} fps | "
                f"已用 {format_duration(elapsed)} | "
                f"ETA {format_duration(eta)}"
            )
        else:
            message = (
                f"[进度] {count} 帧 | "
                f"{speed:.3f} fps | "
                f"已用 {format_duration(elapsed)}"
            )

        print(message, flush=True)
        self.last_report_at = now
        self.last_count = count


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

    # 当前 DPIR master 的 UNetRes.__init__ 不接收 bias 参数；
    # 某些旧版/分支可能接收。按实际签名动态兼容。
    model_kwargs = {
        "in_nc": 4,
        "out_nc": 3,
        "nc": [64, 128, 256, 512],
        "nb": 4,
        "act_mode": "R",
        "downsample_mode": "strideconv",
        "upsample_mode": "convtranspose",
    }

    if "bias" in inspect.signature(UNetRes.__init__).parameters:
        model_kwargs["bias"] = False

    model = UNetRes(**model_kwargs)

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
        candidates = [1024, 896, 768, 640, 512, 384, 256]
    elif total_vram_gb >= 20:
        candidates = [1024, 896, 768, 640, 512]
    elif total_vram_gb >= 12:
        candidates = [896, 768, 640, 512, 384]
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

    # ---------- 预计算 tile 布局、feather mask 和总权重 ----------
    # 这些内容对整段视频固定，只计算一次。
    tile_plan: list[tuple[int, int, int, int, torch.Tensor]] = []
    weight_sum_cpu = np.zeros((1, height, width), dtype=np.float32)

    for y0 in range(0, height, stride):
        y1 = min(y0 + tile_size, height)

        for x0 in range(0, width, stride):
            x1 = min(x0 + tile_size, width)

            mask_cpu = feather_mask(
                y1 - y0,
                x1 - x0,
                y0 > 0,
                y1 < height,
                x0 > 0,
                x1 < width,
            )

            # HWC(1通道) -> CHW，之后常驻 GPU。
            mask_gpu = (
                torch.from_numpy(mask_cpu)
                .permute(2, 0, 1)
                .to(device=device, dtype=torch.float32)
            )

            tile_plan.append((y0, y1, x0, x1, mask_gpu))
            weight_sum_cpu[:, y0:y1, x0:x1] += (
                mask_cpu.transpose(2, 0, 1)
            )

    if np.any(weight_sum_cpu <= 0):
        raise RuntimeError("tile 权重出现未覆盖像素。")

    weight_sum_gpu = torch.from_numpy(weight_sum_cpu).to(
        device=device,
        dtype=torch.float32,
    )

    # 每种 padded tile 尺寸的强度图只创建一次。
    level_map_cache: dict[tuple[int, int], torch.Tensor] = {}

    # 输出缓冲区复用，避免每帧反复申请整张 GPU tensor。
    output_gpu = torch.empty(
        (3, height, width),
        dtype=torch.float32,
        device=device,
    )

    @torch.inference_mode()
    def process_frame(frame_u16: np.ndarray) -> np.ndarray:
        # 整帧只进行一次 CPU -> GPU 传输；
        # 原版本每个 tile 都单独传输，重复开销较大。
        frame_gpu = (
            torch.from_numpy(np.ascontiguousarray(frame_u16))
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            .div_(65535.0)
        )

        output_gpu.zero_()

        for y0, y1, x0, x1, mask_gpu in tile_plan:
            tile = frame_gpu[:, y0:y1, x0:x1].unsqueeze(0)
            original_h = y1 - y0
            original_w = x1 - x0

            pad_h = (-original_h) % 8
            pad_w = (-original_w) % 8

            if pad_h or pad_w:
                # 只在右侧和底部补齐到 8 的倍数，与原逻辑一致。
                tile = F.pad(
                    tile,
                    (0, pad_w, 0, pad_h),
                    mode="reflect",
                )

            padded_h = tile.shape[2]
            padded_w = tile.shape[3]
            level_key = (padded_h, padded_w)

            level_map = level_map_cache.get(level_key)
            if level_map is None:
                level_map = torch.full(
                    (1, 1, padded_h, padded_w),
                    deblock_level,
                    dtype=torch.float32,
                    device=device,
                )
                level_map_cache[level_key] = level_map

            prediction = model(torch.cat((tile, level_map), dim=1))
            prediction = prediction[
                0, :, :original_h, :original_w
            ].clamp_(0.0, 1.0)

            output_gpu[:, y0:y1, x0:x1].add_(
                prediction * mask_gpu
            )

        output_gpu.div_(weight_sum_gpu)

        # 整帧只进行一次 GPU -> CPU 传输。
        return (
            output_gpu
            .permute(1, 2, 0)
            .clamp_(0.0, 1.0)
            .mul(65535.0)
            .round_()
            .to(torch.int32)
            .cpu()
            .numpy()
            .astype("<u2")
        )

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
        "-x265-params", "aq-mode=3:log-level=error",
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
    print(
        f"进度日志间隔: {config.get('progress_interval_seconds', 60)} 秒 | "
        f"解码预取: {config.get('prefetch_frames', 2)} 帧"
    )
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

    progress_interval = float(
        config.get("progress_interval_seconds", 60)
    )
    prefetch_frames = max(
        1,
        int(config.get("prefetch_frames", 2)),
    )

    progress = RateLimitedProgress(
        total=total_frames,
        interval_seconds=progress_interval,
    )

    # 后台线程预取少量已解码帧，使 FFmpeg 解码和 GPU 推理重叠。
    frame_queue: queue.Queue[bytes | BaseException | None] = queue.Queue(
        maxsize=prefetch_frames
    )

    def decoder_worker() -> None:
        try:
            while True:
                raw_frame = read_exact(decoder.stdout, frame_bytes)

                if raw_frame is None:
                    frame_queue.put(None)
                    return

                frame_queue.put(raw_frame)

        except BaseException as exc:
            frame_queue.put(exc)

    decode_thread = threading.Thread(
        target=decoder_worker,
        name="ffmpeg-frame-prefetch",
        daemon=True,
    )
    decode_thread.start()

    try:
        while True:
            item = frame_queue.get()

            if item is None:
                break

            if isinstance(item, BaseException):
                raise item

            frame = (
                np.frombuffer(item, dtype="<u2")
                .reshape(height, width, 3)
                .copy()
            )

            enhanced = process_frame(frame)

            try:
                encoder.stdin.write(enhanced.tobytes(order="C"))
            except BrokenPipeError as exc:
                raise RuntimeError(
                    "FFmpeg 编码进程提前退出。"
                ) from exc

            processed += 1
            progress.update(processed)

    except KeyboardInterrupt:
        print("\n收到中断，正在关闭子进程。", flush=True)
        decoder.terminate()
        encoder.terminate()
        raise

    finally:
        progress.update(processed, force=True)

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
