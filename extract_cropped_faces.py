"""Crop every video in a directory to a fixed square face box.

Two stages, deliberately split because they bottleneck on different resources:

  1. **Detect** (GPU, serial). Sample every Nth frame, run torchlm's
     faceboxesv2 + pipnet, median the per-frame boxes into one synthetic box per
     video. `torchlm.runtime.bind` is process-global state and the models hold a
     CUDA context, so this stage cannot be threaded and is only worth
     multi-processing across *several* GPUs (see --num_workers).
  2. **Crop** (CPU, pooled). One ffmpeg transcode per video, fanned across a
     process pool. This is the stage that scales with core count.

The stages are pipelined, not barriered: a video's crop is submitted to the pool
as soon as its box is known, so ffmpeg saturates the cores while the GPU keeps
detecting. The original version of this script interleaved the two serially with
a blocking `os.system`, which left the GPU idle during every encode and the
cores idle during every detect.

The crop geometry is unchanged from that original: inflate the median box by
(1 + expanded_ratio), squarify to the longer side clamped to the frame, place it
horizontally centred and vertically by face_top_ratio, then slide it back inside
the frame. scripts/filter_data.py mirrors the same geometry.
"""

import argparse

# torchlm 0.1.6.x does `from scipy.integrate import simps` at import time (in
# torchlm/metrics/metrics.py, reached via torchlm.models). SciPy removed that
# deprecated alias in 1.14; `simpson` is the same function. Must run before
# `import torchlm`. .venv-bw also carries this in sitecustomize.py, but keep it
# here so the script works in a freshly built env too.
import scipy.integrate as _scipy_integrate
if not hasattr(_scipy_integrate, "simps"):
    _scipy_integrate.simps = _scipy_integrate.simpson

import torchlm
import torch
import cv2
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tqdm import tqdm
import os
import random
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait


def bind_models(device):
    """Bind faceboxesv2 + pipnet onto `device`. Process-global (torchlm design)."""
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18", pretrained=True, num_nb=10, num_lms=68,
            net_stride=32, input_size=256, meanface_type="300w",
            map_location=device, checkpoint=None
        )
    )


def resolve_crop_workers(requested):
    """Worker count for the ffmpeg stage. `requested=None` means auto.

    ffmpeg already multithreads a single encode, so one worker per core
    oversubscribes and loses to context switching. Half the cores keeps workers
    busy through ffmpeg's *serial* phases (open, header parse, mux, disk wait)
    without fighting its own thread pool. Capped at 16: past that the shared
    disk and the single detect stage, not the cores, set the ceiling.

    Mirrors src/common/parallel.resolve_cpu_workers, with a higher cap because
    this pool is the script's only parallelism and gets the whole box.
    """
    if requested is not None:
        return max(1, requested)
    return max(1, min(16, (os.cpu_count() or 1) // 2))


def derive_box(from_mp4_file_path, skip_per_frame, expanded_ratio, face_top_ratio):
    """Return (x1, y1, size) for the fixed square crop, or None if no face.

    Geometry identical to the pre-pool version of this script.
    """
    video = cv2.VideoCapture(from_mp4_file_path)
    index = 0
    bboxes_lists = []
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while video.isOpened():
        success = video.grab()
        if not success:
            break
        if index % skip_per_frame == 0:
            success, frame = video.retrieve()
            if not success:
                break
            landmarks, bboxes = torchlm.runtime.forward(frame)

            if bboxes.shape == (1, 5):
                bboxes_lists.append(bboxes[0])
            elif bboxes.shape[0] > 0:
                # If multiple persons exist, select the one with the largest width
                max_bboxes = max(bboxes, key=lambda bbox: bbox[2] - bbox[0])
                bboxes_lists.append(max_bboxes)
        index += 1
    video.release()

    x_center_lists, y_center_lists, width_lists, height_lists = [], [], [], []
    for bbox in bboxes_lists:
        x1, y1, x2, y2 = bbox[:4]
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        x_center_lists.append(x_center)
        y_center_lists.append(y_center)
        width_lists.append(x2 - x1)
        height_lists.append(y2 - y1)

    if not (x_center_lists and y_center_lists and width_lists and height_lists):
        return None

    x_center = sorted(x_center_lists)[len(x_center_lists) // 2]
    y_center = sorted(y_center_lists)[len(y_center_lists) // 2]
    median_width = sorted(width_lists)[len(width_lists) // 2]
    median_height = sorted(height_lists)[len(height_lists) // 2]

    expanded_width = int(median_width * (1 + expanded_ratio))
    expanded_height = int(median_height * (1 + expanded_ratio))

    fixed_cropped_width = min(max(expanded_width, expanded_height), width, height)

    x1 = int(x_center - fixed_cropped_width / 2)
    y1 = int(y_center - fixed_cropped_width * face_top_ratio)
    x1 = min(max(x1, 0), width - fixed_cropped_width)
    y1 = min(max(y1, 0), height - fixed_cropped_width)

    return x1, y1, fixed_cropped_width


def crop_one(job):
    """ffmpeg worker. Returns (mp4_name, status, detail). Runs in a subprocess.

    Writes to a hidden `.partial` sibling and renames on success, so an
    interrupted run never leaves a truncated mp4 that the `os.path.exists` skip
    at the top of the next run would treat as done. The original wrote the final
    path directly and had exactly that failure mode.

    argv is a list, not a shell string: the previous `os.system` f-string broke
    on any path containing a space, quote or `$`.
    """
    mp4_name, from_path, to_path, size, x1, y1, threads = job
    tmp_path = os.path.join(
        os.path.dirname(to_path), f".{os.path.basename(to_path)}.partial.mp4"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", from_path,
        "-filter:v",
        f"crop={size}:{size}:{x1}:{y1},"
        f"pad={size}:{size}:{x1 + size}:{y1 + size}",
        # Each worker is capped so N concurrent ffmpegs don't each try to grab
        # the whole box and thrash. 0 = ffmpeg's own default (unbounded).
        *(("-threads", str(threads)) if threads else ()),
        "-c:a", "copy",
        tmp_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        detail = (proc.stderr.strip().splitlines() or ["ffmpeg error"])[-1]
        return mp4_name, "fail", detail
    os.replace(tmp_path, to_path)
    return mp4_name, "ok", f"{size}x{size}+{x1}+{y1}"


def main(from_dir_prefix, output_dir_prefix, expanded_ratio, skip_per_frame, shuffle,
         face_top_ratio, num_workers, ffmpeg_threads, device_arg, debug):
    os.makedirs(output_dir_prefix, exist_ok=True)

    device = torch.device(device_arg)
    bind_models(device)

    mp4_files = os.listdir(from_dir_prefix)
    if shuffle:
        random.shuffle(mp4_files)

    workers = resolve_crop_workers(num_workers)
    print(f"[INFO] device={device} crop_workers={workers} "
          f"ffmpeg_threads={ffmpeg_threads or 'auto'} cores={os.cpu_count()}")

    written = skipped = failed = 0
    # Bound the queue of in-flight crops. Detection is much faster than encoding,
    # so an unbounded submit loop would finish detecting every video and pile
    # thousands of pending futures into memory before the pool drained.
    max_pending = workers * 4
    pending = {}

    def drain(block_until):
        """Reap finished futures until at most `block_until` remain pending."""
        nonlocal written, skipped, failed
        while len(pending) > block_until:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                mp4_name, status, detail = fut.result()
                del pending[fut]
                if status == "ok":
                    written += 1
                else:
                    failed += 1
                    tqdm.write(f"[fail] {mp4_name}: {detail}")
                    if debug:
                        raise SystemExit(1)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mp4_name in tqdm(mp4_files, desc="videos"):
            from_mp4_file_path = os.path.join(from_dir_prefix, mp4_name)
            to_mp4_file_path = os.path.join(output_dir_prefix, mp4_name)

            if os.path.exists(to_mp4_file_path):
                skipped += 1
                continue

            box = derive_box(
                from_mp4_file_path, skip_per_frame, expanded_ratio, face_top_ratio
            )
            if box is None:
                print(f"Face may not exist, please check the video: {mp4_name}")
                failed += 1
                if debug:
                    raise SystemExit(1)
                continue

            x1, y1, size = box
            job = (mp4_name, from_mp4_file_path, to_mp4_file_path, size, x1, y1,
                   ffmpeg_threads)
            if debug:
                print(f"[debug] {mp4_name} -> crop={size}:{size}:{x1}:{y1}")

            drain(max_pending - 1)
            pending[pool.submit(crop_one, job)] = mp4_name

        drain(0)

    print(f"[INFO] done: {written} written, {skipped} already present, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--from_dir_prefix', type=str, default='/teamspace/studios/this_studio/imt/assets/identities',
                        help='input directory where raw videos are stored')
    parser.add_argument('--output_dir_prefix', type=str, default='data_processing/cropped_faces/',
                        help='output directory where cropped faces will be stored')
    parser.add_argument('--expanded_ratio', type=float, default=1.2,
                        help='ratio to expand the bounding box for cropping, the larger the value, the smaller the face')
    parser.add_argument('--skip_per_frame', type=int, default=25,
                        help='number of frames to skip before detecting the face again, here it defaults to detecting the face position every 25 frames, only a rough calculation of the face position in the frame is needed, not too large')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, if you want to print the ffmpeg command, you can turn on this switch, note that it will automatically exit when an exception occurs after enabling.')
    parser.add_argument('--shuffle', action='store_true', help='Enable shuffling of the input data')
    parser.add_argument('--face_top_ratio', type=float, default=0.35,
                        help='vertical position of the face center within the cropped square, as a fraction from the top (0 = top edge, 0.5 = vertically centered, 1 = bottom edge)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='parallel ffmpeg crop workers. Default: half the cores, capped at 16. 1 runs the crops serially.')
    parser.add_argument('--ffmpeg_threads', type=int, default=4,
                        help='-threads passed to each ffmpeg worker; 0 leaves ffmpeg to decide. num_workers * ffmpeg_threads is roughly the core count you will use.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='torch device for the detection stage, e.g. cuda:0 or cpu')
    args = parser.parse_args()

    main(args.from_dir_prefix, args.output_dir_prefix, args.expanded_ratio,
         args.skip_per_frame, args.shuffle, args.face_top_ratio, args.num_workers,
         args.ffmpeg_threads, args.device, args.debug)
