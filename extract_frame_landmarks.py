import argparse

# torchlm 0.1.6.x does `from scipy.integrate import simps` at import time (in
# torchlm/metrics/metrics.py, pulled in by torchlm/models). SciPy removed that
# deprecated alias in 1.14; `simpson` is the same function under its real name.
# Must run before `import torchlm`.
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
import numpy as np
import multiprocessing as mp

def save_lmds(dict_item, txt_path):
    with open(txt_path, 'w') as obj:
        for name, landmarks in dict_item.items():
            obj.write(name + " ")
            for x, y in landmarks:
                obj.write(f"{int(x)}_{int(y)} ")
            obj.write("\n")

def _worker(rank, world_size, gpu, from_dir, lmd_output_dir, skip_existing, check_and_padding):
    """Process the shard of clips belonging to `rank`.

    torchlm.runtime is process-global module state (bind() mutates a singleton
    RuntimeWrapper), and torchlm.runtime.forward() takes exactly one HWC image
    -- there is no batched entrypoint. So the unit of parallelism has to be the
    process, not the tensor: each worker binds its own detector+landmark pair on
    its own GPU and walks a disjoint slice of the clip list.
    """
    os.makedirs(lmd_output_dir, exist_ok=True)
    device = torch.device(f"cuda:{gpu}")
    torchlm.runtime.bind(faceboxesv2(device=device))

    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True,
                num_nb=10, num_lms=68, net_stride=32, input_size=256,
                meanface_type="300w", map_location=device, checkpoint=None)
    ) 

    # Deterministic disjoint shards. sorted() first so every worker derives the
    # same ordering from the same directory listing, then stride by world_size:
    # unlike the old np.random.shuffle + --skip_existing race, no two workers can
    # ever pick the same clip, so nothing is processed twice.
    clip_dirs = sorted(os.listdir(from_dir))[rank::world_size]

    # position/leave keep the concurrent tqdm bars from overwriting each other.
    for clip_dir in tqdm(clip_dirs, desc=f"[gpu{gpu}:w{rank}] clips",
                         position=rank, leave=True):
        lmd_path = os.path.join(lmd_output_dir, f'{clip_dir}.txt')
        frames_path = os.path.join(from_dir, clip_dir)

        img_lists = sorted(os.listdir(frames_path))
        if check_and_padding and os.path.exists(lmd_path):
            
            with open(lmd_path, 'r') as file:
                lines = file.readlines()
                if len(img_lists) == len(lines):
                    continue
                else:
                    print(f'{lmd_path} has not aligned landmark size.{len(img_lists)}!={len(lines)} checking....')

        elif skip_existing and os.path.exists(lmd_path):
            continue

        
        # img_lists = sorted(os.listdir(frames_path))

        current_dict = {}

        last_landmarks = None
        for image_name in img_lists:
            if not (image_name.endswith('.png') or image_name.endswith('.jpg') or image_name.endswith('.jpeg')):
                continue
            frame = cv2.imread(os.path.join(frames_path, image_name))
            if frame is None:
                break
            landmarks, bboxes = torchlm.runtime.forward(frame)

            if len(bboxes) == 0:
                

                if check_and_padding:
                    
                    if last_landmarks is None:
                        print(f"{clip_dir}'s {image_name} does not have first frame. Passing ...")
                        break
                    print(f"{clip_dir}'s {image_name} padds the missing landmarks using last frames.")
                    landmarks = last_landmarks
                else:
                    print(f"{clip_dir}'s {image_name} is missing, later frames will not be processed!")
                    break

            current_dict[image_name] = [(x, y) for x, y in landmarks[0][:68]]
            last_landmarks = landmarks
        save_lmds(current_dict, lmd_path)


def main(from_dir, lmd_output_dir, skip_existing, check_and_padding,
         num_gpus=None, workers_per_gpu=1):
    """Fan the clip list out across GPUs (and processes per GPU).

    Each frame costs one detector pass plus one landmark pass on a batch of one,
    so a single worker leaves most of a modern GPU idle waiting on imread/decode.
    workers_per_gpu > 1 overlaps one worker's disk I/O with another's compute on
    the same device; both models are small (resnet18-class), so several fit.
    """
    os.makedirs(lmd_output_dir, exist_ok=True)

    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("No CUDA devices visible; this script requires a GPU.")
    num_gpus = available if num_gpus is None else min(num_gpus, available)
    world_size = num_gpus * workers_per_gpu

    if world_size == 1:
        # Single worker: run inline. Avoids a pointless subprocess and keeps
        # tracebacks attached to the parent, which matters when debugging.
        _worker(0, 1, 0, from_dir, lmd_output_dir, skip_existing, check_and_padding)
        return

    print(f"[INFO] {len(os.listdir(from_dir))} clips -> {world_size} workers "
          f"({num_gpus} GPUs x {workers_per_gpu} per GPU)")

    # spawn, not fork: CUDA contexts do not survive fork(). Each worker
    # initialises its own context after the process starts.
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(world_size):
        gpu = rank % num_gpus
        pr = ctx.Process(target=_worker,
                         args=(rank, world_size, gpu, from_dir, lmd_output_dir,
                               skip_existing, check_and_padding),
                         daemon=False)
        pr.start()
        procs.append(pr)

    failed = []
    try:
        for rank, pr in enumerate(procs):
            pr.join()
            if pr.exitcode != 0:
                failed.append((rank, pr.exitcode))
    except KeyboardInterrupt:
        # Don't leave orphaned workers holding GPU memory on Ctrl-C.
        print("\n[WARN] interrupted; terminating workers ...")
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
        for pr in procs:
            pr.join()
        raise

    if failed:
        # Surface partial failure instead of exiting 0 -- a silently dead worker
        # means a whole shard of clips has no landmarks, which would otherwise
        # only show up much later as missing files at training time.
        raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")
    print(f"[INFO] all {world_size} workers finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frame landmarks.')
    parser.add_argument('--from_dir', type=str, default='./data_processing/specified_formats/videos/video_frames/',
                        help='Directory where video frames are stored')
    parser.add_argument('--lmd_output_dir', type=str, default='./data_processing/specified_formats/videos/landmarks/',
                        help='Directory where landmarks will be saved')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing if landmarks file already exists')
    parser.add_argument('--check_and_padding', action='store_true',
                        help='Check and pad frames.')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='GPUs to use (default: all visible)')
    parser.add_argument('--workers_per_gpu', type=int, default=2,
                        help='Processes per GPU; >1 overlaps frame decode with inference')
    args = parser.parse_args()

    main(args.from_dir, args.lmd_output_dir, args.skip_existing, args.check_and_padding,
         num_gpus=args.num_gpus, workers_per_gpu=args.workers_per_gpu)
