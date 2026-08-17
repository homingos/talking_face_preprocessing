"""Convert a directory of videos to the fixed training format: 25fps video,
16kHz mono audio, and PNG frame folders.

Three independent stages, each one ffmpeg invocation per video, fanned across a
process pool. The stages run one after another because stage 3 consumes stage
1's output, but *within* a stage the videos are embarrassingly parallel -- the
original version of this script ran them with a blocking `subprocess.run` in a
plain for-loop, which used roughly one core's worth of a many-core box.

The three stages do not want the same worker count, so each gets its own:

  1. **25fps convert** (`--video_workers`). A full CPU re-encode; the heaviest
     stage. ffmpeg self-threads, so workers * ffmpeg_threads is the core budget.
  2. **16kHz audio** (`--audio_workers`). Cheap demux + resample, mostly I/O.
     Runs more workers with one ffmpeg thread each.
  3. **PNG frames** (`--frame_workers`). Write-amplifying: one PNG per frame
     turns a few MB of video into hundreds. Disk, not cores, sets the ceiling,
     so this pool is deliberately the smallest.

Mirrors the pool idiom in extract_cropped_faces.py rather than importing
src/common/parallel.py: this directory is a git submodule and must keep working
standalone, so the worker heuristic is duplicated here on purpose (same reason
extract_cropped_faces.resolve_crop_workers is a local copy).
"""

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm


def resolve_ffmpeg_workers(requested, cap):
    """Worker count for an ffmpeg stage. `requested=None` means auto.

    ffmpeg already multithreads a single encode, so one worker per core
    oversubscribes and loses to context switching. Half the cores keeps workers
    busy through ffmpeg's *serial* phases (open, header parse, mux, disk wait)
    without fighting its own thread pool.

    `cap` is per-stage: the encode stage scales with cores, the frame-dump stage
    saturates the disk long before the cores. Mirrors
    src/common/parallel.resolve_cpu_workers and
    extract_cropped_faces.resolve_crop_workers.
    """
    if requested is not None:
        return max(1, requested)
    return max(1, min(cap, (os.cpu_count() or 1) // 2))


def _run_ffmpeg(cmd):
    """Run ffmpeg, returning (ok, detail).

    `-nostdin` and `-y` both matter in a pool: without them a stray existing
    output or a malformed input makes ffmpeg wait on a tty prompt that no worker
    has, hanging the slot forever. The original script passed `-y` *after* the
    output path, where it is not a global flag and does not reliably suppress
    the prompt.
    """
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        # ffmpeg's own diagnosis is in the last few lines; the banner is noise.
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
        return False, f"ffmpeg exit {proc.returncode}: {tail}"
    return True, ""


def _threads_args(ffmpeg_threads):
    return ["-threads", str(ffmpeg_threads)] if ffmpeg_threads else []


def convert_video_one(video_file, source_folder, target_folder, ffmpeg_threads):
    """Re-encode one video to 25fps. Returns (name, status, detail)."""
    source_path = os.path.join(source_folder, video_file)
    target_path = os.path.join(target_folder, video_file)
    if os.path.exists(target_path):
        return video_file, "skip", ""

    # Encode to a temp name and rename on success. A worker killed mid-encode
    # would otherwise leave a truncated mp4 that the exists() check above treats
    # as finished forever -- silent dataset corruption, and far more likely with
    # a pool than with the original serial loop.
    tmp_path = target_path + ".part.mp4"
    ok, detail = _run_ffmpeg(
        ["ffmpeg", "-nostdin", "-y", "-i", source_path, "-r", "25"]
        + _threads_args(ffmpeg_threads)
        + [tmp_path]
    )
    if not ok:
        _unlink_quiet(tmp_path)
        return video_file, "fail", detail
    os.replace(tmp_path, target_path)
    return video_file, "ok", ""


def convert_audio_one(video_file, source_folder, target_folder, ffmpeg_threads):
    """Extract one 16kHz mono wav. Returns (name, status, detail)."""
    source_path = os.path.join(source_folder, video_file)
    audio_file = os.path.splitext(video_file)[0] + '.wav'
    target_path = os.path.join(target_folder, audio_file)
    if os.path.exists(target_path):
        return video_file, "skip", ""

    tmp_path = target_path + ".part.wav"
    ok, detail = _run_ffmpeg(
        ["ffmpeg", "-nostdin", "-y", "-i", source_path, "-ar", "16000", "-ac", "1"]
        + _threads_args(ffmpeg_threads)
        + [tmp_path]
    )
    if not ok:
        _unlink_quiet(tmp_path)
        return video_file, "fail", detail
    os.replace(tmp_path, target_path)
    return video_file, "ok", ""


def extract_frames_one(video_file, source_folder, target_folder, ffmpeg_threads):
    """Dump one video's frames to a PNG folder. Returns (name, status, detail)."""
    source_path = os.path.join(source_folder, video_file)
    frame_folder = os.path.splitext(video_file)[0]
    frame_target_folder = os.path.join(target_folder, frame_folder)
    if os.path.exists(frame_target_folder):
        return video_file, "skip", ""

    # The original created the real folder up front, so an interrupted run left a
    # half-full directory that the exists() check skipped on every later pass.
    # Fill a temp dir and rename it into place, so the final path only ever
    # appears complete.
    tmp_folder = frame_target_folder + ".part"
    shutil.rmtree(tmp_folder, ignore_errors=True)
    os.makedirs(tmp_folder, exist_ok=True)
    ok, detail = _run_ffmpeg(
        ["ffmpeg", "-nostdin", "-y", "-i", source_path, "-vf", "fps=25"]
        + _threads_args(ffmpeg_threads)
        + [os.path.join(tmp_folder, "%06d.png")]
    )
    if not ok:
        shutil.rmtree(tmp_folder, ignore_errors=True)
        return video_file, "fail", detail
    os.replace(tmp_folder, frame_target_folder)
    return video_file, "ok", ""


def _unlink_quiet(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def list_mp4s(source_folder):
    """Sorted .mp4 names in `source_folder`.

    Sorted because os.listdir order is filesystem-dependent; with a pool the
    completion order is arbitrary anyway, but a stable *submission* order keeps
    reruns and logs comparable.
    """
    if not os.path.isdir(source_folder):
        raise SystemExit(f"[FATAL] source folder does not exist: {source_folder}")
    return sorted(f for f in os.listdir(source_folder) if f.endswith('.mp4'))


def run_stage(label, worker_fn, source_folder, target_folder, workers,
              ffmpeg_threads, debug):
    """Fan `worker_fn` over every mp4 in `source_folder`. Returns failure count.

    imap-style unordered completion via ProcessPoolExecutor.map would preserve
    input order and stall the progress bar behind the slowest early video, so
    submit-then-as-completed is used instead. Videos are small and uniform here,
    so a bounded submit queue (as in extract_cropped_faces) is not needed -- the
    whole job list is a few thousand tuples at most.
    """
    os.makedirs(target_folder, exist_ok=True)
    videos = list_mp4s(source_folder)
    if not videos:
        print(f"[WARN] {label}: no .mp4 files in {source_folder}, nothing to do")
        return 0

    print(f"[INFO] {label}: {len(videos)} videos, workers={workers}, "
          f"ffmpeg_threads={ffmpeg_threads or 'auto'}, cores={os.cpu_count()}")

    task = partial(worker_fn, source_folder=source_folder,
                   target_folder=target_folder, ffmpeg_threads=ffmpeg_threads)

    written = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, status, detail in tqdm(
            pool.map(task, videos), total=len(videos), desc=label
        ):
            if status == "ok":
                written += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                tqdm.write(f"[fail] {name}: {detail}")
                if debug:
                    raise SystemExit(1)

    print(f"[INFO] {label}: {written} written, {skipped} skipped, {failed} failed")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Process videos and extract data.")
    # BooleanOptionalAction, not type=bool: argparse applies `type` to the string,
    # and bool("False") is True, so the original `--convert_video False` silently
    # enabled the stage. These now take --no-convert-video etc.
    parser.add_argument("--convert_video", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable video conversion to 25fps.")
    parser.add_argument("--convert_audio", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable audio conversion to 16kHz.")
    parser.add_argument("--extract_frames", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable frame extraction to PNG format.")
    parser.add_argument("--source_folder", type=str, default='data_processing/cropped_faces/', help="Source folder path.")
    parser.add_argument("--audio_source_folder", type=str, default=None,
                        help="Source folder for audio extraction. Defaults to "
                             "--source_folder; the README's pipeline takes audio "
                             "from raw_data, since cropped videos may carry no "
                             "audio track.")
    parser.add_argument("--video_target_folder", type=str, default='data_processing/specified_formats/videos/videos_25fps/', help="Target folder path for videos.")
    parser.add_argument("--audio_target_folder", type=str, default='data_processing/specified_formats/audios/audios_16k/', help="Target folder path for audios.")
    parser.add_argument("--frames_target_folder", type=str, default='data_processing/specified_formats/videos/video_frames/', help="Target folder path for video frames.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Workers for every stage; per-stage flags below win. "
                             "Default: auto (half the cores, per-stage cap).")
    parser.add_argument("--video_workers", type=int, default=None,
                        help="Workers for the 25fps encode stage. Auto-caps at 16.")
    parser.add_argument("--audio_workers", type=int, default=None,
                        help="Workers for the 16kHz audio stage. Auto-caps at 32 "
                             "(cheap and I/O-bound, so it takes more).")
    parser.add_argument("--frame_workers", type=int, default=None,
                        help="Workers for the PNG frame stage. Auto-caps at 8: "
                             "dumping PNGs is disk-bound, not CPU-bound.")
    parser.add_argument("--ffmpeg_threads", type=int, default=4,
                        help="-threads passed to each ffmpeg worker; 0 leaves "
                             "ffmpeg to decide. workers * ffmpeg_threads is "
                             "roughly the core count you will use.")
    parser.add_argument("--debug", action="store_true",
                        help="Abort on the first ffmpeg failure instead of "
                             "logging it and continuing.")

    args = parser.parse_args()

    def workers_for(stage_specific, cap):
        return resolve_ffmpeg_workers(
            stage_specific if stage_specific is not None else args.num_workers, cap
        )

    failed = 0
    if args.convert_video:
        failed += run_stage(
            "25fps", convert_video_one, args.source_folder, args.video_target_folder,
            workers_for(args.video_workers, 16), args.ffmpeg_threads, args.debug,
        )
    if args.convert_audio:
        # Audio is a demux+resample, not an encode: one thread each, more workers.
        failed += run_stage(
            "16k audio", convert_audio_one,
            args.audio_source_folder or args.source_folder,
            args.audio_target_folder,
            workers_for(args.audio_workers, 32), 1, args.debug,
        )
    if args.extract_frames:
        # Reads the 25fps output, so this must follow the video stage.
        failed += run_stage(
            "frames", extract_frames_one, args.video_target_folder,
            args.frames_target_folder,
            workers_for(args.frame_workers, 8), args.ffmpeg_threads, args.debug,
        )

    if failed:
        raise SystemExit(f"[FATAL] {failed} ffmpeg task(s) failed; see [fail] lines above")


if __name__ == "__main__":
    main()
