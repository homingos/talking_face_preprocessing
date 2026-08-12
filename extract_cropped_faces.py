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

def main(from_dir_prefix, output_dir_prefix, expanded_ratio, skip_per_frame, shuffle, face_top_ratio):
    os.makedirs(output_dir_prefix, exist_ok=True)

    device = torch.device("cuda:0")
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18", pretrained=True, num_nb=10, num_lms=68,
            net_stride=32, input_size=256, meanface_type="300w",
            map_location=device, checkpoint=None
        )
    )
    if shuffle:
        mp4_files = os.listdir(from_dir_prefix)
        random.shuffle(mp4_files)
    else:
        mp4_files = os.listdir(from_dir_prefix)

    for mp4_name in tqdm(mp4_files):
        from_mp4_file_path = os.path.join(from_dir_prefix, mp4_name)
        to_mp4_file_path = os.path.join(output_dir_prefix, mp4_name)

        if os.path.exists(to_mp4_file_path):
            continue

        video = cv2.VideoCapture(from_mp4_file_path)
        index = 0
        bboxes_lists = []
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while video.isOpened():
            success = video.grab()
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

        x_center_lists, y_center_lists, width_lists, height_lists = [], [], [], []
        for bbox in bboxes_lists:
            x1, y1, x2, y2 = bbox[:4]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_lists.append(x_center)
            y_center_lists.append(y_center)
            width_lists.append(x2 - x1)
            height_lists.append(y2 - y1)

        if not (x_center_lists and y_center_lists and width_lists and height_lists):
            print(f"Face may not exist, please check the video: {mp4_name}")
            if args.debug:
                exit(1)
            continue

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

        if args.debug:
            print(cmd)
            cmd = (
                f'ffmpeg -i {from_mp4_file_path} -filter:v "crop={fixed_cropped_width}:{fixed_cropped_width}:{x1}:{y1},'
                f'pad={fixed_cropped_width}:{fixed_cropped_width}:{x1 + fixed_cropped_width}:{y1 + fixed_cropped_width}" '
                f'-c:a copy {to_mp4_file_path} -y'
            )
        else:
            cmd = (
                f'ffmpeg -i {from_mp4_file_path} -filter:v "crop={fixed_cropped_width}:{fixed_cropped_width}:{x1}:{y1},'
                f'pad={fixed_cropped_width}:{fixed_cropped_width}:{x1 + fixed_cropped_width}:{y1 + fixed_cropped_width}" '
                f'-c:a copy {to_mp4_file_path} -y -loglevel quiet'
            )

        if os.system(cmd) != 0:
            print(f"Error executing command: {cmd}, please check")
            if args.debug:
                exit(1)
            continue

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
    args = parser.parse_args()

    main(args.from_dir_prefix, args.output_dir_prefix, args.expanded_ratio, args.skip_per_frame, args.shuffle, args.face_top_ratio)