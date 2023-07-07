import os
from typing import Dict

import cv2
# from pprint import pprint
from striprtf.striprtf import rtf_to_text


def sample(
    path: str,
    sampling_interval: float,
    adjust_sampling_interval: bool,
    remove_image_before: bool,
    remove_video_after: bool,
) -> None:
    for folder_path in [f.path for f in os.scandir(path) if f.is_dir()]:

        if remove_image_before:
            for file in os.scandir(folder_path):
                if file.path.endswith(".png"):
                    os.remove(file.path)

        for file in os.scandir(folder_path):
            if file.path.endswith(".mp4"):
                vidcap = cv2.VideoCapture(file.path)
                count = 0
                success = True
                fps = int(vidcap.get(cv2.CAP_PROP_FPS))
                frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                if adjust_sampling_interval:
                    while (sampling_interval / duration) * 100 > 1:
                        sampling_interval /= 2
                while success:
                    success, image = vidcap.read()
                    if count % (sampling_interval * fps) == 0:
                        if success:
                            cv2.imwrite(f"""{file.path[:-4]}.{count}.png""", image)
                    count += 1

        if remove_video_after:
            for file in os.scandir(folder_path):
                if file.path.endswith(".mp4"):
                    os.remove(file.path)


def annotate(annotation_path: str) -> Dict[str, Dict[str, int | [int]]]:
    json_output = {}
    for folder_path in [f.path for f in os.scandir(annotation_path) if f.is_dir()]:
        for annotation in os.scandir(folder_path):
            if annotation.path.endswith(".rtf"):
                with open(annotation, "r") as content:
                    text = rtf_to_text(content.read())
                    seconds, *classes = text.split(",")
                    classes = list(map(lambda c: c.lower(), classes))
                    json_output.update(
                        {
                            annotation.path: {
                                "start": seconds,
                                "labels": classes,
                            }
                        }
                    )
    return json_output


# Driver code
#
# pprint(annotate(annotation_path="GT"))
#
# sample(path=os.path.join(os.getcwd(), "TRAINING_SET_DEV_1"),
#        sampling_interval=5,
#        adjust_sampling_interval=False,
#        remove_image_before=True,
#        remove_video_after=True)
