# Approach : 0
# Use zero-shot image classification model (CLIP) to predict whether there is fire or smoke in a video
#   Step 1 : Sample the video into frames with a fixed sample interval (may vary across videos)
#   Step 2 : Apply model prediction on each frame
#   Step 3 : Compute the detection by aggregating the prediction in a timeline
#   Step 4 : Classify the video based on the percentage of detection time over the total video time
#   Step 5 : Maximize the overall performance of this approach by varying the detection threshold

import os
import collections
from pprint import pprint
from hashlib import sha512
from functools import reduce
from itertools import groupby

import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


SAMPLING_INTERVAL = 5  # seconds


def predicate(a, b):
    if isinstance(a, int | float):
        if isinstance(b, int | float):
            return a + b
        return a + b[1] - b[0]
    if isinstance(b, int | float):
        return a[1] - a[0] + b
    return a[1] - a[0] + b[1] - b[0]


def process_videos(model, processor, prompts, folder_path):
    videos = []

    for movie_file in os.scandir(folder_path):
        if movie_file.path.endswith(".mp4"):
            vidcap = cv2.VideoCapture(movie_file.path)
            count = 0
            success = True
            fps = int(vidcap.get(cv2.CAP_PROP_FPS))
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = frame_count / fps
            adjusted_sampling_interval: float = SAMPLING_INTERVAL
            while (adjusted_sampling_interval / total_duration) * 100 > 1:
                adjusted_sampling_interval /= 2
            while success:
                success, image = vidcap.read()
                if count % (adjusted_sampling_interval * fps) == 0:
                    if success:
                        cv2.imwrite(f"""{movie_file.path[:-4]}.{count}.png""", image)
                count += 1

            images, indexes = {}, []

            for file in os.scandir(folder_path):
                if file.path.endswith(".png"):
                    frame = int(file.name.split(".")[1])
                    image = Image.open(file.path)
                    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to("cuda")
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = list(map(float, list(logits_per_image.softmax(dim=1)[0])))
                    images.update({frame: bool(sum(probs[:2]) >= 0.5)})
                    indexes.append(frame)

            images, results = collections.OrderedDict(sorted(images.items())), []

            for k, g in groupby(iterable=enumerate(images.values()), key=lambda x: x[1]):
                if k:
                    g = list(g)
                    results.append(
                        [
                            int(g[0][0]) * adjusted_sampling_interval,
                            int(g[0][0] + len(g)) * adjusted_sampling_interval,
                        ]
                    )

            detection_time = reduce(predicate, [*results, 0])
            detection_time_relative_to_total_duration = (detection_time / total_duration) * 100

            videos.append(detection_time_relative_to_total_duration)

            for file in os.scandir(folder_path):
                if file.path.endswith(".png"):
                    os.remove(file.path)

    return videos


if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts = [
        "a picture of an area with fire or smoke",
        "a photo of an area with fire or smoke",
        "a picture of an area without fire or smoke",
        "a photo of an area without fire or smoke",
    ]
    data_dir = "TEST_SET_DEV"
    positive_video_dir, negative_video_dir = "1", "0"
    positive_video_scores, negative_video_scores = [], []
    positive_video_number, negative_video_number = 0, 0
    true_positive, false_negative, true_negative, false_positive = 0, 0, 0, 0
    detection_threshold_range = list(map(float, np.around(np.arange(start=0, stop=30, step=0.01), decimals=2)))
    performances = {}

    for folder_path in [f.path for f in os.scandir(data_dir) if f.is_dir()]:
        for file in os.scandir(folder_path):
            if file.path.endswith(".png"):
                os.remove(file.path)

        if os.path.basename(folder_path) == positive_video_dir:
            positive_video_score_test_set_hash = sha512(
                "".join(
                    sorted(
                        os.path.basename(file.path) for file in os.scandir(folder_path) if file.path.endswith(".mp4")
                    )
                ).encode()
            ).hexdigest()
            positive_video_score_file = f"{positive_video_score_test_set_hash}.positive.md"
            positive_video_number = len(
                [movie_file for movie_file in os.scandir(folder_path) if movie_file.path.endswith(".mp4")]
            )

            if not os.path.exists(os.path.join(os.getcwd(), positive_video_score_file)):
                positive_video_scores = process_videos(
                    model=model,
                    processor=processor,
                    prompts=prompts,
                    folder_path=folder_path,
                )
                with open(positive_video_score_file, "w+") as file:
                    file.write(("\n".join(str(positive_video_score) for positive_video_score in positive_video_scores)))
            else:
                with open(positive_video_score_file, "r") as file:
                    positive_video_scores = list(map(float, file.readlines()))

        if os.path.basename(folder_path) == negative_video_dir:
            negative_video_score_test_set_hash = sha512(
                "".join(
                    sorted(
                        os.path.basename(file.path) for file in os.scandir(folder_path) if file.path.endswith(".mp4")
                    )
                ).encode()
            ).hexdigest()
            negative_video_score_file = f"{negative_video_score_test_set_hash}.negative.md"
            negative_video_number = len(
                [movie_file for movie_file in os.scandir(folder_path) if movie_file.path.endswith(".mp4")]
            )

            if not os.path.exists(os.path.join(os.getcwd(), negative_video_score_file)):
                negative_video_scores = process_videos(
                    model=model,
                    processor=processor,
                    prompts=prompts,
                    folder_path=folder_path,
                )
                with open(negative_video_score_file, "w+") as file:
                    file.write(("\n".join(str(negative_video_score) for negative_video_score in negative_video_scores)))
            else:
                with open(negative_video_score_file, "r") as file:
                    negative_video_scores = list(map(float, file.readlines()))

    for detection_threshold in detection_threshold_range:
        negative_videos_detected = [
            detection_time_relative_to_total_duration > detection_threshold
            for detection_time_relative_to_total_duration in negative_video_scores
        ]
        positive_videos_detected = [
            detection_time_relative_to_total_duration > detection_threshold
            for detection_time_relative_to_total_duration in positive_video_scores
        ]

        true_positive = sum(1 for positive_video in positive_videos_detected if positive_video)
        false_negative = positive_video_number - true_positive
        true_negative = sum(1 for negative_video in negative_videos_detected if not negative_video)
        false_positive = negative_video_number - true_negative

        recall = true_positive / (true_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        specificity = true_negative / (true_negative + false_positive)
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

        performances.update(
            {
                detection_threshold: {
                    "accuracy": accuracy,
                    "precision": precision,
                    "specificity": specificity,
                    "recall": recall,
                    "score": accuracy + precision + specificity + recall,
                }
            }
        )

    pprint(sorted(performances.items(), key=lambda item: item[1]["score"]))
