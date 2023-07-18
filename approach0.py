# Approach : 0
# Use zero-shot image classification model (CLIP) to predict whether there is fire or smoke in a video
#   Step 1 : Sample the video into frames with a fixed sample interval (may vary across videos)
#   Step 2 : Apply model prediction on each frame
#   Step 3 : Compute the detection by aggregating the prediction in a timeline
#   Step 4 : Classify the video based on the percentage of detection time over the total video time
#   Step 5 : Maximize the overall performance of this approach by varying the detection threshold

import os
import json
from hashlib import sha512

import cv2
import torch
import open_clip
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

from utils import slugify


SAMPLING_INTERVAL = 5  # seconds


def predicate(a, b):
    if isinstance(a, int | float):
        if isinstance(b, int | float):
            return a + b
        return a + b[1] - b[0]
    if isinstance(b, int | float):
        return a[1] - a[0] + b
    return a[1] - a[0] + b[1] - b[0]


def process_videos(model, prompts, frame_detection_threshold, folder_path):
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

            frames = []

            for image_file in os.scandir(folder_path):
                if image_file.path.endswith(".png"):
                    image = preprocess(Image.open(image_file.path)).cuda().unsqueeze(0)

                    with torch.no_grad():
                        image_features = model.encode_image(image)
                        text_features = model.encode_text(prompts)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0].tolist()
                    frames.append(bool(sum(probs[:2]) >= frame_detection_threshold))

            detection_time = min(sum(1 for frame in frames if frame) * adjusted_sampling_interval, total_duration)
            detection_time_relative_to_total_duration = (detection_time / total_duration) * 100
            videos.append(detection_time_relative_to_total_duration)

            for image_file in os.scandir(folder_path):
                if image_file.path.endswith(".png"):
                    os.remove(image_file.path)

    return videos


if __name__ == "__main__":
    for model_name, pretrained in open_clip.list_pretrained():
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
        model = model.cuda()

        tokenizer = open_clip.get_tokenizer(model_name=model_name)
        prompts = tokenizer(
            [
                "a picture of an area with fire or smoke",
                "a photo of an area with fire or smoke",
                "a picture of an area without fire or smoke",
                "a photo of an area without fire or smoke",
            ]
        ).cuda()

        data_dir = "VAL_SET_DEV"
        positive_video_dir, negative_video_dir = "1", "0"
        positive_video_scores, negative_video_scores = [], []
        positive_video_number, negative_video_number = 0, 0
        true_positive, false_negative, true_negative, false_positive = 0, 0, 0, 0
        frame_detection_threshold = 0.5
        video_detection_threshold_range = list(map(float, np.around(np.arange(start=0, stop=30, step=0.01), decimals=2)))
        performances = {}

        for folder_path in [f.path for f in os.scandir(data_dir) if f.is_dir()]:
            for image_file in os.scandir(folder_path):
                if image_file.path.endswith(".png"):
                    os.remove(image_file.path)

            if os.path.basename(folder_path) == positive_video_dir:
                positive_video_score_validation_set_hash = sha512("".join(sorted(os.path.basename(file.path) for file in os.scandir(folder_path) if file.path.endswith(".mp4"))).encode()).hexdigest()
                positive_video_score_file = f"{positive_video_score_validation_set_hash}.{slugify(model_name)}.{slugify(pretrained)}.positive.md"
                positive_video_number = len([movie_file for movie_file in os.scandir(folder_path) if movie_file.path.endswith(".mp4")])

                if not os.path.exists(os.path.join(os.getcwd(), positive_video_score_file)):
                    positive_video_scores = process_videos(
                        model=model,
                        prompts=prompts,
                        frame_detection_threshold=frame_detection_threshold,
                        folder_path=folder_path,
                    )
                    with open(positive_video_score_file, "w+") as file:
                        file.write(("\n".join(str(positive_video_score) for positive_video_score in positive_video_scores)))
                else:
                    with open(positive_video_score_file, "r") as file:
                        positive_video_scores = list(map(float, file.readlines()))

            if os.path.basename(folder_path) == negative_video_dir:
                negative_video_score_validation_set_hash = sha512("".join(sorted(os.path.basename(file.path) for file in os.scandir(folder_path) if file.path.endswith(".mp4"))).encode()).hexdigest()
                negative_video_score_file = f"{negative_video_score_validation_set_hash}.{slugify(model_name)}.{slugify(pretrained)}.negative.md"
                negative_video_number = len([movie_file for movie_file in os.scandir(folder_path) if movie_file.path.endswith(".mp4")])

                if not os.path.exists(os.path.join(os.getcwd(), negative_video_score_file)):
                    negative_video_scores = process_videos(
                        model=model,
                        prompts=prompts,
                        frame_detection_threshold=frame_detection_threshold,
                        folder_path=folder_path,
                    )
                    with open(negative_video_score_file, "w+") as file:
                        file.write(("\n".join(str(negative_video_score) for negative_video_score in negative_video_scores)))
                else:
                    with open(negative_video_score_file, "r") as file:
                        negative_video_scores = list(map(float, file.readlines()))

        for video_detection_threshold in video_detection_threshold_range:
            negative_videos_detected = [detection_time_relative_to_total_duration > video_detection_threshold for detection_time_relative_to_total_duration in negative_video_scores]
            positive_videos_detected = [detection_time_relative_to_total_duration > video_detection_threshold for detection_time_relative_to_total_duration in positive_video_scores]
            y_true = negative_video_number * [0] + positive_video_number * [1]
            y_pred = list(map(int, negative_videos_detected + positive_videos_detected))

            performances.update({video_detection_threshold: classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)["macro avg"]})

        output_data = {detection_threshold: classification_metrics for detection_threshold, classification_metrics in sorted(performances.items(), key=lambda item: item[1]["f1-score"])[-50:]}

        with open(f"""output.{slugify(model_name)}.{slugify(pretrained)}.json""", "w", encoding="utf-8") as output_file:
            json.dump(output_data, output_file, ensure_ascii=False, indent=4)
