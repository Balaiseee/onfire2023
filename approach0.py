# Approach : 0
# Use zero-shot image classification model (CLIP) to predict whether there is fire or smoke in a video
#   Step 1 : Sample the video into frames with a fixed sample interval
#   Step 2 : Apply model prediction on each frame
#   Step 3 : Compute the detection by aggregating the prediction in a timeline
#   Step 4 : Classify the video based on the percentage of detection time over the total video time

import os
import collections
from functools import reduce
from itertools import groupby

import cv2
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


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
prompts = [
    "a picture of an area with fire or smoke",
    "a photo of an area with fire or smoke",
    "a picture of an area without fire or smoke",
    "a photo of an area without fire or smoke",
]
videos = []
data_dir = "TRAINING_SET_DEV_1"

for folder_path in [f.path for f in os.scandir(data_dir) if f.is_dir()]:

    for file in os.scandir(folder_path):
        if file.path.endswith(".png"):
            os.remove(file.path)

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
                    # print(probs)
                    images.update({frame: bool(sum(probs[:2]) >= 0.5)})
                    # images.update({frame: bool(probs.index(max(probs)) <= 1)})
                    indexes.append(frame)

            images, results = collections.OrderedDict(sorted(images.items())), []

            for k, g in groupby(enumerate(images.values()), key=lambda x: x[1]):
                if k:
                    g = list(g)
                    results.append(
                        [int(g[0][0]) * adjusted_sampling_interval, int(g[0][0] + len(g)) * adjusted_sampling_interval]
                    )

            detection_time = reduce(predicate, [*results, 0])
            detection_time_relative_to_total_duration = (detection_time / total_duration) * 100
            videos.append(detection_time_relative_to_total_duration > 3)

            for file in os.scandir(folder_path):
                if file.path.endswith(".png"):
                    os.remove(file.path)
print(videos)
