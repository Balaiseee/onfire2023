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

videos = []
for folder_name in ("0", "1"):
    folder_path = os.path.join(os.getcwd(), "TRAINING_SET", "1")
    # for file in os.listdir(folder_path):
    #     if file.endswith(".jpg"):
    #         os.remove(os.path.join(folder_path, file))
    # continue
    for movie_file in os.listdir(folder_path):
        if movie_file.endswith(".mp4"):
            vidcap = cv2.VideoCapture(os.path.join(folder_path, movie_file))
            count = 0
            success = True
            fps = int(vidcap.get(cv2.CAP_PROP_FPS))
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            adjusted_sampling_interval: float = SAMPLING_INTERVAL
            while (adjusted_sampling_interval / duration) * 100 > 1:
                adjusted_sampling_interval /= 2
            while success:
                success, image = vidcap.read()
                if count % (adjusted_sampling_interval * fps) == 0:
                    if success:
                        cv2.imwrite(os.path.join(folder_path, f"""{movie_file[:-4]}.{count}.jpg"""), image)
                count += 1

            images, indexes = {}, []

            for file in os.listdir(folder_path):
                if file.endswith(".jpg"):
                    frame = int(file.split(".")[1])
                    image = Image.open(os.path.join(folder_path, file))
                    inputs = processor(text=["a picture of an area with fire or smoke",
                                             "a photo of an area with fire or smoke",
                                             "a picture of an area without fire or smoke",
                                             "a photo of an area without fire or smoke"],
                                       images=image, return_tensors="pt",
                                       padding=True).to("cuda")
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
                    results.append([int(g[0][0]) * adjusted_sampling_interval, int(g[0][0] + len(g)) * adjusted_sampling_interval])

            detection_time = reduce(predicate, [*results, 0])
            total_time = duration
            detection_time_relative_to_total_time = (detection_time / total_time) * 100

            # print((adjusted_sampling_interval / total_time) * 100)
            # print(detection_time_relative_to_total_time)
            videos.append(detection_time_relative_to_total_time > 3)

            for file in os.listdir(folder_path):
                if file.endswith(".jpg"):
                    os.remove(os.path.join(folder_path, file))
print(videos)
