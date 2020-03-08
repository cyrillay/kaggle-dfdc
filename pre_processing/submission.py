# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np
import pandas as pd
import os
import cv2
import json
from pathlib import Path
import face_recognition
from fastai.vision import ImageList, DatasetType, load_learner


def save_frames(video_path: str, output_base_path: str):
    Path(output_base_path).mkdir(parents=True, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem
    print('total number frames =', vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success, image = vidcap.read()
    while success:
        # save 1 frame every 60 frames
        if count % 60 == 0:
            output_path = f'{output_base_path}/{video_name}_frame{count}.jpg'
            print(f'Writing at : {output_path}')
            cv2.imwrite(output_path, image)
            print('Read a new frame: ', success)
            count += 1
        else:
            print('Skipping frame ', count)
        success, image = vidcap.read()
        count += 1


def extract_face(image_path: Path, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    face_locations = face_recognition.face_locations(image)
    file_name = image_path.stem
    print(f'Number of face(s) in this image: {len(face_locations)}.')
    faces = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        print(face_location)
        # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cropped_face = image[top-30:bottom+30, left-30:right+30]
        faces.append(cropped_face)
    for i in range(len(faces)):
        output_image_path = f'{output_path}/face_{file_name}_{i}.jpg'
        print(f'Writing at {output_image_path}')
        cv2.imwrite(output_image_path, faces[i])


for dirname, _, filenames in os.walk('/kaggle/input/deepfake-detection-challenge/test_videos/'):
    for filename in filenames:
        f = os.path.join(dirname, filename)
        save_frames(video_path=f, output_base_path='/kaggle/working/frames')

for dirname, _, filenames in os.walk('/kaggle/working/frames/'):
    for filename in filenames:
        f = os.path.join(dirname, filename)
        extract_face(image_path=Path(f), output_path=Path('/kaggle/working/faces'))

learn = load_learner(path='/kaggle/input/fastai-5-epochs-trained-cnn-dfdc/',
                     test=ImageList.from_folder('/kaggle/working/faces'))

preds,_ = learn.get_preds(ds_type=DatasetType.Test)
pred_labels = np.argmax(preds, 1)
test_paths = learn.data.test_ds.items
path_to_predicted_label = dict(zip(test_paths, pred_labels))
# TODO : group predictions by video and aggregate the dominant prediction

print(preds)
pd.DataFrame.from_dict(file_name_to_prediction, orient='index', columns=['file_name', 'prediction'])