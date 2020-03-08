import cv2
import json
import os
from pathlib import Path

INPUT_PATH = '/Users/cyril/Documents/software/workspace/kaggle/deepfake/train_sample_videos'
OUTPUT_BASE_PATH = '/Users/cyril/Documents/software/workspace/kaggle/deepfake/processed_data'


def get_real_fake_video_names(metadata_path: str):
    real_video_names = []
    fake_video_names = []
    with open(metadata_path) as data:
        metadata = json.load(data)
        for video_name, metadata in metadata.items():
            label = metadata['label']
            if label == 'FAKE':
                fake_video_names.append(video_name)
            elif label == 'REAL':
                real_video_names.append(video_name)
            else:
                raise AttributeError(f'Unknown label : {label}')
    return real_video_names, fake_video_names


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
        success, image = vidcap.read()
        count += 1

if __name__ == '__main__':
    real_names, fake_names = get_real_fake_video_names(f'{INPUT_PATH}/metadata.json')

    for name in fake_names:
        save_frames(video_path=f'{INPUT_PATH}/{name}', output_base_path=f'{OUTPUT_BASE_PATH}/frames/fake')
    for name in real_names:
        save_frames(video_path=f'{INPUT_PATH}/{name}', output_base_path=f'{OUTPUT_BASE_PATH}/frames/real')



