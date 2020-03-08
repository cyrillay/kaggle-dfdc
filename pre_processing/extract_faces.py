import face_recognition
import cv2
from typing import List
from pathlib import Path
from convert_video_to_frames import INPUT_PATH, OUTPUT_BASE_PATH


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



if __name__ == '__main__':

    real_frames_paths = Path(f'{OUTPUT_BASE_PATH}/frames/real').glob('**/*.jpg')
    fake_frames_paths = Path(f'{OUTPUT_BASE_PATH}/frames/fake').glob('**/*.jpg')

    for path in real_frames_paths:
        extract_face(path, Path(f'{OUTPUT_BASE_PATH}/faces/real'))

    for path in fake_frames_paths:
        extract_face(path, Path(f'{OUTPUT_BASE_PATH}/faces/fake'))

    # extract_faces(fake_frames_paths, f'{OUTPUT_BASE_PATH}/faces/fake')
