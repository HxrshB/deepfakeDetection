import cv2
from mtcnn import MTCNN
import os
import tensorflow as tf
print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

physical_devices = tf.config.list_physical_devices('GPU:1')
print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Base paths for the fake and real datasets
fake_videos_path = './fake_videos/'
real_videos_path = './real_videos/'

# Process directories for face extraction
def process_directories(folder_path):
    for subfolder in os.listdir(folder_path):
        tmp_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(tmp_path):
            continue

        print('Processing Directory: ' + tmp_path)
        frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
        faces_path = os.path.join(tmp_path, 'faces')
        print('Creating Directory: ' + faces_path)
        os.makedirs(faces_path, exist_ok=True)
        print('Cropping Faces from Images...')

        for frame in frame_images:
            print('Processing ', frame)
            detector = MTCNN()
            image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(image)
            print('Face Detected: ', len(results))
            count = 0

            for result in results:
                bounding_box = result['box']
                print(bounding_box)
                confidence = result['confidence']
                print(confidence)
                if len(results) < 2 or confidence > 0.95:
                    margin_x = bounding_box[2] * 0.3  # 30% as the margin
                    margin_y = bounding_box[3] * 0.3  # 30% as the margin
                    x1 = int(bounding_box[0] - margin_x)
                    if x1 < 0:
                        x1 = 0
                    x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
                    if x2 > image.shape[1]:
                        x2 = image.shape[1]
                    y1 = int(bounding_box[1] - margin_y)
                    if y1 < 0:
                        y1 = 0
                    y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
                    if y2 > image.shape[0]:
                        y2 = image.shape[0]
                    print(x1, y1, x2, y2)
                    crop_image = image[y1:y2, x1:x2]
                    new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, os.path.splitext(frame)[0]), count)
                    count += 1
                    cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
                else:
                    print('Skipped a face..')

# Main processing
print("Processing fake videos...")
process_directories('./output/fake/')

print("Processing real videos...")
process_directories('./output/real/')

print("All directories processed!")
