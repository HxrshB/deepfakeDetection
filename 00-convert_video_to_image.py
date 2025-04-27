import os
import cv2
import math

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

# Base paths for the fake and real datasets
fake_videos_path = './fake_videos/'
real_videos_path = './real_videos/'

# Process videos in a given folder
def process_videos(folder_path, output_base_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_file = os.path.join(folder_path, filename)
            output_folder = os.path.join(output_base_path, get_filename_only(filename))

            print(f'Creating Directory: {output_folder}')
            os.makedirs(output_folder, exist_ok=True)

            print(f'Converting Video to Images for {filename}...')
            count = 0
            cap = cv2.VideoCapture(video_file)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate

            while cap.isOpened():
                frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Current frame number
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frames at 1-second intervals
                if frame_id % math.floor(frame_rate) == 0:
                    print('Original Dimensions: ', frame.shape)

                    # Determine scale ratio based on frame width
                    if frame.shape[1] < 300:
                        scale_ratio = 2
                    elif frame.shape[1] > 1900:
                        scale_ratio = 0.33
                    elif frame.shape[1] > 1000:
                        scale_ratio = 0.5
                    else:
                        scale_ratio = 1

                    print('Scale Ratio: ', scale_ratio)

                    width = int(frame.shape[1] * scale_ratio)
                    height = int(frame.shape[0] * scale_ratio)
                    dim = (width, height)
                    new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                    print('Resized Dimensions: ', new_frame.shape)

                    new_filename = os.path.join(output_folder, f'{get_filename_only(filename)}-{count:03d}.png')
                    count += 1
                    cv2.imwrite(new_filename, new_frame)

            cap.release()
            print(f"Done processing {filename}!")

# Main processing
output_fake_path = './output/fake/'
output_real_path = './output/real/'

print("Processing fake videos...")
process_videos(fake_videos_path, output_fake_path)

print("Processing real videos...")
process_videos(real_videos_path, output_real_path)

print("All videos processed!")
