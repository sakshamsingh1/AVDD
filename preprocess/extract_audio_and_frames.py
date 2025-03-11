# Description: Extract audio and frames from AVE dataset
# Will need to make some changes to adjust for other datasets

import ffmpeg
import os
import pandas as pd
from tqdm import tqdm

# Extract audio from a video using ffmpeg
def extract_frames(video_path, output_folder, start_time, end_time, fps=8):
    if os.path.exists(output_folder):
        print(f'Folder {output_folder} already exists. Skipping...')
        return

    os.makedirs(output_folder, exist_ok=True)
    video_id = os.path.basename(video_path).split('.')[0]

    for second in range(start_time, end_time):
        folder_path = os.path.join(output_folder, str(second))
        os.makedirs(folder_path, exist_ok=True)  # Create folder for each second
        
        output_path = os.path.join(folder_path, f'{video_id}_{second}_%02d.jpg')  # Output path with a pattern to save frames as images
        (
            ffmpeg
            .input(video_path, ss=second)
            # .output(output_path, vf='fps=' + str(fps), t=1)
            .output(output_path, vf='fps=' + str(fps) + ',scale=224:224', t=1)
            .global_args('-loglevel', 'error')
            .run()
        )

def extract_audio(video_path, output_folder, start_time, end_time, afps=11000):
    if os.path.exists(output_folder):
        print(f'Folder {output_folder} already exists. Skipping...')
        return

    os.makedirs(output_folder, exist_ok=True)
    video_id = os.path.basename(video_path).split('.')[0]

    for second in range(start_time, end_time):
        
        output_path = os.path.join(output_folder, f'{video_id}_{second}.wav') 
        (
            ffmpeg
            .input(video_path, ss=second, t=1)
            .output(output_path, acodec='pcm_s16le', ac=2, ar=str(afps))
            .global_args('-loglevel', 'error')
            .run()
        )

#constants
FPS = 8  
AFPS = 11000
base_path = '/mnt/data0/datasets/AVE_Dataset'

label_path = os.path.join(base_path, 'Annotations.txt')
ave_path = os.path.join(base_path, 'AVE')
labels_df = pd.read_csv(label_path, delimiter='&')

output_base = '/mnt/data1/saksham/AVDD/frames'
output_base_audio = '/mnt/data1/saksham/AVDD/audio'

for index, row in tqdm(labels_df.iterrows()):
    video_path = os.path.join(ave_path, row['VideoID'] + '.mp4')
    output_folder_img = os.path.join(output_base, row['VideoID'])
    start_time = row['StartTime']
    end_time = row['EndTime']
    extract_frames(video_path, output_folder_img, start_time, end_time, fps=FPS)
    
    output_folder_aud = os.path.join(output_base_audio, row['VideoID'])
    extract_audio(video_path, output_folder_aud, start_time, end_time, afps=AFPS)