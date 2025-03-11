import os
import pandas as pd
import random
import torchvision
import torch
from tqdm import tqdm

import torchaudio
import torchaudio.transforms as T

base_path = '/mnt/data0/datasets/AVE_Dataset'
label_train_path = os.path.join(base_path, 'trainSet.txt')
label_test_path = os.path.join(base_path, 'testSet.txt')
label_idx_to_map_path = 'meta_data/cat_id_map.pt'

frame_path = '/mnt/data1/saksham/AVDD/frames'
audio_path = '/mnt/data1/saksham/AVDD/audio'

IMG_SIZE = [224, 224]
AUDIO_SIZE = [128, 56]

AFPS = 11000

def resize_audio(waveform, target_length=11000):
    current_length = waveform.shape[-1]
    if current_length < target_length:
        pad_length = target_length - current_length
        waveform = torch.cat((waveform, torch.zeros(pad_length)), dim=-1)
    elif current_length > target_length:
        waveform = waveform[:target_length]
    return waveform

def get_mel_spec(file):
    waveform, sample_rate = torchaudio.load(file)
    waveform = torch.mean(waveform, dim=0)
    waveform = resize_audio(waveform, target_length=AFPS)
    mel_spec = T.MelSpectrogram(sample_rate)(waveform)
    mel_spec = T.AmplitudeToDB()(mel_spec)
    mel_spec = mel_spec.unsqueeze_(0)
    return mel_spec

def get_label_idx(label):
    label = correct_label[label]
    return label_to_id[label]

correct_label = {
'Church bell':'Church_bell',
'Male speech, man speaking': 'Male_speech',
'Bark':'Bark',
'Fixed-wing aircraft, airplane':'airplane',
'Race car, auto racing':'Race_car',
'Female speech, woman speaking':'Female_speech',
'Helicopter':'Helicopter',
'Violin, fiddle':'Violin',
'Flute':'Flute',
'Ukulele': 'Ukulele', 
'Frying (food)': 'Frying',
'Truck': 'Truck', 
'Shofar': 'Shofar',
'Motorcycle': 'Motorcycle',
'Acoustic guitar': 'Acoustic_guitar', 
'Train horn': 'Train_horn', 
'Clock': 'Clock', 
'Banjo': 'Banjo',
'Goat': 'Goat', 
'Baby cry, infant cry': 'Baby_cry', 
'Bus': 'Bus', 
'Chainsaw': 'Chainsaw', 
'Cat': 'Cat', 
'Horse': 'Horse',
'Toilet flush': 'Toilet_flush', 
'Rodents, rats, mice': 'Rodents', 
'Accordion': 'Accordion', 
'Mandolin': 'Mandolin'
}

label_to_id = torch.load(label_idx_to_map_path) 
df_train = pd.read_csv(label_train_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])
#remove some videos with processing issues
df_train = df_train[df_train['video_id']!='VWi2ENBuTbw']
df_train = df_train[df_train['video_id']!='0-I1-DOC-r8']
df_train = df_train[df_train['video_id']!='TTYevyM_tUw']
df_train = df_train[df_train['video_id']!='K6F1sogt46U']
df_train = df_train[df_train['video_id']!='MWFQerde_h8']
df_train = df_train[df_train['video_id']!='GmetnCLxFHE']

# df_val = pd.read_csv(label_val_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])
df_test = pd.read_csv(label_test_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])

train_count, val_count, test_count = 0, 0, 0
for index, row in df_train.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']
    frame_count = end_time - start_time
    train_count += frame_count

# if VAL_SPLIT:
#     for index, row in df_val.iterrows():
#         start_time = row['start_time']
#         end_time = row['end_time']
#         frame_count = end_time - start_time
#         val_count += frame_count

for index, row in df_test.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']
    frame_count = end_time - start_time
    test_count += frame_count    
    
print(f'train:{train_count}, val:{val_count}, test:{test_count}')

audio_train = torch.Tensor(train_count, 1, AUDIO_SIZE[0], AUDIO_SIZE[1])
images_train = torch.Tensor(train_count, 3, IMG_SIZE[0], IMG_SIZE[1])
labels_train = torch.zeros(train_count, dtype=torch.int32)

print('Creating train data...')
curr_count = 0
for index, row in tqdm(df_train.iterrows(),total=len(df_train)):
    video_id = row['video_id']
    category = row['category']
    start_time = row['start_time']
    end_time = row['end_time']
    
    for curr_sec in range(start_time, end_time):
        img_file = os.path.join(frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_04.jpg')
        images_train[curr_count] = torchvision.io.read_image(img_file)

        aud_file = os.path.join(audio_path, video_id, f'{video_id}_{curr_sec}.wav')
        audio_train[curr_count] = get_mel_spec(aud_file)

        labels_train[curr_count] = get_label_idx(category)
        curr_count +=1

# if VAL_SPLIT:
#     audio_val = torch.Tensor(val_count, 1, AUDIO_SIZE[0], AUDIO_SIZE[1])
#     images_val = torch.Tensor(val_count, 3, IMG_SIZE[0], IMG_SIZE[1])
#     labels_val = torch.zeros(val_count, dtype=torch.int32)

#     print('Creating val data...')
#     curr_count = 0
#     for index, row in tqdm(df_val.iterrows(),total=len(df_val)):
#         video_id = row['vid']
#         category = row['category']
#         start_time = row['start_time']
#         end_time = row['end_time']

#         for curr_sec in range(start_time, end_time):
#             img_file = os.path.join(frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_04.jpg')
#             images_val[curr_count] = torchvision.io.read_image(img_file)

#             aud_file = os.path.join(audio_path, video_id, f'{video_id}_{curr_sec}.wav')
#             audio_val[curr_count] = get_mel_spec(aud_file)

#             labels_val[curr_count] = get_label_idx(category)
#             curr_count +=1

audio_test = torch.Tensor(test_count, 1, AUDIO_SIZE[0], AUDIO_SIZE[1])
images_test = torch.Tensor(test_count, 3, IMG_SIZE[0], IMG_SIZE[1])
labels_test = torch.zeros(test_count, dtype=torch.int32)

print('Creating test data...')
curr_count = 0
for index, row in tqdm(df_test.iterrows(),total=len(df_test)):
    video_id = row['video_id']
    category = row['category']
    start_time = row['start_time']
    end_time = row['end_time']
    
    for curr_sec in range(start_time, end_time):
        img_file = os.path.join(frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_04.jpg')
        images_test[curr_count] = torchvision.io.read_image(img_file)

        aud_file = os.path.join(audio_path, video_id, f'{video_id}_{curr_sec}.wav')
        audio_test[curr_count] = get_mel_spec(aud_file)

        labels_test[curr_count] = get_label_idx(category)
        curr_count +=1

input_data = {}
input_data['classes'] = list(label_to_id.keys())
input_data['images_train'] = images_train
input_data['audio_train'] = audio_train
input_data['labels_train'] = labels_train
# if VAL_SPLIT:
#     input_data['images_val'] = images_val
#     input_data['audio_val'] = audio_val
#     input_data['labels_val'] = labels_val
input_data['images_test'] = images_test
input_data['audio_test'] = audio_test
input_data['labels_test'] = labels_test

torch.save(input_data, 'data/ave_train.pt')