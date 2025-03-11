import os
import pandas as pd
import random
import torchvision
import torch
from tqdm import tqdm
import numpy as np
import pickle

import torchaudio
import torchaudio.transforms as T
from torchaudio.transforms import Resample

# labels = '''chicken crowing	
# toilet flushing	
# playing acoustic guitar
# playing piano	
# ocean burbling	
# fireworks banging
# child speech, kid speaking
# basketball bounce
# police radio chatter
# driving buses'''

# labels = labels.split('\n')
# labels = [label.strip() for label in labels]
# #make a dict of number to label
# label_dict = {}
# for i, label in enumerate(labels):
#     label_dict[i] = label

label_path = '/mnt/data0/datasets/saksham/vgg/vgg_subset.csv'

base_path = '/mnt/data0/datasets/vggs_data'
frame_path = os.path.join(base_path, 'frames')
audio_path = os.path.join(base_path, 'audio')

label_df = pd.read_csv(label_path)

IMG_SIZE = [224, 224] #[32, 32]#
AUDIO_SIZE = [128, 56]

VAL_SPLIT = False
AFPS = 11000

def resize_audio(waveform, sec=4, AFPS=11000, orig_APFS=11430):
    resampler = Resample(orig_freq=orig_APFS, new_freq=AFPS)
    resampled_waveform = resampler(waveform)
    curr_wav = resampled_waveform[sec*AFPS:(sec+1)*AFPS]
    return curr_wav

def get_mel_spec(file, sec=4):
    waveform, sample_rate = torchaudio.load(file)
    waveform = torch.mean(waveform, dim=0)
    waveform = resize_audio(waveform, orig_APFS=sample_rate)
    mel_spec = T.MelSpectrogram(sample_rate)(waveform)
    mel_spec = T.AmplitudeToDB()(mel_spec)
    mel_spec = mel_spec.unsqueeze_(0)
    return mel_spec

data_train = label_df[label_df['split'] == 'train']
data_test = label_df[label_df['split'] == 'test']
data_test = data_test[data_test['vid']!='JsxLvhJ4P6w']

train_count, test_count = len(data_train), len(data_test)
print(f'train:{train_count}, test:{test_count}')

audio_train = torch.Tensor(train_count, 1, AUDIO_SIZE[0], AUDIO_SIZE[1])
images_train = torch.Tensor(train_count, 3, IMG_SIZE[0], IMG_SIZE[1])
labels_train = torch.zeros(train_count, dtype=torch.int32)

print('Creating train data...')
curr_count = 0
for index, row in tqdm(data_train.iterrows(), total=len(data_train)):
    vid = row['vid']
    label = row['label_id']
    
    aud_file = os.path.join(audio_path, f'{vid}.wav')
    img_file = os.path.join(frame_path, f'{vid}.jpg')
    
    curr_img = torchvision.io.read_image(img_file)
    images_train[curr_count] = torchvision.transforms.functional.resize(curr_img, IMG_SIZE)
    
    audio_train[curr_count] = get_mel_spec(aud_file)
    labels_train[curr_count] = label
    curr_count += 1

audio_test = torch.Tensor(test_count, 1, AUDIO_SIZE[0], AUDIO_SIZE[1])
images_test = torch.Tensor(test_count, 3, IMG_SIZE[0], IMG_SIZE[1])
labels_test = torch.zeros(test_count, dtype=torch.int32)

print('Creating test data...')
curr_count = 0
for index, row in tqdm(data_test.iterrows(), total=len(data_test)):
    vid = row['vid']
    label = row['label_id']
    
    aud_file = os.path.join(audio_path, f'{vid}.wav')
    img_file = os.path.join(frame_path, f'{vid}.jpg')
    
    curr_img = torchvision.io.read_image(img_file)
    images_test[curr_count] = torchvision.transforms.functional.resize(curr_img, IMG_SIZE)
    
    audio_test[curr_count] = get_mel_spec(aud_file)
    labels_test[curr_count] = label
    curr_count += 1

input_data = {}
input_data['classes'] = list(data_test.keys())
input_data['images_train'] = images_train
input_data['audio_train'] = audio_train
input_data['labels_train'] = labels_train
input_data['images_test'] = images_test
input_data['audio_test'] = audio_test
input_data['labels_test'] = labels_test

torch.save(input_data, 'vgg_subset.pt')