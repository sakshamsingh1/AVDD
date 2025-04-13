import torch
import os
from tqdm import tqdm

path = 'data/train_data/vgg_subset.pt'
base_path = 'data/classwise_train_data/class_wise_train_vgg_subset'

input_data = torch.load(path)

images = input_data['images_train']
audios = input_data['audio_train']
labels = input_data['labels_train']
class_names = input_data['classes']

labels_all = []
num_classes = len(class_names)
# num_classes = 10
indices_class = [[] for c in range(num_classes)]

for i, lab in enumerate(labels):
    indices_class[lab].append(i)

for c in tqdm(range(num_classes)):
    input_data = {}
    curr_img = images[indices_class[c]]
    curr_aud = audios[indices_class[c]]
    input_data['images'] = curr_img
    input_data['audio'] = curr_aud

    curr_path = os.path.join(base_path, f'train_{c}.pt')
    torch.save(input_data, curr_path)    