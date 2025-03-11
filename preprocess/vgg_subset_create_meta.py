import pandas as pd

labels = '''chicken crowing	
toilet flushing	
playing acoustic guitar
playing piano	
ocean burbling	
fireworks banging
child speech, kid speaking
basketball bounce
police radio chatter
driving buses'''

labels = labels.split('\n')
labels = [label.strip() for label in labels]
#make a dict of number to label
label_dict = {}
for i, label in enumerate(labels):
    label_dict[i] = label

# #save the dict as pickle
# import pickle
# save_path = '/mnt/data/datasets/saksham/vgg/label_dict.pkl'
# with open('label_dict.pkl', 'wb') as f:
#     pickle.dump(label_dict, f)

#read and add headers vid, num, label, split
df = pd.read_csv('/mnt/data/datasets/saksham/vgg/vggsound.csv', header=None, names=['vid', 'num', 'label', 'split'])

def is_label_in_dict(label):
    return label in label_dict.values()

df['label_in_dict'] = df['label'].apply(is_label_in_dict)
df = df[df['label_in_dict'] == True]
del df['label_in_dict']

import os
path = '/mnt/data/datasets/vggs_data/audio'
files = os.listdir(path)
files = [file.split('.')[0] for file in files]

def is_vid_in_files(vid):
    return vid in files

df['vid_in_files'] = df['vid'].apply(is_vid_in_files)
df = df[df['vid_in_files'] == True]
del df['vid_in_files']
df['label_id'] = df['label'].apply(lambda x: list(label_dict.keys())[list(label_dict.values()).index(x)])

save_path = 'vgg_subset.csv'
df.to_csv(save_path, index=False)