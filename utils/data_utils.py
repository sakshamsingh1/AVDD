import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import time

def get_test_dataset(dataset, args):
    
    if dataset == 'VGG_subset':
        data = torch.load('data/test_data/vgg_subset_test.pt', map_location='cpu')
        
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        #Test
        aud_test = data['audio_test']
        images_test = data['images_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)

    elif dataset == 'Music_21':
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 21

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]

        data = torch.load('data/test_data/music_21_test.pt', map_location='cpu')
        #Test
        aud_test = data['audio_test']
        images_test = data['images_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)  

    elif dataset == 'AVE':
        data = torch.load('data/test_data/ave_test.pt', map_location='cpu')
        #common
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        #Test
        aud_test = data['audio_test']
        images_test = data['images_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)

    elif dataset == 'VGG':
        data = torch.load('data/test_data/vgg_test.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 309

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # Test
        images_test = data['images_test']
        aud_test = data['audio_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)

    else:
        exit('unknown dataset: %s'%dataset)
    
    bs = 32
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=bs, shuffle=False, num_workers=args.num_workers)

    return channel, im_size, num_classes, testloader


def get_train_dataset(dataset, args):
    
    if dataset == 'VGG_subset':
        data = torch.load('data/train_data/vgg_subset_train.pt', map_location='cpu')
        
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        #Test
        aud_train = data['audio_train']
        images_train = data['images_train']
        labels_train = data['labels_train']
        
        aud_train = aud_train.detach().float()
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach().long()
        for c in range(channel[1]):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]
        dst_train = CombTensorDataset(aud_train, images_train, labels_train, args)

    elif dataset == 'AVE':
        data = torch.load('data/train_data/ave_train.pt', map_location='cpu')
        
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        #Test
        aud_train = data['audio_train']
        images_train = data['images_train']
        labels_train = data['labels_train']
        
        aud_train = aud_train.detach().float()
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach().long()
        for c in range(channel[1]):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]
        dst_train = CombTensorDataset(aud_train, images_train, labels_train, args)

    else:
        exit('unknown dataset: %s'%dataset)
    
    return channel, im_size, num_classes, mean, std, dst_train

def get_herd_path(dataset):
    if dataset == 'VGG_subset':
        return 'data/herding_data/VGG_subset_herd_idx_dict.pkl'
    elif dataset == 'VGG':
        return 'data/herding_data/VGG_herd_idx_dict.pkl'
    elif dataset == 'AVE':
        return 'data/herding_data/AVE_herd_idx_dict.pkl'
    elif dataset == 'Music_21_center' or dataset == 'Music_21':
        return 'data/herding_data/Music_21_center_herd_idx_dict.pkl'
    else:
        exit('unknown dataset: %s'%dataset)

class CombTensorDataset(Dataset):
    def __init__(self, audio, images, labels, args): # images: n x c x h x w tensor
        if args.input_modality == 'a' or args.input_modality == 'av':
            self.audio = audio.detach().float()
        if args.input_modality == 'v' or args.input_modality == 'av':
            self.images = images.detach().float()
        self.labels = labels.detach()
        self.args = args

    def __getitem__(self, index):
        audio, frame = torch.zeros(1), torch.zeros(1)
        label = self.labels[index]
        if self.args.input_modality == 'a' or self.args.input_modality == 'av':
            audio = self.audio[index]
        if self.args.input_modality == 'v' or self.args.input_modality == 'av':
            frame = self.images[index] 
        ret_dict = {'frame': frame, 'audio': audio, 'label':label}
        return ret_dict

    def __len__(self):
        return self.labels.shape[0]

def number_sign_augment(image_syn):
    half_h, half_w = image_syn.shape[2]//2, image_syn.shape[3]//2
    a, b, c, d = image_syn[:, :, :half_h, :half_w].clone(), image_syn[:, :, half_h:, :half_w].clone(), image_syn[:, :, :half_h, half_w:].clone(), image_syn[:, :, half_h:, half_w:].clone()
    a, b, c, d = F.upsample(a, scale_factor=2, mode='bilinear'), F.upsample(b, scale_factor=2, mode='bilinear'), \
        F.upsample(c, scale_factor=2, mode='bilinear'), F.upsample(d, scale_factor=2, mode='bilinear')
    image_syn_augmented = torch.concat([a, b, c, d], dim=0)
    return image_syn_augmented

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))