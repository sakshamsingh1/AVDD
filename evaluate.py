import os
import copy
import argparse
import numpy as np
import torch
from utils.train_utils_DM import evaluate_synset_av, get_network
from utils.data_utils import get_test_dataset, ParamDiffAug, number_sign_augment

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def main(args):
    channel, im_size, num_classes, testloader = get_test_dataset(args.dataset, args)
    args.cls_num = num_classes

    accs_all_exps = []

    def get_syn_data(exp_num, iteration=5000, modality='av'):
        audio_syn, image_syn = None, None
        audio_path = os.path.join(args.base_dir, f'exp_{exp_num}_audSyn_{iteration}.pth')
        if os.path.exists(audio_path):
            audio_syn = torch.load(audio_path)

        elif modality == 'a' or modality == 'av':
            audio_syn = torch.randn(size=(num_classes*args.ipc, channel[0], im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=False, device=args.device)
        
            for c in range(num_classes):
                path = os.path.join(args.base_dir, f'class:{c}.pt')
                data = torch.load(path)
                audio_syn.data[c*args.ipc:(c+1)*args.ipc] = data[exp_num]['audio_syn'][iteration].detach().data

        image_path = os.path.join(args.base_dir, f'exp_{exp_num}_imgSyn_{iteration}.pth')
        if os.path.exists(image_path):
            image_syn = torch.load(image_path)

        elif modality == 'v' or modality == 'av':
            image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float, requires_grad=True, device=args.device)
            for c in range(num_classes):
                path = os.path.join(args.base_dir, f'class:{c}.pt')
                data = torch.load(path)
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = data[exp_num]['image_syn'][iteration].detach().data
        return audio_syn, image_syn

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        # base_seed = 178645
        # seed = (base_seed + exp) % 100000
        # set_seed(seed)

        aud_syn_eval, image_syn_eval = get_syn_data(exp, modality='av')
        label_syn_eval = copy.deepcopy(label_syn.detach())

        if args.idm_aug:
            if args.input_modality == 'a' or args.input_modality == 'av':
                aud_syn_eval = number_sign_augment(aud_syn_eval)
            if args.input_modality == 'v' or args.input_modality == 'av':
                image_syn_eval = number_sign_augment(image_syn_eval)
            label_syn_eval = label_syn_eval.repeat(4)

        accs = []
        for it_eval in range(args.num_eval):
            nets, net_eval = get_network(args)

            acc = evaluate_synset_av(nets, net_eval, aud_syn_eval, image_syn_eval, label_syn_eval, testloader, args)
            accs.append(acc)
            print(f'it_eval: {it_eval} Val acc: {acc:.2f}%')

        print(f'Mean eval at it: {exp} Val acc: {np.mean(accs):.2f}%')
        accs_all_exps += accs            

    print('\n==================== Final Results ====================\n')
    accs = accs_all_exps
    print('Run %d experiments, random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, len(accs), np.mean(accs), np.std(accs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='VGG_subset', help='dataset')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--epoch_eval_train', type=int, default=30, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='iteration to evaluate the synthetic data')
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')    
    parser.add_argument('--base_dir', type=str, default='', help='place to save buffer')
    parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    
    parser.add_argument('--arch_sound', type=str, default='convNet', help='resnet18, convNet')
    parser.add_argument('--weights_sound', type=str, default='', help='weights for sound')   
    parser.add_argument('--arch_frame', type=str, default='convNet', help='resnet18, convNet, CustomResnet')
    parser.add_argument('--weights_frame', type=str, default='', help='weights for frame')
    parser.add_argument('--arch_classifier', type=str, default='ensemble', help='concat, sum, ensemble')
    parser.add_argument('--weights_classifier', type=str, default='', help='weights for classifier')

    parser.add_argument('--lr_frame', type=float, default=1e-4, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_sound', type=float, default=1e-3, help='sound learning rate')    
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='classifier learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='classifier learning rate')
    parser.add_argument('--batch_syn', type=int, default=32, help='batch size for syn data')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--idm_aug', action='store_true', help='use IDM or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lr_steps_interval = 10
    args.lr_steps = np.arange(args.lr_steps_interval, args.epoch_eval_train, args.lr_steps_interval).tolist()
    
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    main(args)