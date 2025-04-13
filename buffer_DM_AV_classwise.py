import os
import torch
import numpy as np
import argparse
import wandb
import copy

from utils.data_utils import get_class_train_dataset, get_time, DiffAugment, ParamDiffAug
from utils.train_utils_DM import get_network

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def main(args):
    channel, im_size, dst_train, _, _, num_classes = get_class_train_dataset(args.dataset, args.class_num, args)
    args.cls_num = num_classes

    save_dict = {}
    for exp in range(args.num_exp):
        save_dict[exp] = {}
        print(f'\n================== Exp {exp} for class {args.class_num} ==================\n')
        
        if args.wandb_disable:
            wandb.init(mode="disabled")
        else:
            wandb.init(sync_tensorboard=False,
                    project=f"AVDD_buffer_{args.dataset}",
                    config=args,
                    group=f'Exp_{exp}',
                    name = f'class_{args.class_num}')

        base_seed = 178645
        seed = (base_seed + exp) % 100000
        set_seed(seed)

        def get_aud_images(n): 
            idx_aud, idx_img = None, None
            all_indices = list(range(len(dst_train)))
            if len(dst_train) < n:
                idx_shuffle = np.random.permutation(all_indices)
            else:
                idx_shuffle = np.random.permutation(all_indices)[:n]

            if args.input_modality == 'a' or args.input_modality == 'av':
                idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)

            if args.input_modality == 'v' or args.input_modality == 'av':
                idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
            return idx_aud, idx_img

        ''' initialize the synthetic data '''
        image_syn, audio_syn = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            audio_syn = torch.randn(size=(args.ipc, channel[0], im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            image_syn = torch.randn(size=(args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float, requires_grad=True, device=args.device)

        print('initialize synthetic data from random real images')
        
        aud_real_init, img_real_init = get_aud_images(args.ipc)
        
        if args.input_modality == 'a' or args.input_modality == 'av':
            aud_real_init = aud_real_init.detach().data
            audio_syn.data = aud_real_init

        if args.input_modality == 'v' or args.input_modality == 'av':
            img_real_init = img_real_init.detach().data
            image_syn.data = img_real_init

        ##SAVE
        if args.input_modality == 'a' or args.input_modality == 'av':
            save_dict[exp]['syn_aud'] = copy.deepcopy(audio_syn.detach().cpu())
        if args.input_modality == 'v' or args.input_modality == 'av':
            save_dict[exp]['syn_img'] = copy.deepcopy(image_syn.detach().cpu())

        def get_syn_optimizer(aud_syn, img_syn):
            param_groups = []
            if args.input_modality == 'a' or args.input_modality == 'av':
                param_groups += [{'params': aud_syn, 'lr': args.lr_syn_aud}]
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                param_groups += [{'params': img_syn, 'lr': args.lr_syn_img}]
            return torch.optim.SGD(param_groups, momentum=0.5)

        # ''' training '''
        optimizer_comb = get_syn_optimizer(audio_syn, image_syn)  

        print('%s training begins'%get_time())
        for it in range(args.Iteration+1):
            save_dict[exp][it] = {}

            base_seed = 178645
            seed = (base_seed + it + exp) % 100000
            set_seed(seed)

            nets, _ = get_network(args)
            (net_audio, net_frame, _) = nets
            if args.input_modality == 'a' or args.input_modality == 'av':
                net_audio.to(args.device)
                net_audio.train()
                for param in list(net_audio.parameters()):
                    param.requires_grad = False
                audio_embd = net_audio.module.embed if torch.cuda.device_count() > 1 else net_audio.embed                 

            if args.input_modality == 'v' or args.input_modality == 'av': 
                net_frame.to(args.device)   
                net_frame.train()        
                for param in list(net_frame.parameters()):
                    param.requires_grad = False
                image_embd = net_frame.module.embed if torch.cuda.device_count() > 1 else net_frame.embed 

            ''' Train synthetic data '''
            save_dict[exp][it] = {}
            loss_c = torch.tensor(0.0).to(args.device)
            aud_real, img_real = get_aud_images(args.batch_real)

            if args.input_modality == 'a' or args.input_modality == 'av':
                aud_real = aud_real.to(args.device)
            if args.input_modality == 'v' or args.input_modality == 'av':
                img_real = img_real.to(args.device)
            
            if args.input_modality == 'a' or args.input_modality == 'av':
                curr_aud_syn = audio_syn.reshape((args.ipc, channel[0], im_size[0][0], im_size[0][1]))
            if args.input_modality == 'v' or args.input_modality == 'av':
                curr_img_syn = image_syn.reshape((args.ipc, channel[1], im_size[1][0], im_size[1][1]))

            if args.dsa:
                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_real = DiffAugment(aud_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    curr_aud_syn = DiffAugment(curr_aud_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    curr_img_syn = DiffAugment(curr_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

            if args.input_modality == 'a' or args.input_modality == 'av':
                embd_aud_real = audio_embd(aud_real).detach()
                save_dict[exp][it]['embd_aud_real'] = torch.mean(embd_aud_real, dim=0)
                embd_aud_syn = audio_embd(curr_aud_syn)
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                embd_img_real = image_embd(img_real).detach()
                save_dict[exp][it]['embd_img_real'] = torch.mean(embd_img_real, dim=0)
                embd_img_syn = image_embd(curr_img_syn)

            ## Embedding matching
            if args.input_modality == 'av':
                loss_c += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                loss_c += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)

            elif args.input_modality == 'a':
                loss_c += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
            
            elif args.input_modality == 'v':
                loss_c += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)
            
            optimizer_comb.zero_grad()
            loss_c.backward()
            optimizer_comb.step()
            
            loss_avg = loss_c.item()

            if it%10 == 0:
                print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}, lr = {optimizer_comb.param_groups[0]["lr"]:.6f}')
                wandb.log({'train_loss': loss_avg}, step=it)
            
        wandb.finish()

    torch.save(save_dict, args.buff_ckpt + f'/class:{args.class_num}.pth')
    print('experiment run save')            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='VGG_subset', help='dataset')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--Iteration', type=int, default=5000, help='training iterations')
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')    
    parser.add_argument('--wandb_disable', action='store_true', help='wandb disable')
    parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')

    parser.add_argument('--class_num', type=int, default=0, help='the class number')

    parser.add_argument('--lr_syn_aud', type=float, default=0.2, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_syn_img', type=float, default=0.2, help='learning rate for updating synthetic audio specs')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--batch_real', type=int, default=128, help='batch size for real data')

    parser.add_argument('--arch_sound', type=str, default='convNet', help='convNet')
    parser.add_argument('--weights_sound', type=str, default='', help='weights for sound')   
    parser.add_argument('--arch_frame', type=str, default='convNet', help='convNet')
    parser.add_argument('--weights_frame', type=str, default='', help='weights for frame')
    parser.add_argument('--arch_classifier', type=str, default='ensemble', help='ensemble')
    parser.add_argument('--weights_classifier', type=str, default='', help='weights for classifier')

    args = parser.parse_args()        
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.dsa_param = ParamDiffAug()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.id = f'classWiseBuffer_{args.dataset}_mod-{args.input_modality}_ipc-{args.ipc}_iter:{args.Iteration}'
    args.buff_ckpt = os.path.join('data/buffers', args.id)
    if not os.path.exists(args.buff_ckpt):
        os.mkdir(args.buff_ckpt)
    
    main(args)