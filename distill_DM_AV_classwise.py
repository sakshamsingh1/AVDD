import os
import torch
import numpy as np
import argparse
import copy
import wandb
import torch.nn.functional as F
import pickle
import torchvision

from utils.data_utils import get_class_train_dataset, get_time, DiffAugment, \
    ParamDiffAug, get_herd_path_classwise, number_sign_augment
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
    viz_it_pool = np.arange(0, args.Iteration+1, args.interval).tolist()
    channel, im_size, dst_train, mean, std, num_classes = get_class_train_dataset(args.dataset, args.class_num, args)
    args.cls_num = num_classes
    
    buff_path = args.buff_ckpt_curr + f'/class:{args.class_num}.pth'
    buff_dict = torch.load(buff_path)
    print('Hyper-parameters: \n', args.__dict__)

    save_data = {}
    for exp in range(args.num_exp):
        save_data[exp] = {}
        
        if args.wandb_disable:
            wandb.init(mode="disabled")
        else:
            wandb.init(sync_tensorboard=False,
                project=f"AVDD_distill_{args.dataset}",
                config=args,
                group=f'ipc:{args.ipc}',
                name = f'Exp_{exp}_class:{args.class_num}')

        base_seed = 178645
        seed = (base_seed + exp) % 100000
        set_seed(seed)

        def downscale(image_syn, scale_factor):
            image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
            return image_syn
        
        if args.init_herding:
            args.herd_path = get_herd_path_classwise(args.dataset)
            print('initialize synthetic data from herding')

        def get_aud_images_init(n): 
            if args.init_herding and args.dataset=='Music_21':
                _, _, dst_train_center, _, _, _ = get_class_train_dataset('Music_21_center', args.class_num, args)
                with open(args.herd_path, 'rb') as f:
                    herd_idx_dict = pickle.load(f)
                if len(herd_idx_dict[args.class_num]['av']) < n:
                    idx_shuffle = herd_idx_dict[args.class_num]['av']
                    remain_n = n - len(herd_idx_dict[args.class_num]['av'])
                    idx_shuffle += herd_idx_dict[args.class_num]['av'][:remain_n]
                else:
                    idx_shuffle = herd_idx_dict[args.class_num]['av'][:n] 
                idx_aud = dst_train_center[idx_shuffle]['audio'].to(args.device)
                idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
            else:
                idx_aud, idx_img = None, None
                all_indices = list(range(len(dst_train)))
                if args.init_herding:
                    with open(args.herd_path, 'rb') as f:
                        herd_idx_dict = pickle.load(f)
                    if len(herd_idx_dict[args.class_num]['av']) < n:
                        idx_shuffle = herd_idx_dict[args.class_num]['av']
                        remain_n = n - len(herd_idx_dict[args.class_num]['av'])
                        idx_shuffle += herd_idx_dict[args.class_num]['av'][:remain_n]
                    else:
                        idx_shuffle = herd_idx_dict[args.class_num]['av'][:n] 
                else:
                    if len(dst_train) < n:
                        idx_shuffle = np.random.permutation(all_indices)
                    else:
                        idx_shuffle = np.random.permutation(all_indices)[:n]
                if args.input_modality == 'a' or args.input_modality == 'av':
                    idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
                if args.input_modality == 'v' or args.input_modality == 'av':
                    idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
            return idx_aud, idx_img

        print(f'\n================== Exp {exp} for class {args.class_num} ==================\n')
        ''' initialize the synthetic data '''

        image_syn, audio_syn = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            audio_syn = torch.randn(size=(args.ipc, channel[0], im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=True, device=args.device)
            
        if args.input_modality == 'v' or args.input_modality == 'av':
            image_syn = torch.randn(size=(args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float, requires_grad=True, device=args.device)

        print('initialize synthetic data from random real images')
        if not args.idm_aug:
            aud_real_init, img_real_init = get_aud_images_init(args.ipc)

            if args.input_modality == 'a' or args.input_modality == 'av':
                audio_syn.data = aud_real_init.detach().data
                audio_syn.data = aud_real_init
                save_data[exp]['audio_syn'] = {}

            if args.input_modality == 'v' or args.input_modality == 'av':
                img_real_init = img_real_init.detach().data
                image_syn.data = img_real_init
                save_data[exp]['image_syn'] = {}
        else:
            assert args.input_modality == 'av'
            save_data[exp]['audio_syn'] = {}
            save_data[exp]['image_syn'] = {}
            
            a_half_h, a_half_w = im_size[0][0]//2, im_size[0][1]//2
            v_half_size = im_size[1][0]//2
            auds_real, imgs_real = get_aud_images_init(args.ipc*args.idm_aug_count*args.idm_aug_count)
            
            start,end = 0, args.ipc
            audio_syn.data[:, :, :a_half_h, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
            audio_syn.data[:, :, a_half_h:, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
            audio_syn.data[:, :, :a_half_h, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
            audio_syn.data[:, :, a_half_h:, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc

            v_half_size = im_size[1][0]//2; start,end = 0, args.ipc
            image_syn.data[:, :, :v_half_size, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
            image_syn.data[:, :, v_half_size:, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
            image_syn.data[:, :, :v_half_size, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
            image_syn.data[:, :, v_half_size:, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc

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

            if it in viz_it_pool:
                ''' visualize and save '''
                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_syn_vis = copy.deepcopy(audio_syn.detach().cpu())
                    save_data[exp]['audio_syn'][it] = aud_syn_vis
                    grid = torchvision.utils.make_grid(aud_syn_vis, nrow=max(5, args.ipc), normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Audio": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    save_data[exp]['image_syn'][it] = image_syn_vis
                    for ch in range(channel[1]):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                    image_syn_vis[image_syn_vis<0] = 0.0
                    image_syn_vis[image_syn_vis>1] = 1.0
                    grid = torchvision.utils.make_grid(image_syn_vis, nrow=max(5, args.ipc), normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Image": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

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
            loss_c = torch.tensor(0.0).to(args.device)
            
            if args.input_modality == 'a' or args.input_modality == 'av':
                curr_aud_syn = audio_syn.reshape((args.ipc, channel[0], im_size[0][0], im_size[0][1]))
            if args.input_modality == 'v' or args.input_modality == 'av':
                curr_img_syn = image_syn.reshape((args.ipc, channel[1], im_size[1][0], im_size[1][1]))
            
            if args.idm_aug:
                if args.input_modality == 'a' or args.input_modality == 'av':
                    curr_aud_syn = number_sign_augment(curr_aud_syn)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    curr_img_syn = number_sign_augment(curr_img_syn)

            if args.dsa:
                if args.input_modality == 'a' or args.input_modality == 'av':
                    curr_aud_syn = DiffAugment(curr_aud_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    curr_img_syn = DiffAugment(curr_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

            if args.input_modality == 'a' or args.input_modality == 'av':
                embd_aud_real_mean = buff_dict[exp][it]['embd_aud_real']
                embd_aud_syn = audio_embd(curr_aud_syn)
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                embd_img_real_mean = buff_dict[exp][it]['embd_img_real']
                embd_img_syn = image_embd(curr_img_syn)

            ## Embedding matching
            if args.input_modality == 'av':
                loss_c += torch.sum((embd_aud_real_mean - torch.mean(embd_aud_syn, dim=0))**2)
                loss_c += torch.sum((embd_img_real_mean - torch.mean(embd_img_syn, dim=0))**2)
                
                # Implicit Cross Matching
                real_mean_aud_vis = (embd_aud_real_mean + embd_img_real_mean)
                syn_mean_aud_vis = (torch.mean(embd_aud_syn, dim=0) + torch.mean(embd_img_syn, dim=0))
                loss_c += args.lam_icm*torch.sum((real_mean_aud_vis - syn_mean_aud_vis)**2)

                # Cross-Modal Gap Matching
                cross_mean_Raud_Svis = (embd_aud_real_mean + torch.mean(embd_img_syn, dim=0))
                cross_mean_Rvis_Saud = (embd_img_real_mean + torch.mean(embd_aud_syn, dim=0))
                loss_c += args.lam_cgm*torch.sum((cross_mean_Raud_Svis - cross_mean_Rvis_Saud)**2)

            elif args.input_modality == 'a':
                loss_c += torch.sum((embd_aud_real_mean - torch.mean(embd_aud_syn, dim=0))**2)
            
            elif args.input_modality == 'v':
                loss_c += torch.sum((embd_img_real_mean - torch.mean(embd_img_syn, dim=0))**2)
            
            optimizer_comb.zero_grad()
            loss_c.backward()
            optimizer_comb.step()
            
            loss_avg = loss_c.item()

            if it%10 == 0:
                print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}, lr = {optimizer_comb.param_groups[0]["lr"]:.6f}')
                wandb.log({'train_loss': loss_avg}, step=it)
            
        wandb.finish()
    
    save_dir = f'Distill_{args.dataset}_ipc:{args.ipc}'
    save_dir = os.path.join(args.distill_path, save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, f'class:{args.class_num}.pt')
    torch.save(save_data, save_path)
    print('experiment run save')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='VGG', help='dataset')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--Iteration', type=int, default=5000, help='training iterations')
    parser.add_argument('--interval', type=int, default=1000, help='interval to evaluate the synthetic data')
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')    
    parser.add_argument('--wandb_disable', action='store_true', help='wandb disable')
    parser.add_argument('--init_herding', action='store_true', help='init using herding or not')
    parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')
    parser.add_argument('--class_num', type=int, default=0, help='the class number')
    parser.add_argument('--idm_aug', action='store_true', help='use IDM or not')

    parser.add_argument('--lam_cgm', type=float, default=10.0, help='weight for cross-modal gap matching loss')
    parser.add_argument('--lam_icm', type=float, default=10.0, help='weight for implicit cross matching loss')
    parser.add_argument('--lr_syn_aud', type=float, default=0.2, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_syn_img', type=float, default=0.2, help='learning rate for updating synthetic image')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

    parser.add_argument('--arch_sound', type=str, default='convNet', help='convNet')
    parser.add_argument('--weights_sound', type=str, default='', help='weights for sound')   
    parser.add_argument('--arch_frame', type=str, default='convNet', help='convNet')
    parser.add_argument('--weights_frame', type=str, default='', help='weights for frame')
    parser.add_argument('--arch_classifier', type=str, default='ensemble', help='ensemble')
    parser.add_argument('--weights_classifier', type=str, default='', help='weights for classifier')

    args = parser.parse_args()
    args.id = f'classWiseBuffer_{args.dataset}_mod-{args.input_modality}_ipc-1_iter:{args.Iteration}'    
    args.buff_ckpt_curr = os.path.join('data/buffers', args.id)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.idm_aug_count=2
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.dsa_param = ParamDiffAug()
    args.distill_path = 'data/distilled_data'

    if not os.path.exists(args.buff_ckpt_curr):
        raise ValueError(f'Buffer path {args.buff_ckpt_curr} does not exist!')

    main(args)