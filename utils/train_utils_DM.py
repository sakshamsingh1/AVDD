import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import ModelBuilder
from utils.train_utils import NetWrapper, adjust_learning_rate
from utils.data_utils import CombTensorDataset, DiffAugment

def get_network(args):
    builder = ModelBuilder()

    net_sound = builder.build_sound(
        arch=args.arch_sound,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        weights=args.weights_frame)
    net_classifier = builder.build_classifier(
        arch=args.arch_classifier,
        cls_num=args.cls_num,
        weights=args.weights_classifier,
        input_modality=args.input_modality)

    if args.input_modality == 'av':
        nets = (net_sound, net_frame, net_classifier)
    elif args.input_modality == 'a':
        nets = (net_sound, None, net_classifier)
    elif args.input_modality == 'v':
        nets = (None, net_frame, net_classifier)    
    netWrapper = NetWrapper(args, nets)
    return nets, netWrapper

def create_optimizer(nets, args):
    (net_sound, net_frame, net_classifier) = nets
    param_groups = [{'params': net_classifier.parameters(), 'lr': args.lr_classifier}]
    if net_sound is not None:
        param_groups += [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    if net_frame is not None:
        param_groups += [{'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.Adam(param_groups, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)    

def evaluate(netWrapper, loader, args):
    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    netWrapper.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for i, batch_data in enumerate(loader):

        audio, frame = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            audio = batch_data['audio'].float().to(args.device)
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            frame = batch_data['frame'].float().to(args.device)
        
        gt = batch_data['label'].to(args.device)
        
        # forward pass
        if args.arch_classifier == 'ensemble':
            out_a, out_v = netWrapper.forward(audio, frame)
            err = criterion(out_a, gt) + criterion(out_v, gt)
            preds_a = F.softmax(out_a, dim=1)
            preds_v = F.softmax(out_v, dim=1)
            preds = (preds_a + preds_v) / 2
        else:
            preds = netWrapper(audio, frame)
            err = criterion(preds, gt)

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        total_loss += err.item()

    acc = 100 * correct / total
    average_loss = total_loss / len(loader)
    return average_loss, acc

def train(netWrapper, loader, optimizer, args):
    torch.set_grad_enabled(True) 
    criterion = nn.CrossEntropyLoss()

    # switch to train mode
    netWrapper.train()
    correct = 0
    total = 0
    total_loss = 0.0

    # main loop
    torch.cuda.synchronize()
    for i, batch_data in enumerate(loader):
        torch.cuda.synchronize()

        audio, frame = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            audio = batch_data['audio'].float().to(args.device)
            audio = DiffAugment(audio, args.dsa_strategy, param=args.dsa_param)

        if args.input_modality == 'v' or args.input_modality == 'av':
            frame = batch_data['frame'].float().to(args.device)
            frame = DiffAugment(frame, args.dsa_strategy, param=args.dsa_param)
        gt = batch_data['label'].to(args.device)

        # forward pass
        netWrapper.zero_grad()
        if args.arch_classifier == 'ensemble':
            out_a, out_v = netWrapper.forward(audio, frame)
            err = criterion(out_a, gt) + criterion(out_v, gt)
            preds_a = F.softmax(out_a, dim=1)
            preds_v = F.softmax(out_v, dim=1)
            preds = (preds_a + preds_v) / 2
        else:
            preds = netWrapper.forward(audio, frame)
            err = criterion(preds, gt)

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        # backward
        err.backward()
        optimizer.step()

        torch.cuda.synchronize()
        total_loss += err.item()
    
    average_loss = total_loss / len(loader)
    accuracy = correct*100 / total
    return average_loss, accuracy

def evaluate_synset_av(nets, net_eval, auds_train, images_train, labels_train, testloader, args):
    reset_params(args)

    net_eval = net_eval.to(args.device)
    optimizer = create_optimizer(nets, args)
    
    if args.input_modality == 'av' or args.input_modality == 'v':    
        images_train = images_train.to(args.device)
    if args.input_modality == 'av' or args.input_modality == 'a':
        auds_train = auds_train.to(args.device)
    labels_train = labels_train.to(args.device)

    dst_train = CombTensorDataset(auds_train, images_train, labels_train, args)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_syn, shuffle=True, num_workers=0)
    for e in range(args.epoch_eval_train):
        train_loss, train_acc = train(net_eval, trainloader, optimizer, args)
        # print(f'Epoch {e+1}/{args.epoch_eval_train} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}')

        if e in args.lr_steps:
            adjust_learning_rate(optimizer, args)
    
    val_loss, val_acc = evaluate(net_eval, testloader, args)
    val_acc = round(val_acc, 2)
    return val_acc

def reset_params(args):
    args.weights_sound = ''
    args.weights_frame = ''
    args.weights_classifier = ''

    args.lr_sound = 1e-3
    args.lr_frame = 1e-4
    args.lr_classifier = 1e-3