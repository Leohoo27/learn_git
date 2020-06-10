#!/usr/bin/env python3
import torch
import argparse
import torchvision
import os, shutil

from advertorch import attacks
import foolbox as fb
import torch.nn as nn

from utils.data_loader import load_data


parser = argparse.ArgumentParser(description='generate adversarial samples leveraging off-the-shelf tools')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--input_data', type=str, default='/home/leohoo/work/code/tlfuzz/datasets/office_caltech/caltech/test',
                    help='input data dir to generate adversarial samples')
                    
parser.add_argument('--input_model', type=str, default='./models/models_office_caltech/resnet50_amazon_caltech.model.pt',
                    help='source model to generate adversarial samples')
                    
parser.add_argument('--target_model', type=str, default='./models/models_office_caltech/target_densenet121.pt',
                    help='target model to be attacked')
                    
parser.add_argument('--output_path', type=str, default="./data/adversarial_seeds",
                    help='output path of the adversarial samples')

parser.add_argument('--model_type', type=int, default=0, choices=[0, 1],
                    help='type of the model (transfer learning (0) or normal DNN (1))')

parser.add_argument('--base_net', type=str, default='resnet50', help='base net of the transfer learning',
                    choices=['alexnet', 'resnet18', 'resnet34', 'resnet50',
                             'resnet101', 'resnet152', 'vgg16', 'vgg19', 'inception_v3']
                    )

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(0)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def main_advertorch():

    data_loaders, class_to_idx, dataset_sizes = load_data(args.input_data, args.batch_size, train_flag=False, kwargs=kwargs)
    print('length of dataset: {}'.format(dataset_sizes))

    if args.model_type == 0:
        # transfer learning based model
        from TransferNet import Transfer_Net
        model_target = Transfer_Net(num_class=10, base_net=args.base_net)
        model_target.load_state_dict(torch.load(args.input_model))
        model_target.to(DEVICE)
        model_target = model_target.predict
      
    elif args.model_type == 1:
        # normal DNN
        model_target = torch.load(args.target_model)
        model_target.to(DEVICE)
        model_target = model_target.eval()
    
    # SinglePixelAttack
    adversary = attacks.DDNL2Attack(model_target, nb_iter=100, gamma=0.05, init_norm=1.0,
                                    quantize=True, levels=256, clip_min=0.0, clip_max=1.0, targeted=False,
                                    loss_fn=nn.CrossEntropyLoss(reduction="sum"))
    # adversary = attacks.LinfPGDAttack(model_target, loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
    #        eps=0.05, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    
    running_corrects_adv_untargeted = 0

    if os.path.isdir(args.output_path):
        shutil.rmtree(args.output_path)

    for batch_idx, (inputs, labels) in enumerate(data_loaders):
    
        cln_data, true_label = inputs.to(DEVICE), labels.to(DEVICE)
        print()
        print('clean data shape: {}, true label: {}'.format(cln_data.shape, true_label))

        adv_untargeted = adversary.perturb(cln_data, true_label)

        # predict adversarial samples
        outputs = model_target(adv_untargeted.to(DEVICE))
        _, predicted = torch.max(outputs, 1)
        running_corrects_adv_untargeted += torch.sum(predicted == true_label.data)
        print('untargeted perturb predict label: ', predicted)

        # save adversarial images to local
        for idx, adver_seed in enumerate(adv_untargeted):
            for key, value in class_to_idx.items():
                if true_label[idx].item() == value:
                    adver_seed_dir = os.path.join(args.output_path, key)
                    if not os.path.isdir(adver_seed_dir):
                        os.makedirs(adver_seed_dir)

            adver_seed_path = os.path.join(adver_seed_dir, str(batch_idx) + '_' + str(idx) + '.jpg')
            torchvision.utils.save_image(adver_seed, adver_seed_path, normalize=True, scale_each=True)
                        
    print('running_corrects_adver_untargeted: {}'.format(running_corrects_adv_untargeted))


def main_foolbox():

    data_loaders, dataset_sizes, class_to_idx = load_data(args.test_data, args.batch_size, train_flag=False, kwargs=kwargs)
    print('length of dataset: {}'.format(dataset_sizes))

    if args.model_type == 0:

        # transfer learning based model

        from TransferNet import Transfer_Net
        model_target = Transfer_Net(num_class=10, base_net=args.base_net)
        model_target.load_state_dict(torch.load(args.input_model))
        model_target.to(DEVICE)
        model_target = model_target.predict

    elif args.model_type == 1:
        # normal DNN
        model_target = torch.load(args.target_model)
        model_target.to(DEVICE)
        model_target = model_target.eval()

    fmodel = fb.PyTorchModel(model_target, bounds=(0, 1))
    epsilons = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.1, 0.3, 0.5,1.0]

    attacks = [
            fb.attacks.L2RepeatedAdditiveGaussianNoiseAttack(), 
            # fb.attacks.LinfDeepFoolAttack(steps=50, candidates=10, overshoot=0.02, loss='logits'), 
            ]
    attack = attacks[0]

    running_corrects_adv_untargeted = 0

    if os.path.isdir(args.output_path):
        shutil.rmtree(args.output_path)

    for batch_idx, (inputs, labels) in enumerate(data_loaders):

        cln_data, true_label = inputs.to(DEVICE), labels.to(DEVICE)
        print()
        print('clean data shape: {}, true label: {}'.format(cln_data.shape, true_label))
        print()

        advs, _, success = attack(fmodel, cln_data, true_label, epsilons=epsilons)
        adv_images = advs[4].clone().detach().requires_grad_(True)

        # predict adversarial samples
        outputs = model_target(adv_images.to(DEVICE))
        _, predicted = torch.max(outputs, 1)
        running_corrects_adv_untargeted += torch.sum(predicted == true_label.data)
        print('perturbed data predict label: ', predicted)

        # save adversarial images to local
        for idx, adver_seed in enumerate(adv_images):
            for key, value in class_to_idx.items():
                if true_label[idx].item() == value:
                    adver_seed_dir = os.path.join(args.output_path, key)
                    if not os.path.isdir(adver_seed_dir):
                        os.makedirs(adver_seed_dir)

            adver_seed_path = os.path.join(adver_seed_dir, str(batch_idx) + '_' + str(idx) + '.jpg')
            torchvision.utils.save_image(adver_seed, adver_seed_path, normalize=True, scale_each=True)

    print('running_corrects_adver: {}'.format(running_corrects_adv_untargeted))


if __name__ == '__main__':
  
    main_advertorch()
