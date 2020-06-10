"""usage:
python3 test_fuzz_adversarial_samples.py --batch_size 32 --model_type 1 --seeds_path ./data/adversarial_seeds_test/ --output_path ./data/generated_fuzz_samples_test/ 
"""
from torchvision import models
from collections import defaultdict
import torch.nn as nn
from torchvision import datasets, transforms
import torch
import random
import torch.nn.functional as F

import torchvision
from utils.config import bcolors
import numpy as np
import os
import argparse

from utils.mutators import image_noise, constraint_occl, constraint_black, image_blur


parser = argparse.ArgumentParser(description='Generate adversarial samples by using DeepXplore')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--model_type', type=int, default=1,
        help='model type for fuzzing (normal DNN: 0, transfer learning: 1)')

parser.add_argument('--seeds_path', type=str, default=None, help='seeds data path')

parser.add_argument('--output_path', type=str, default=None, help='output fuzzed adversarial samples path')

args = parser.parse_args()

torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(model):
    for idx, (name, item) in enumerate(model.named_modules()):
        # print(name, item)
        class_name = item.__class__.__name__
        if class_name.find('Conv') != -1:
            # print(class_name, item.weight.data.shape)
            conv_weight = nn.init.normal_(item.weight.data, 0.0, 0.02)
            print(conv_weight.shape)
        elif class_name.find('BatchNorm') != -1:
            bn_weight = nn.init.normal_(item.weight.data, 1.0, 0.02)
            bn_bias = nn.init.constant_(item.bias.data, 0)
            print(bn_weight.shape, bn_bias.shape)


def init_coverage_table(model):

    model_layer_dict = defaultdict(bool)
    layer_names = {}

    for idx, (name, item) in enumerate(model.named_modules()):
        # if item.state_dict().get('weight') is not None:
        # print(name, item)
        if len(name) > 0 and 'bottleneck_layer' not in name:
            # layer_size = item.state_dict().get('weight').size()[0]
            # # print(name, item, item.state_dict().get('weight').size()[0])
            # layer_names[name] = item
            #
            # for index in range(layer_size):
            #     model_layer_dict[(name, index)] = False

            if isinstance(item, nn.Conv2d):
                # print(name, item)
                layer_names[name] = item
                for index in range(item.out_channels):
                    model_layer_dict[(name, index)] = False

            if isinstance(item, nn.ReLU):
                # print(name, item)
                layer_names[name] = item

            if isinstance(item, nn.BatchNorm2d):
                layer_names[name] = item
                for index in range(item.numfeatures):
                    model_layer_dict[(name, index)] = False

            if isinstance(item, nn.AdaptiveAvgPool2d):
                # print(name, item)
                layer_names[name] = item

            if isinstance(item, nn.MaxPool2d):
                # print(name, item)
                layer_names[name] = item

            if isinstance(item, nn.Linear):
                # print(name, item)
                layer_names[name] = item
                for index in range(item.out_features):
                    model_layer_dict[(name, index)] = False

            if isinstance(item, nn.ReLU6):
                # print(name, item)
                layer_names[name] = item

    return model_layer_dict, layer_names




def load_data(data_folder, batch_size, train_flag, kwargs):

    '''
    load image data by leveraging torchvision.datasets.ImageFolder from
    data folders.
    '''
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.5, hue=0),
             # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 0.9), shear=1, fillcolor=(50, 1, 1)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])],
             # transforms.RandomErasing(ratio=(0.1, 0.2), value='random'),
             )
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train_flag else 'test'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs,
                                              drop_last=True if train_flag else False)
    
    return data_loader, data.class_to_idx


def scale(intermediate_layer_output, rmax=1, rmin=0):

    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin

    return X_scaled


def update_coverage(model, input_data, model_layer_dict, layer_names, threshold=0):

    model_name = model.__class__.__name__
    input = input_data
    identity = input_data
    for idx, (key, value) in enumerate(layer_names.items()):

        if model_name in ['AlexNet', 'VGG', 'Transfer_Net'] and isinstance(layer_names.get(key), nn.Linear):
            input = torch.flatten(input, 1)

        # print(input.shape, key, layer_names.get(key))
        intermediate_layer_output = layer_names.get(key).forward(input)

        scaled = scale(intermediate_layer_output[0])

        for num_neuron in range(scaled.shape[-1]):
            if torch.mean(scaled[..., num_neuron]) > threshold and \
                    not model_layer_dict[(list(layer_names.keys())[idx], num_neuron)]:
                model_layer_dict[(list(layer_names.keys())[idx], num_neuron)] = True

        input = intermediate_layer_output


def neuron_covered(model_layer_dict):

    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)

    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def neuron_to_cover(model_layer_dict):

    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())

    print("layer_name: {0}, index: {1}\n".format(layer_name, index))

    return layer_name, index


def init_coverage_tables(model_1, model_2, model_3):

    model_layer_dict_1, model_1_layer_names = init_coverage_table(model_1)
    model_layer_dict_2, model_2_layer_names = init_coverage_table(model_2)
    model_layer_dict_3, model_3_layer_names = init_coverage_table(model_3)

    return model_layer_dict_1, model_layer_dict_2, model_layer_dict_3, \
    model_1_layer_names, model_2_layer_names, model_3_layer_names


def main():

    model_path = './models/models_office_caltech'
    data_loader, class_to_idx = load_data(data_folder=args.seeds_path, batch_size=1, train_flag=False, kwargs={'num_workers': 0}, )

    # alexnet, vgg19, mobilenet_v2, vgg16
    if args.model_type == 0:
        # normal DDN model
        model_1 = models.alexnet(pretrained=True).to(DEVICE)
        model_1.eval()
        model_2 = models.vgg16(pretrained=True).to(DEVICE)
        model_2.eval()
        model_3 = models.vgg19(pretrained=True).to(DEVICE)
        model_3.eval()

    if args.model_type == 1:
        # transfer learning model
        from TransferNet import Transfer_Net
        model_1 = Transfer_Net(num_class=10, base_net='alexnet')
        model_1.load_state_dict(torch.load(os.path.join(model_path, 'alexnet_amazon_caltech.model.pt')))
        model_1.to(DEVICE)
        model_1.eval()
        
        model_2 = Transfer_Net(num_class=10, base_net='vgg16')
        model_2.load_state_dict(torch.load(os.path.join(model_path, 'vgg16_amazon_caltech.model.pt')))
        model_2.to(DEVICE)
        model_2.eval()

        model_3 = Transfer_Net(num_class=10, base_net='vgg19')
        model_3.load_state_dict(torch.load(os.path.join(model_path, 'vgg19_amazon_caltech.model.pt')))
        model_3.to(DEVICE)
        model_3.eval()

    # weights_init(model)

    # neuron coverage
    model_layer_dict_1, model_layer_dict_2, model_layer_dict_3, \
    model_1_layer_names, model_2_layer_names, model_3_layer_names = init_coverage_tables(model_1, model_2, model_3)
    
    for idx, (image, labels) in enumerate(data_loader):
        
        image.requires_grad = True
        labels = labels.to(DEVICE)
        if args.model_type == 0:
            output_1, output_2, output_3 = model_1.forward(image.to(DEVICE)), model_2.forward(image.to(DEVICE)), model_3(image.to(DEVICE))
        elif args.model_type == 1:
            output_1, output_2, output_3 = model_1.predict(image.to(DEVICE)), model_2.predict(image.to(DEVICE)), model_3.predict(image.to(DEVICE))
        pred_1, pred_2, pred_3 = torch.max(output_1, 1)[1], torch.max(output_2, 1)[1], torch.max(output_3, 1)[1]
        loss_1, loss_2, loss_3 = F.nll_loss(output_1, labels), F.nll_loss(output_2, labels), \
                                 F.nll_loss(output_3, labels)

        # adversarial images save path
        for key, value in class_to_idx.items():
            if labels.item() == value:
                image_dir = os.path.join(args.output_path, key)
                if not os.path.isdir(image_dir):
                    os.makedirs(image_dir)

        if pred_1 != pred_2 and pred_1 != pred_3 and pred_2 != pred_3:
            image = image.to(DEVICE)
            print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(
                pred_1, pred_2, pred_3) + bcolors.ENDC)

            update_coverage(model_1, image, model_layer_dict_1, model_1_layer_names, threshold=0)
            update_coverage(model_2, image, model_layer_dict_2, model_2_layer_names, threshold=0)
            update_coverage(model_3, image, model_layer_dict_3, model_3_layer_names, threshold=0)

            covered_neurons_1, total_neurons_1, covered_neurons_rate_1 = neuron_covered(model_layer_dict_1)
            covered_neurons_2, total_neurons_2, covered_neurons_rate_2 = neuron_covered(model_layer_dict_2)
            covered_neurons_3, total_neurons_3, covered_neurons_rate_3 = neuron_covered(model_layer_dict_3)

            averaged_nc = (covered_neurons_rate_1 + covered_neurons_rate_2 + covered_neurons_rate_3) / 3

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (total_neurons_1, covered_neurons_1,
                     total_neurons_2, covered_neurons_2,
                     total_neurons_3, covered_neurons_3) +
                  bcolors.ENDC)
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)
            
            image_name = str(idx) + '.jpg'
            torchvision.utils.save_image(image, os.path.join(image_dir, image_name), normalize=True, scale_each=True)
            
            continue

        else:

            print('********************* start fuzzing *********************')

            layer_name_1, index_1 = neuron_to_cover(model_layer_dict_1)
            layer_name_2, index_2 = neuron_to_cover(model_layer_dict_2)
            layer_name_3, index_3 = neuron_to_cover(model_layer_dict_3)

            loss1_neuron = torch.mean(model_1_layer_names.get(layer_name_1).state_dict().get('weight')[index_1, ...])
            loss2_neuron = torch.mean(model_2_layer_names.get(layer_name_2).state_dict().get('weight')[index_2, ...])
            loss3_neuron = torch.mean(model_3_layer_names.get(layer_name_3).state_dict().get('weight')[index_3, ...])

            iters_num = 20
            for iter in range(1, iters_num + 1):
                
                print(" iteration: {0} \n".format(iter))

                final_loss = (-1.0 * loss_1 + loss_2 + loss_3) + 0.1 * (loss1_neuron + loss2_neuron + loss3_neuron)
                # print(final_loss)

                # target model
                model_1.zero_grad()
                final_loss.backward()
                
                grads = F.normalize(image.grad.data)[0]

                # mutation strategy
                grads_value = constraint_black(gradients=grads)
                # grads_value = constraint_occl(gradients=grads, start_point=(0, 0), rect_shape=(50, 50))
                
                # perturb noise to image
                image = torch.add(image, grads_value * 1.0)
                param = 3  # type of mutator
                image = image_noise(image, param)
                # image = image_blur(image, param)
                image = torch.tensor(image, requires_grad=True).to(torch.float)
                
                if args.model_type == 0:
                    output_1, output_2, output_3 = model_1.forward(image.to(DEVICE)), model_2.forward(image.to(DEVICE)), model_3(image.to(DEVICE))
                elif args.model_type == 1:
                    output_1, output_2, output_3 = model_1.predict(image.to(DEVICE)), model_2.predict(image.to(DEVICE)), \
                                               model_3.predict(image.to(DEVICE))
                pred_1, pred_2, pred_3 = torch.max(output_1, 1)[1], torch.max(output_2, 1)[1], \
                                         torch.max(output_3, 1)[1]
                print(pred_1, pred_2, pred_3)
                loss_1, loss_2, loss_3 = F.nll_loss(output_1, labels), F.nll_loss(output_2, labels), \
                                         F.nll_loss(output_3, labels)

                if pred_1 != pred_2 and pred_1!= pred_3 and pred_2 != pred_3:

                    image = image.to(DEVICE)

                    update_coverage(model_1, image, model_layer_dict_1, model_1_layer_names, threshold=0)
                    update_coverage(model_2, image, model_layer_dict_2, model_2_layer_names, threshold=0)
                    update_coverage(model_3, image, model_layer_dict_3, model_3_layer_names, threshold=0)

                    covered_neurons_1, total_neurons_1, covered_neurons_rate_1 = neuron_covered(model_layer_dict_1)
                    covered_neurons_2, total_neurons_2, covered_neurons_rate_2 = neuron_covered(model_layer_dict_2)
                    covered_neurons_3, total_neurons_3, covered_neurons_rate_3 = neuron_covered(model_layer_dict_3)

                    averaged_nc = (covered_neurons_rate_1 + covered_neurons_rate_2 + covered_neurons_rate_3) / 3

                    print(
                        bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                        % (total_neurons_1, covered_neurons_1,
                           total_neurons_2, covered_neurons_2,
                           total_neurons_3, covered_neurons_3) +
                        bcolors.ENDC)
                    print()
                    print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

                    image_name = str(idx) + '.jpg'
                    torchvision.utils.save_image(image, os.path.join(image_dir, image_name))

                    break

                elif iter < iters_num:
                    print("Not predict_1 != predict_2 != predict_3! Keep continue!\n")

                else:
                    print("Over {} times fuzz iteration.".format(iters_num))
                    image_name = str(idx) + '.jpg'
                    torchvision.utils.save_image(image, os.path.join(image_dir, image_name), normalize=True, scale_each=True)
    return 0


if __name__ == '__main__':

    main()
