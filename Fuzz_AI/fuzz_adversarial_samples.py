"""usage:
python3 fuzz_adversarial_samples.py --batch_size 32 --model_type 1 --models_path ./models/models_office_caltech/ --input_data ./data/adversarial_seeds/ --output_path ./data/generated_fuzz_samples_from_advertorch/
"""
from torchvision import models
import torch
import os
import argparse

from utils.data_loader import load_data
from coverage import CoverageTable
from lib import Fuzzer


parser = argparse.ArgumentParser(description='Generate adversarial samples by improving '
                                             'DeepXplore fuzzing method')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--model_type', type=int, default=1, choices=[0, 1],
                    help='model type for fuzzing (normal DNN: 0, transfer learning: 1)')

parser.add_argument('--input_data', type=str, default=None,
                    help='path of image seeds data')

parser.add_argument('--output_path', type=str, default=None,
                    help='output fuzzed adversarial samples path')

parser.add_argument('--models_path', type=str, default=None,
                    help='models dir for fuzzing')

parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 233)')

args = parser.parse_args()

torch.manual_seed(args.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    model_1, model_2, model_3 = None, None, None
    if args.model_type == 0:
        # normal DDN model
        model_1 = models.alexnet(pretrained=True).to(DEVICE).eval()
        model_2 = models.vgg16(pretrained=True).to(DEVICE).eval()
        model_3 = models.vgg19(pretrained=True).to(DEVICE).eval()

    elif args.model_type == 1:
        # transfer learning model
        from TransferNet import Transfer_Net
        model_1 = Transfer_Net(num_class=10, base_net='alexnet')
        model_1.load_state_dict(torch.load(os.path.join(args.models_path, 'alexnet_amazon_caltech.model.pt')))
        model_1.to(DEVICE).eval()

        model_2 = Transfer_Net(num_class=10, base_net='vgg16')
        model_2.load_state_dict(torch.load(os.path.join(args.models_path, 'vgg16_amazon_caltech.model.pt')))
        model_2.to(DEVICE).eval()

        model_3 = Transfer_Net(num_class=10, base_net='vgg19')
        model_3.load_state_dict(torch.load(os.path.join(args.models_path, 'vgg19_amazon_caltech.model.pt')))
        model_3.to(DEVICE).eval()

    return model_1, model_2, model_3


def main():

    data_loader, class_to_idx, dataset_sizes = load_data(data_folder=args.input_data, batch_size=1,
                                          train_flag=False, kwargs={'num_workers': 0})

    # models for fuzzing: alexnet, vgg19, mobilenet_v2, vgg16
    model_1, model_2, model_3 = load_model()

    # neuron coverage
    coverage_table_init = CoverageTable.CoverageTableInit()
    model_layer_dict_1, model_1_layer_names, \
    model_layer_dict_2, model_2_layer_names, \
    model_layer_dict_3, model_3_layer_names = \
        coverage_table_init.init_deepxplore_coverage_tables(model_1, model_2, model_3)

    # params of Fuzzer
    data = [data_loader, class_to_idx]
    models = [model_1, model_2, model_3]
    model_layer_dicts = [model_layer_dict_1, model_layer_dict_2, model_layer_dict_3]
    model_layer_names = [model_1_layer_names, model_2_layer_names, model_3_layer_names]

    fuzzer = Fuzzer.Fuzzer(data, args, models, model_layer_dicts, model_layer_names, DEVICE)
    fuzzer.loop()


if __name__ == '__main__':

    main()
