"""usage:
python3 test_target_model.py --batch_size 32 --target_model ./models/models_office_caltech/target_densenet121.pt --test_data_path ./data/adversarial_seeds/
"""
#!/usr/bin/env python3
import torch
import argparse
from torchvision import transforms, datasets


parser = argparse.ArgumentParser(description='Target Model Training')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--target_model', type=str, default='./model/models_office_caltech/target_densenet121.pt',
                    help='path of target model to be attacked')

parser.add_argument('--test_data_path', type=str, default=None,
                    help='test data path')

parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def load_data(data_dir, batch_size):
    data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = datasets.ImageFolder(data_dir, data_transforms)

    data_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
                                               shuffle=True, **kwargs)

    dataset_sizes = len(image_datasets)

    return data_loaders, dataset_sizes, image_datasets.class_to_idx


def model_test():
    print()
    print('offline testing')
    print()

    model = torch.load(args.target_model)
    model = model.eval()

    data_loaders, dataset_sizes, class_to_idx = load_data(args.test_data_path, args.batch_size)
    print('number of the adversarial samples: {}'.format(dataset_sizes))

    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in data_loaders:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the target model on the test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':

    model_test()
