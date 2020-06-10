import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from lib import models

import os
from lib.models import Transfer_Net

use_cuda = True
image_nc = 3
batch_size = 1

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "./models/alexnet_amazon_caltech.model.pth"
# targeted_model, input_size = initialize_model(model_name='alexnet', num_classes=10, feature_extract=True, use_pretrained=True)
target_model = Transfer_Net(num_class=10, base_net='alexnet')
target_model.load_state_dict(torch.load(pretrained_model))
target_model.to(device)
print(target_model)
target_model = target_model.predict
model_num_labels = 10


# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_120.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in office_caltech training dataset
office_dataset = torchvision.datasets.ImageFolder(root='../../../datasets/office_caltech/amazon/train/',
                                                      transform=transforms.Compose(
                                                          [
                                                              transforms.Resize([256, 256]),
                                                              transforms.RandomCrop(224),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])
                                                          ]
                                                      ))
train_dataloader = DataLoader(office_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('Office_caltech amazon training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(office_dataset)))

# test adversarial examples in office_caltech testing dataset
office_dataset_test = torchvision.datasets.ImageFolder(root='../../../datasets/office_caltech/caltech/test/',
                                                      transform=transforms.Compose(
                                                          [
                                                              transforms.Resize([224, 224]),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])
                                                          ]
                                                      ))
class_to_idx = office_dataset_test.class_to_idx
test_dataloader = DataLoader(office_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)

num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    # save adv image
    for key, value in class_to_idx.items():
        if test_label.item() == value:
            image_dir = os.path.join('./output', key)
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)

    torchvision.utils.save_image(adv_img, os.path.join(image_dir, 'adv_img_{}.jpg'.format(str(i))), normalize=True, scale_each=True)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in Office_caltech caltech testing set: %f\n'%(num_correct.item()/len(office_dataset_test)))

