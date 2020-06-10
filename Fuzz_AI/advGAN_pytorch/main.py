import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack

from train_target_model import initialize_model, set_parameter_requires_grad
from models import Transfer_Net


use_cuda = True
image_nc = 3
epochs = 120
batch_size = 32
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./models/alexnet_amazon_caltech.model.pth"
# targeted_model, input_size = initialize_model(model_name='alexnet', num_classes=10, feature_extract=True, use_pretrained=True)
targeted_model = Transfer_Net(num_class=10, base_net='alexnet') 
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.to(device)
print(targeted_model)
targeted_model = targeted_model.predict
model_num_labels = 10

# office_caltech train dataset and dataloader declaration
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
dataloader = DataLoader(office_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
print(dataloader)
advGAN = AdvGAN_Attack(device,
                       targeted_model,
                       model_num_labels,
                       image_nc,
                       BOX_MIN,
                       BOX_MAX)

advGAN.train(dataloader, epochs)
