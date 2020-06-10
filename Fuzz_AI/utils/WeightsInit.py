import torch.nn as nn


class WeightsInit:
    def __init__(self):
        pass

    def weights_init(self, model):
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


if __name__ == '__main__':
    print('test...')