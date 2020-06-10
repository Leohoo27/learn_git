import torch.nn as nn
import collections


class CoverageTableInit():
    def __init__(self):
        pass

    def init_deepxplore_coverage_table(self, model):
        """
        initial DeepXplore coverage table
        :param model: DNN model (AlexNet, VGG)
        :return: 1. model layer dict. 2. each layer name
        """
        model_layer_dict = collections.OrderedDict(bool)
        layer_names = collections.OrderedDict()

        for idx, (name, item) in enumerate(model.named_modules()):

            if len(name) > 0 and 'bottleneck_layer' not in name:
                # layer_size = item.state_dict().get('weight').size()[0]
                # # print(name, item, item.state_dict().get('weight').size()[0])
                # layer_names[name] = item

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

    def init_deepxplore_coverage_tables(self, model_1, model_2, model_3):

        model_layer_dict_1, model_1_layer_names = self.init_deepxplore_coverage_table(model_1)
        model_layer_dict_2, model_2_layer_names = self.init_deepxplore_coverage_table(model_2)
        model_layer_dict_3, model_3_layer_names = self.init_deepxplore_coverage_table(model_3)

        return model_layer_dict_1, model_1_layer_names, \
               model_layer_dict_2, model_2_layer_names, \
               model_layer_dict_3, model_3_layer_names

    def init_deephunter_coverage_table(self, model):
        """
        initial DeepHunter coverage table
        :param model: DNN model (AlexNet, VGG)
        :return: 1. model layer dict. 2. each layer name
        """
        model_layer_dict = collections.OrderedDict()
        layer_names = collections.OrderedDict()

        flag = 0

        for idx, (name, item) in enumerate(model.named_modules()):
            if len(name) > 0:

                if isinstance(item, nn.Conv2d):
                    layer_names[name] = item
                    flag = item.out_channels
                    for index in range(item.out_channels):
                        model_layer_dict[(name, index)] = [0.0, 0.0, 0.0, None, None]

                if isinstance(item, nn.ReLU):
                    layer_names[name] = item
                    for index in range(flag):
                        model_layer_dict[(name, index)] = [0.0, 0.0, 0.0, None, None]

                if isinstance(item, nn.BatchNorm2d):
                    layer_names[name] = item
                    flag = item.num_features
                    for index in range(item.num_features):
                        model_layer_dict[(name, index)] = [0.0, 0.0, 0.0, None, None]

                if isinstance(item, nn.AdaptiveAvgPool2d):
                    layer_names[name] = item
                    for index in range(flag):
                        model_layer_dict[(name, index)] = [0.0, 0.0, 0.0, None, None]

                if isinstance(item, nn.MaxPool2d):
                    layer_names[name] = item
                    for index in range(flag):
                        model_layer_dict[(name, index)] = [0.0, 0.0, 0.0, None, None]

                if isinstance(item, nn.Linear):
                    layer_names[name] = item
                    flag = item.out_features
                    # print(item, item.out_features)
                    for index in range(item.out_features):
                        model_layer_dict[(name, index)] = [0.0, 0.0, 0.0, None, None]

                if isinstance(item, nn.ReLU6):
                    layer_names[name] = item
                    for index in range(flag):
                        model_layer_dict[(name, index)] = [0.0, 0.0, 0.0, None, None]

        return model_layer_dict, layer_names
