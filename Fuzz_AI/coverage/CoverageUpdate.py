import torch
import torch.nn as nn
import random

from utils.utils import scale


class CoverageUpdate():
    def __init__(self):
        pass

    def update_nc_coverage(self, model, input_data, model_layer_dict, layer_names, threshold=0):
        """
        Neuron coverage update proposed from DeepXplore.
        :param model: DNN model
        :param input_data: image
        :param model_layer_dict:
        :param layer_names:
        :param threshold:
        :return:
        """
        model_name = model.__class__.__name__
        input = input_data

        for idx, (key, value) in enumerate(layer_names.items()):

            if model_name in ['AlexNet', 'VGG', 'Transfer_Net'] and isinstance(layer_names.get(key), nn.Linear):
                input = torch.flatten(input, 1)

            intermediate_layer_output = layer_names.get(key).forward(input)

            scaled = scale(intermediate_layer_output[0])

            for num_neuron in range(scaled.shape[-1]):
                if torch.mean(scaled[..., num_neuron]) > threshold and \
                        not model_layer_dict[(list(layer_names.keys())[idx], num_neuron)]:
                    model_layer_dict[(list(layer_names.keys())[idx], num_neuron)] = True

            input = intermediate_layer_output

        return model_layer_dict

    def neuron_covered(self, model_layer_dict):

        covered_neurons = len([v for v in model_layer_dict.values() if v])
        total_neurons = len(model_layer_dict)

        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def neuron_to_cover(self, model_layer_dict):

        not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(model_layer_dict.keys())

        print("layer_name: {0}, index: {1}\n".format(layer_name, index))

        return layer_name, index