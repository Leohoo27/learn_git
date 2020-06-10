import torch
import torch.nn.functional as F
import torchvision
import os

from coverage import CoverageUpdate
from utils.config import bcolors
from utils.mutators import constraint_black, constraint_occl


class Fuzzer(object):
    def __init__(self, data, args, models, model_layer_dicts, model_layer_names, DEVICE):

        self.data_loader = data[0]
        self.class_to_idx = data[1]
        self.model_type = args.model_type
        self.output_path = args.output_path
        self.device = DEVICE

        self.model_1 = models[0]
        self.model_2 = models[1]
        self.model_3 = models[2]

        self.model_layer_dicts_1 = model_layer_dicts[0]
        self.model_layer_dicts_2 = model_layer_dicts[1]
        self.model_layer_dicts_3 = model_layer_dicts[2]

        self.model_layer_names_1 = model_layer_names[0]
        self.model_layer_names_2 = model_layer_names[1]
        self.model_layer_names_3 = model_layer_names[2]

        self.iters_num = 50

    def loop(self):
        for idx, (image, labels) in enumerate(self.data_loader):

            image.requires_grad = True
            labels = labels.to(self.device)
            output_1, output_2, output_3 = None, None, None
            if self.model_type == 0:
                output_1, output_2, output_3 = self.model_1(image.to(self.device)), \
                                               self.model_2(image.to(self.device)), \
                                               self.model_3(image.to(self.device))
            elif self.model_type == 1:
                output_1, output_2, output_3 = self.model_1.predict(image.to(self.device)), \
                                               self.model_2.predict(image.to(self.device)), \
                                               self.model_3.predict(image.to(self.device))
            pred_1, pred_2, pred_3 = torch.max(output_1, 1)[1], \
                                     torch.max(output_2, 1)[1], \
                                     torch.max(output_3, 1)[1]
            loss_1, loss_2, loss_3 = F.nll_loss(output_1, labels), F.nll_loss(output_2, labels), \
                                     F.nll_loss(output_3, labels)

            # adversarial images save path
            global image_dir
            for key, value in self.class_to_idx.items():
                if labels.item() == value:
                    image_dir = os.path.join(self.output_path, key)
                    if not os.path.isdir(image_dir):
                        os.makedirs(image_dir)

            update_coverage = CoverageUpdate.CoverageUpdate()

            if pred_1 != pred_2 and pred_1 != pred_3 and pred_2 != pred_3:
                print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(
                    pred_1, pred_2, pred_3) + bcolors.ENDC)

                self.model_layer_dict_1 = update_coverage.update_nc_coverage(
                    self.model_1, image.to(self.device), self.model_layer_dict_1, self.model_layer_names_1, threshold=0)
                self.model_layer_dict_2 = update_coverage.update_nc_coverage(
                    self.model_2, image.to(self.device), self.model_layer_dict_2, self.model_layer_names_2, threshold=0)
                self.model_layer_dict_3 = update_coverage.update_nc_coverage(
                    self.model_3, image.to(self.device), self.model_layer_dict_3, self.model_layer_names_3, threshold=0)

                covered_neurons_1, total_neurons_1, covered_neurons_rate_1 = update_coverage.neuron_covered(
                    self.model_layer_dict_1)
                covered_neurons_2, total_neurons_2, covered_neurons_rate_2 = update_coverage.neuron_covered(
                    self.model_layer_dict_2)
                covered_neurons_3, total_neurons_3, covered_neurons_rate_3 = update_coverage.neuron_covered(
                    self.model_layer_dict_3)

                averaged_nc = (covered_neurons_rate_1 + covered_neurons_rate_2 + covered_neurons_rate_3) / 3

                print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                      % (total_neurons_1, covered_neurons_1,
                         total_neurons_2, covered_neurons_2,
                         total_neurons_3, covered_neurons_3) +
                      bcolors.ENDC)
                print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

                # save already causes different outputs
                torchvision.utils.save_image(image, os.path.join(image_dir, str(idx) + '.jpg'),
                                             normalize=True, scale_each=True)
                continue

            else:

                print('********************* start fuzzing *********************')

                layer_name_1, index_1 = update_coverage.neuron_to_cover(self.model_layer_dict_1)
                layer_name_2, index_2 = update_coverage.neuron_to_cover(self.model_layer_dict_2)
                layer_name_3, index_3 = update_coverage.neuron_to_cover(self.model_layer_dict_3)

                loss1_neuron = torch.mean(
                    self.model_layer_names_1.get(layer_name_1).state_dict().get('weight')[index_1, ...])
                loss2_neuron = torch.mean(
                    self.model_layer_names_2.get(layer_name_2).state_dict().get('weight')[index_2, ...])
                loss3_neuron = torch.mean(
                    self.model_layer_names_3.get(layer_name_3).state_dict().get('weight')[index_3, ...])

                for iter in range(1, self.iters_num + 1):

                    print(" iteration: {0} \n".format(iter))

                    final_loss = (-1.0 * loss_1 + loss_2 + loss_3) + \
                                 0.1 * (loss1_neuron + loss2_neuron + loss3_neuron)

                    # target model
                    self.model_1.zero_grad()
                    final_loss.backward()

                    grads = F.normalize(image.grad.data)[0]

                    # mutation strategy
                    # grads_value = constraint_black(gradients=grads)
                    grads_value = constraint_occl(gradients=grads, start_point=(0, 0),
                                                  rect_shape=(50, 50))
                    # perturb noise to image
                    image = torch.tensor(torch.add(image, grads_value * 1.0), requires_grad=True)

                    if self.model_type == 0:
                        output_1, output_2, output_3 = self.model_1(image.to(self.device)), \
                                                       self.model_2(image.to(self.device)), \
                                                       self.model_3(image.to(self.device))
                    elif self.model_type == 1:
                        output_1, output_2, output_3 = self.model_1.predict(image.to(self.device)), \
                                                       self.model_2.predict(image.to(self.device)), \
                                                       self.model_3.predict(image.to(self.device))
                    pred_1, pred_2, pred_3 = torch.max(output_1, 1)[1], torch.max(output_2, 1)[1], \
                                             torch.max(output_3, 1)[1]
                    print('Predictions after fuzzing: {}, {}, {}'.format(pred_1, pred_2, pred_3))
                    loss_1, loss_2, loss_3 = F.nll_loss(output_1, labels), F.nll_loss(output_2, labels), \
                                             F.nll_loss(output_3, labels)

                    if pred_1 != pred_2 and pred_1 != pred_3 and pred_2 != pred_3:

                        self.model_layer_dict_1 = update_coverage.update_nc_coverage(
                            self.model_1, image.to(self.device), self.model_layer_dict_1,
                            self.model_layer_names_1, threshold=0)

                        self.model_layer_dict_2 = update_coverage.update_nc_coverage(
                            self.model_2, image.to(self.device), self.model_layer_dict_2,
                            self.model_layer_names_2, threshold=0)

                        self.model_layer_dict_3 = update_coverage.update_nc_coverage(
                            self.model_3, image.to(self.device), self.model_layer_dict_3,
                            self.model_layer_names_3, threshold=0)

                        covered_neurons_1, total_neurons_1, covered_neurons_rate_1 = update_coverage.neuron_covered(
                            self.model_layer_dict_1)
                        covered_neurons_2, total_neurons_2, covered_neurons_rate_2 = update_coverage.neuron_covered(
                            self.model_layer_dict_2)
                        covered_neurons_3, total_neurons_3, covered_neurons_rate_3 = update_coverage.neuron_covered(
                            self.model_layer_dict_3)

                        averaged_nc = (covered_neurons_rate_1 + covered_neurons_rate_2 + covered_neurons_rate_3) / 3

                        print(
                            bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                            % (total_neurons_1, covered_neurons_1,
                               total_neurons_2, covered_neurons_2,
                               total_neurons_3, covered_neurons_3) +
                            bcolors.ENDC)
                        print()
                        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

                        torchvision.utils.save_image(image, os.path.join(image_dir, str(idx) + '.jpg'),
                                                     normalize=True, scale_each=True)

                        break

                    elif iter < self.iters_num:
                        print("Not predict_1 != predict_2 != predict_3! Keep continue!\n")

                    elif iter > self.iters_num:
                        print("Over {} times fuzz iteration.".format(self.iters_num))
                        image_name = str(idx) + '.jpg'
                        torchvision.utils.save_image(image, os.path.join(image_dir, str(idx) + '.jpg'),
                                                     normalize=True, scale_each=True)
        return 0
