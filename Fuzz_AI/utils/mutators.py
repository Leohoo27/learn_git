from __future__ import print_function
import cv2
import numpy as np
import random

import torch
import torch.nn as nn
from scipy.ndimage import filters


def image_noise(img, params, grads, pixel=800):
    up = nn.Upsample(size=(pixel, pixel), mode='bilinear', align_corners=True)
    img_up = up(img)
    
    if params == 0:
        return img

    elif params == 1:  # Gaussian-distributed additive noise.
        batch, ch, row, col = img_up.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = torch.tensor(np.random.normal(mean, sigma, (batch, ch, row, col))).to(dtype=torch.float)
        new_img = torch.add(img_up, gauss * 0.5)

    elif params == 2:  
        # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
        alpha = 0.01
        imhist, bins = np.histogram(img_up.detach().numpy().flatten())
        cdf = imhist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        img_temp = np.interp(img_up.detach().numpy().flatten(), bins[:-1], cdf * alpha)
        img_up = torch.tensor(img_temp.reshape(img_up.shape))

        mean = 0
        var = 0.1
        sigma = var ** 0.5

        batch, ch, row, col = img_up.shape
        gauss = torch.tensor(np.random.randn(mean, sigma, (batch, ch, row, col))).to(dtype=torch.float)
        new_img = torch.add(img_up, img_up * gauss * 0.5)

    elif params == 3:
        # Multiplicative noise using out = image + noise * alpha, 
        # where n is uniform noise with specified mean & variance.
        batch, ch, row, col = img_up.shape
        
        mean = torch.mean(grads)
        var = torch.var(grads)
        sigma = var ** 0.5
        
        gauss = torch.normal(mean, sigma, (batch, ch, row, col))
        new_img = torch.add(img_up, gauss * 0.1)

    elif params == 4:
        img_up_temp = img_up.detach().numpy()

        alpha = 0.01
        imhist, bins = np.histogram(img_up_temp.flatten())
        cdf = imhist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        img_temp = np.interp(img_up_temp.flatten(), bins[:-1], cdf * alpha)
        img_up = img_temp.reshape(img_up_temp.shape)

        s_vs_p = 0.5
        amount = 0.004
        new_img = np.copy(img_up)

        # Salt mode
        num_salt = np.ceil(amount * img_up.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                  for i in img_up.shape]
        new_img[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img_up.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                  for i in img_up.shape]
        new_img[tuple(coords)] = 0
        
        batch, ch, row, col = img.shape
        down = nn.Upsample(size=(row, col), mode='bilinear', align_corners=True)
        new_img = down(torch.tensor(new_img))

        new_img += grads * 0.5
        
        return new_img
    
    elif params == 5:
        imx = np.zeros(img_up.shape)
        filters.sobel(img_up.detach().numpy(), 1, imx)

        imy = np.zeros(img_up.shape)
        filters.sobel(img_up.detach().numpy(), 0, imy)
        magnitude = np.sqrt(imx ** 2 + imy ** 2) / 255.0
        
        new_img = torch.add(img_up, torch.tensor(magnitude))

    batch, ch, row, col = img.shape
    down = nn.Upsample(size=(row, col), mode='bilinear', align_corners=True)
    new_img = down(new_img)

    return new_img



def image_blur(image, params):

    # print("blur")
    img = image[0].detach().numpy()
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    # if params == 9:
    #     blur = cv2.blur(img, (6, 6))
    if params == 9:
        blur = cv2.bilateralFilter(img, 6, 50, 50)
        # blur = cv2.bilateralFilter(img, 9, 75, 75)
    image_blur = torch.tensor(blur).view(image.shape)
    return image_blur.detach().numpy()


def constraint_black(gradients, rect_shape=(10, 10)):

    start_point = (torch.randint(0, gradients.shape[1] - rect_shape[0], (1,)),
                   torch.randint(0, gradients.shape[2] - rect_shape[1], (1,)))

    new_grads = torch.zeros_like(gradients)

    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
            start_point[1]:start_point[1] + rect_shape[1]]

    if torch.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -torch.ones_like(patch)

    return new_grads


def constraint_occl(gradients, start_point, rect_shape):

    new_grads = torch.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads

