#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 6 - Denoising
"""

import imageio
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception as e:
    pass


def adaptive_filtering_local_noise_filter():
    pass


def adaptive_median_filter():
    pass


def contraharmonic_mean_filter():
    pass



def normalize(f):
    fmax = np.max(f)
    fmin = np.min(f)
    f = (f-fmin)/(fmax-fmin)
    f = (f*255).astype(np.uint8)
    return f


def rmse(predictions, targets):
    error = ((predictions - targets) ** 2)
    error = np.sum(error)
    error *= 1 / (predictions.shape[0] * predictions.shape[1])
    error = np.sqrt(error)
    print("{0:.5f}".format(error), end='')


def plot(image1, image2):
    try:
        plt.subplot(1, 2, 1)
        plt.imshow(image1, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.5)

        plt.show()
    except Exception as e:
        pass


def main():
    # receber parâmetros
    original_filename = str(input()).rstrip()
    try:
        # run.codes
        original = imageio.imread(original_filename)
    except Exception as e:
        # local
        original = imageio.imread("tests/"+original_filename)

    noisy_filename = str(input()).rstrip()
    try:
        # run.codes
        noisy = imageio.imread(noisy_filename)
    except Exception as e:
        # local
        noisy = imageio.imread("tests/"+noisy_filename)

    method = int(input())
    filter_size = int(input())

    # restauração
    if (method == 1):
        noisy_dist = int(input())
        restored = option1()
    elif (method == 2):
        filter_size_max = int(input())
        restored = option2()
    elif (method == 3):
        filter_order = int(input())
        restored = option3()

    # exibir imagens para comparação
    plot(original, restored)

    # exibir erro entre original e recuperada
    rmse(restored, original)


if __name__ == '__main__':
    main()
