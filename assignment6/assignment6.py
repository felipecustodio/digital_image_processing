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


def adaptive_filtering_local(noisy, sigma, filter_size):
    # inicializar resultado
    result = np.zeros(noisy.shape)
    # inicializar filtro
    sigma = np.power(sigma, 2)
    # para cada posição, um filtro (vizinhança)
    neighbors = np.zeros((noisy.shape[0], noisy.shape[1], filter_size, filter_size))

    # filtro = andar metade do tamanho para cada direção
    filter_step = int((filter_size - 1) / 2)
    for x in range(noisy.shape[0]):
        for y in range(noisy.shape[1]):
            # wrap circular com módulo
            x_start = x - filter_step
            x_end = (x + filter_step + 1) % noisy.shape[0]
            y_start = y - filter_step
            y_end = (y + filter_step + 1) % noisy.shape[1]
            neighbors[x][y] = noisy[x_start:x_end, y_start:y_end]

    # aplicar fórmula final
    filter_variance = np.var(neighbors)
    mean_diff = noisy - np.mean(neighbors)
    result = noisy - (sigma / filter_variance) * mean_diff
    return result


def adaptive_median_A(noisy, filter_size_current, filter_size_max, x, y):
    filter_step = int((filter_size_current - 1) / 2)

    x_start = x - filter_step
    x_end = x + filter_step + 1 % noisy.shape[0]
    y_start = y - filter_step
    y_end = y + filter_step + 1 % noisy.shape[1]

    neighbors = noisy[x_start:x_end, y_start:y_end]

    z_med = np.median(neighbors)
    z_min = np.min(neighbors)
    z_max = np.max(neighbors)

    A1 = (z_med - z_min)
    A2 = (z_med - z_max)

    if(A1 > 0 and A2 < 0):
        # ETAPA B
        return adaptive_median_B(noisy, z_min, z_med, z_max, x, y)
    else:
        # aumentar filtro
        filter_size_current += 1
        if(filter_size_current <= filter_size_max):
            return adaptive_median_A(noisy, filter_size_current, filter_size_max, x, y)
        else:
            return z_med


def adaptive_median_B(noisy, z_min, z_med, z_max, x, y):
    B1 = noisy[x][y] - z_min
    B2 = z_med - z_max

    if (B1 > 0 and B2 < 0):
        return noisy[x][y]
    else:
        return z_med


def adaptive_median(noisy, filter_size_max, filter_size):
    result = np.zeros(noisy.shape)
    for x in range(noisy.shape[0]):
        for y in range(noisy.shape[1]):
            result[x][y] = adaptive_median_A(noisy, filter_size, filter_size_max, x, y)
    return result


def contraharmonic_mean(noisy, filter_order, filter_size):
    # preencher bordas com 0
    result = np.zeros(noisy.shape)
    noisy_pad = np.pad(noisy, [int((filter_size-1) / 2), int((filter_size-1) / 2)], mode='constant')

    for x in range(noisy.shape[0]):
        for y in range(noisy.shape[1]):
            # gerar filtro para o pixel atual
            # (vizinhança do tamanho do filtro)
            neighbors = noisy_pad[x:(x + int(filter_size)), y:(y + int(filter_size))]
            neighbors[neighbors == 0] += 1

            numerator = np.sum(np.power(neighbors, filter_order + 1))
            denominator = np.sum(np.power(neighbors, filter_order))
            result[x][y] = numerator / denominator

    return result


def rmse(predictions, targets):
    error = ((predictions - targets) ** 2)
    error = np.sum(error)
    error *= 1 / (predictions.shape[0] * predictions.shape[1])
    error = np.sqrt(error)
    print("{0:.4f}".format(error), end='')


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
        sigma = float(input())
        restored = adaptive_filtering_local(noisy, sigma, filter_size)
    elif (method == 2):
        filter_size_max = int(input())
        restored = adaptive_median(noisy, filter_size_max, filter_size)
    elif (method == 3):
        filter_order = float(input())
        restored = contraharmonic_mean(noisy, filter_order, filter_size)

    # exibir imagens para comparação
    plot(original, restored)

    # exibir erro entre original e recuperada
    rmse(restored, original)


if __name__ == '__main__':
    main()
