#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 4 - Filtragem 2D
"""

import math
import imageio
import numpy as np
import matplotlib.pyplot as plt


def rmse(predictions, targets):
    error = ((predictions - targets) ** 2)
    error = np.sum(error)
    error *= 1 / (predictions.shape[0] * predictions.shape[1])
    error = np.sqrt(error)
    print("{0:.4f}".format(error), end='')


def normalize(f):
    fmax = np.max(f)
    fmin = np.min(f)
    f = ((f - fmin)/(fmax-fmin)).astype(np.uint8)
    return f


def main():
    # receber parâmetros
    filename_image = str(input()).rstrip()

    # ler imagem original
    # image = imageio.imread("tests/"+filename_image)  # local
    image = imageio.imread(filename_image)  # run.codes

    # visualizar imagens
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(filtered_image, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplots_adjust(wspace=0.5)
    # plt.savefig("images/result_" + str(filter_op) + "_" + str(filter_domain) + ".png")

    # calcular erro entre imagem filtrada e original
    # print("result_{}_{}: ".format(filter_op, filter_domain), end='')
    rmse(filtered_image, image)


if __name__ == '__main__':
    main()
