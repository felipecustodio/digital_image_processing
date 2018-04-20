#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 4 - Filtragem 2D
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt


def arbitrary_filter(image):
    fourier = np.fft.fft2(image)
    pass


def laplacian_of_gaussian(size, sigma):
    log2d = np.zeros((size,size))
    # dividir equações em partes
    log2d_a = (-1) / (np.pi * np.power(sigma, 4))

    # positivos/negativos para normalização
    positive = 0
    negative = 0

    # gerar filtro
    # TODO: mapear o quadrado de lado 10
    for x in range(size):
        for y in range(size):
            log2d_b = 1 - (np.power(x,2) + np.power(y,2) / (2 * np.power(sigma, 2)))
            log2d_c = np.exp((-1)*((np.power(x,2) + np.power(y,2))/(2 * np.power(sigma, 2))))
            log2d[x][y] = log2d_a * log2d_b * log2d_c

    # normalizar
    positive = sum(i for i in log2d if i > 0)
    negative = sum(i for i in log2d if i < 0)
    positive *= (-1)
    normalize = positive/negative
    for x in range(size):
        for y in range(size):
            if (log2d[x][y] < 0):
                log2d[x][y] = log2d[x][y] * normalize
    return log2d


def sobel_operator(image):
    image_x = np.copy(image)
    image_y = np.copy(image)
    image_out = np.zeros((image.shape))

    fx = [[1,0,-1],[2,0,-2],[1,0,-1]]
    fy = [[1,2,1],[0,0,0],[-1,-2,-1]]

    image_x = np.multiply(image_x, fx)
    image_y = np.multiply(image_y), fy

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a = np.power(image_x[i][j], 2)
            b = np.power(image_y[i][j], 2)
            image_out[i][j] = np.sqrt(a + b)

    fourier = np.fft.fft2(image_out)
    return fourier


def cutting(image, hlb, hub, wlb, wub):
    height = image.shape[0]
    width = image.shape[1]
    image_cut1 = image[0:(height/2), 0:(width/2)]

    image_cut1_size = image_cut1.shape[0]
    hlb = hlb * image_cut1_size
    hub = hub * image_cut1_size
    wlb = wlb * image_cut1_size
    wub = wub * image_cut1_size

    image_cut2 = image_cut1[hlb:hub, wlb:wub]
    return image_cut1, image_cut2


def classification():
    pass


def matrix2vector(image):
    return np.asarray(image).reshape(-1)


def vector2matrix(vector, image):
    return np.reshape(vector, image.shape)


def rmse(predictions, targets):
    error = ((predictions - targets) ** 2)
    error = np.sum(error)
    error *= 1 / (predictions.shape[0] * predictions.shape[1])
    error = np.sqrt(error)
    print("{0:.4f}".format(error), end='')


def main():
    # receber parâmetros
    image_filename = str(input()).rstrip()
    image = imageio.imread("tests/"+image_filename)  # local
    # image = imageio.imread(str(input()).rstrip())  # run.codes
    method = int(input())
    if (method == 1):
        line = input().split()
        height = int(line[0])
        width = int(line[1])
        filter = np.zeros((width, height))
        for i in range(height):
            line = input().split()
            for j in range(width):
                filter[i][j] = line[j]
    if (method == 2):
        filter_size = int(input())
        sigma = float(input())
    positions = input().split()
    hlb = positions[0]
    hub = positions[1]
    wlb = positions[2]
    wub = positions[3]

    # run.codes
    # dataset = np.load(str(input()).rstrip())
    # dataset_labels = np.load(str(input()).rstrip())

    # local
    dataset = np.load("tests/"+str(input()).rstrip())
    dataset_labels = np.load("tests/"+str(input()).rstrip())

    filtered_image = np.copy(image)

    # visualizar imagens
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.5)

    # plot something
    figure = plt.gcf()  # get current figure
    # 800 x 600
    figure.set_size_inches(8, 6)
    # save with high DPI
    plt.savefig("plots/" + image_filename, dpi=100)

    # calcular erro entre imagem filtrada e original
    rmse(filtered_image, image)


if __name__ == '__main__':
    try:
        import IPython.core.ultratb
    except ImportError:
        # No IPython. Use default exception printing.
        pass
    else:
        import sys
        sys.excepthook = IPython.core.ultratb.ColorTB()
        main()
