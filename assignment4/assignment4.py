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
try:
    import matplotlib.pyplot as plt
except Exception as e:
    pass


def frequency_convolution(image, filter_matrix):
    # deixar filtro do tamanho da imagem
    # preenchido com 0s
    new_filter = np.zeros((image.shape))
    for i in range(filter_matrix.shape[0]):
        for j in range(filter_matrix.shape[1]):
            new_filter[i][j] = filter_matrix[i][j]

    image_fourier = np.real(np.fft.fft2(image))
    filter_fourier = np.real(np.fft.fft2(new_filter))

    for i in range(new_filter.shape[0]):
        for j in range(new_filter.shape[1]):
            filter_fourier[i][j] = new_filter[i][j]

    image_out = np.multiply(image_fourier, filter_fourier)
    return image_out


def laplacian_of_gaussian(size, sigma):
    log2d = np.zeros((size, size))
    # positivos/negativos para normalização
    positive = 0
    negative = 0

    # mapear o quadrado de lado 10 com o tamanho do filtro
    indices = np.linspace(-5, 5, num=size, dtype=float)

    # gerar filtro
    # dividir equações em partes
    log2d_a = (-1) / (np.pi * np.power(sigma, 4))
    for x in range(size):
        for y in range(size):
            index_x = indices[y]
            index_y = indices[size-x-1]
            log2d_b = 1 - (np.power(index_x,2) + np.power(index_y,2) / (2 * np.power(sigma, 2)))
            log2d_c = np.exp((-1)*((np.power(index_x,2) + np.power(index_y,2))/(2 * np.power(sigma, 2))))
            log2d[x][y] = log2d_a * log2d_b * log2d_c
            if (log2d[x][y] > 0):
                positive += log2d[x][y]
            else:
                negative += log2d[x][y]

    # normalizar
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

    # valores do filtro
    fx_values = np.asarray(([[1,0,-1],[2,0,-2],[1,0,-1]]),dtype=float)
    fy_values = np.asarray(([[1,2,1],[0,0,0],[-1,-2,-1]]),dtype=float)

    # preencher filtro com 0
    fx = np.zeros((image.shape))
    fy = np.zeros((image.shape))
    for i in range(fx_values.shape[0]):
        for j in range(fx_values.shape[0]):
            fx[i][j] = fx_values[i][j]
            fy[i][j] = fy_values[i][j]

    # multiplicar por filtros
    image_x = np.multiply(image_x, fx)
    image_y = np.multiply(image_y, fy)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a = np.power(image_x[i][j], 2)
            b = np.power(image_y[i][j], 2)
            image_out[i][j] = np.sqrt(a + b)

    #  realizar transformada de fourier
    fourier = np.real(np.fft.fft2(image_out))
    return fourier


def cutting(image, hlb, hub, wlb, wub):
    height = int(image.shape[0])
    width = int(image.shape[1])
    image_cut1 = image[0:int(height/2), 0:int(width/2)]

    image_cut1_size = image_cut1.shape[0]
    hlb = int(hlb * image_cut1_size)
    hub = int(hub * image_cut1_size)
    wlb = int(wlb * image_cut1_size)
    wub = int(wub * image_cut1_size)

    image_cut2 = image_cut1[hlb:hub, wlb:wub]

    return image_cut2


def classification(image, dataset, labels):
    vector = np.real(matrix2vector(image))
    min_distance = math.inf
    min_index = -1
    # percorrer linhas do dataset
    for index, sample in enumerate(dataset):
        sample = np.real(sample)
        # calcular distância euclidiana entre
        # vetor (imagem cortada) e amostra do dataset
        euclidian = np.linalg.norm(vector-sample)
        if (euclidian < min_distance):
            min_distance = euclidian
            min_index = index
    min_label = labels[min_index]
    print(min_label)
    print(min_index)


def matrix2vector(image):
    vector = np.asarray(image).reshape(-1)
    return vector


def vector2matrix(vector, image):
    matrix = np.reshape(vector, image.shape)
    return matrix


def main():
    # receber parâmetros
    image_filename = str(input()).rstrip()
    try:
        # run.codes
        image = imageio.imread(image_filename)
    except Exception as e:
        image = imageio.imread("tests/"+image_filename)

    method = int(input())
    if (method == 1):
        line = input().split()
        height = int(line[0])
        width = int(line[1])
        arbitrary_filter = np.zeros((width, height))
        for i in range(height):
            line = input().split()
            for j in range(width):
                arbitrary_filter[i][j] = float(line[j])
    if (method == 2):
        filter_size = int(input())
        sigma = float(input())
    positions = input().split()
    hlb = float(positions[0])
    hub = float(positions[1])
    wlb = float(positions[2])
    wub = float(positions[3])

    dataset_filename = str(input()).rstrip()
    labels_filename = str(input()).rstrip()
    try:
        # run.codes
        dataset = np.load(dataset_filename)
        labels = np.load(labels_filename)
    except Exception as e:
        # local
        dataset = np.load("tests/"+dataset_filename)
        labels = np.load("tests/"+labels_filename)

    filtered_image = np.copy(image)

    # aplicar filtro
    if (method == 1):
        filtered_image = frequency_convolution(image, arbitrary_filter)
    elif (method == 2):
        log2d = laplacian_of_gaussian(filter_size, sigma)
        filtered_image = frequency_convolution(image, log2d)
    elif (method == 3):
        filtered_image = sobel_operator(image)

    # extrair características com corte
    cut_image = cutting(filtered_image, hlb, hub, wlb, wub)

    # classificar com 1-NN
    classification(cut_image, dataset, labels)

    # visualizar imagens
    try:
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(filtered_image, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.5)

        plt.subplot(1, 3, 3)
        plt.imshow(cut_image, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.5)

        figure = plt.gcf()  # get current figure
        # 800 x 600
        figure.set_size_inches(8, 6)
        # salvar com DPI alto
        plt.savefig("plots/" + image_filename, dpi=100)
    except Exception as e:
        pass


if __name__ == '__main__':
    main()
