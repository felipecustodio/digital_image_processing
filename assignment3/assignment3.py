#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 3 - Filtragem 1D
"""

import math
import imageio
import numpy as np
import matplotlib.pyplot as plt


def arbitrary_filter_space(image, filter_vector):
    new_image = np.copy(image)
    filter_size = len(filter_vector)
    # inverter filtro para convolução
    inv_filter = np.flip(filter_vector)
    # aplicar convolução para cada posição do vetor
    for index, position in enumerate(image):
        conv_mult = np.zeros(filter_size)
        # descobrir centro do filtro para posicionar
        if filter_size % 2 == 0:
            center = int(filter_size/2)
        else:
            center = int((filter_size-1/2)-1)
        # multiplicar valores do vetor pelo filtro
        # filtro posicionado com image wrap


        conv_sum = np.sum(conv_mult)
        new_image[index] = conv_sum
    # retornar imagem filtrada
    return new_image


def arbitrary_filter_frequency(image, filter_vector):
    pass


def gaussian_equation(x, delta):
    return (1/np.sqrt(2*np.pi)*delta)*(math.exp((-(x**2)/2*(delta**2)))


def gaussian_filter_space(image, size, delta):
    # gerar filtro gaussiano baseado nas posições
    filter_vector = np.zeros(size)
    center = int(size/2)
    for i in range(size):
        filter_vector[i] = gaussian_equation((i-center), delta)
    # normalizar filtro gaussiano
    filter_vector = normalize_1(filter_vector)


def gaussian_filter_frequency(image, size, delta):
    pass


def DFT_1D(image):
    # inicializar com 0
    fourier = np.zeros(image.shape, dtype=np.complex64)
    n = image.shape[0]
    # para cada frequencia
    for u in np.arange(n):
        # para cada elemento
        for x in range(n):
            fourier[u] += image[x] * np.exp((-1j * 2 * np.pi * u * x) / n)
    return fourier


def invDFT_1D(fourier):
    # armazena a transformada
    A = np.zeros(fourier.shape, dtype=np.float32)
    n = fourier.shape[0]
    # criar indices para 'u' (cada frequencia)
    u = np.arange(n)
    # para cada valor do vetor (x)
    for x in np.arange(n):
        A[x] = np.real(np.sum(np.multiply(fourier, np.exp((1j*2* np.pi * u * x)/n))))

    return A/n


def rmse(predictions, targets):
    error = ((predictions - targets) ** 2)
    error = np.sum(error)
    error *= 1 / (predictions.shape[0] * predictions.shape[1])
    error = np.sqrt(error)
    print("{0:.4f}".format(error), end='')


def normalize_1(f):
    fmax = np.max(f)
    fmin = np.min(f)
    f = (f - fmin)/(fmax-fmin)
    f = (f*1).astype(np.uint8)
    return f


def normalize_255(f):
    fmax = np.max(f)
    fmin = np.min(f)
    f = (f - fmin)/(fmax-fmin)
    f = (f*255).astype(np.uint8)
    return f


def main():
    # get images and parameters
    filename_image = str(input()).rstrip()
    filter_op = int(input())
    filter_size = int(input())
    if (filter_op == 1):
        filter_vector = []
        weights = input()
        for i in range(filter_size):
            filter_vector.append(weights[i])
    else:
        filter_delta = float(input())
    filter_domain = int(input())

    image = imageio.imread("tests/"+filename_image)  # local
    # image = imageio.imread(filename_image)  # run.codes

    # format image as vector
    vector = np.asarray(image).reshape(-1)

    # apply filter
    if (filter_op == 1):
        if (filter_domain == 1):
            vector = arbitrary_filter_space(vector, filter_vector)
        elif (filter_domain == 2):
            vector = arbitrary_filter_frequency(vector, filter_vector)
    elif (filter_op == 2):
        if (filter_domain == 1):
            vector = gaussian_filter_space(vector, filter_size, filter_delta)
        elif (filter_domain == 2):
            vector = gaussian_filter_frequency(vector, filter_size, filter_delta)

    # format vector back as image
    filtered_image = np.reshape(vector, image.shape)

    # calculate error between generated image and original image
    rmse(filtered_image, image)


if __name__ == '__main__':
    main()
