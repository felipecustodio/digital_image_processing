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


def arbitrary_filter_spatial(image, filter_vector):
    # realizar convolução no domínio espacial
    # utilizando filtro pré-definido
    new_image = spatial_convolution(image, filter_vector)
    return new_image


def arbitrary_filter_frequency(image, filter_vector):
    # realizar convolução no domínio das frequências
    # utilizando filtro pré-definido
    new_image = frequency_convolution(image, filter_vector)
    return new_image


def gaussian_filter_spatial(image, size, delta):
    # gerar filtro gaussiano baseado nas posições
    filter_vector = np.zeros(size)
    center = int(size/2)
    for i in range(size):
        filter_vector[i] = gaussian_equation((i-center), delta)
    # normalizar filtro gaussiano
    filter_vector = normalize(filter_vector)
    # realizar convolução no domínio espacial
    # utilizando filtro gaussiano
    new_image = spatial_convolution(image, filter_vector)
    return new_image


def gaussian_filter_frequency(image, size, delta):
    # gerar filtro gaussiano baseado nas posições
    filter_vector = np.zeros(size)
    center = int(size/2)
    for i in range(size):
        filter_vector[i] = gaussian_equation((i-center), delta)
    # normalizar filtro gaussiano
    filter_vector = normalize(filter_vector)
    # realizar convolução no domínio das frequências
    # utilizando filtro gaussiano
    new_image = frequency_convolution(image, filter_vector)
    return new_image


def spatial_convolution(image, filter_vector):
    new_image = np.copy(image)
    filter_size = len(filter_vector)
    # inverter filtro para convolução
    filter_vector = filter_vector[::-1]
    # descobrir centro do filtro para posicionar
    center = int(np.floor(filter_size/2))
    # aplicar convolução para cada posição do vetor
    for index, position in enumerate(image):
        conv_mult = np.zeros(filter_size)
        # multiplicar valores do vetor pelo filtro
        # percorre do meio ao começo
        for i in range(index, index - center, -1):
            conv_mult[i-index+(filter_size-1)] = image[i] * filter_vector[i-index+(filter_size-1)]
        # percorre do meio ao fim
        for i in range(index + 1, index + center):
            conv_mult[(i-index+(filter_size-1)) % (filter_size-1)] = image[i % (len(image) - 1)] * filter_vector[(i-index+(filter_size-1)) % (filter_size-1)]
        # novo valor da posição = soma dos resultados da multiplicação
        new_image[index] = np.sum(conv_mult)
    # retornar imagem filtrada
    return new_image


def frequency_convolution(image, filter_vector):
    # calcular tamanhos
    image_size = len(image)
    filter_size = len(filter_vector)

    # gerar filtro com tamanho correto
    # preenchido com 0
    new_filter = np.zeros(image.shape[0])
    for i in range(filter_size):
        new_filter[i] = filter_vector[i]

    # ir p/ dominio das frequencias
    image_fourier = DFT_1D(image)
    filter_fourier = DFT_1D(new_filter)

    # convolução
    new_image = np.multiply(image_fourier, filter_fourier)
    # inverter transformada (voltar p/ domínio espacial)
    new_image = np.real(invDFT_1D(new_image))
    return new_image


def gaussian_equation(x, delta):
    return (1/np.sqrt(2*np.pi)*delta)*(math.exp((-(x**2)/2*(delta**2))))


def DFT_1D(image):
    # inicializar com 0
    fourier = np.empty_like(image).astype(np.complex64)
    # fourier = np.zeros(image, dtype=np.complex64)
    n = (np.asarray(image)).shape[0]
    # para cada frequencia
    for u in np.arange(n):
        # para cada elemento
        for x in range(n):
            fourier[u] += image[x] * np.exp((-1j * 2 * np.pi * u * x) / n)
    return fourier


def invDFT_1D(fourier):
    # armazena a transformada
    # A = np.zeros(fourier, dtype=np.float32)
    fourier = np.empty_like(fourier).astype(np.complex64)

    n = fourier.shape[0]
    # criar indices para 'u' (cada frequencia)
    u = np.arange(n)
    # para cada valor do vetor (x)
    for x in np.arange(n):
        fourier[x] = (np.real(np.sum(np.multiply(fourier, np.exp((1j * 2 * np.pi * u * x) / n))))) / n
    return fourier


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
    filter_op = int(input())
    filter_size = int(input())
    if (filter_op == 1):
        filter_vector = []
        weights = input()
        weights = weights.split()
        for i in range(filter_size):
            filter_vector.append(float(weights[i]))
    else:
        filter_delta = float(input())
    filter_domain = int(input())

    # ler imagem original
    image = imageio.imread("tests/"+filename_image)  # local
    # image = imageio.imread(filename_image)  # run.codes

    # formatar matriz da imagem como vetor
    vector = np.asarray(image).reshape(-1)

    # apply filter
    if (filter_op == 1):
        if (filter_domain == 1):
            vector = arbitrary_filter_spatial(vector, filter_vector)
        elif (filter_domain == 2):
            vector = arbitrary_filter_frequency(vector, filter_vector)
    elif (filter_op == 2):
        if (filter_domain == 1):
            vector = gaussian_filter_spatial(vector, filter_size, filter_delta)
        elif (filter_domain == 2):
            vector = gaussian_filter_frequency(vector, filter_size, filter_delta)

    # transformar vetor de volta em matriz
    filtered_image = np.reshape(vector, image.shape)

    # visualizar imagens
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.savefig("images/result_" + str(filter_op) + "_" + str(filter_domain) + ".png")

    # calcular erro entre imagem filtrada e original
    print("result_{}_{}: ".format(filter_op, filter_domain), end='')
    rmse(filtered_image, image)


if __name__ == '__main__':
    main()
