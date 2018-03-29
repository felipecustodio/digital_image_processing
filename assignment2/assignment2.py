#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 1 - Gerador de Imagens
"""

import math
import imageio
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def histogram_individual_transfer(image):
    # inicializar imagem equalizada
    equalized = np.zeros(image.shape).astype(np.uint8)
    # histograma de tons de cinza da imagem original
    histogram, bins = np.histogram(image, range(256))
    # calcular histograma acumulado
    cumulative_hist = np.zeros((histogram.shape[0])).astype(int)
    cumulative_hist[0] = histogram[0]
    for i in range(1, 255):
        cumulative_hist[i] = histogram[i] + cumulative_hist[i-1]
    # aplicar equalização de histograma
    for value in range(255):
        # calcular novo valor
        new_value = ((255)/float(image.shape[0] *  image.shape[1])) * cumulative_hist[value]
        # substituir todos os pixels com valores 'value'
        # pela sua frequência no histograma acumulado
        equalized[np.where(image == value)] = new_value
    return equalized


def histogram_joint_transfer(images):
    # inicializar lista com as novas imagens equalizadas
    equalized = []
    for image in images:
        equalized.append(np.zeros((image.shape)))

    # inicializar histograma acumulado único
    cumulative_hist = np.zeros(256).astype(int)
    for image in images:
        histogram, bins =  np.histogram(image, range(256))
        # acumular valores do histograma atual no
        # histograma acumulado único
        for value, frequency in enumerate(histogram):
            cumulative_hist[value] += frequency

    """
    Aplicar equalização utilizando histograma ac. único
    assumindo que todas as imagens de baixa resolução
    possuem o mesmo tamanho
    """
    for value in range(255):
        new_value = ((255)/float(images[0].shape[0] * images[0].shape[1])) * cumulative_hist[value]
        for index, image in enumerate(images):
            equalized[index][np.where(image == value)] = new_value

    return equalized


def gamma_adjust(image, gamma):
    new_image = np.copy(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # aplicar ajuste gamma
            new_image[x, y] = math.floor(255 * (np.power((image[x, y]/255.0), (1/gamma))))
    return new_image


def superresolution(images):
    # dimensões da imagem (low)
    size = images[0].shape[0]

    # inicializar imagem (high)
    super_image = np.zeros((size*2, size*2))

    """
    Queremos andar na matriz (nova) com passos de 2x2 pixels
    cada pixel da nova imagem equivale a um quadrante
    formado pelas 4 imagens de baixa resolução.
    Para não sair da matriz, voltamos para o início
    do último quadrante.
    Como em Python fazer uma atribuição array = outra_array
    significa criar um ponteiro, podemos editar o quadrante
    separadamente, garantindo que a imagem nova terá suas
    posições corretas editadas.
    """
    current_row = 0
    current_col = 0
    while current_row <= size * 2 - 2:
        while current_col <= size * 2 - 2:
            # cortar nova imagem em quadrante 2x2
            quadrant = super_image[current_row:current_row + size, current_col:current_col + size]
            # atribuir valores das imagens de baixa resolução
            quadrant[0, 0] = images[0][int(current_row / 2), int(current_col / 2)]
            quadrant[0, 1] = images[2][int(current_row / 2), int(current_col / 2)]
            quadrant[1, 0] = images[1][int(current_row / 2), int(current_col / 2)]
            quadrant[1, 1] = images[3][int(current_row / 2), int(current_col / 2)]
            # andar para o próximo quadrante
            current_col += 2
        current_row += 2
        current_col = 0
    return super_image


def rmse(predictions, targets):
    error = ((predictions - targets) ** 2)
    error = np.sum(error)
    error *= 1 / (predictions.shape[0] * predictions.shape[1])
    error = np.sqrt(error)
    print("{0:.4f}".format(error), end='')


def normalize(f):
    fmax = np.max(f)
    fmin = np.min(f)
    f = (f - fmin)/(fmax-fmin)
    f = (f*255).astype(np.uint8)
    return f


def main():
    # input
    filename_low = str(input()).rstrip()
    filename_high = str(input()).rstrip()
    method = int(input())
    gamma = float(input())

    imglow = []
    # ler 4 imagens de baixa resolução
    imglow.append(imageio.imread("tests/"+filename_low+"1.png"))
    imglow.append(imageio.imread("tests/"+filename_low+"2.png"))
    imglow.append(imageio.imread("tests/"+filename_low+"3.png"))
    imglow.append(imageio.imread("tests/"+filename_low+"4.png"))
    # carregar imagem de alta resolução
    imghigh_ref = imageio.imread("tests/"+filename_high+".png")

    # equalizar imagens de baixa resolução
    equalized = []
    for image in imglow:
        current = np.copy(image)
        # realce
        if method != 0:
            # não utilizar realce se opção 0
            if method == 1:
                # função de transferência individual
                current = histogram_individual_transfer(current)
                pass
            elif method == 3:
                # método de realce gamma
                current = gamma_adjust(image, gamma)
        current = normalize(current) # normalizar (0-255)
        equalized.append(current)

    # função de transferência conjunta
    if (method == 2):
        equalized = histogram_joint_transfer(imglow)

    # superresolução
    imghigh = superresolution(equalized)

    # comparar com imagem final
    sns.set()
    plt.figure(figsize=(100, 100))

    plt.subplot(121)
    plt.imshow(imghigh, cmap="gray")
    plt.colorbar()
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(imghigh_ref, cmap="gray")
    plt.colorbar()
    plt.axis('off')
    # salvar como imagem + método usado
    plt.savefig("images/"+filename_high+"_"+str(method)+".png")

    # erro médio quadrático entre superres e a referência
    rmse(imghigh, imghigh_ref)


if __name__ == '__main__':
    main()
