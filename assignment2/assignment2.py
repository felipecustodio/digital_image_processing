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
import matplotlib.pyplot as plt


def histogram_individual_transfer(image):
    # histograma acumulado
    return image


def histogram_joint_transfer(image):
    # histograma acumulado único
    return image


def gamma_adjust(image, gamma):
    new_image = np.copy(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # aplicar ajuste gamma
            new_image[x, y] = math.floor(255*((image[x, y]/255.0)**1/gamma))
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
            quadrant[0, 0] = images[0][current_row % size, current_col % size]
            quadrant[0, 1] = images[2][current_row % size, current_col % size]
            quadrant[1, 0] = images[1][current_row % size, current_col % size]
            quadrant[1, 1] = images[3][current_row % size, current_col % size]
            # andar para o próximo quadrante
            current_col += 2
        current_row += 2
        current_col = 0

    return super_image


def rmse(predictions, targets):
    error = (np.sqrt(((predictions - targets) ** 2).mean()))
    print("{0:.4f}".format(error), end='')


def submatrix(f, x, y, i):
    sub = []
    for row in range(x-i, x+i):
        for col in range(y-i, y+i):
            sub.append(f[row, col])
    return sub


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
                pass
            elif method == 2:
                current = histogram_joint_transfer(image)
                # função de transferência conjunta
                pass
            elif method == 3:
                current = gamma_adjust(image, gamma)
        equalized.append(current)

    # superresolução
    imghigh = superresolution(equalized)
    print(type(imghigh))

    # exibir imagem final
    plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.imshow(imghigh, cmap="gray")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(imghigh_ref, cmap="gray")
    plt.axis('off')
    plt.show()

    # erro médio quadrático entre superres e a referência
    rmse(imghigh, imghigh_ref)


if __name__ == '__main__':
    main()
