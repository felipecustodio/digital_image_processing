#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 1 - Gerador de Imagens
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt


def histogram_individual_transfer(image):
    # histograma acumulado
    # equalizar imagem
    return image


def histogram_joint_transfer(image):
    # histograma acumulado único
    # equalizar todas as imagens i
    return image


def gamma_adjust(image):
    return image


def superresolution(image):
    return image


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
    gamma = int(input())

    # ler 4 imagens de baixa resolução
    # TODO transformar para vetor de imagens
    imglow1 = imageio.imread("tests/"+filename_low+"1.png")
    imglow2 = imageio.imread("tests/"+filename_low+"2.png")
    imglow3 = imageio.imread("tests/"+filename_low+"3.png")
    imglow4 = imageio.imread("tests/"+filename_low+"4.png")
    # carregar imagem de alta resolução
    imghigh = imageio.imread("tests/"+filename_high+".png")  # (Repo)

    # realce
    if method != 0:
        # gamma
        if method == 1:
            # histogram_individual_transfer(image)
            pass
        elif method == 2:
            pass
        elif method == 3:
            pass

    # superresolução
    new_high = superresolution(imglow1)

    # imglow = np.load(filename_low)  # (run.codes)
    # imghigh = np.load(filename_high)  # (run.codes)

    # exibir imagens
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(imglow1, cmap="gray")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(imglow2, cmap="gray")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(imglow3, cmap="gray")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(imglow4, cmap="gray")
    plt.axis('off')
    plt.show()


    # erro médio quadrático entre g e a referência
    rmse(imghigh, imghigh)


if __name__ == '__main__':
    main()
