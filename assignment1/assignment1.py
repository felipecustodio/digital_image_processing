#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 1 - Gerador de Imagens
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt


def f1(f_image, C):
    for x in range(C):
        for y in range(C):
            f_image[x, y] = x + y
    return f_image


def f2(f_image, C, Q):
    for x in range(C):
        for y in range(C):
            f_image[x, y] = abs(math.sin(x/Q) + math.sin(y/Q))
    return f_image


def f3(f_image, C, Q):
    for x in range(C):
        for y in range(C):
            f_image[x, y] = (x/Q - math.sqrt(y/Q))
    return f_image


def f4(f_image, C, S):
    random.seed(S)
    for x in range(C):
        for y in range(C):
            f_image[x, y] = random.random()
    # normalizar
    return f_image


def f5(f_image, C, S):
    random.seed(S)
    x = 0
    y = 0
    f_image[x, y] = 1
    # TODO
    for i in range(int(1 + (C**2/2))):
        # passo em x
        dx = random.randint(-1, 1)
        x = ((x + dx) % C)
        f_image[x, y] = 1
        # passo em y
        dy = random.randint(-1, 1)
        y = ((y + dy) % C)
        f_image[x, y] = 1
    # normalizar
    f_image = (f_image*65535).astype(np.uint8)
    return f_image


def g(N, C, B):
    # TODO
    g_image = np.zeros([N, N])
    # d = C/N
    for x in range(N):
        for y in range(N):
            # g_image[x, y] = max(f_image[:, :])
            pass
    # B bits menos significativos
    return g_image


def rmse(predictions, targets):
    print(np.sqrt(((predictions - targets) ** 2).mean()))


def main():
    # carregar imagem de referência
    filename = str(input()).rstrip()
    R = np.load(filename)

    # parâmetros
    C = int(input())  # tamanho lateral da cena
    f = int(input())  # função a ser utilizada
    Q = int(input())  # parâmetro Q
    N = int(input())  # tamanho lateral da imagem digital
    B = int(input())  # bits por pixel
    S = int(input())  # semente para random

    # gerar cena a partir de f
    f_image = np.zeros([C, C])  # inicializar cena
    if f == 1:
        f_image = f1(f_image, C)
    elif f == 2:
        f_image = f2(f_image, C, Q)
    elif f == 3:
        f_image = f3(f_image, C, Q)
    elif f == 4:
        f_image = f4(f_image, C, S)
    elif f == 5:
        f_image = f5(f_image, C, S)

    # gerar imagem digital g
    g_image = g(N, C, B)

    # visualizar imagens
    plt.subplot(1, 2, 1)
    plt.title("Cena (F{})".format(f))
    plt.imshow(f_image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Imagem digital (G)")
    plt.imshow(g_image, cmap='gray')

    plt.subplots_adjust(wspace=0.5)
    plt.suptitle('Trabalho 1', fontsize=16)
    # plt.show()

    # erro médio quadrático entre g e a referência
    rmse(g_image, R)


if __name__ == '__main__':
    main()
