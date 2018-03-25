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
# import matplotlib.pyplot as plt


def f1(f_image, C):
    for x in range(C):
        for y in range(C):
            f_image[x, y] = x + y
    # normalizar
    f_image = normalize(f_image)
    return f_image


def f2(f_image, C, Q):
    for x in range(C):
        for y in range(C):
            f_image[x, y] = abs(math.sin(x/Q) + math.sin(y/Q))
    # normalizar
    f_image = normalize(f_image)
    return f_image


def f3(f_image, C, Q):
    for x in range(C):
        for y in range(C):
            f_image[x, y] = (x/Q - math.sqrt(y/Q))
    # normalizar
    f_image = normalize(f_image)
    return f_image


def f4(f_image, C, S):
    random.seed(S)
    for x in range(C):
        for y in range(C):
            f_image[x, y] = random.random()
    # normalizar
    f_image = normalize(f_image)
    return f_image


def f5(f_image, C, S):
    random.seed(S)
    x = 0
    y = 0
    f_image[x, y] = 1
    for i in range(int(1 + (C*C/2))):
        # passo em x
        dx = random.randint(-1, 1)
        x = ((x + dx) % C)
        f_image[x, y] = 1
        # passo em y
        dy = random.randint(-1, 1)
        y = ((y + dy) % C)
        f_image[x, y] = 1
    # normalizar
    fmax = np.max(f_image)
    fmin = np.min(f_image)
    f_image = (f_image - fmin)/(fmax-fmin)
    f_image = (f_image*65535).astype(np.uint8)
    return f_image


def g(f_image, N, C, B):
    g_image = np.zeros([N, N])
    d = int(C/N)  # quantidade relativa de pixels
    # operação de máximo local
    for x in range(N):
        for y in range(N):
            sub = submatrix(f_image, x, y, d)
            g_image[x, y] = max(sub)
    # normalizar para inteiro de 8 bits
    g_image = normalize(g_image)
    # B bits menos significativos
    g_image << B
    return g_image


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
    # carregar imagem de referência
    filename = str(input()).rstrip()
    R = np.load(filename)  # (Run.Codes)
    # R = np.load("examples/"+filename)  # (Repo)

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
    g_image = g(f_image, N, C, B)

    # plt.style.use('Solarize_Light2')
    # # visualizar imagens
    # plt.subplot(1, 2, 1)
    # plt.title("Cena (F{})".format(f))
    # plt.imshow(f_image, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title("Imagem digital (G)")
    # plt.imshow(g_image, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplots_adjust(wspace=0.5)
    # plt.suptitle('Trabalho 1', fontsize=16)
    # plt.show()

    # erro médio quadrático entre g e a referência
    rmse(g_image, R)


if __name__ == '__main__':
    main()
