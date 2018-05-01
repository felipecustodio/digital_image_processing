#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 4 - Inpainting usando FFTs
"""

import imageio
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception as e:
    pass


def gerchberg_papoulis(image, mask, T):
    # obter fourier da máscara
    M = np.fft.fft2(mask)
    # encontrar magnitude máxima (máscara)
    magnitude_mask = np.real(np.max(M))
    # filtro de média 7x7
    mean_filter = np.zeros((49), dtype=float)
    # Gk[0] = imagem deteriorada
    # Gk[1...T] = iterações do algoritmo
    g = {}
    g[0] = np.copy(image)
    # gerchberg-papoulis
    for k in range(1, T+1):
        # obter fourier da imagem Gk
        g[k] = np.fft.fft2(g[k-1])
        # encontrar magnitude máxima (gk)
        magnitude_gk = np.real(np.max(g[k]))
        # filtrar (limitar frequências)
        for x in range(g[k].shape[0]):
            for y in range(g[k].shape[1]):
                if (g[k][x][y] >= (0.9 * magnitude_mask)):
                    g[k][x][y] = 0
                if (g[k][x][y] <= (0.01 * magnitude_gk)):
                    g[k][x][y] = 0
        # obter transformada inversa
        g[k] = np.real(np.fft.ifft2(g[k]))
        # convolução com filtro de média
        for x in range(g[k].shape[0]):
            for y in range(g[k].shape[1]):
                filter_pos = 0
                # gerar filtro de média para posição (x,y)
                for i in range(x-3, (x+4) % g[k].shape[0]):
                    for j in range(y-3, (y+4) % g[k].shape[1]):
                        mean_filter[filter_pos] = (g[k][i][j])
                        filter_pos += 1
                # encontrar média e atribuir ao pixel atual
                g[k][x][y] = np.sum(mean_filter) / 49
        # renormalizar
        g[k] = normalize(g[k])
        # inserir pixels na estimativa k
        g[k] = np.multiply((1-(mask/255)), g[0]) + np.multiply((mask/255), g[k])
    # retornar imagem recuperada (última iteração)
    return g[T]


def normalize(f):
    fmax = np.max(f)
    fmin = np.min(f)
    f = (f-fmin)/(fmax-fmin)
    f = (f*255).astype(np.uint8)
    return f


def rmse(predictions, targets):
    error = ((predictions - targets) ** 2)
    error = np.sum(error)
    error *= 1 / (predictions.shape[0] * predictions.shape[1])
    error = np.sqrt(error)
    print("{0:.5f}".format(error), end='')


def main():
    # receber parâmetros
    original_filename = str(input()).rstrip()
    try:
        # run.codes
        original = imageio.imread(original_filename)
    except Exception as e:
        # local
        original = imageio.imread("tests/"+original_filename)

    deteriorated_filename = str(input()).rstrip()
    try:
        # run.codes
        deteriorated = imageio.imread(deteriorated_filename)
    except Exception as e:
        # local
        deteriorated = imageio.imread("tests/"+deteriorated_filename)

    mask_filename = str(input()).rstrip()
    try:
        # run.codes
        mask = imageio.imread(mask_filename)
    except Exception as e:
        # local
        mask = imageio.imread("tests/"+mask_filename)
    iterations = int(input())

    # algoritmo de inpainting
    recovered = gerchberg_papoulis(deteriorated, mask, iterations)

    # visualizar imagens
    try:
        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(deteriorated, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.5)

        plt.subplot(1, 3, 3)
        plt.imshow(recovered, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.5)

        plt.show()
    except Exception as e:
        pass

    # exibir erro entre original e recuperada
    rmse(recovered, original)


if __name__ == '__main__':
    main()
