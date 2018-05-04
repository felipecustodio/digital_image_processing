#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0251 - Processamento de Imagens - 2018/1
ICMC - USP
Professor Moacir Ponti
Aluno: Felipe Scrochio Custódio - 9442688
Trabalho 6 - Denoising
"""

import imageio
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception as e:
    pass


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

    # exibir erro entre original e recuperada
    rmse(recovered, original)


if __name__ == '__main__':
    main()
