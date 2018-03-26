# superresolução:
# L1[0,0] L3[0,0] L1[0,1] L3[0,1]
# L2[0,0] L4[0,0] L2[0,1] L4[0,1]
# L1[1,0] L3[1,0] L1[1,1] L3[1,1]
# L2[1,0] L4[1,0] L2[1,1] L4[1,1]
#
#
# L1[] L3[] (...)
# L2[] L4[] (...)
# L1[] L3[] (...)
# L2[] L4[] (...)
# (...)
# Superresolution algorithm sandbox

import numpy as np

size = 4
image1 = np.ones((size, size))
image2 = np.ones((size, size)) * 2
image3 = np.ones((size, size)) * 3
image4 = np.ones((size, size)) * 4

super_image = np.zeros((size*2, size*2))

current_row = 0
current_col = 0
while current_row <= size * 2 - 2:
    while current_col <= size * 2 - 2:
        quadrant = super_image[current_row:current_row + size, current_col:current_col + size]
        quadrant[0, 0] = image1[current_row % size, current_col % size]
        quadrant[0, 1] = image3[current_row % size, current_col % size]
        quadrant[1, 0] = image2[current_row % size, current_col % size]
        quadrant[1, 1] = image4[current_row % size, current_col % size]
        print(current_row / 2, current_col / 2)
        current_col += 2
    current_row += 2
    current_col = 0

print(super_image)
