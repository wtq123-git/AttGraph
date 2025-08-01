# import numpy as np
#
# def ComputeCoordinate(image_size, stride, indice, ratio):
#     size = int(image_size / stride)
#     column_window_num = (size - ratio[1]) + 1
#     x_indice = indice // column_window_num
#     y_indice = indice % column_window_num
#     x_lefttop = x_indice * stride - 1
#     y_lefttop = y_indice * stride - 1
#     x_rightlow = x_lefttop + ratio[0] * stride
#     y_rightlow = y_lefttop + ratio[1] * stride
#     # for image
#     if x_lefttop < 0:
#         x_lefttop = 0
#     if y_lefttop < 0:
#         y_lefttop = 0
#     coordinate = np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow)).reshape(1, 4)
#
#     return coordinate
#
#
# def indices2coordinates(indices, stride, image_size, ratio):
#     batch, _ = indices.shape
#     coordinates = []
#
#     for j, indice in enumerate(indices):
#         coordinates.append(ComputeCoordinate(image_size, stride, indice, ratio))
#
#     coordinates = np.array(coordinates).reshape(batch,4).astype(int)       # [N, 4]
#     return coordinates
#

import numpy as np


def ComputeCoordinate(image_size, stride, indice, ratio):
    size = int(image_size / stride)
    column_window_num = (size - ratio[1]) + 1
    x_indice = indice // column_window_num
    y_indice = indice % column_window_num

    # Calculate window coordinates
    x_lefttop = x_indice * stride
    y_lefttop = y_indice * stride
    x_rightlow = x_lefttop + ratio[0] * stride
    y_rightlow = y_lefttop + ratio[1] * stride

    # Ensure the coordinates don't go out of bounds
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    if x_rightlow > image_size:
        x_rightlow = image_size
    if y_rightlow > image_size:
        y_rightlow = image_size

    coordinate = np.array([x_lefttop, y_lefttop, x_rightlow, y_rightlow]).reshape(1, 4).astype(int)
    return coordinate


def indices2coordinates(indices, stride, image_size, ratio):
    batch, _ = indices.shape
    coordinates = []

    for j in range(batch):
        # Ensure `indice` is a scalar and pass it to `ComputeCoordinate`
        indice = indices[j, 0]  # indices[j] is an array, we want the scalar value
        coordinates.append(ComputeCoordinate(image_size, stride, indice, ratio))

    coordinates = np.concatenate(coordinates, axis=0)  # Stack the coordinates along the batch axis
    return coordinates
