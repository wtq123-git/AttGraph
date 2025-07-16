# def compute_window_nums(ratios, stride, input_size):
#     size = input_size / stride
#     window_nums = []
#
#     for _, ratio in enumerate(ratios):
#         window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))
#
#     return window_nums

import math


def compute_window_nums(ratios, stride, input_size):
    size = input_size / stride
    window_nums = []

    for ratio in ratios:
        # 确保窗口数计算时不丢失精度，向下取整
        width_window_num = math.floor(size - ratio[0] + 1)
        height_window_num = math.floor(size - ratio[1] + 1)

        # 窗口总数 = 宽度窗口数 * 高度窗口数
        window_nums.append(width_window_num * height_window_num)

    return window_nums
