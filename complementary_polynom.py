from copy import deepcopy
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


DELIMITER = '#' * 180


def quadratic_equation_roots(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]

    print('a', a, 'b', b, 'c', c)
    discriminant = b * b - 4 * a * c
    print("discriminant", discriminant)

    # if discriminant < 0:
    #     return []

    # discriminant_sqrt = sqrt(discriminant)
    discriminant_sqrt = np.emath.sqrt(discriminant)

    x_1 = ((-1) * b - discriminant_sqrt) / (2 * a)
    x_2 = ((-1) * b + discriminant_sqrt) / (2 * a)

    return [x_1, x_2]


def qubic_equation_roots(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]

    print('a', a, 'b', b, 'c', c, 'd', d)
    result = []

    # Cardano's formula
    p = (3 * a * c - b * b) / (3 * a * a)
    q = (27 * a * a * d - 9 * a * b * c + 2 * pow(b, 3)) / (27 * pow(a, 3))
    print('p', p, 'q', q)

    Q = pow(p/3, 3) + pow(q/2, 2)
    print("Q", Q)

    alpha = pow((-0.5 * q + np.emath.sqrt(Q)), 1/3)
    beta = pow((-0.5 * q - np.emath.sqrt(Q)), 1/3)

    y_1 = alpha + beta
    y_2 = -0.5 * (alpha + beta) + np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)
    y_3 = -0.5 * (alpha + beta) - np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)

    for y in [y_1, y_2, y_3]:
        x = y - b / (3 * a)
        result.append(x)

    return result


def polynome(coeffs, x):
    sum = 0
    power_biggest = len(coeffs) - 1

    for i in range(len(coeffs)):
        # print(i)
        sum += (coeffs[i] * pow(x, power_biggest - i))

    return sum


def build_data(x_start, x_end, x_step, coeffs):
    data = [[], []]
    x = x_start

    while x <= x_end:
        data[0].append(x)
        data[1].append(polynome(coeffs, x))
        x += x_step

    return data


def build_graphs(x_start, x_end, x_step, coeff_arrays):
    for array in coeff_arrays:
        data = build_data(x_start, x_end, x_step, array)
        plt.plot(data[0], data[1], label=str(array))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


def convexity_direction(coeffs, value):
    result = polynome(coeffs, value)

    if result > 0:
        return "down"
    elif result < 0:
        return "up"
    else:
        return "value is 0"


def get_extremums(coeffs):
    extremums = qubic_equation_roots(coeffs)
    print("extremums", extremums)

    return extremums


def get_inflection_points(coeffs):
    inflection_points = sorted(quadratic_equation_roots(coeffs))
    print("inflection_points", inflection_points)

    inflection_points_len = len(inflection_points)
    print("inflection_points_len", inflection_points_len)

    if inflection_points_len == 2:
        print(
            "convexity",
            f"{convexity_direction(coeffs, inflection_points[0] - 1)} | {inflection_points[0]} | "
            f"{convexity_direction(coeffs, 0.5 * (inflection_points[0] + inflection_points[1]))} | {inflection_points[1]} | " 
            f"{convexity_direction(coeffs, inflection_points[1] + 1)}")

    return inflection_points


def main():
    x_step = 0.1

    coeffs_1 = [-1, -3, 1, 6, 0]

    qubick_coeffs = [4 * coeffs_1[0], 3 * coeffs_1[1], 2 * coeffs_1[2], 1 * coeffs_1[3]]
    quadratic_coeffs = [12 * coeffs_1[0], 6 * coeffs_1[1], 2 * coeffs_1[2]]

    extremums = get_extremums(qubick_coeffs)
    inflection_points = get_inflection_points(quadratic_coeffs)

    index_to_change = 1
    variation_diapason = 5

    start = coeffs_1[index_to_change]

    coeffs = [deepcopy(coeffs_1)]

    extremums = extremums

    for i in range(-1 * variation_diapason, variation_diapason, 1):
        coeffs_new = coeffs_1
        coeffs_new[index_to_change] = start + i
        # coeff_sum = [sum(i) for i in zip(coeffs_1, coeffs_new)]
        coeffs.append(deepcopy(coeffs_new))
        extremums += get_extremums(coeffs_new)

    print("coeffs", coeffs)

    extremums_sorted = sorted(extremums)
    print("extremums_sorted", extremums_sorted)

    # x_start = extremums_sorted[0] - 0.1
    x_start = -2
    x_end = extremums_sorted[-1] + 0.1

    build_graphs(x_start, x_end, x_step, coeffs)

    # convex condition
    # zeros = quadratic_equation_roots()


if __name__ == '__main__':
    main()
