from copy import deepcopy
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# DEBUG = False
DEBUG = True

DELIMITER = '#' * 180


def get_var_name(var):
    for name, value in locals().items():
        if value is var:
            return name


def print_custom(*args):
    if DEBUG:
        # print(f"{get_var_name(value)}:", value)
        print(' '.join([str(i) for i in args]))


def quadratic_equation_roots(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]

    print_custom('a:', a, 'b:', b, 'c:', c)
    discriminant = b * b - 4 * a * c
    print_custom("discriminant:", discriminant)

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

    print_custom('a:', a, 'b:', b, 'c:', c, 'd:', d)
    result = []

    # Cardano's formula
    p = (3 * a * c - b * b) / (3 * a * a)
    q = (27 * a * a * d - 9 * a * b * c + 2 * pow(b, 3)) / (27 * pow(a, 3))
    print_custom('p:', p, 'q:', q)

    Q = pow(p/3, 3) + pow(q/2, 2)
    print_custom("Q:", Q)

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
    print_custom("extremums:", extremums)

    return extremums


def get_inflection_points(coeffs):
    inflection_points = sorted(quadratic_equation_roots(coeffs))
    print_custom("inflection_points:", inflection_points)

    inflection_points_len = len(inflection_points)
    print_custom("inflection_points_len:", inflection_points_len)

    if inflection_points_len == 2:
        print_custom(
            "convexity:",
            f"{convexity_direction(coeffs, inflection_points[0] - 1)} | {inflection_points[0]} | "
            f"{convexity_direction(coeffs, 0.5 * (inflection_points[0] + inflection_points[1]))} | "
            f"{inflection_points[1]} | " 
            f"{convexity_direction(coeffs, inflection_points[1] + 1)}")

    return inflection_points


def play_with_coeffs(coeffs):
    x_step = 0.1
    index_to_change = 1
    variation_diapason = 5

    first_derivative_coeffs = [4 * coeffs[0], 3 * coeffs[1], 2 * coeffs[2], 1 * coeffs[3]]
    second_derivative_coeffs = [12 * coeffs[0], 6 * coeffs[1], 2 * coeffs[2]]

    extremums = get_extremums(first_derivative_coeffs)
    inflection_points = get_inflection_points(second_derivative_coeffs)

    start = coeffs[index_to_change]
    coeffs_full = [deepcopy(coeffs)]
    extremums_full = deepcopy(extremums)

    for i in range(-1 * variation_diapason, variation_diapason, 1):
        coeffs_new = deepcopy(coeffs)
        coeffs_new[index_to_change] = start + i
        # coeff_sum = [sum(i) for i in zip(coeffs_1, coeffs_new)]
        coeffs_full.append(deepcopy(coeffs_new))
        extremums_full += get_extremums(coeffs_new)

    print_custom("coeffs:", coeffs)

    extremums_sorted = sorted(extremums_full)
    print_custom("extremums_sorted:", extremums_sorted)

    # x_start = extremums_sorted[0] - 0.1
    x_start = -2.5
    x_end = extremums_sorted[-1] + 0.1

    build_graphs(x_start, x_end, x_step, coeffs_full)


def main():
    coeffs_1 = [-1, -3, 1, 6, 0]

    play_with_coeffs(coeffs_1)

    # convex condition
    # zeros = quadratic_equation_roots()


if __name__ == '__main__':
    main()
