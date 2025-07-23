import math
import operator
import time
from copy import deepcopy
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False
# DEBUG = True

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


def get_cardano_coeff_p(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]

    # print_custom('a:', a, 'b:', b, 'c:', c, 'd:', d)
    return (3 * a * c - b * b) / (3 * a * a)


def get_cardano_coeff_q(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]

    # print_custom('a:', a, 'b:', b, 'c:', c, 'd:', d)
    return (27 * a * a * d - 9 * a * b * c + 2 * pow(b, 3)) / (27 * pow(a, 3))


def get_cardano_coeff_Q(p, q):
    # print_custom('p:', p, 'q:', q)
    return pow(p/3, 3) + pow(q/2, 2)


def qubic_equation_roots(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]

    print_custom('a:', a, 'b:', b, 'c:', c, 'd:', d)
    result = []

    # Cardano's formula
    p = get_cardano_coeff_p(coeffs)
    q = get_cardano_coeff_q(coeffs)
    print_custom('p:', p, 'q:', q)

    Q = get_cardano_coeff_Q(p, q)
    print_custom("Q:", Q)

    alpha = pow((-0.5 * q + np.emath.sqrt(Q)), 1/3)
    beta = pow((-0.5 * q - np.emath.sqrt(Q)), 1/3)

    y_1 = alpha + beta
    y_2 = -0.5 * (alpha + beta) + np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)
    y_3 = -0.5 * (alpha + beta) - np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)

    for y in [y_1, y_2, y_3]:
        x = y - b / (3 * a)
        result.append(x)

    return sorted(result)


def polynome(coeffs, x):
    sum = 0
    # power_biggest = len(coeffs) - 1
    power_biggest = 4

    for i in range(len(coeffs)):
        # print(i)
        sum += (coeffs[i] * pow(x, power_biggest - i))

    return sum


def build_data(x_start, x_end, x_step, coeffs, func):
    print("####coeffs:", coeffs)
    data = [[], []]
    x = x_start

    while x <= x_end:
        data[0].append(x)
        data[1].append(func(coeffs, x))
        x += x_step

    # print(data[0])
    # print(data[1])

    return data


def build_graphs(x_start, x_end, x_step, coeff_arrays, func):
    print("coeff_arrays:", coeff_arrays)
    for array in coeff_arrays:
        data = build_data(x_start, x_end, x_step, array, func)
        plt.plot(data[0], data[1], label=str(array))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.show()


def convexity_direction(coeffs, value):
    result = polynome(coeffs, value)

    if result > 0:
        return "down"
    elif result < 0:
        return "up"
    else:
        return "value is 0"


def get_first_derivative_coeffs(coeffs):
    return [4 * coeffs[0], 3 * coeffs[1], 2 * coeffs[2], 1 * coeffs[3]]


def get_second_derivative_coeffs(coeffs):
    return [12 * coeffs[0], 6 * coeffs[1], 2 * coeffs[2]]


def get_extremums(coeffs):
    extremums = qubic_equation_roots(get_first_derivative_coeffs(coeffs))
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
    index_to_change = 1
    variation_diapason = 5

    start = coeffs[index_to_change]
    coeffs_full = [deepcopy(coeffs)]

    for i in range(-1 * variation_diapason, variation_diapason, 1):
        coeffs_new = deepcopy(coeffs)
        coeffs_new[index_to_change] = start + i
        coeffs_full.append(deepcopy(coeffs_new))

    build_graph_group(coeffs_full, 0.1, 0.001, polynome)


def build_graph_group(coeffs_group, x_step, x_margin, func):
    print("coeffs_group:", coeffs_group)
    extremums = []

    for coeffs in coeffs_group:
        extremums += get_extremums(coeffs)

    extremums_sorted = sorted(extremums)
    print_custom("extremums_sorted:", extremums_sorted)

    # x_start = extremums_sorted[0] - x_margin
    x_start = -100
    # x_end = extremums_sorted[-1] + x_margin
    x_end = 100

    build_graphs(x_start, x_end, x_step, coeffs_group, func)


def build_coeff_visualization_group(coeffs_group):
    print("coeffs_group:", coeffs_group)

    for array in coeffs_group:
        indexes = [i + 1 for i in range(len(array))]
        print("indexes:", indexes)
        print("array:", array)
        plt.plot(indexes, array, label=str(array))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='upper left')
    plt.xlim(0, 10)
    plt.ylim(-100, 100)
    plt.show()


def fix_and_search_coeffs(coeffs):
    extremums = sorted(get_extremums(coeffs))
    # global_extremum = max(polynome(coeffs_1, i) for i in extremums)
    global_extremum = max(extremums)
    print("global_extremum:", global_extremum)

    # margin = (extremums[-1] - extremums[0]) * 0.1
    # x_min = extremums[0] - margin
    # x_max = extremums[-1] + margin
    # build_graphs(x_min, x_max, 0.1, [coeffs_1])

    # take global extremum
    # fix a and b coeffs, play with c coeff searching for d coeff having same extremum
    a = coeffs[0]
    b = coeffs[1]

    coeffs_full = [deepcopy(coeffs)]

    for c in range(5, 10, 1):
        d = -4 * a * pow(global_extremum, 3) - 3 * b * pow(global_extremum, 2) - 2 * c * pow(global_extremum, 1)
        coeffs_full.append(deepcopy([a, b, c, d]))

    print_custom("coeffs_full:", coeffs_full)

    build_graph_group(coeffs_full, 0.1, 0.001, polynome)

    # take global extremum
    # fix a and d coeffs, play with b coeff searching for c coeff having same extremum
    a = coeffs[0]
    d = coeffs[3]

    coeffs_full = [deepcopy(coeffs)]

    for b in range(-8, -3, 1):
        c = (-4 * a * pow(global_extremum, 3) - 3 * b * pow(global_extremum, 2) - d) / (2 * global_extremum)
        coeffs_full.append(deepcopy([a, b, c, d]))

    print_custom("coeffs_full:", coeffs_full)

    build_graph_group(coeffs_full, 0.1, 0.001, polynome)

    # take global extremum
    # fix c and d coeffs, play with a coeff searching for b coeff having same extremum
    c = coeffs[2]
    d = coeffs[3]

    coeffs_full = [deepcopy(coeffs)]

    for a in range(-8, -3, 1):
        b = (-4 * a * pow(global_extremum, 3) - 2 * c * pow(global_extremum, 1) - d) / (3 * pow(global_extremum, 2))
        coeffs_full.append(deepcopy([a, b, c, d]))

    print_custom("coeffs_full:", coeffs_full)

    build_graph_group(coeffs_full, 0.1, 0.001, polynome)


def get_d_interval(a, b, c):
    a1 = (27 * 27 * pow(a, 4)) / 4
    a2 = 27 * pow(a, 2) * pow(b, 3)
    a3 = 27 * pow(a, 6) * pow(c, 3) - 27 * pow(a, 2) * pow(b, 2) * pow(c, 2) - pow(b, 6) + pow(b, 2)

    discriminant = a2 * a2 - 4 * a1 * a3

    if discriminant < 0:
        return None

    elif discriminant == 0:
        d = -a2 / (2 * a1)
        return [d, d]

    else:
        d1 = -a2 - sqrt(discriminant) / (2 * a1)
        d2 = -a2 + sqrt(discriminant) / (2 * a1)
        return [d1, d2]


def get_complementary_polynom_coeffs(coeffs):
    # for i in range(-10, 10, 1):
    #     coeffs[0] = i
    #
    #     if coeffs[0] == 0:
    #         continue

    # one extremum condition
    # Q = 0 (p = q = 0) in Cardano's formula

    extremums = sorted(get_extremums(coeffs))
    x = max([[i, polynome(coeffs, i)] for i in extremums], key=operator.itemgetter(1))[0]
    print("x:", x)

    coeffs_full = [deepcopy(coeffs)]

    a3 = (-3 / 20) * x
    b3 = (3/5) * pow(x, 2)
    c3 = -0.9 * pow(x, 3)
    d3 = (3/5) * pow(x, 4)

    print_custom('a3:', a3, 'b3:', b3, 'c3:', c3, 'd3:', d3)

    coeffs_3 = deepcopy([a3, b3, c3, d3])

    # move graph down to zero
    e3 = polynome(coeffs_3, x)
    coeffs_3.append((-1) * e3)
    # coeffs_full.append(coeffs_3)

    print_custom("coeffs_full:", coeffs_full)

    # build_graph_group(coeffs_full, 0.1, 0.001, polynome)

    return coeffs_3


def build_polynoms_with_one_same_extremum(coeffs):
    coeffs_full = []

    #####################################################

    # a1 = -1
    # b1 = -2
    #
    # c1 = (3 * pow(b1, 2)) / (8 * a1)
    # d1 = pow(b1, 3) / (16 * pow(a1, 2))
    #
    # coeffs_1 = [a1, b1, c1, d1]
    #
    # coeffs_full.append(deepcopy(coeffs_1))
    #
    # # for a2 in range(-100, -1, 1):
    # # (b * c) / (a * d) = 6
    # for i in range(0, 10, 1):
    #     k = 0.9 - 0.05 * i
    #     a2 = a1 * k
    #
    #     if a2 == 0:
    #         continue
    #
    #     b2 = (a2 / a1) * b1
    #     c2 = (3 * pow(b2, 2)) / (8 * a2)
    #     d2 = pow(b2, 3) / (16 * pow(a2, 2))
    #
    #     coeffs_full.append([a2, b2, c2, d2])

    #####################################################

    # rate = 0.5
    # step = 0.1
    #
    # for k in range(0, 5, 1):
    #     coeffs_2 = deepcopy(coeffs)
    #     base = rate - k * step
    #     print("base:", base)
    #     coeffs_2[1] *= base
    #     coeffs_full.append(coeffs_2)

    #####################################################

    # for a in range(-10, 0, 1):
    a = -1
    # for b in range(-10, 0, 1):
    b = -1
    for c in range(-10, 0, 1):
        d = 1 * (c * b) / a

        coeffs_full.append([a, b, c, d])

    #####################################################

    build_graph_group(coeffs_full, 0.01, 0.001, polynome)

    coeffs_full.clear()


def main():
    coeffs_1 = [-1, -3, 1, 6, 0]

    # play_with_coeffs(coeffs_1)
    # fix_and_search_coeffs(coeffs_1)
    # get_complementary_polynom_coeffs(coeffs_1)

    # search for polynoms with 3 extremums
    # for a in range(-2, -1, 1):
    #     b = 0
    #     # for b in range(-5, -1, 1):
    #     aa = -27 * pow(a, 6)
    #     bb = 27 * pow(a, 2) * pow(b, 2)
    #     cc = 0
    #     dd = pow(b, 6) - pow(b, 2)
    #
    #     c_zeros = qubic_equation_roots(deepcopy([aa, bb, cc, dd]))
    #     print(c_zeros)
    #
    #     # for c in range(1, 5, 1):
    #     #     if get_d_interval(a, b, c):
    #     #         [d1, d2] = get_d_interval(a, b, c)
    #     #         print(a, b, c, d1, d2)
    #
    #         # for d in range(math.ceil(d1), math.floor(d2), 1):
    #         #     get_complementary_polynom(deepcopy([a, b, c, d]))

    #############################################################################################

    # coeffs_full = [deepcopy(coeffs_1)]
    # coeffs_full.append(get_complementary_polynom_coeffs(coeffs_1))
    # build_coeff_visualization_group(coeffs_full)

    #############################################################################################

    coeffs_full = [deepcopy(coeffs_1)]

    rate = 0.5
    step = 0.1

    for k in range(0, 5, 1):
        base = rate - k * step
        print("base:", base)
        coeffs = [coeffs_1[i] * pow(base, 4 - i) for i in range(len(coeffs_1))]

        coeffs_full.append(coeffs)
        coeffs_full.append(get_complementary_polynom_coeffs(coeffs))
        # build_coeff_visualization_group([coeffs, get_complementary_polynom_coeffs(coeffs)])

    build_graph_group(coeffs_full, 0.1, 0.001, polynome)

    #############################################################################################

    # build_polynoms_with_one_same_extremum(coeffs_1)


if __name__ == '__main__':
    main()
