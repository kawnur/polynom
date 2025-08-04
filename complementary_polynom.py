import math
import operator
import time
from collections import defaultdict
from copy import deepcopy
from math import sqrt
import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy import var

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


def cubic_equation_roots(coeffs):
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

    # return sorted(result)
    return result


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
    extremums = cubic_equation_roots(get_first_derivative_coeffs(coeffs))
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


def get_supplemented_polynom_coeffs(coeffs):
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


def search_for_polynoms_with_3_extremums():
    # search for polynoms with 3 extremums
    for a in range(-2, -1, 1):
        b = 0
        # for b in range(-5, -1, 1):
        aa = -27 * pow(a, 6)
        bb = 27 * pow(a, 2) * pow(b, 2)
        cc = 0
        dd = pow(b, 6) - pow(b, 2)

        c_zeros = cubic_equation_roots(deepcopy([aa, bb, cc, dd]))
        print(c_zeros)

        for c in range(1, 5, 1):
            if get_d_interval(a, b, c):
                [d1, d2] = get_d_interval(a, b, c)
                print(a, b, c, d1, d2)

                for d in range(math.ceil(d1), math.floor(d2), 1):
                    get_supplemented_polynom_coeffs(deepcopy([a, b, c, d]))


def build_coeff_visualization(coeffs):
    coeffs_full = [coeffs]
    coeffs_full.append(get_supplemented_polynom_coeffs(coeffs))
    build_coeff_visualization_group(coeffs_full)


def build_base_and_supplemented_polynoms(coeffs):
    coeffs_full = [coeffs]

    rate = 0.5
    step = 0.1

    for k in range(0, 5, 1):
        base = rate - k * step
        print("base:", base)
        new_coeffs = [coeffs[i] * pow(base, 4 - i) for i in range(len(coeffs))]

        coeffs_full.append(new_coeffs)
        coeffs_full.append(get_supplemented_polynom_coeffs(new_coeffs))
        build_coeff_visualization_group([new_coeffs, get_supplemented_polynom_coeffs(new_coeffs)])

    build_graph_group(coeffs_full, 0.1, 0.001, polynome)


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


def get_roots_order_key(indexes):
    result = 0

    for index in range(3):
        result += pow(indexes[index] + 1, 2 - index)

    return result


def define_extremums_and_values(coeffs):
    first_derivative_coeffs = get_first_derivative_coeffs(coeffs)
    extremums = cubic_equation_roots(first_derivative_coeffs)

    result = []

    for index in range(len(extremums)):
        value = polynome(coeffs, extremums[index])
        result.append([index, extremums[index], value])

    sorted(result, key=operator.itemgetter(0))
    indexes = [i[0] for i in result]


def get_extremum_value_distribution():
    root_number_distribution = defaultdict(int)
    distribution = defaultdict(int)

    coeffs_full = []

    for a in range(-20, 0, 1):
        if a == 0:
            continue

        for b in range(-10, 10, 1):
            for c in range(-10, 10, 1):

                d = b * c / a
                # for d in range(-10, 10, 1):
                coeffs = [a, b, c, d]
                # print(coeffs)

                first_derivative_coeffs = get_first_derivative_coeffs(coeffs)
                # extremums = qubic_equation_roots(first_derivative_coeffs)

                # Cardano's formula
                p = get_cardano_coeff_p(first_derivative_coeffs)
                q = get_cardano_coeff_q(first_derivative_coeffs)
                print_custom('p:', p, 'q:', q)

                Q = get_cardano_coeff_Q(p, q)
                print_custom("Q:", Q)

                if Q > 0:
                    root_number_distribution[1] += 1
                elif Q == 0:
                    root_number_distribution[2] += 1
                elif Q < 0:
                    root_number_distribution[3] += 1

                    coeffs_full.append(coeffs)

                    alpha = pow((-0.5 * q + np.emath.sqrt(Q)), 1 / 3)
                    beta = pow((-0.5 * q - np.emath.sqrt(Q)), 1 / 3)

                    y_1 = alpha + beta
                    y_2 = -0.5 * (alpha + beta) + np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)
                    y_3 = -0.5 * (alpha + beta) - np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)

                    extremums = []

                    a3 = first_derivative_coeffs[0]
                    b3 = first_derivative_coeffs[1]

                    for y in [y_1, y_2, y_3]:
                        x = y - b3 / (3 * a3)
                        extremums.append(x)

                    result = []

                    for index in range(len(extremums)):
                        value = polynome(coeffs, extremums[index])
                        result.append([index, extremums[index], value])

                    # print(result)
                    result = sorted(result, key=operator.itemgetter(2))
                    # print(result)

                    indexes = [i[0] for i in result]

                    key = get_roots_order_key(indexes)
                    # print(key, indexes)

                    distribution[key] += 1
                    # print(coeffs, key, indexes, result, distribution)
                    # print(coeffs, key, indexes, result, distribution)

    print(root_number_distribution)
    print(distribution)

    coeffs_group = []

    for i in range(len(coeffs_full)):
        if not coeffs_full[i][0] == -14:
            continue

        coeffs_group.append(coeffs_full[i])

        if len(coeffs_group) == 50:
            build_graph_group(coeffs_group, 0.01, 0.001, polynome)
            coeffs_group.clear()


def get_nonlinear_equation_system_solution(coeffs):
    """Get solution of system

        a1 * x1 + b1 * x2 + c1 * x1 * x2 = y1
        a2 * x1 + b2 * x2 + c2 * x1 * x2 = y2

        coeffs: [[a1, b1, c1, y1], [a2, b2, c2, y2]]
    """
    a1, b1, c1, y1 = coeffs[0]
    a2, b2, c2, y2 = coeffs[1]

    n = (c1 * a2 - c2 * a1)
    p = (c1 * y2 - c2 * y1) / n
    q = (c2 * b1 - c1 * b2) / n

    x2 = quadratic_equation_roots([c1 * q, a1 * q + b1 + c1 * p, a1 * p - y1])
    result = [[p + q * x, x] for x in x2]

    return result


def get_supplemented_polynom_coeffs_by_known_points(coeffs):
    first_derivative_coeffs = get_first_derivative_coeffs(coeffs)
    # extremums = qubic_equation_roots(first_derivative_coeffs)

    # Cardano's formula
    p = get_cardano_coeff_p(first_derivative_coeffs)
    q = get_cardano_coeff_q(first_derivative_coeffs)
    print_custom('p:', p, 'q:', q)

    Q = get_cardano_coeff_Q(p, q)
    print_custom("Q:", Q)

    if Q > 0:
        print("Q > 0")
    elif Q == 0:
        print("Q == 0")
    elif Q < 0:
        print("Q < 0")

        alpha = pow((-0.5 * q + np.emath.sqrt(Q)), 1 / 3)
        beta = pow((-0.5 * q - np.emath.sqrt(Q)), 1 / 3)

        y_1 = alpha + beta
        y_2 = -0.5 * (alpha + beta) + np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)
        y_3 = -0.5 * (alpha + beta) - np.emath.sqrt(-1) * 0.5 * (alpha - beta) * sqrt(3)

        extremums = []

        a3 = first_derivative_coeffs[0]
        b3 = first_derivative_coeffs[1]

        for y in [y_1, y_2, y_3]:
            x = y - b3 / (3 * a3)
            extremums.append(x)

        result = []

        for index in range(len(extremums)):
            value = polynome(coeffs, extremums[index])
            result.append([index, extremums[index], value])

        # print(result)
        result = sorted(result, key=operator.itemgetter(1))
        print(result)

        if not result[1][2] < result[0][2] and result[1][2] < result[2][2]:
            print("Minimum is not in the middle")
        else:
            addition_y = 0
            maximums_delta_x = result[2][1] - result[0][1]
            maximums_delta_y = abs(result[2][2] - result[0][2])

            min_of_maximums_y_result_item = min([result[0], result[2]], key=operator.itemgetter(2))
            min_of_maximums_y = min_of_maximums_y_result_item[2]
            min_of_maximums_x = min_of_maximums_y_result_item[1]
            print("min_of_maximums_x", min_of_maximums_x)
            print("min_of_maximums_y", min_of_maximums_y)

            # points
            x1 = result[0][1]  # first max
            y1 = polynome(coeffs, x1) + addition_y

            x2 = result[2][1]  # second max
            y2 = polynome(coeffs, x2) + addition_y

            # x3 = result[1][1]  # min
            # y3_lin = min_of_maximums_y + abs(x3 - min_of_maximums_x) * maximums_delta_y / maximums_delta_x

            # coeff = 0.1
            # y3 = y3_lin + (min_of_maximums_y + maximums_delta_y - y3_lin) * coeff

            # x4 = min_of_maximums_x - 0.1 * maximums_delta_x  # 4th point
            # y4 = polynome(coeffs, x4) + addition_y

            print(x1, y1)
            print(x2, y2)
            # print(x3, y3)
            # print(x4, y4)

            # matrix_a = np.array([
            #     [x1**4, x1**3, x1**2, x1**1],
            #     [x2**4, x2**3, x2**2, x2**1],
            #     [x3**4, x3**3, x3**2, x3**1],
            #     [x4**4, x4**3, x4**2, x4**1]
            # ])
            # matrix_a = np.array([
            #     [x1**3, x1**2, x1**1],
            #     [x2**3, x2**2, x2**1],
            #     [x3**3, x3**2, x3**1]
            # ])
            #
            # vector_y = np.array([y1 - coeffs[0] * x1**4, y2 - coeffs[0] * x2**4, y3 - coeffs[0] * x3**4])
            #
            # x = numpy.linalg.solve(matrix_a, vector_y)

            coeffs_system = [
                [x1**3, x1**2, x1**1 / coeffs[0], y1 - coeffs[0] * x1**4],
                [x2**3, x2**2, x2**1 / coeffs[0], y2 - coeffs[0] * x2**4],
            ]

            solution = get_nonlinear_equation_system_solution(coeffs_system)

            print(solution)
            coeffs_group = [coeffs]

            for s in solution:
                print(x1, polynome(s, x1))
                print(x2, polynome(s, x2))

                new_coeffs = [
                    coeffs[0], s[0], s[1], s[0] * s[1] / coeffs[0]
                ]
                coeffs_group.append(new_coeffs)

                print(y1 / polynome(new_coeffs, x1))
                print(y2 / polynome(new_coeffs, x2))

            print(coeffs_group)
            build_graph_group(coeffs_group, 0.01, 0.001, polynome)

        # indexes = [i[0] for i in result]

        # key = get_roots_order_key(indexes)
        # print(key, indexes)

        # distribution[key] += 1
        # print(coeffs, key, indexes, result, distribution)
        # print(coeffs, key, indexes, result, distribution)


def find_supplemented_polynom_coeffs_using_equality_of_extremums(coeffs):
    """
    Biggest extremum of base polynom is equal to supplemented polynom extremum.

    alpha1 + beta1 - b1 / (3 * a1) = -b2 / (3 * a2)
    """
    first_derivative_coeffs = get_first_derivative_coeffs(coeffs)
    p = get_cardano_coeff_p(first_derivative_coeffs)
    q = get_cardano_coeff_q(first_derivative_coeffs)
    print_custom('p:', p, 'q:', q)

    Q = get_cardano_coeff_Q(p, q)
    print_custom("Q:", Q)

    if Q > 0:
        print("Q > 0")
    elif Q == 0:
        print("Q == 0")
    elif Q < 0:
        print("Q < 0")

        alpha1 = pow((-0.5 * q + np.emath.sqrt(Q)), 1 / 3)
        beta1 = pow((-0.5 * q - np.emath.sqrt(Q)), 1 / 3)

        a1 = coeffs[0]
        b1 = coeffs[1]

        a2 = a1
        b2 = 3 * a2 * (b1 / (3 * a1) - alpha1 - beta1)
        c2 = 9 * b2**2 / (32 * a2)
        d2 = (4 * a2 * b2 * c2 - b2**3) / (8 * a2**2)

        new_coeffs = [a2, b2, c2, d2]

        print(cubic_equation_roots(first_derivative_coeffs))
        print(cubic_equation_roots(get_first_derivative_coeffs(new_coeffs)))

        build_graph_group([coeffs, new_coeffs], 0.01, 0.001, polynome)


def main():
    coeffs_1 = [-1, -3, 1, 6, 0]

    # play_with_coeffs(coeffs_1)
    # fix_and_search_coeffs(coeffs_1)
    # get_supplemented_polynom_coeffs(coeffs_1)
    # search_for_polynoms_with_3_extremums()
    # build_coeff_visualization(coeffs_1)
    # build_base_and_supplemented_polynoms(coeffs_1)
    # build_polynoms_with_one_same_extremum(coeffs_1)
    # define_extremums_and_values(coeffs_1)
    # get_extremum_value_distribution()
    # get_supplemented_polynom_coeffs_by_known_points(coeffs_1)
    # find_supplemented_polynom_coeffs_using_equality_of_extremums(coeffs_1)


if __name__ == '__main__':
    main()
