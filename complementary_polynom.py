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

    return sorted(result)


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
    plt.legend(loc='lower right')
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
    index_to_change = 1
    variation_diapason = 5

    start = coeffs[index_to_change]
    coeffs_full = [deepcopy(coeffs)]

    for i in range(-1 * variation_diapason, variation_diapason, 1):
        coeffs_new = deepcopy(coeffs)
        coeffs_new[index_to_change] = start + i
        coeffs_full.append(deepcopy(coeffs_new))

    build_graph_group(coeffs_full, 0.1, 0.001)


def build_graph_group(coeffs_group, x_step, x_margin):
    extremums = []

    for coeffs in coeffs_group:
        extremums += get_extremums(coeffs)

    extremums_sorted = sorted(extremums)
    print_custom("extremums_sorted:", extremums_sorted)

    x_start = extremums_sorted[0] - x_margin
    # x_start = -2.5
    x_end = extremums_sorted[-1] + x_margin

    build_graphs(x_start, x_end, x_step, coeffs_group)


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

    build_graph_group(coeffs_full, 0.1, 0.001)

    # take global extremum
    # fix a and d coeffs, play with b coeff searching for c coeff having same extremum
    a = coeffs[0]
    d = coeffs[3]

    coeffs_full = [deepcopy(coeffs)]

    for b in range(-8, -3, 1):
        c = (-4 * a * pow(global_extremum, 3) - 3 * b * pow(global_extremum, 2) - d) / (2 * global_extremum)
        coeffs_full.append(deepcopy([a, b, c, d]))

    print_custom("coeffs_full:", coeffs_full)

    build_graph_group(coeffs_full, 0.1, 0.001)

    # take global extremum
    # fix c and d coeffs, play with a coeff searching for b coeff having same extremum
    c = coeffs[2]
    d = coeffs[3]

    coeffs_full = [deepcopy(coeffs)]

    for a in range(-8, -3, 1):
        b = (-4 * a * pow(global_extremum, 3) - 2 * c * pow(global_extremum, 1) - d) / (3 * pow(global_extremum, 2))
        coeffs_full.append(deepcopy([a, b, c, d]))

    print_custom("coeffs_full:", coeffs_full)

    build_graph_group(coeffs_full, 0.1, 0.001)


def main():
    coeffs_1 = [-1, -3, 1, 6, 0]
    first_derivative_coeffs = [4 * coeffs_1[0], 3 * coeffs_1[1], 2 * coeffs_1[2], 1 * coeffs_1[3]]
    second_derivative_coeffs = [12 * coeffs_1[0], 6 * coeffs_1[1], 2 * coeffs_1[2]]

    # play_with_coeffs(coeffs_1)
    # fix_and_search_coeffs(coeffs_1)

    # one extremum condition
    # Q = 0 (p = q = 0) in Cardano's formula
    # fix a, from one extremum condition get expressions for b and c
    # from 'same extremum' condition get qubic equation and find d
    coeffs_full = [deepcopy(coeffs_1)]

    extremums = sorted(get_extremums(coeffs_1))
    x = max(extremums)
    print("x:", x)

    a = coeffs_1[0]

    c3 = 1
    c2 = 2 * x * pow(27 * a, 1/3)
    c1 = 3 * pow(x, 2) * np.emath.sqrt(3 * a) * pow(27 * a, 1/6)
    c0 = 4 * a * pow(x, 3)
    t = qubic_equation_roots([c3, c2, c1, c0])
    print("t:", t)

    ds = [pow(i, 1/3) for i in t]
    print("ds:", ds)

    for d in ds:
        c = pow(27 * a * d * d, 1/3)
        b = np.emath.sqrt(3 * a * c)
        coeffs_full.append(deepcopy([a, b, c, d]))

    print_custom("coeffs_full:", coeffs_full)

    build_graph_group(coeffs_full, 0.1, 5)

    # convex condition
    # zeros = quadratic_equation_roots()


if __name__ == '__main__':
    main()
