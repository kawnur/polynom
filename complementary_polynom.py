from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


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


def main():


    x_start = -2.5
    x_end = 1.5
    x_step = 0.1

    coeffs_1 = [-1, -3, 1, 6, 0]
    coeffs_2 = [-1, -3, 0, 0, 0]

    qubick_coeffs = [4 * coeffs_1[0], 3 * coeffs_1[1], 2 * coeffs_1[2], 1 * coeffs_1[3]]
    quadratic_coeffs = [12 * coeffs_1[0], 6 * coeffs_1[1], 2 * coeffs_1[2]]

    extremums = qubic_equation_roots(qubick_coeffs)
    print("extremums", extremums)

    inflection_points = sorted(quadratic_equation_roots(quadratic_coeffs))
    print("inflection_points", inflection_points)

    inflection_points_len = len(inflection_points)
    print("inflection_points_len", inflection_points_len)

    if inflection_points_len == 2:
        print(
            "convexity",
            f"{convexity_direction(quadratic_coeffs, inflection_points[0] - 1)} | {inflection_points[0]} | "
            f"{convexity_direction(quadratic_coeffs, 0.5 * (inflection_points[0] + inflection_points[1]))} | {inflection_points[1]} | " 
            f"{convexity_direction(quadratic_coeffs, inflection_points[1] + 1)}")

    # print(quadratic_equation_roots(1, -3, 2))

    # coeff_sum = [sum(i) for i in zip(coeffs_1, coeffs_2)]

    # start = -3

    # print(coeff_sum)
    # for i in range(5):
    #     coeffs_new = coeffs_2
    #     coeffs_new[0] = start + i
    #     coeff_sum = [sum(i) for i in zip(coeffs_1, coeffs_new)]
    #     build_graphs(x_start, x_end, x_step, [coeffs_1, coeffs_new, coeff_sum])

    build_graphs(x_start, x_end, x_step, [coeffs_1])

    # convex condition
    # zeros = quadratic_equation_roots()


if __name__ == '__main__':
    main()
