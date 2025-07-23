from sympy import *


def main():
    # init_session()

    # x = symbols("x")
    # expr = x ** 2 - 2 * x + 1
    # print(factor(expr))

    a, b, c, d, p1, q1, alpha1, beta1 = symbols("a b c d p1 q1 alpha1 beta1")

    p = (3 * a * c - b * b) / (3 * a * a)
    q = (2 * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * pow(a, 3))
    Q = pow(p/3, 3) + pow(q/2, 2)

    # print(Q)

    alpha = pow((-0.5 * q + sqrt(Q)), 1/3)
    beta = pow((-0.5 * q - sqrt(Q)), 1/3)

    y2 = -0.5 * (alpha + beta) + 0.5 * sqrt(3) * 1j * (alpha - beta)
    y3 = -0.5 * (alpha + beta) - 0.5 * sqrt(3) * 1j * (alpha - beta)

    x2 = y2 - b / (3 * a)
    x3 = y3 - b / (3 * a)

    expr = (a * x3 ** 4 + b * x3 ** 3 + c * x3 ** 2 + d * x3) - (a * x2 ** 4 + b * x2 ** 3 + c * x2 ** 2 + d * x2)

    # expr = expr.subs(p, p1)
    # expr = expr.subs(q, q1)
    expr = expr.subs(alpha, alpha1)
    expr = expr.subs(beta, beta1)

    # simplify(expr)
    expand(expr)

    pprint(expr, use_unicode=false, num_columns=10000)


if __name__ == '__main__':
    main()
