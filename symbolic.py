import numpy
from sympy import *


def cardano():
    # init_session()

    # x = symbols("x")
    # expr = x ** 2 - 2 * x + 1
    # print(factor(expr))

    a, b, c, d, p, q, alpha, beta = symbols("a b c d p q alpha beta")

    p1 = (3 * a * c - b * b) / (3 * a * a)
    q1 = (2 * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * pow(a, 3))
    Q = pow(p1/3, 3) + pow(q1/2, 2)

    # print(Q)

    alpha1 = pow((-0.5 * q1 + sqrt(Q)), 1/3)
    beta1 = pow((-0.5 * q1 - sqrt(Q)), 1/3)

    y2 = -0.5 * (alpha1 + beta1) + 0.5 * sqrt(3) * 1j * (alpha1 - beta1)
    y3 = -0.5 * (alpha1 + beta1) - 0.5 * sqrt(3) * 1j * (alpha1 - beta1)

    x2 = y2 - b / (3 * a)
    x3 = y3 - b / (3 * a)

    expr = (a * x3 ** 4 + b * x3 ** 3 + c * x3 ** 2 + d * x3) - (a * x2 ** 4 + b * x2 ** 3 + c * x2 ** 2 + d * x2)

    # expr = expr.subs(p1, p)
    # expr = expr.subs(q1, q)
    expr = expr.subs(alpha1, alpha)
    expr = expr.subs(beta1, beta)

    expr = expr.expand()
    # expr = expr.simplify()

    # m, n = symbols("m n")
    # m1 = alpha * beta
    # n1 = alpha / beta
    # expr = expr.subs(m1, m)
    # expr = expr.subs(n1, n)
    # expr = expr.simplify()
    #
    # r, s = symbols("r s")
    # r1 = alpha**4 - beta**4
    # s1 = alpha**2 - beta**2
    # expr = expr.subs(r1, r)
    # expr = expr.subs(s1, s)
    # expr = expr.simplify()

    expr = factor(expr)

    pprint(expr, use_unicode=false, num_columns=10000)

    # solution = solve(Eq(expr, 0), d, dict=True)
    #
    # for s in solution:
    #     expr1 = s[d]
    #
    #     pprint(expr1, use_unicode=false, num_columns=10000)


def try_equations1():
    x = symbols("x")

    expr = x ** 2 - 2 * x + 1
    print(solveset(Eq(expr, 0), x))


def try_equations2():
    x, a, b, c, d, e, p, q, alpha, beta = symbols("x a b c d e p q alpha beta")

    expr = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x

    p1 = (3 * a * c - b * b) / (3 * a * a)
    q1 = (2 * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * pow(a, 3))
    Q = pow(p1/3, 3) + pow(q1/2, 2)

    alpha1 = pow((-0.5 * q1 + sqrt(Q)), 1/3)
    beta1 = pow((-0.5 * q1 - sqrt(Q)), 1/3)

    # solution = solveset(Eq(expr, 0), x)
    solution = solve(Eq(expr, 0), x, dict=True)

    # solution = solution.subs(p1, p)
    # solution = solution.subs(q1, q)

    # pprint(solution, use_unicode=false, num_columns=10000)

    # print(type(expr))
    # print(type(solution))

    for s in solution:
        expr1 = s[x]

        # t1, t2, t3, t4, t5, t6, t7 = symbols("t1 t2 t3 t4 t5 t6 t7")
        # t_1 = b / a
        # t_2 = (3 * c) / a
        # t_3 = (27 * d) / (1 * a)
        # t_4 = sqrt(-4 * pow(t1**2 - t2, 3) + pow(2 * t1**3 - 3 * t1 * t2 + t3, 2))
        # t_5 = pow(t1, 3) - (3 * t1 * t2) / 2 + t3 / 2 + t4 / 2
        # t_6 = -1/2 - (sqrt(3) * I) / 2
        # t_7 = -1/2 + (sqrt(3) * I) / 2
        #
        # for i, j in ((t_1, t1), (t_2, t2), (t_3, t3), (t_4, t4), (t_5, t5), (t_6, t6), (t_7, t7)):
        #     expr1 = expr1.subs(i, j)

        # expr1 = expr1.subs(p1, p).subs(q1, q).subs(alpha1, alpha).subs(beta1, beta)
        # expr1 = expr1.subs(alpha1, alpha).subs(beta1, beta)
        # expr1 = expr1.subs(p1, p)
        # expr1 = expr1.subs(q1, q)
        # pprint(expr1, use_unicode=false, num_columns=10000)

        expr2 = a * expr1 ** 4 + b * expr1 ** 3 + c * expr1 ** 2 + d * expr1
        expr2 = expr2.expand()
        # expr2 = expr2.simplify()

        f, g = symbols("f g")
        f1 = sqrt(729 * d**2 / a**2 - 486 * b * c * d / a**3 + 108 * c**3 / a**3 + 108 * b**3 * d / a**4 - 27 * b**2 * c**2 / a**4)
        expr2 = expr2.subs(f1, f)

        g1 = pow((f/2 + (27 * d) / (2 * a) - (9 * b * c) / (2 * a**2) + b**3 / a**3), 1/3)
        expr2 = expr2.subs(g1, g)

        pprint(expr2, use_unicode=false, num_columns=10000)


def main():
    # cardano()
    try_equations2()


if __name__ == '__main__':
    main()
