import numpy as np


def fraction_subtractions(num1, num2, den1, den2):
    """
    Calculates the value of [(num1 - num2) / (der1 - der2)]

    :param num1: numerator 1st term
    :param num2: numerator 2nd term
    :param den1: denominator 1st term
    :param den2: denominator 2nd term
    :return: the value of [(num1 - num2) / (der1 - der2)]
    """

    return (num1 - num2) / (den1 - den2)


def trimf(dot_set, abc):
    """
    Forms a set of dots, which represent Triangular Membership Function

    :param dot_set: A set of dots for parsing through conditions
    :param abc: A set of boundary dots
    :return: A set of dots, which unite in a triangular
    :raises ValueError: if triangular is not correct. (a < b < c)
    """

    a, b, c = abc

    if not (a < b < c):
        raise ValueError("You set a wrong points for triangle. "
                         "Each number should be sharply bigger than previous one.")

    triangle_dots = np.zeros(len(dot_set))

    for i, x in enumerate(dot_set):
        if a <= x <= b:
            triangle_dots[i] = fraction_subtractions(x, a, b, a)
        elif b <= x <= c:
            triangle_dots[i] = fraction_subtractions(c, x, c, b)

    return triangle_dots


def trapmf(dot_set, abcd):
    """
    Function for creating Trapezoid Membership Function

    :param dot_set: A set of dots for parsing through conditions
    :param abcd: A set of boundary dots
    :return: A set of dots, which unite in a trapezoid
    :raises ValueError: if trapezoid is not correct. (a <= b <= c <= d)
    """
    trapezoid_dots = np.zeros(len(dot_set))
    a, b, c, d = abcd

    if not (a <= b <= c <= d):
        raise ValueError("You set a wrong points for trapezoid. "
                         "Each number should be bigger or equally bigger than previous one.")

    for i, x in enumerate(dot_set):

        if a <= x < b:
            trapezoid_dots[i] = fraction_subtractions(x, a, b, a)
        elif b <= x <= c:
            trapezoid_dots[i] = 1
        elif c < x <= d:
            trapezoid_dots[i] = fraction_subtractions(d, x, d, c)

    return trapezoid_dots


def gaussmf(x, c, sigma):
    """
    Computes values using a Gaussian membership function

    :param x: set of x-axi coordinates
    :param c: center of the curve
    :param sigma: width of the curve
    :return: values using a Gaussian membership function
    """
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))


def gauss2mf(dot_set, c1, sigma1, c2, sigma2):
    """
    Computes values using a combination of two Gaussian membership functions.

    :param dot_set: set of x-axi coordinates
    :param c1: center of the curve 1
    :param sigma1: width of the curve 1
    :param c2: center of the curve 2
    :param sigma2: width of the curve 2
    :return: values using a combination of two Gaussian membership functions
    """
    gauss2mf_dots = np.ones(len(dot_set))

    for i, x in enumerate(dot_set):
        if x <= c1:
            gauss2mf_dots[i] = gaussmf(x, c1, sigma1)
        elif x > c2:
            gauss2mf_dots[i] = gaussmf(x, c2, sigma2)

    return gauss2mf_dots


def gbellmf(x, a, b, c):
    """
    Computes values using a generalized bell-shaped membership function

    :param x: set of x-axi coordinates
    :param a: width of the membership function,
    :param b: shape of the curve on either side of plateau
    :param c: center of the membership function
    :return: values using a generalized bell-shaped membership function
    """

    return 1 / (1 + np.abs(((x - c) / a))**(2 * b))


def sigmf(x, c, a):
    """
    Computes values using a sigmoidal membership function

    :param x: set of x-axi coordinates
    :param c: center of the membership function
    :param a: width of the transition area
    :return: values using a sigmoidal membership function
    """

    return 1.0 / (1.0 + np.exp(-a * (x - c)))


def dsigmf(x, c1, a1, c2, a2):
    """
    Computes values using the difference between two sigmoidal membership functions

    :param x: set of x-axi coordinates
    :param a1: width of the transition area 1
    :param c1: center of the membership function 1
    :param a2: width of the transition area 2
    :param c2: center of the membership function 2
    :return: values using the difference between two sigmoidal membership functions
    """

    return sigmf(x, c1, a1) - sigmf(x, c2, a2)


def psigmf(x, c1, a1, c2, a2):
    """
    Computes values using the product of two sigmoidal membership functions

    :param x: set of x-axi coordinates
    :param a1: width of the transition area 1
    :param c1: center of the membership function 1
    :param a2: width of the transition area 2
    :param c2: center of the membership function 2
    :return: values using the product of two sigmoidal membership functions
    """
    return sigmf(x, c1, a1) * sigmf(x, c2, a2)


def zmf(dot_set, a, b):
    """
    Computes values using a spline-based Z-shaped membership function

    :param dot_set: set of x-axi coordinates
    :param a: shoulder of function
    :param b: foot of function
    :return: values using a spline-based Z-shaped membership function
    """
    zmf_dots = np.ones(len(dot_set))

    sum_mid = (a + b) / 2

    for i, x in enumerate(dot_set):
        if a <= x <= sum_mid:
            zmf_dots[i] = 1 - 2 * fraction_subtractions(x, a, b, a) ** 2
        elif sum_mid <= x <= b:
            zmf_dots[i] = 2 * fraction_subtractions(x, b, b, a) ** 2
        elif x >= b:
            zmf_dots[i] = 0

    return zmf_dots


def pimf(dot_set, a, b, c, d):
    """
    Computes values using a spline-based Pi-shaped membership function

    :param dot_set: set of x-axi coordinates
    :param a: left foot of function
    :param b: left shoulder of function
    :param c: right shoulder of function
    :param d: right foot of function
    :return: values using a spline-based Pi-shaped membership function
    """
    pimf_dots = np.zeros(len(dot_set))

    sum_ab_mid = (a + b) / 2
    sum_cd_mid = (c + d) / 2

    for i, x in enumerate(dot_set):
        if a <= x <= sum_ab_mid:
            pimf_dots[i] = 2 * fraction_subtractions(x, a, b, a) ** 2
        elif sum_ab_mid <= x <= b:
            pimf_dots[i] = 1 - 2 * fraction_subtractions(x, b, b, a) ** 2
        elif b <= x <= c:
            pimf_dots[i] = 1
        elif c <= x <= sum_cd_mid:
            pimf_dots[i] = 1 - 2 * fraction_subtractions(x, c, d, c) ** 2
        elif sum_cd_mid <= x <= d:
            pimf_dots[i] = 2 * fraction_subtractions(x, d, d, c) ** 2

    return pimf_dots


def smf(dot_set, a, b):
    """
    Computes values using a spline-based S-shaped membership function

    :param dot_set: set of x-axi coordinates
    :param a: foot of function
    :param b: shoulder of function
    :return: values using a spline-based S-shaped membership function
    """
    smf_dots = np.zeros(len(dot_set))

    sum_mid = (a + b) / 2

    for i, x in enumerate(dot_set):
        if a <= x <= sum_mid:
            smf_dots[i] = 2 * fraction_subtractions(x, a, b, a) ** 2
        elif sum_mid <= x <= b:
            smf_dots[i] = 1 - 2 * fraction_subtractions(x, b, b, a) ** 2
        elif x >= b:
            smf_dots[i] = 1

    return smf_dots
