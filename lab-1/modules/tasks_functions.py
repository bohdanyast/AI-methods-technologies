from .myskfuzzy import *
from .graphbuilder import *


def task1():
    x = np.linspace(0, 15, 100)

    try:
        tri_mf = trimf(x, [1, 5, 13])
        trap_mf = trapmf(x, [1, 5, 7, 12])

        data = {
            1: {
                'title': 'Triangular MF',
                'lines': [
                    {'y': tri_mf}
                ]
            },
            2: {
                'title': 'Trapezoid MF',
                'lines': [
                    {'y': trap_mf}
                ]
            }
        }

        build_plot(x, data)
    except ValueError as ve:
        print(ve)


def task2():
    x = np.linspace(-40, 40, 100)

    gauss = gaussmf(x, 10, 10)
    gauss21 = gauss2mf(x, 12, 4, 13, 6)
    gauss22 = gauss2mf(x, 5, 7, 15, 3)
    gauss23 = gauss2mf(x, 14, 8, 25, 8)

    data = {
        1: {
            'title': 'Gaussian MF',
            'lines': [
                {'y': gauss}
            ]
        },
        2: {
            'title': 'Gaussian two-combined MF',
            'lines': [
                {'y': gauss21, 'label': '[12, 4, 13, 6)]'},
                {'y': gauss22, 'label': '[5, 7, 15, 3]'},
                {'y': gauss23, 'label': '[14, 8, 25, 8]'}
            ]
        }
    }

    build_plot(x, data)


def task3():
    x = np.linspace(-5, 15, 100)
    bell = gbellmf(x, 2, 4, 0)

    data = {
        1: {
            'title': 'Generalized Bell MF',
            'lines': [
                {'y': bell}
            ]
        }
    }

    build_plot(x, data)


def task4():
    x = np.linspace(-10, 15, 100)

    one_side = sigmf(x, 0, 5)
    two_side = dsigmf(x, 2, 3, 5, 7)
    add_asym = psigmf(x, 4, 5, 8, 4)

    data = {
        1: {
            'title': 'Основна одностороння',
            'lines': [
                {'y': one_side}
            ]
        },
        2: {
            'title': 'Додаткова двостороння',
            'lines': [
                {'y': two_side}
            ]
        },
        3: {
            'title': 'Додаткова асиметрична',
            'lines': [
                {'y': add_asym}
            ]
        }
    }

    build_plot(x, data)


def task5():
    x = np.linspace(-5, 15, 100)

    z = zmf(x, 6, 7)
    pi = pimf(x, 6, 7, 8, 9)
    s = smf(x, 8, 9)

    data = {
        1: {
            'title': 'Z-function',
            'lines': [
                {'y': z}
            ]
        },
        2: {
            'title': 'Pi-function',
            'lines': [
                {'y': pi}
            ]
        },
        3: {
            'title': 'S-function',
            'lines': [
                {'y': s}
            ]
        }
    }

    build_plot(x, data)


def task6():
    x = np.linspace(-5, 15, 100)

    f1 = gaussmf(x, 5, 4)
    f2 = gaussmf(x, 7.5, 6)

    fmin = np.fmin(f1, f2)
    fmax = np.fmax(f1, f2)

    data = {
        1: {
            'title': 'Min-interpretation',
            'lines': [
                {'y': f1},
                {'y': f2},
                {'y': fmin, 'marker': 'x'}
            ]
        },
        2: {
            'title': 'Max-interpretation',
            'lines': [
                {'y': f1},
                {'y': f2},
                {'y': fmax, 'marker': 'x'}
            ]
        }
    }

    build_plot(x, data)


def task7():
    x = np.linspace(-5, 15, 100)

    f1 = gaussmf(x, 5, 4)
    f2 = gaussmf(x, 7.5, 6)

    and_f = f1 * f2
    or_f = f1 + f2 - and_f

    data = {
        1: {
            'title': 'Кон\'юкція',
            'lines': [
                {'y': f1},
                {'y': f2},
                {'y': and_f, 'marker': 'x'}
            ]
        },
        2: {
            'title': 'Диз\'юнкція',
            'lines': [
                {'y': f1},
                {'y': f2},
                {'y': or_f, 'marker': 'x'}
            ]
        }
    }

    build_plot(x, data)


def task8():
    x = np.linspace(0, 10, 100)

    f = pimf(x, 6, 7, 8, 9)
    not_f = 1 - f

    data = {
        1: {
            'title': 'Функція/доповнення',
            'lines': [
                {'y': f, 'label': 'Функція'},
                {'y': not_f, 'linestyle': '--', 'label': 'Доповнення'}
            ]
        }
    }

    build_plot(x, data)
