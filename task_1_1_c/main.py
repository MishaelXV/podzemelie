import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300})


def exact_solution(x, t, u, D, m):
    if D == 0:
        return np.where(x < u * t / m, 1.0, 0.0)
    else:
        return 0.5 * erfc((x - u * t / m) / (2 * np.sqrt(D * t / m)))


def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_star = np.zeros(n)
    d_star = np.zeros(n)

    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_star[i - 1]
        c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d_star[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    return x


def solve_convection_diffusion(D, m, u, x, t_max, dt):
    n = len(x)
    if n < 3:
        raise ValueError("Сетка должна содержать как минимум 3 точки")

    dx = np.diff(x)

    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)

    for i in range(1, n - 1):
        h_prev = x[i] - x[i - 1]
        h_next = x[i + 1] - x[i]
        alpha[i - 1] = D / (h_prev * (h_prev + h_next) / 2)
        beta[i - 1] = u / (h_prev + h_next)

    c = np.zeros(n)
    c_new = np.zeros(n)
    c[0] = 1.0

    for t in np.arange(0, t_max, dt):
        a = np.zeros(n)
        b = np.zeros(n)
        c_coef = np.zeros(n)
        d = np.zeros(n)

        for i in range(1, n - 1):
            a[i] = -alpha[i - 1] - beta[i - 1]
            b[i] = m / dt + alpha[i - 1] + alpha[i] + beta[i - 1] - beta[i]
            c_coef[i] = -alpha[i] + beta[i]
            d[i] = m / dt * c[i]

        b[0], c_coef[0], d[0] = 1.0, 0.0, 1.0
        a[-1], b[-1], d[-1] = 0.0, 1.0, 0.0

        c_new = thomas_algorithm(a, b, c_coef, d)
        c = c_new.copy()

    return c


def plot_concentration_profiles():
    x = np.linspace(0, 10, 200)
    u, m, D = 1.0, 0.2, 0.1
    time_points = [0.1, 0.5, 1.0]

    plt.figure(figsize=(12, 6))

    for t in time_points:
        c_num = solve_convection_diffusion(D, m, u, x, t_max=t, dt=0.01)

        c_exact = exact_solution(x, t, u, D, m)

        plt.plot(x, c_num, '--', lw=2, label=f'Численное, t={t}')
        plt.plot(x, c_exact, '-', lw=1, alpha=0.7, label=f'Точное, t={t}')

    plt.title('Профили концентрации для различных моментов времени (D=0.1)', fontsize=14)
    plt.xlabel('Расстояние, x', fontsize=12)
    plt.ylabel('Концентрация, c', fontsize=12)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, linestyle='--', linewidth=0.5)

    filename = 'plots/concentration_profiles.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def compare_solutions():
    x = np.linspace(0, 10, 200)
    u, m, t_max = 1.0, 0.2, 1.0
    D_values = [0.01, 0.1, 1.0]

    for D in D_values:
        plt.figure(figsize=(10, 6))

        c_num = solve_convection_diffusion(D, m, u, x, t_max, dt=0.005)

        c_exact = exact_solution(x, t_max, u, D, m)

        plt.plot(x, c_num, 'r--', lw=2, label='Численное решение')
        plt.plot(x, c_exact, 'b-', lw=1.5, label='Точное решение')

        plt.title(f'Сравнение решений для D = {D}', fontsize=14)
        plt.xlabel('Расстояние, x', fontsize=12)
        plt.ylabel('Концентрация, c', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', linewidth=0.5)

        filename = f'plots/comparison_D_{D}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def show_dispersion_effect():
    x = np.linspace(0, 10, 500)
    u, m, t_max = 1.0, 0.2, 1.0
    D_values = [0.0, 0.01, 0.1, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(D_values)))

    plt.figure(figsize=(12, 6))

    for D, color in zip(D_values, colors):
        c = solve_convection_diffusion(D, m, u, x, t_max, dt=0.001)
        plt.plot(x, c, '-', color=color, lw=2, label=f'D = {D}')

    plt.title('Влияние коэффициента дисперсии на распределение концентрации', fontsize=14)
    plt.xlabel('Расстояние, x', fontsize=12)
    plt.ylabel('Концентрация, c', fontsize=12)
    plt.legend(fontsize=10, title='Коэффициент дисперсии')
    plt.grid(True, linestyle='--', linewidth=0.5)

    filename = 'plots/dispersion_effect.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    plot_concentration_profiles()
    compare_solutions()
    show_dispersion_effect()


if __name__ == "__main__":
    main()