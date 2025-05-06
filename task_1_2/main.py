import numpy as np
import matplotlib.pyplot as plt
import os


def read_grid_and_initialize(name):
    with open(name, 'r') as file1:
        x = np.loadtxt(file1)
    n = np.size(x)

    p_0 = 1
    p_1 = 0

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    b = np.zeros(n)
    p_n = np.zeros(n)

    return x, n, p_0, p_1, A, B, C, b, p_n


def fill_matrices_with_perm(x, n, p_0, p_1, A, B, C, b, p_n, k_field):
    # Граничные условия
    p_n[0] = p_0
    p_n[n - 1] = p_1

    k_values = k_field(x)  # Значения проницаемости в узлах
    k_interfaces = 2 * k_values[:-1] * k_values[1:] / (k_values[:-1] + k_values[1:])  # Гармоническое среднее

    for i in range(1, n - 1):
        dx_minus = x[i] - x[i - 1]
        dx_plus = x[i + 1] - x[i]

        A[i] = k_interfaces[i - 1] / dx_minus
        C[i] = k_interfaces[i] / dx_plus
        B[i] = A[i] + C[i]

    # Граничные условия для матриц
    A[0] = 0
    C[n - 1] = 0
    B[0] = 1
    B[n - 1] = 1

    # Правая часть
    b[0] = p_n[0]
    b[n - 1] = p_n[n - 1]

    return A, B, C, b


def solve_tridiagonal(A, B, C, b):
    n = len(B)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    # Прямой ход
    alpha[0] = C[0] / B[0]
    beta[0] = b[0] / B[0]

    for i in range(1, n):
        alpha[i] = C[i] / (B[i] - A[i] * alpha[i - 1])
        beta[i] = (b[i] + A[i] * beta[i - 1]) / (B[i] - A[i] * alpha[i - 1])

    # Обратный ход
    p_n = np.zeros(n)
    p_n[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        p_n[i] = beta[i] + alpha[i] * p_n[i + 1]

    return p_n


def analytical_pressure(x, case):
    if case == "Однородный":
        return 1 - x
    elif case == "Зонально неоднородный":
        p = np.zeros_like(x)
        mask = x < 0.5
        p[mask] = 1 - (2 / 11) * x[mask]
        p[~mask] = (20 / 11) * (1 - x[~mask])
        return p
    elif case == "Аналитическое K":
        p = np.zeros_like(x)
        mask = x < 0.75
        p[mask] = 1 - (2 / 11) * x[mask]
        p[~mask] = (38 / 11) * (1 - x[~mask])
        return p
    else:
        raise ValueError("Неизвестный тип проницаемости")


def calculate_residual(p_n, p_a):
    E = np.sqrt(np.mean((p_n - p_a) ** 2))
    print(f"Невязка E = {E:.6f}")
    return E


def plot_pressure(x, p_n, p_a, case):
    plt.figure(figsize=(10, 6))
    plt.plot(x, p_n, 'ro', markersize=5, label='Численное решение')
    plt.plot(x, p_a, 'b-', linewidth=2, label='Аналитическое решение')
    plt.xlabel("x")
    plt.ylabel("p", rotation=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title(f'Распределение давления ({case})')

    if not os.path.exists('plots'):
        os.makedirs('plots')

    filename = f'plots/pressure_{case.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_dissipation_profile(x, p_n, k_field, case):
    dp_dx = np.gradient(p_n, x)
    epsilon = k_field(x) * dp_dx ** 2

    plt.figure(figsize=(12, 6))
    plt.plot(x, epsilon, 'r-', linewidth=2, label='Численное решение')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Удельная диссипация, ε", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.title(f'Профиль диссипации энергии ({case})', fontsize=14)
    plt.ylim(0, 3)
    plt.tight_layout()

    filename = f'plots/dissipation_{case.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_average_perm(x, k_field):
    dx = np.diff(x)
    x_mid = (x[:-1] + x[1:]) / 2
    k_mid = k_field(x_mid)
    harmonic_mean = np.sum(dx / k_mid) / np.sum(dx)
    return 1 / harmonic_mean


def calculate_velocity_and_time(L, k_avg, m, delta_p, mu, case):
    k_avg_m2 = k_avg * 1e-12

    u = k_avg_m2 * delta_p / (mu * L)

    v = u / m

    T = L / v

    print(f"{case}:")
    print(f"Средняя проницаемость k_avg = {k_avg_m2:.3e} м^2")
    print(f"Скорость фильтрации u = {u:.3e} м/с")
    print(f"Истинная скорость v = {v:.3e} м/с")
    print(f"Время прохождения T = {T:.3e} с ({T / 86400:.3f} сут)\n")


def k_field_uniform(x):
    return np.ones_like(x)


def k_field_zonal(x):
    return np.where(x < 0.5, 1.0, 0.1)


def k_field_heterogeneous(x):
    K = 1 / 19
    return np.where(x < 0.75, 1.0, K)


def calculate_residual_vs_nodes(name, case, k_field, analytical_solution, max_nodes=100, step=10):
    with open(name, 'r') as f:
        x_full = np.loadtxt(f)

    nodes_range = range(10, max_nodes + 1, step)
    residuals = []

    for n_nodes in nodes_range:
        x = np.linspace(x_full.min(), x_full.max(), n_nodes)

        # Инициализация
        p_0, p_1 = 1, 0
        A = np.zeros(n_nodes)
        B = np.zeros(n_nodes)
        C = np.zeros(n_nodes)
        b = np.zeros(n_nodes)
        p_n = np.zeros(n_nodes)

        # Решение
        A, B, C, b = fill_matrices_with_perm(x, n_nodes, p_0, p_1, A, B, C, b, p_n, k_field)
        p_n = solve_tridiagonal(A, B, C, b)
        p_a = analytical_solution(x, case)

        # Расчет невязки
        E = np.sqrt(np.mean((p_n - p_a) ** 2))
        residuals.append(E)

    plt.figure(figsize=(10, 6))
    plt.plot(nodes_range, residuals, 'bo-', markersize=5)
    plt.xlabel("Число узлов сетки", fontsize=12)
    plt.ylabel("Невязка E", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Зависимость невязки от числа узлов ({case})', fontsize=14)
    plt.tight_layout()

    if not os.path.exists('plots'):
        os.makedirs('plots')
    filename = f'plots/residuals_{case.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return nodes_range, residuals


def main():
    # Параметры задачи
    L = 100
    m = 0.2
    delta_p = 1e6
    mu = 1e-3
    name = 'grid.txt'

    cases = [
        ("Однородный", k_field_uniform),
        ("Зонально неоднородный", k_field_zonal),
        ("Аналитическое K", k_field_heterogeneous)
    ]

    for case, k_field in cases:
        x, n, p_0, p_1, A, B, C, b, p_n = read_grid_and_initialize(name)

        A, B, C, b = fill_matrices_with_perm(x, n, p_0, p_1, A, B, C, b, p_n, k_field)
        p_n = solve_tridiagonal(A, B, C, b)
        p_a = analytical_pressure(x, case)

        plot_pressure(x, p_n, p_a, case)
        calculate_residual(p_n, p_a)
        plot_dissipation_profile(x, p_n, k_field, case)
        k_avg = calculate_average_perm(x, k_field)
        calculate_velocity_and_time(L, k_avg, m, delta_p, mu, case)

        calculate_residual_vs_nodes(name, case, k_field, analytical_pressure, max_nodes=10, step=1)


if __name__ == "__main__":
    main()