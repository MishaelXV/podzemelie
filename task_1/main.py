import numpy as np
import matplotlib.pyplot as plt

def read_grid_and_initialize(name):
    # Чтение сетки
    with open(name, 'r') as file1:
        x = np.loadtxt(file1)
    n = np.size(x)

    # Граничные условия
    p_0 = 1
    p_1 = 0

    # Инициализация массивов
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    b = np.zeros(n)
    p_n = np.zeros(n)

    return x, n, p_0, p_1, A, B, C, b, p_n

def fill_matrices_and_solve(x, n, p_0, p_1, A, B, C, b, p_n):
    # Заполнение массивов
    A[0] = 0
    A[n - 1] = 0
    for i in range(1, n - 1):
        A[i] = 1 / (x[i] - x[i - 1])

    C[0] = 0
    C[n - 1] = 0
    for i in range(1, n - 1):
        C[i] = 1 / (x[i + 1] - x[i])

    B[0] = 1
    B[n - 1] = 1
    for i in range(1, n - 1):
        B[i] = A[i] + C[i]

    p_n[0] = p_0
    p_n[n - 1] = p_1

    b[0] = p_n[0]
    b[n - 1] = p_n[n - 1]

    # Решение системы методом прогонки
    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = C[0] / B[0]
    beta[0] = b[0] / B[0]

    for i in range(1, n):
        alpha[i] = C[i] / (B[i] - A[i] * alpha[i - 1])
        beta[i] = (b[i] + A[i] * beta[i - 1]) / (B[i] - A[i] * alpha[i - 1])

    for i in range(n - 2, 0, -1):
        p_n[i] = beta[i] + alpha[i] * p_n[i + 1]

    return p_n

def visualize_first_task(x, p_n):
    # Аналитическое решение
    p_a = 1 - x

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(x, p_a, linewidth=3, color='blue', antialiased=True, label='p_a')
    plt.plot(x, p_n, marker='o', markersize='8', linestyle='none', color='red', antialiased=True, label='p_n')
    plt.xlabel("x")
    plt.ylabel("p", rotation=0)
    plt.grid(True)
    plt.legend()
    plt.title('Зависимость p от x')
    plt.savefig('1.png', format = 'png', dpi = 300)
    plt.show()

def calculate_residual(p_n, p_a, n):
    E = np.sqrt(1 / n * np.sum(np.power(p_n - p_a, 2)))
    print("Невязка E =", E)

def plot_dissipation_profile(x, p_n):
    dp_dx = np.gradient(p_n, x)
    epsilon = dp_dx**2
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(x, epsilon, linewidth=3, color='blue', antialiased=True)
    plt.xlabel("x")
    plt.ylabel("Dv", rotation=90)
    plt.grid(True)
    plt.ylim(0, 2)
    plt.title('Профиль удельной объемной диссипации энергии')
    plt.savefig('3.png', format='png', dpi=300)
    plt.show()

def total_dissipation(delta_p, L, k, m, mu):
    return k * (delta_p ** 2) / (m * mu * L)

def plot_total_dissipation():
    delta_p_default = 1.0
    L_default = 1.0
    k_default = 1e-12
    m = 0.002
    mu = 1e-3

    delta_p_range = np.linspace(0.1, 10, 100)
    L_range = np.linspace(1, 10, 100)
    k_range = np.logspace(-12, -9, 100)

    E_delta_p = total_dissipation(delta_p_range, L_default, k_default, m, mu)
    E_L = total_dissipation(delta_p_default, L_range, k_default, m, mu)
    E_k = total_dissipation(delta_p_default, L_default, k_range, m, mu)

    plt.figure(figsize=(18, 6))

    # График E(Δp)
    plt.subplot(1, 3, 1)
    plt.plot(delta_p_range, E_delta_p, linewidth=3, antialiased=True)
    plt.xlabel('Δp')
    plt.ylabel('E', rotation=90)
    plt.title('Зависимость E от Δp')
    plt.grid(True)

    # График E(L)
    plt.subplot(1, 3, 2)
    plt.plot(L_range, E_L, linewidth=3, antialiased=True)
    plt.xlabel('L')
    plt.ylabel('E', rotation=90)
    plt.title('Зависимость E от L')
    plt.grid(True)

    # График E(k)
    plt.subplot(1, 3, 3)
    plt.loglog(k_range, E_k, linewidth=3, antialiased=True)
    plt.xlabel('k')
    plt.ylabel('E', rotation=90)
    plt.title('Зависимость E от k')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('4.png', format='png', dpi=300)
    plt.show()

def calculate_velocity_and_time():
    L = 100
    k = 1e-12
    m = 0.02
    delta_p = 1e+6
    mu = 1e-3

    # Скорость фильтрации
    u = k * delta_p / (mu * L)
    print("Скорость фильтрации u =", u)

    # Истинная скорость
    v = u / m
    print("Истинная скорость v =", v)

    # Время прохождения частиц между галереями
    T_a = m * mu * L**2 / (k * delta_p)
    print("Точное время прохождения частиц между галереями T =", T_a / 86400, "сут")

def main():
    name = 'grid.txt'
    x, n, p_0, p_1, A, B, C, d, p_n = read_grid_and_initialize(name)
    p_n = fill_matrices_and_solve(x, n, p_0, p_1, A, B, C, d, p_n)
    visualize_first_task(x, p_n)

    p_a = 1 - x
    calculate_residual(p_n, p_a, n)
    plot_dissipation_profile(x, p_n)
    plot_total_dissipation()
    calculate_velocity_and_time()

if __name__ == "__main__":
    main()