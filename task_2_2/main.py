import numpy as np
import os
import matplotlib.pyplot as plt

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

# Параметры задачи
R = 100.0  # радиус контура питания, м
rw = 0.1  # радиус скважины, м
k0 = 1e-12  # проницаемость, м^2
m = 0.2  # пористость
delta_p = 1e6  # перепад давления, Па
mu = 1e-3  # вязкость, Па*с

rw_bar = rw / R

def solve_radial_flow(N, f_func, rw_bar=0.001, R_bar=1.0, log_grid=False):
    if log_grid:
        r = np.logspace(np.log10(rw_bar), np.log10(R_bar), N)
    else:
        r = np.linspace(rw_bar, R_bar, N)

    h = np.diff(r)

    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    for i in range(1, N - 1):
        r_plus = 0.5 * (r[i] + r[i + 1])
        r_minus = 0.5 * (r[i] + r[i - 1])

        f_plus = f_func(r_plus)
        f_minus = f_func(r_minus)

        a[i] = f_minus * r_minus / h[i - 1]
        c[i] = f_plus * r_plus / h[i]
        b[i] = -(a[i] + c[i])
        d[i] = 0.0

    # Граничные условия
    # На скважине (r=rw): p = 0
    b[0] = 1.0
    c[0] = 0.0
    d[0] = 0.0

    # На контуре (r=R): p = 1
    a[-1] = 0.0
    b[-1] = 1.0
    d[-1] = 1.0

    p = thomas_algorithm(a, b, c, d)

    u = np.zeros(N)
    for i in range(N - 1):
        f_half = f_func(0.5 * (r[i] + r[i + 1]))
        u[i] = -f_half * (p[i + 1] - p[i]) / (r[i + 1] - r[i])
    u[-1] = u[-2]

    return r, p, u


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


def analytical_solution_homogeneous(r, rw_bar):
    return np.log(r / rw_bar) / np.log(1.0 / rw_bar)


def analytical_solution_heterogeneous(r, rw, re=1.0, rho=0.5, K=0.1, model='two_zone', **params):
    if model == 'two_zone':
        k1 = 1.0
        k2 = K

        # Граничные условия:
        # 1) p1(rw) = 0
        # 2) p2(re) = 1
        # Условия сопряжения на rho:
        # 3) p1(rho) = p2(rho)
        # 4) k1*dp1/dr|rho = k2*dp2/dr|rho

        # Общее решение:
        # p1(r) = A1*ln(r) + B1
        # p2(r) = A2*ln(r) + B2

        # Из условий получаем систему:
        # A1*ln(rw) + B1 = 0
        # A2*ln(re) + B2 = 1
        # A1*ln(rho) + B1 = A2*ln(rho) + B2
        # k1*A1/rho = k2*A2/rho => A2 = (k1/k2)*A1

        A1 = 1 / ((k1 / k2) * np.log(re / rho) + np.log(rho / rw))
        B1 = -A1 * np.log(rw)

        A2 = (k1 / k2) * A1
        B2 = 1 - A2 * np.log(re)

        p = np.where(r < rho,
                     A1 * np.log(r) + B1,
                     A2 * np.log(r) + B2)

    elif model == 'composite':
        r1, r2 = params.get('boundaries', [0.3, 0.6])
        k = params.get('permeabilities', [1.0, 0.5, 0.1])

        # Коэффициенты для каждой зоны
        A = np.zeros(3)
        B = np.zeros(3)

        # Условия:
        # 1) p1(rw)=0
        # 2) p3(re)=1
        # 3) p1(r1)=p2(r1), k1*A1=k2*A2
        # 4) p2(r2)=p3(r2), k2*A2=k3*A3

        A[0] = 1 / (np.log(r1 / rw) + (k[0] / k[1]) * np.log(r2 / r1) + (k[0] / k[2]) * np.log(re / r2))
        B[0] = -A[0] * np.log(rw)

        A[1] = (k[0] / k[1]) * A[0]
        B[1] = A[0] * np.log(r1) + B[0] - A[1] * np.log(r1)

        A[2] = (k[0] / k[2]) * A[0]
        B[2] = A[1] * np.log(r2) + B[1] - A[2] * np.log(r2)

        p = np.piecewise(r,
                         [r < r1, (r >= r1) & (r < r2), r >= r2],
                         [lambda x: A[0] * np.log(x) + B[0],
                          lambda x: A[1] * np.log(x) + B[1],
                          lambda x: A[2] * np.log(x) + B[2]])

    elif model == 'radial_k':
        alpha = params.get('alpha', -0.5)
        k0 = params.get('k0', 1.0)

        if np.abs(alpha + 1) < 1e-12:
            p = np.log(r / rw) / np.log(re / rw)
        else:
            p = (r ** (alpha + 1) - rw ** (alpha + 1)) / (re ** (alpha + 1) - rw ** (alpha + 1))

    elif model == 'linear_k':
        beta = params.get('beta', 0.5)

        C = np.log(r / rw) / np.log(re / rw)
        p = (-1 + np.sqrt(1 + 2 * beta * C)) / beta

    else:
        raise ValueError(f"Unknown model type: {model}")

    return p


def calculate_travel_time(r, u, m, rw, R, k0, delta_p, mu):
    u_dim = u * (k0 * delta_p) / (mu * R)

    # Интегрирование (dr / (u/m))
    time = 0.0
    for i in range(len(r) - 1, 0, -1):
        dr = r[i] - r[i - 1]
        u_avg = 0.5 * (u_dim[i] + u_dim[i - 1])
        time += (dr * R) / (u_avg / m)

    return time


def f1(r):
    return 1.0

def f2(r, rho=0.5, K=0.1):
    return np.where(r < rho, 1.0, K)


def find_K_for_equal_flow(N=1000):
    r2, p2, u2 = solve_radial_flow(N, lambda r: f2(r))
    q_ref = u2[0] * r2[0]

    def target(K):
        def f3(r, K=K):
            return np.where(r < 0.75, 1.0, K)

        r3, p3, u3 = solve_radial_flow(N, f3)
        q = u3[0] * r3[0]
        return q - q_ref

    K_low, K_high = 0.01, 1.0
    tolerance = 1e-6
    while K_high - K_low > tolerance:
        K_mid = 0.5 * (K_low + K_high)
        if target(K_mid) < 0:
            K_high = K_mid
        else:
            K_low = K_mid

    return 0.5 * (K_low + K_high)


K3 = find_K_for_equal_flow()
print(f"Найденное K для варианта 3: {K3:.6f}")

def f3(r, K=K3):
    return np.where(r < 0.75, 1.0, K)


def plot_graph(r1, r2, r3, p1, p2, p3, p1_analytical, p2_analytical, p3_analytical, u1, u2, u3):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(r1, p1, label='Вариант 1 (числ)')
    plt.plot(r1, p1_analytical, '--', label='Вариант 1 (аналит)')
    plt.plot(r2, p2, label='Вариант 2')
    plt.plot(r3, p3, label=f'Вариант 3 (K={K3:.3f})')
    plt.plot(r2, p2_analytical, '--', label='Вариант 2 (аналит)')
    plt.plot(r3, p3_analytical, '--', label='Вариант 3 (аналит)')
    plt.xlabel('Радиус')
    plt.ylabel('Давление')
    plt.title('Распределение давления')
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.subplot(2, 2, 2)
    plt.plot(r1, u1 * r1, label='Вариант 1')
    plt.plot(r2, u2 * r2, label='Вариант 2')
    plt.plot(r3, u3 * r3, label='Вариант 3')
    plt.xlabel('Радиус')
    plt.ylabel('Скорость')
    plt.title('Распределение скорости')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5)

    N_values = np.array([10, 20, 50, 100, 200, 500, 1000])
    errors = []

    for N in N_values:
        r, p_num, _ = solve_radial_flow(N, f1, rw_bar)
        p_analytical = analytical_solution_homogeneous(r, rw_bar)
        error = np.sqrt(np.mean((p_num - p_analytical) ** 2))
        errors.append(error)

    plt.subplot(2, 2, 3)
    plt.loglog(N_values, errors, 'o-')
    plt.xlabel('Число узлов N')
    plt.ylabel('Невязка E')
    plt.title('Зависимость невязки от числа узлов')
    plt.grid(True, linestyle='--', linewidth=0.5)

    rhos = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75]
    Ks = np.linspace(0.1, 0.9, 9)
    q_values = np.zeros((len(rhos), len(Ks)))

    for i, rho in enumerate(rhos):
        for j, K in enumerate(Ks):
            r, p, u = solve_radial_flow(100, lambda r: f2(r, rho, K), rw_bar)
            q_values[i, j] = u[0] * r[0]

    plt.subplot(2, 2, 4)
    for i, rho in enumerate(rhos):
        plt.plot(Ks, q_values[i, :], label=f'ρ={rho}')
    plt.xlabel('K')
    plt.ylabel('Расход q')
    plt.title('Зависимость расхода от K и ρ')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_dir='plots'
    filename = os.path.join(output_dir, 'task_reg.png')
    plt.savefig(filename)
    plt.close()


def plot_res_graph():
    plt.figure(figsize=(12, 8))
    N_values = np.array([10, 20, 50, 100, 200, 500, 1000])

    # Вариант 1
    errors1 = []
    for N in N_values:
        r, p_num, _ = solve_radial_flow(N, f1, rw_bar)
        p_analytical = analytical_solution_homogeneous(r, rw_bar)
        error = np.sqrt(np.mean((p_num - p_analytical) ** 2))
        errors1.append(error)

    plt.subplot(2, 2, 1)
    plt.loglog(N_values, errors1, 'o-', label='Вариант 1', color='blue')
    plt.xlabel('Число узлов N')
    plt.ylabel('Невязка E')
    plt.title('Невязка vs N (вариант 1)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    # Вариант 2
    errors2 = []
    for N in N_values:
        r, p_num, _ = solve_radial_flow(N, f2, rw_bar)
        p_analytical = analytical_solution_heterogeneous(r, rw_bar, rho=0.5, K=0.1)
        error = np.sqrt(np.mean((p_num - p_analytical) ** 2))
        errors2.append(error)

    plt.subplot(2, 2, 2)
    plt.loglog(N_values, errors2, 'o-', label='Вариант 2', color='green')
    plt.xlabel('Число узлов N')
    plt.ylabel('Невязка E')
    plt.title('Невязка vs N (вариант 2)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    # Вариант 3
    errors3 = []
    for N in N_values:
        r, p_num, _ = solve_radial_flow(N, f3, rw_bar)
        p_analytical = analytical_solution_heterogeneous(r, rw_bar, rho=0.75, K=K3)
        error = np.sqrt(np.mean((p_num - p_analytical) ** 2))
        errors3.append(error)

    plt.subplot(2, 2, 3)
    plt.loglog(N_values, errors3, 'o-', label='Вариант 3', color='red')
    plt.xlabel('Число узлов N')
    plt.ylabel('Невязка E')
    plt.title('Невязка vs N (вариант 3)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    output_dir = 'plots'
    filename = os.path.join(output_dir, 'task_reg_res.png')
    plt.savefig(filename)
    plt.close()


def main():
    N = 1000
    r1, p1, u1 = solve_radial_flow(N, f1, rw_bar)
    r2, p2, u2 = solve_radial_flow(N, f2, rw_bar)
    r3, p3, u3 = solve_radial_flow(N, f3, rw_bar)

    p1_analytical = analytical_solution_homogeneous(r1, rw_bar)
    p2_analytical = analytical_solution_heterogeneous(r2, rw_bar, rho=0.5, K=0.1)
    p3_analytical = analytical_solution_heterogeneous(r3, rw_bar, rho=0.75, K=K3)

    time1 = calculate_travel_time(r1, u1, m, rw, R, k0, delta_p, mu)
    time2 = calculate_travel_time(r2, u2, m, rw, R, k0, delta_p, mu)
    time3 = calculate_travel_time(r3, u3, m, rw, R, k0, delta_p, mu)

    print(f"Время прохождения (вариант 1): {time1:.2f} с")
    print(f"Время прохождения (вариант 2): {time2:.2f} с")
    print(f"Время прохождения (вариант 3): {time3:.2f} с")

    plot_graph(r1, r2, r3, p1, p2, p3, p1_analytical, p2_analytical, p3_analytical, u1, u2, u3)
    plot_res_graph()

if __name__ == "__main__":
    main()