import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

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


def ensure_plots_directory_structure(task, mesh_type, correction):
    base_dir = "plots"
    task_dir = f"task{task}"
    mesh_dir = f"mesh_{mesh_type}"
    corr_dir = f"corr_{correction}"

    os.makedirs(os.path.join(base_dir, task_dir, mesh_dir, corr_dir), exist_ok=True)
    return os.path.join(base_dir, task_dir, mesh_dir, corr_dir)


def create_mesh(n, r_w=0.001, mesh_type='reg'):
    if mesh_type == 'reg':
        return np.linspace(r_w, 1, n)
    else:
        r = np.zeros(n)
        for i in range(n):
            power = i / (n - 1)
            r[i] = r_w * (1 / r_w) ** power
        return r


def calculate_analytical_solution(r, r_w):
    p_a = -np.log(1 / r) / np.log(1 / r_w) + 1
    u_a = -1 / r_w / np.log(1 / r_w)
    return p_a, u_a


def solve_pressure_task(r, r_w, correction=True):
    n = len(r)
    rp = (r[:-1] + r[1:]) / 2

    theta_p = np.zeros(n - 1)
    for i in range(n - 1):
        theta_p[i] = 2 * (r[i + 1] - r[i]) / (r[i] + r[i + 1]) / np.log(r[i + 1] / r[i])

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    d = np.zeros(n)

    A[0] = A[-1] = C[0] = C[-1] = 0
    B[0] = B[-1] = 1
    d[0] = 0
    d[-1] = 1

    for i in range(1, n - 1):
        if correction:
            A[i] = rp[i - 1] / (r[i] - r[i - 1]) * theta_p[i - 1]
            C[i] = rp[i] / (r[i + 1] - r[i]) * theta_p[i]
        else:
            A[i] = rp[i - 1] / (r[i] - r[i - 1])
            C[i] = rp[i] / (r[i + 1] - r[i])
        B[i] = A[i] + C[i]

    alpha = np.zeros(n)
    beta = np.zeros(n)
    p_n = np.zeros(n)

    alpha[0] = C[0] / B[0]
    beta[0] = d[0] / B[0]

    for i in range(1, n):
        alpha[i] = C[i] / (B[i] - A[i] * alpha[i - 1])
        beta[i] = (d[i] + A[i] * beta[i - 1]) / (B[i] - A[i] * alpha[i - 1])

    p_n[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        p_n[i] = beta[i] + alpha[i] * p_n[i + 1]

    return p_n


def calculate_velocity(r, p_n, correction=True):
    rp = (r[:-1] + r[1:]) / 2
    dp_dx = (p_n[1:] - p_n[:-1]) / (r[1:] - r[:-1])
    u = -dp_dx

    if correction:
        theta_p = np.zeros(len(r) - 1)
        for i in range(len(r) - 1):
            theta_p[i] = 2 * (r[i + 1] - r[i]) / (r[i] + r[i + 1]) / np.log(r[i + 1] / r[i])
        u *= theta_p

    return u, rp


def calculate_dissipation(u):
    return u ** 2

def plot_pressure_distribution(r, p_a, p_n, task, mesh_type, correction, save=False):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(r, p_a, 'b-', linewidth=2, label="Аналитическое решение")
    plt.plot(r, p_n, 'ro', markersize=5, label="Численное решение")
    plt.xlabel("r/R")
    plt.ylabel("p")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Распределение давления в пласте')
    plt.legend()
    if save:
        save_dir = ensure_plots_directory_structure(task, mesh_type, correction)
        plt.savefig(os.path.join(save_dir, f'pressure_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_velocity_distribution(rp, u, task, mesh_type, correction, save=False):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(rp, u, 'b-', linewidth=2)
    plt.xlabel("r")
    plt.ylabel("u", rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Распределение скорости')
    if save:
        save_dir = ensure_plots_directory_structure(task, mesh_type, correction)
        plt.savefig(os.path.join(save_dir, f'velocity_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_dissipation_profile(rp, dissipation, task, mesh_type, correction, save=False):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(rp, dissipation, 'b-', linewidth=2)
    plt.xlabel("r")
    plt.ylabel("$\mathrm{D_V}$", rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Профиль удельной объемной диссипации энергии')
    if save:
        save_dir = ensure_plots_directory_structure(task, mesh_type, correction)
        plt.savefig(os.path.join(save_dir, f'dissipation_profile.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_total_dissipation(dissipation, rp, task, mesh_type, correction, save=False):
    delta_p_default = 1e+6
    R_default = 100.0
    k_default = 1e-12
    m = 0.2
    mu = 1e-3

    integral = simpson(dissipation, rp)

    def total_dissipation(delta_p, R, k, mu, m):
        return integral * k * delta_p ** 2 / (R * mu * m)

    delta_p_range = np.linspace(1e+5, 1e+6, 100)
    R_range = np.linspace(100, 1000, 100)
    k_range = np.logspace(-12, -9, 100)

    D_delta_p = total_dissipation(delta_p_range, R_default, k_default, mu, m)
    D_R = total_dissipation(delta_p_default, R_range, k_default, mu, m)
    D_k = total_dissipation(delta_p_default, R_default, k_range, mu, m)

    fig = plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(delta_p_range, D_delta_p, 'b-', linewidth=2)
    plt.xlabel('Δp')
    plt.ylabel('D')
    plt.title('Зависимость D от Δp')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 2)
    plt.plot(R_range, D_R, 'b-', linewidth=2)
    plt.xlabel('R')
    plt.ylabel('D')
    plt.title('Зависимость D от R')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.loglog(k_range, D_k, 'b-', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('D')
    plt.title('Зависимость D от k')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save:
        save_dir = ensure_plots_directory_structure(task, mesh_type, correction)
        plt.savefig(os.path.join(save_dir, f'total_dissipation.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_error_convergence(n_n, E_n, task, mesh_type, correction, save=False):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(n_n, E_n, 'b-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("n")
    plt.ylabel("E")
    plt.title('Зависимость невязки от n')
    if save:
        save_dir = ensure_plots_directory_structure(task, mesh_type, correction)
        plt.savefig(os.path.join(save_dir, f'error_convergence.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_flow_rate(n_n, q_n, u_a, task, mesh_type, correction, save=False):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(n_n, q_n, 'b-', linewidth=2, label="Численный дебит")
    plt.plot([n_n[0], n_n[-1]], [u_a, u_a], 'r--', linewidth=2, label="Аналитический дебит")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("n")
    plt.ylabel("q")
    plt.title('Численный дебит скважин')
    plt.legend()
    if correction:
        plt.ylim(-200, 0)
    if save:
        save_dir = ensure_plots_directory_structure(task, mesh_type, correction)
        plt.savefig(os.path.join(save_dir, f'flow_rate.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_bottomhole_pressure(n_n, p_n_n, p_a_0, task, mesh_type, correction, save=False):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(n_n, p_n_n, 'b-', linewidth=2, label="Численное решение")
    plt.plot([n_n[0], n_n[-1]], [p_a_0, p_a_0], 'r--', linewidth=2, label="Аналитическое решение")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("n")
    plt.ylabel("p")
    plt.title('Зависимость забойного давления от n')
    plt.legend()
    if save:
        save_dir = ensure_plots_directory_structure(task, mesh_type, correction)
        plt.savefig(os.path.join(save_dir, f'bottomhole_pressure.png'), dpi=200, bbox_inches='tight')
    plt.close()


def calculate_particle_travel_time(u, r, R=100, k=1e-12, m=0.2, delta_p=1e6, mu=1e-3):
    u_dim = u * delta_p / R * k / mu
    v = u_dim / m
    dx_dim = np.diff(r) * R
    segm_time = dx_dim / v
    total_time = -np.sum(segm_time)
    return total_time / 86400  # в сутках


def main():
    # Параметры расчета
    task = 1
    mesh_type = 'reg'
    correction = True
    r_w = 0.001
    n_start = 11
    n_stop = 1001
    n_plot = 21
    save = True

    n_n = np.linspace(n_start, n_stop, n_stop - n_start)
    E_n = np.zeros(n_stop - n_start)
    q_n = np.zeros(n_stop - n_start)
    p_n_n = np.zeros(n_stop - n_start)

    if task == 1:
        for n in range(n_start, n_stop):
            r = create_mesh(n, r_w, mesh_type)
            p_a, u_a = calculate_analytical_solution(r, r_w)
            p_n = solve_pressure_task(r, r_w, correction)

            E_n[n - n_start] = np.sqrt(1 / n * np.sum((p_n - p_a) ** 2))

            u, rp = calculate_velocity(r, p_n, correction)
            dissipation = calculate_dissipation(u)

            if n == n_plot:
                plot_pressure_distribution(r, p_a, p_n, task, mesh_type, correction, save)
                plot_velocity_distribution(rp, u, task, mesh_type, correction, save)
                plot_dissipation_profile(rp, dissipation, task, mesh_type, correction, save)
                plot_total_dissipation(dissipation, rp, task, mesh_type, correction, save)

            q_n[n - n_start] = -u[0] * rp[0] / r_w + u_a * 2

        plot_error_convergence(n_n, E_n, task, mesh_type, correction, save)
        plot_flow_rate(n_n, q_n, u_a, task, mesh_type, correction, save)

    elif task == 2:
        u_a = -1 / r_w / np.log(1 / r_w)
        U = u_a
        p_1 = 1

        for n in range(n_start, n_stop):
            r = create_mesh(n, r_w, mesh_type)
            p_a, _ = calculate_analytical_solution(r, r_w)

            rp = (r[:-1] + r[1:]) / 2
            theta_p = np.zeros(len(r) - 1)
            for i in range(len(r) - 1):
                theta_p[i] = 2 * (r[i + 1] - r[i]) / (r[i] + r[i + 1]) / np.log(r[i + 1] / r[i])

            u = np.zeros(len(r) - 1)
            u[0] = r_w * U / rp[0]
            for i in range(len(r) - 2):
                u[i + 1] = u[i] * rp[i] / rp[i + 1]

            p_n = np.zeros(len(r))
            p_n[-1] = p_1
            for i in range(len(r) - 2, -1, -1):
                if correction and i == 0:
                    p_n[i] = u[i] * (r[i + 1] - r[i]) / theta_p[i] + p_n[i + 1]
                else:
                    p_n[i] = u[i] * (r[i + 1] - r[i]) + p_n[i + 1]

            if n == n_plot:
                plot_pressure_distribution(r, p_a, p_n, task, mesh_type, correction, save)

            E_n[n - n_start] = np.sqrt(1 / n * np.sum((p_n - p_a) ** 2))
            p_n_n[n - n_start] = p_n[0]

        plot_error_convergence(n_n, E_n, task, mesh_type, correction, save)
        plot_bottomhole_pressure(n_n, p_n_n, p_a[0], task, mesh_type, correction, save)

    if 'u' in locals() and 'r' in locals():
        travel_time = calculate_particle_travel_time(u, r)
        print(f"Время прохождения частиц между галереями T = {travel_time:.2f} сут")


if __name__ == "__main__":
    main()