import os
import numpy as np
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

# --- Глобальные параметры задачи ---
R = 10.0       # радиус контура питания [м]
rw = 0.1        # радиус скважины [м]
k0 = 1e-12      # проницаемость [м^2]
mu = 1e-3       # вязкость [Па*с]
dp = 1e6        # перепад давления [Па]
m = 0.2         # пористость
D_values = [0, 0.1, 1.0]  # коэффициенты дисперсии

# --- Параметры сетки и времени ---
N = 100
t_max = 1.1
dt = 0.01
nt = int(t_max / dt)

time_points = [0.1, 0.5, 1.0]

r = np.linspace(rw, R, N)
dr = r[1] - r[0]
Q = 2 * np.pi * k0 * dp / (mu * np.log(R / rw))
u = Q / (2 * np.pi * r)


def solve_tridiagonal(a, b, c, d):
    n = len(d)
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def solve_transport(D):
    c = np.zeros(N)
    c[0] = 1.0
    profiles = []
    for n in range(nt):
        t = n * dt
        a = np.zeros(N)
        b = np.zeros(N)
        c_diag = np.zeros(N)
        d = np.zeros(N)

        for i in range(1, N - 1):
            u_plus = 0.5 * (u[i] + u[i + 1])
            u_minus = 0.5 * (u[i] + u[i - 1])
            D_plus = D_minus = D
            r_plus = 0.5 * (r[i] + r[i + 1])
            r_minus = 0.5 * (r[i] + r[i - 1])

            a[i] = -dt * (r_minus * u_minus / (2 * m * r[i] * dr) + r_minus * D_minus / (m * r[i] * dr**2))
            b[i] = 1 + dt * (r_plus * D_plus + r_minus * D_minus) / (m * r[i] * dr**2)
            c_diag[i] = dt * (r_plus * u_plus / (2 * m * r[i] * dr) - r_plus * D_plus / (m * r[i] * dr**2))
            d[i] = c[i]

        b[0], c_diag[0], d[0] = 1.0, 0.0, 1.0
        a[-1], b[-1], d[-1] = 0.0, 1.0, 0.0

        c_new = solve_tridiagonal(a, b, c_diag, d)
        c = c_new.copy()

        if any(np.isclose(t, tp, rtol=1e-3) for tp in time_points):
            profiles.append((t, c.copy()))
    return profiles


def plot_profiles(profiles_dict, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    for D, profiles in profiles_dict.items():
        plt.figure(figsize=(10, 6))
        for t, profile in profiles:
            label = f't={t}'
            plt.plot(r, profile, label=label)
        plt.xlabel('Радиус')
        plt.ylabel('Концентрация')
        plt.title(f'Концентрация')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        filename = os.path.join(output_dir, f'concentration_D_{D}.png')
        plt.savefig(filename)
        plt.close()


def main():
    all_profiles = {}
    for D in D_values:
        print(f"Решение для D = {D}")
        all_profiles[D] = solve_transport(D)
    plot_profiles(all_profiles)


# --- Точка входа ---
if __name__ == "__main__":
    main()