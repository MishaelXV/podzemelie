import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

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

Q = 100
k = 1
x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)

def potential_case1(x, y):
    r = np.sqrt(x**2 + y**2)
    r = np.where(r == 0, 1e-6, r)
    return -Q / (2 * np.pi * k) * np.log(r)


def potential_case2(x, y):
    x0, y0 = 0, 2
    r1 = np.sqrt((x - x0)**2 + (y - y0)**2)
    r2 = np.sqrt((x - x0)**2 + (y + y0)**2)
    r1 = np.where(r1 == 0, 1e-6, r1)
    r2 = np.where(r2 == 0, 1e-6, r2)
    return -Q / (2 * np.pi * k) * (np.log(r1) + np.log(r2))


def potential_case3(x, y):
    x0, y0 = 0, 2
    r1 = np.sqrt((x - x0)**2 + (y - y0)**2)
    r2 = np.sqrt((x - x0)**2 + (y + y0)**2)
    r1 = np.where(r1 == 0, 1e-6, r1)
    r2 = np.where(r2 == 0, 1e-6, r2)
    return -Q / (2 * np.pi * k) * np.log(r1) + Q / (2 * np.pi * k) * np.log(r2)


wells = [(0, 0, -100), (3, 3, 50), (-3, -3, 50)]

def potential_case4(x, y):
    phi = np.zeros_like(x)
    for xw, yw, qw in wells:
        r = np.sqrt((x - xw)**2 + (y - yw)**2)
        r = np.where(r == 0, 1e-6, r)
        phi += -qw / (2 * np.pi * k) * np.log(r)
    return phi

wells_32 = [(0, 0, -100), (1, 0, -100), (0, 1, -100)]

def potential_case32(x, y):
    phi = np.zeros_like(x)
    for xw, yw, qw in wells_32:
        r = np.sqrt((x - xw)**2 + (y - yw)**2)
        r = np.where(r == 0, 1e-6, r)
        phi += -qw / (2 * np.pi * k) * np.log(r)

        x_mirror = -2 - xw
        r_m = np.sqrt((x - x_mirror)**2 + (y - yw)**2)
        r_m = np.where(r_m == 0, 1e-6, r_m)
        phi += qw / (2 * np.pi * k) * np.log(r_m)

        y_mirror = -2 - yw
        r_p = np.sqrt((x - xw)**2 + (y - y_mirror)**2)
        r_p = np.where(r_p == 0, 1e-6, r_p)
        phi += -qw / (2 * np.pi * k) * np.log(r_p)

    return phi


def potential_case33(x, y):
    xw, yw = 1, 1
    qw = -100
    phi = np.zeros_like(x)

    r1 = np.sqrt((x - xw)**2 + (y - yw)**2)
    r2 = np.sqrt((x + xw)**2 + (y - yw)**2)
    r3 = np.sqrt((x - xw)**2 + (y + yw)**2)
    r4 = np.sqrt((x + xw)**2 + (y + yw)**2)
    for r in [r1, r2, r3, r4]:
        r[:] = np.where(r == 0, 1e-6, r)
    phi += -qw / (2 * np.pi * k) * (np.log(r1) + np.log(r2) + np.log(r3) + np.log(r4))
    return phi


def plot_potential(phi_func, title, filename):
    phi = phi_func(X, Y)
    dy, dx = np.gradient(-phi)

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, phi, levels=30, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.streamplot(X, Y, dx, dy, color='black', linewidth=1, density=1.5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.colorbar(contour, label="Потенциал")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.close()


def main():
    plot_potential(potential_case1, "Случай 1: Одиночная скважина", "case1.png")
    plot_potential(potential_case2, "Случай 2: Питающий контур", "case2.png")
    plot_potential(potential_case3, "Случай 3: Непроницаемая граница", "case3.png")
    plot_potential(potential_case4, "Случай 4: Несколько скважин", "case4.png")
    plot_potential(potential_case32, "Задача 3.2: Смешанные границы", "case32.png")
    plot_potential(potential_case33, "Задача 3.3: Биссектриса и отражения", "case33.png")

if __name__ == "__main__":
    main()