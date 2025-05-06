import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Параметры задачи
p0 = pG = 100 * 0.101325  # МПа (перевод из атм)
H = 50  # м
F = 10e6  # м2
m = 0.2
k = 50 * 0.986923e-15  # м2 (перевод из мД)
mu = 0.05  # Па·с
beta_zh = 5e-10  # Па^-1
beta_s = 2e-10  # Па^-1
beta_star = beta_zh + beta_s
q_oil = 10 / 86400  # м3/с (перевод из м3/сут)
n_wells = 50
p_sat = 20 * 0.101325  # МПа

# Расчет коэффициентов
V = F * H  # объем пласта
chi = k / (mu * m * beta_star)  # коэффициент пьезопроводности
Q_total = n_wells * q_oil  # общий отбор


# Вариант I: все границы непроницаемы
def model_I(p, t):
    dpdt = -Q_total / (V * m * beta_star)
    return dpdt


# Вариант II: непроницаемые границы кроме подошвы (p=pG)
def model_II(p, t):
    dpdt = chi * F * (pG - p) / (V * H) - Q_total / (V * m * beta_star)
    return dpdt


# Вариант III: непроницаемые кровля и подошва, p=pG на боковых границах
L = np.sqrt(F)  # характерный размер


def model_III(p, t):
    dpdt = 2 * chi * L * (pG - p) / (V * L) - Q_total / (V * m * beta_star)
    return dpdt


# Временной интервал (5 лет в секундах)
t = np.linspace(0, 5 * 365 * 86400, 1000)

# Решение для всех вариантов
p_I = odeint(model_I, p0, t)
p_II = odeint(model_II, p0, t)
p_III = odeint(model_III, p0, t)


# Поиск времени достижения давления насыщения
def find_time_to_saturation(t, p, p_sat):
    idx = np.argmax(p < p_sat)
    if idx == 0:
        return float('inf')
    return t[idx] / (365 * 86400)  # в годах


time_I = find_time_to_saturation(t, p_I, p_sat)
time_II = find_time_to_saturation(t, p_II, p_sat)
time_III = find_time_to_saturation(t, p_III, p_sat)

# Расчет необходимого количества нагнетательных скважин
q_inj = 50 / 86400  # м3/с


def find_inj_wells(model_func, p_sat):
    def balance_model(p, t, n_inj):
        return model_func(p, t) + n_inj * q_inj / (V * m * beta_star)

    # Находим минимальное количество скважин, при котором p >= p_sat
    n_inj = 0
    while True:
        p = odeint(balance_model, p0, [0, 5 * 365 * 86400], args=(n_inj,))[-1][0]
        if p >= p_sat:
            return n_inj
        n_inj += 1


n_inj_I = find_inj_wells(model_I, p_sat)
n_inj_II = find_inj_wells(model_II, p_sat)
n_inj_III = find_inj_wells(model_III, p_sat)

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(t / (365 * 86400), p_I, label='Вариант I')
plt.plot(t / (365 * 86400), p_II, label='Вариант II')
plt.plot(t / (365 * 86400), p_III, label='Вариант III')
plt.axhline(y=p_sat, color='r', linestyle='--', label='Давление насыщения')
plt.xlabel('Время, годы')
plt.ylabel('Среднее давление, МПа')
plt.title('Динамика среднего пластового давления')
plt.legend()
plt.grid()
plt.show()

# Вывод результатов
print("Результаты:")
print(f"Вариант I: время до p_sat = {time_I:.2f} лет, нужно нагнетательных скважин: {n_inj_I}")
print(f"Вариант II: время до p_sat = {time_II:.2f} лет, нужно нагнетательных скважин: {n_inj_II}")
print(f"Вариант III: время до p_sat = {time_III:.2f} лет, нужно нагнетательных скважин: {n_inj_III}")