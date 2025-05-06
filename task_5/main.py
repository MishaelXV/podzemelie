import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Параметры задачи
p0 = pr = 100 * 0.986923  # атм -> МПа (1 атм = 0.986923 МПа)
H = 50  # м
F = 10e6  # м²
m = 0.2
k = 50e-15  # мД -> м² (1 мД = 1e-15 м²)
mu = 0.05  # Па·с
beta_x = 5e-10  # Па⁻¹
beta_c = 2e-10  # Па⁻¹
q_o = 10 / 86400  # м³/сут -> м³/с
N_prod = 50  # количество добывающих скважин
q_i = 50 / 86400  # м³/сут -> м³/с
ps = 20 * 0.986923  # давление насыщения в МПа

# Общий дебит
Q_total = N_prod * q_o

# Эффективная сжимаемость
beta_star = m * beta_x + beta_c

# Объем порового пространства
V_p = F * H * m

# Функции для решения ОДУ для разных граничных условий
def model1(p, t):
    # Вариант I: все границы непроницаемы
    dpdt = -Q_total / (V_p * beta_star)
    return dpdt

def model2(p, t):
    # Вариант II: все границы непроницаемы, кроме подошвы (p = pr)
    # Упрощенная модель с учетом перетока через подошву
    # Используем приближение, что переток пропорционален разности давлений
    alpha = k * F / (mu * H)  # коэффициент перетока
    dpdt = (-Q_total + alpha * (pr - p)) / (V_p * beta_star)
    return dpdt

def model3(p, t):
    # Вариант III: кровля и подошва непроницаемы, на боковых границах p = pr
    # Упрощенная модель с учетом перетока через боковые границы
    # Площадь боковой поверхности (примерно для круговой залежи)
    L = np.sqrt(F / np.pi)  # характерный размер
    A_lateral = 2 * np.pi * L * H  # площадь боковой поверхности
    alpha = k * A_lateral / (mu * L)  # коэффициент перетока
    dpdt = (-Q_total + alpha * (pr - p)) / (V_p * beta_star)
    return dpdt

# Временной интервал (10 лет в секундах)
t_max = 10 * 365 * 86400  # 10 лет в секундах
t = np.linspace(0, t_max, 1000)

# Решаем ОДУ для всех трех моделей
p1 = odeint(model1, p0, t).flatten()
p2 = odeint(model2, p0, t).flatten()
p3 = odeint(model3, p0, t).flatten()

# Находим время, когда давление достигает ps
def find_time_to_ps(t, p, ps):
    idx = np.argmax(p < ps)
    if idx == 0 and p[0] < ps:
        return 0
    elif idx == 0:
        return float('inf')
    return t[idx]

time_to_ps1 = find_time_to_ps(t, p1, ps)
time_to_ps2 = find_time_to_ps(t, p2, ps)
time_to_ps3 = find_time_to_ps(t, p3, ps)

# Функция для расчета необходимого количества нагнетательных скважин
def calculate_inj_wells(model_func, Q_total, q_i, V_p, beta_star, p0, pr, ps):
    # Находим Q_inj, необходимый для поддержания dp/dt = 0 при p = ps
    if model_func == model1:
        # Для модели 1: Q_inj должен компенсировать Q_total
        N_inj = np.ceil(Q_total / q_i)
    elif model_func == model2:
        # Для модели 2: Q_inj = Q_total - alpha*(pr - ps)
        alpha = k * F / (mu * H)
        Q_inj = Q_total - alpha * (pr - ps)
        N_inj = np.ceil(Q_inj / q_i) if Q_inj > 0 else 0
    elif model_func == model3:
        # Для модели 3: аналогично модели 2, но с другим alpha
        L = np.sqrt(F / np.pi)
        A_lateral = 2 * np.pi * L * H
        alpha = k * A_lateral / (mu * L)
        Q_inj = Q_total - alpha * (pr - ps)
        N_inj = np.ceil(Q_inj / q_i) if Q_inj > 0 else 0
    return int(max(N_inj, 0))

N_inj1 = calculate_inj_wells(model1, Q_total, q_i, V_p, beta_star, p0, pr, ps)
N_inj2 = calculate_inj_wells(model2, Q_total, q_i, V_p, beta_star, p0, pr, ps)
N_inj3 = calculate_inj_wells(model3, Q_total, q_i, V_p, beta_star, p0, pr, ps)

# Вывод результатов
print("Результаты для варианта I (все границы непроницаемы):")
print(f"Время снижения давления до насыщения: {time_to_ps1/86400:.2f} дней")
print(f"Необходимое количество нагнетательных скважин: {N_inj1}")

print("\nРезультаты для варианта II (подошва открыта):")
print(f"Время снижения давления до насыщения: {time_to_ps2/86400:.2f} дней" if time_to_ps2 != float('inf') else "Давление не достигает давления насыщения")
print(f"Необходимое количество нагнетательных скважин: {N_inj2}")

print("\nРезультаты для варианта III (боковые границы открыты):")
print(f"Время снижения давления до насыщения: {time_to_ps3/86400:.2f} дней" if time_to_ps3 != float('inf') else "Давление не достигает давления насыщения")
print(f"Необходимое количество нагнетательных скважин: {N_inj3}")

# Построение графиков
plt.figure(figsize=(12, 6))
plt.plot(t/(365*86400), p1, label='Вариант I: все границы закрыты')
plt.plot(t/(365*86400), p2, label='Вариант II: подошва открыта')
plt.plot(t/(365*86400), p3, label='Вариант III: боковые границы открыты')
plt.axhline(y=ps, color='r', linestyle='--', label='Давление насыщения')
plt.xlabel('Время, годы')
plt.ylabel('Среднее давление, МПа')
plt.title('Динамика среднего пластового давления')
plt.legend()
plt.grid()
plt.show()