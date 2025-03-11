from time import time
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# Funkcja implementująca odwzorowanie logistyczne
def logistic_map(x, r):
    return r * x * (1 - x)

# a) Diagram bifurkacyjny dla 1 <= r <= 4
r_values = np.linspace(1, 4, 1000)
x_values = []
r_points = []

for r in r_values:
    x = 0.5  # Warunek początkowy
    for _ in range(1000):  # Początkowe iteracje na ustabilizowanie
        x = logistic_map(x, r)
    for _ in range(200):  # Zbieranie wartości po ustabilizowaniu
        x = logistic_map(x, r)
        r_points.append(r)
        x_values.append(x)

plt.figure(figsize=(10, 6))
plt.scatter(r_points, x_values, s=0.1, color='black')
plt.xlabel("r")
plt.ylabel("x")
plt.title("Diagram bifurkacyjny odwzorowania logistycznego")
plt.show()

# b) Porównanie precyzji dla 3.75 <= r <= 3.8
r_values_prec = np.linspace(3.75, 3.8, 5)
for r in r_values_prec:
    x_single = np.float32(0.5)
    x_double = np.float64(0.5)
    iter_count = 100
    traj_single = []
    traj_double = []
    
    for _ in range(iter_count):
        x_single = np.float32(logistic_map(x_single, r))
        x_double = np.float64(logistic_map(x_double, r))
        traj_single.append(x_single)
        traj_double.append(x_double)
    
    plt.plot(traj_single, label=f"Single precision, r={r}")
    plt.plot(traj_double, linestyle='dashed', label=f"Double precision, r={r}")
    plt.legend()
    plt.xlabel("Iteracje")
    plt.ylabel("x")
    plt.title("Porównanie trajektorii dla różnych precyzji")
    plt.show()

# c) Liczba iteracji do osiągnięcia zera dla r=4
x0_values = np.linspace(0.001, 0.999, 50)
iterations_to_zero = []

for x0 in x0_values:
    x = np.float32(x0)
    count = 0
    while x > 1e-6 and count < 1000:  # Próg zera i limit iteracji
        x = logistic_map(x, 4)
        count += 1
    iterations_to_zero.append(count)

plt.figure(figsize=(8, 5))
plt.plot(x0_values, iterations_to_zero, marker='o', linestyle='-', color='blue')
plt.xlabel("x0")
plt.ylabel("Iteracje do zera")
plt.title("Liczba iteracji do osiągnięcia zera dla r=4")
plt.show()