import random

# Параметры сетки
start = 0.0
end = 1.0
num_nodes = 100

inner_nodes = sorted({round(random.uniform(0.001, 0.999), 3) for _ in range(num_nodes - 2)})

nodes = [start] + inner_nodes + [end]

with open('grid.txt', 'w') as file:
    for node in nodes:
        file.write(f"{node}\n")

