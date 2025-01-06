import numpy as np
import matplotlib.pyplot as plt

# Функция для вычисления множества Мандельброта
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return max_iter

# Параметры для визуализации
width, height = 800, 800
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
max_iter = 100

# Создание изображения множества Мандельброта
mandelbrot_set = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        # Перевод координат в комплексную плоскость
        real = xmin + (x / width) * (xmax - xmin)
        imag = ymin + (y / height) * (ymax - ymin)
        c = complex(real, imag)
        mandelbrot_set[x, y] = mandelbrot(c, max_iter)

# Отображение множества
plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_set.T, extent=(xmin, xmax, ymin, ymax), cmap='twilight', interpolation='bilinear')
plt.colorbar(label='Number of iterations')
plt.title("Mandelbrot Set Visualization")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.show()
