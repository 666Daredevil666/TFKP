# Обновленная функция для вычисления итераций метода Ньютона с защитой от деления на ноль
def newton_fractal(z, max_iter):
    # Три корня уравнения z^3 = 1
    roots = [1, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)]
    for n in range(max_iter):
        # Проверка на близость к нулю для избежания деления на ноль
        if abs(z) < 1e-6:
            return -1  # Отмечаем как несходящийся
        # Вычисление следующей итерации
        z = z - (z**3 - 1) / (3 * z**2)
        # Проверка близости к одному из корней
        for k, root in enumerate(roots):
            if abs(z - root) < 1e-6:
                return k  # Возвращаем индекс корня, к которому сошлось
    return -1  # Если не сошлось к какому-либо корню

# Пересоздание изображения с учетом новых условий
newton_set = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        # Перевод координат в комплексную плоскость
        real = xmin + (x / width) * (xmax - xmin)
        imag = ymin + (y / height) * (ymax - ymin)
        z = complex(real, imag)
        # Запись индекса корня для окраски
        newton_set[x, y] = newton_fractal(z, max_iter)

# Отображение бассейнов Ньютона
plt.figure(figsize=(10, 10))
plt.imshow(newton_set.T, extent=(xmin, xmax, ymin, ymax), cmap='twilight', interpolation='bilinear')
plt.colorbar(label='Converged root index')
plt.title("Newton's Basins for z^3 = 1")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.show()
