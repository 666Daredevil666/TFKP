import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def branch_sqrt(z):
    magn = np.sqrt(np.abs(z))
    phi = np.angle(z)
    # Сдвигаем отрицательные углы в [0, 2π)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    half_phi = phi / 2
    return magn * np.exp(1j * half_phi)

def visualize_points(points, color, label, filename, cut_type):
    
    plt.figure(figsize=(7, 7))

   
    plt.plot(points.real, points.imag,
             marker='o', linestyle='none',
             markersize=1.2,
             color=color,
             label=label)

    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(label)

   
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    
    plt.grid(True, alpha=0.3, linestyle=':')

    if cut_type == 1:
        # Отрезок [0, i]
        y_cut = np.linspace(0, 1, 50)
        x_cut = np.zeros_like(y_cut)
        plt.plot(x_cut, y_cut, '--', color='black', label='Разрез [0,i]')
    elif cut_type == 2:
        # Луч от -1 по Re
        x_cut = np.linspace(-1, 5, 100)
        y_cut = np.zeros_like(x_cut)
        plt.plot(x_cut, y_cut, '--', color='black', label='Разрез Re ≥ -1')
    elif cut_type == 3:
        # Луч от 0 по Re
        x_cut = np.linspace(0, 5, 100)
        y_cut = np.zeros_like(x_cut)
        plt.plot(x_cut, y_cut, '--', color='black', label='Разрез Re ≥ 0')

    plt.legend(loc='upper right')

    plt.savefig(filename, dpi=150)
    plt.close()

def make_gif(image_folder, gif_name, image_files, duration=1000):
    image_paths = []
    for fname in image_files:
        full_path = os.path.join(image_folder, fname)
        if os.path.exists(full_path):
            image_paths.append(full_path)
        else:
            print(f"[ВНИМАНИЕ] Файл {full_path} не найден, пропускаю...")

    if not image_paths:
        print("Нет изображений для GIF.")
        return

    frames = [Image.open(img) for img in image_paths]
    first_frame = frames[0]

    output_gif = os.path.join(image_folder, gif_name)
    first_frame.save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # 0 = бесконечное зацикливание
    )
    print(f"[OK] GIF {output_gif} успешно создан.")

def main():
    out_folder = "my_first_mapping_blue"
    os.makedirs(out_folder, exist_ok=True)

   
    x_vals = np.linspace(-4, 4, 400)
    y_vals = np.linspace(0.0001, 4, 400)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    z_points = X_grid + 1j*Y_grid

    # Исключаем разрез [0, i]
    cut_mask = (Y_grid > 0) & ~((X_grid == 0) & (Y_grid <= 1))
    z_points = z_points[cut_mask]

   
    step1 = z_points**2         # z → z^2
    step2 = step1 + 1           # z → z + 1
    step3 = branch_sqrt(step2)  # z → sqrt(...)
    step4 = np.arccosh(step3)   # z → arccosh(...)

    
    color_initial = '#1B263B'  
    color_step1 =   '#2F4E6F'  
    color_step2 =   '#3C6E91'  
    color_step3 =   '#4E89C8'  
    color_step4 =   '#74C2E1'  

    visualize_points(
        z_points, color_initial,
        'Исходная область: Im(z)>0, вырез [0,i]',
        os.path.join(out_folder, "initial.png"),
        cut_type=1
    )

    visualize_points(
        step1, color_step1,
        'Шаг 1: z^2 (разрез Re≥-1)',
        os.path.join(out_folder, "step1.png"),
        cut_type=2
    )

    visualize_points(
        step2, color_step2,
        'Шаг 2: (z^2) + 1 (разрез Re≥0)',
        os.path.join(out_folder, "step2.png"),
        cut_type=3
    )

    visualize_points(
        step3, color_step3,
        'Шаг 3: sqrt(...) → Re>0',
        os.path.join(out_folder, "step3.png"),
        cut_type=0
    )

    visualize_points(
        step4, color_step4,
        'Финал: arcosh(...) → 0<Im(z)<π, Re(z)>0',
        os.path.join(out_folder, "final.png"),
        cut_type=0
    )

    frames_order = [
        "initial.png",
        "step1.png",
        "step2.png",
        "step3.png",
        "final.png"
    ]
    make_gif(
        image_folder=out_folder,
        gif_name="transformation_blue.gif",
        image_files=frames_order,
        duration=1200
    )

if __name__ == "__main__":
    main()
