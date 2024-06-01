'''This Python script is designed to process and transform 2D line data from DXF files into a series of datasets. 
It's particularly useful for spatial data manipulation, offering functionalities like point generation, noise addition, translation, rotation, and masking.

Key Functionalities
    DXF File Processing: Reads line data from a specified DXF file.
    Point Generation: Converts lines into discrete points, with an option to add Gaussian noise.
    Data Transformation: Applies translation and rotation to the points, simulating different spatial orientations.
    Masking: Optional masking of points based on specified x and y ranges, useful for simulating obstructions or data loss.
    Data Visualization: Visualizes the generated datasets in a 2x2 grid of subplots for quick inspection.
    File Operations: Saves the transformed datasets and their respective transformation parameters as .npy and .pkl files, respectively.
Usage
    Setup: Define file paths, output directory, and various parameters (like points per meter, noise standard deviation, etc.) at the beginning of the script.
    Execution: Run the script. It will process the DXF file, generate and transform data points, and then save and visualize the results.
Notes
    Ensure the DXF file path and output directory are correctly set.
    Modify the constants at the start of the script to suit your specific needs (e.g., scaling factor, translation range).
    The script is currently set for a specific file path and output directory, which might need to be adjusted according to your environment.'''

import numpy as np
import ezdxf
import os
import pickle
import matplotlib.pyplot as plt
import time

# Constants
DXF_FILE_PATH = "/home/mr_fusion/lidar_ws/src/pcmf/maps/bay_lines_v3.dxf"
OUTPUT_DIR = "/home/mr_fusion/lidar_ws/src/pcmf/maps/processed_maps"
POINTS_PER_METER = 1
NOISE_STD = 0.01
NUM_DATASETS = 4
SCALING_FACTOR = 1  # 1 => meters
TRANSLATION_RANGE = (-0, -0)
ROTATION_RANGE = (-0, -0)  # Degrees
X_MASK_RANGE = (-0, -0)  # (-2, -0.5)
Y_MASK_RANGE = (-0, 0)  # (-2, -1)
TIME_PLOT_OPEN = 3

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_visualize_datasets(output_dir, num_datasets):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Create a 2x2 grid of subplots
    for i in range(num_datasets):
        data_path = os.path.join(output_dir, f'bay_lines_v3{i}.npy')
        data = np.load(data_path)
        ax = axs[i // 2, i % 2]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=5)
        ax.set_title(f'Dataset {i}')
        ax.axis('equal')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(TIME_PLOT_OPEN) # Keep open for 5 seconds
    plt.close()    

def load_lines(dxf_path):
    document = ezdxf.readfile(dxf_path)
    model_space = document.modelspace()
    return [entity for entity in model_space if entity.dxftype() == 'LINE']

def generate_points(lines, points_per_meter, noise_std):
    points = []
    for line in lines:
        start = np.array((line.dxf.start.x, line.dxf.start.y))
        end = np.array((line.dxf.end.x, line.dxf.end.y))
        line_length = np.linalg.norm(end - start)
        num_points = max(int(line_length * points_per_meter), 1)
        for t in np.linspace(0, 1, num_points):
            point = start * (1 - t) + end * t
            noisy_point = point + np.random.normal(0, noise_std, 2)
            points.append(noisy_point)
    return np.array(points)

def apply_transformation(points, translation_range, rotation_range):
    tx, ty = np.random.uniform(*translation_range, size=2)
    angle = np.random.uniform(*rotation_range)
    rad = np.radians(angle)
    rot_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    points = points @ rot_matrix + np.array([tx, ty])
    return points, (tx, ty), angle

def mask_points(points, x_range=None, y_range=None):
    if x_range is not None:
        points = points[~((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]))]

    if y_range is not None:
        points = points[~((points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]))]

    return points

def save_as_npy(data, index, directory):
    filename = os.path.join(directory, f'bay_lines_v3{index}.npy')
    np.save(filename, data)

def save_transformation(data, index, directory):
    filename = os.path.join(directory, f'transformation_{index}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main(dxf_path, output_dir, num_datasets):
    lines = load_lines(dxf_path)
    base_points = generate_points(lines, POINTS_PER_METER, NOISE_STD) * SCALING_FACTOR
    # Apply the mask to the base_points
    base_points = mask_points(base_points, X_MASK_RANGE, Y_MASK_RANGE)
    
    for i in range(num_datasets):
        transformed_points, translation, rotation = apply_transformation(base_points, TRANSLATION_RANGE, ROTATION_RANGE)
        save_as_npy(transformed_points, i, output_dir)
        transformation_data = {
            'points': transformed_points,
            'translation': translation,
            'rotation': rotation
        }
        # save_transformation(transformation_data, i, output_dir)
        print(f"Dataset {i} saved with translation {translation} and rotation {rotation} degrees.")

if __name__ == "__main__":
    main(DXF_FILE_PATH, OUTPUT_DIR, NUM_DATASETS)
    load_and_visualize_datasets(OUTPUT_DIR, NUM_DATASETS)