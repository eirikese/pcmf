import numpy as np
import matplotlib.pyplot as plt

def load_point_cloud(filename):
    return np.load(filename)

def divide_into_cells(point_cloud, cell_size):
    cell_indices = np.floor(point_cloud / cell_size).astype(int)
    unique_cells = np.unique(cell_indices, axis=0)
    cells = {tuple(cell): [] for cell in unique_cells}
    for point, cell_idx in zip(point_cloud, cell_indices):
        cells[tuple(cell_idx)].append(point)
    return cells

def compute_eigenvectors(cells):
    cell_eigenvectors = {}
    for cell, points in cells.items():
        if len(points) < 6:
            continue
        cov_matrix = np.cov(np.array(points).T)
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        dominant_eig_vec = eig_vecs[:, np.argmax(eig_vals)]
        cell_eigenvectors[cell] = dominant_eig_vec
    return cell_eigenvectors

def is_corner(cells, cell_eigenvectors, cell, min_angle=60, max_angle=180, min_points=4):
    min_angle = np.deg2rad(min_angle)
    max_angle = np.deg2rad(max_angle)
    neighbors = [(cell[0] + dx, cell[1] + dy) 
                 for dx in range(-1, 2) 
                 for dy in range(-1, 2) 
                 if (dx, dy) != (0, 0)]
    
    valid_neighbors = [neighbor for neighbor in neighbors 
                       if neighbor in cell_eigenvectors 
                       and len(cells[neighbor]) >= min_points]

    for i in range(len(valid_neighbors)):
        for j in range(i + 1, len(valid_neighbors)):
            vec1 = cell_eigenvectors[valid_neighbors[i]]
            vec2 = cell_eigenvectors[valid_neighbors[j]]
            angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
            
            if min_angle <= angle <= max_angle:
                return True
    return False

def plot_results(point_cloud, cells, cell_eigenvectors, corners, cell_size):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', 'box')

    # Plot original point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c='blue', s=1)

    # Plot cells and normals (eigenvectors)
    for cell, points in cells.items():
        if len(points) > 0:
            cell_center = np.mean(points, axis=0)
            if cell in cell_eigenvectors:
                eig_vec = cell_eigenvectors[cell]
                ax.quiver(cell_center[0], cell_center[1], eig_vec[0], eig_vec[1], color='green')

    # plot cell boundaries
    for cell in cells:
        x, y = cell
        ax.add_patch(plt.Rectangle((x * cell_size[0], y * cell_size[0]),  cell_size[0],  cell_size[0], fill=False, color='orange'))


    # Plot detected corners
    for corner in corners:
        ax.scatter(*corner, c='red', s=50)

    plt.show()

def main():
    filename = "/home/mr_fusion/lidar_ws/src/pcmf/maps/reference_map_mr_partial.npy"
    point_cloud = load_point_cloud(filename) * 100
    cell_size = np.array([10, 10])  # Cell size in meters (2D)
    cells = divide_into_cells(point_cloud, cell_size)
    cell_eigenvectors = compute_eigenvectors(cells)

    corners = []
    for cell in cell_eigenvectors:
        if is_corner(cells, cell_eigenvectors, cell, min_angle=70, max_angle=100, min_points=10):
            corners.append(np.array(cell) * cell_size + cell_size / 2)

    plot_results(point_cloud, cells, cell_eigenvectors, corners, cell_size)

if __name__ == "__main__":
    main()
