
# def estimate_curvature(point_cloud, k):
#     tree = KDTree(point_cloud)
#     _, idx = tree.query(point_cloud, k=k)
    
#     curvatures = []
#     for indices in idx:
#         # Fit a polynomial (degree 2 for a simple curve) to the neighbors
#         fitted_poly = Polynomial.fit(point_cloud[indices, 0], point_cloud[indices, 1], 2)
#         # Derive the polynomial to get the curvature
#         curvature = np.polyder(fitted_poly.coef, 2)
#         curvatures.append(np.max(np.abs(curvature)))

#     return np.array(curvatures)

# def filter_straight_line_points(point_cloud, curvature_threshold, k):
#     curvatures = estimate_curvature(point_cloud, k)
#     return point_cloud[curvatures < curvature_threshold]

def assign_points_to_boxes(point_cloud, box_size):
    # Divide the coordinates by the box size and floor the result to assign each point to a box
    box_indices = np.floor(point_cloud / box_size).astype(int)
    # Use a dictionary to store points in each box
    boxes = {}
    for point, idx in zip(point_cloud, box_indices):
        idx_tuple = tuple(idx)
        if idx_tuple not in boxes:
            boxes[idx_tuple] = []
        boxes[idx_tuple].append(point)
    return boxes

def detect_curve_in_box(points, k=20, curvature_threshold=0.01):
    if len(points) < k:
        # Not enough points to estimate curvature
        return False
    tree = KDTree(points)
    _, idx = tree.query(points, k=k)
    for indices in idx:
        fitted_poly = Polynomial.fit(points[indices, 0], points[indices, 1], 2)
        curvature = np.polyder(fitted_poly.coef, 2)
        if np.max(np.abs(curvature)) > curvature_threshold:
            return True
    return False

def remove_curved_points(point_cloud, k, curvature_threshold, box_size):
    boxes = assign_points_to_boxes(point_cloud, box_size)
    remaining_points = []
    for box_points in boxes.values():
        if not detect_curve_in_box(np.array(box_points), k, curvature_threshold):
            remaining_points.extend(box_points)
    return np.array(remaining_points)



# source_points2 = remove_curved_points(source_points2, k=100, curvature_threshold=0.1, box_size=3)
