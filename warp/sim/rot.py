import warp as wp
import numpy as np

wp.init()

# Function to apply transformation
@wp.func
def transform_point(point: wp.vec3, R: wp.mat33, T: wp.vec3) -> wp.vec3:
    return R @ point + T  # Apply rotation and translation

# Kernel to apply transformations based on shape indices
@wp.kernel
def apply_transform(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),  # Shape index for each point
    R_matrices: wp.array(dtype=wp.mat33),  # Array of rotation matrices
    T_vectors: wp.array(dtype=wp.vec3),    # Array of translation vectors
    output: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()

    shape_idx = indices[tid]  # Determine the shape index for the current point
    R = R_matrices[shape_idx]  # Get corresponding rotation matrix
    T = T_vectors[shape_idx]  # Get corresponding translation vector

    # output[tid] = transform_point(points[tid], R, T)
    output[tid] = R * points[tid] + T

# Example Data
n_points = 100
n_shapes = 5  # Suppose we have 5 different objects

# Random points
np.random.seed(0)
points_np = np.random.rand(n_points, 3).astype(np.float32)
points = wp.array(points_np, dtype=wp.vec3)

# Assign each point a shape index (randomly assigning points to n_shapes)
indices_np = np.random.randint(0, n_shapes, size=n_points, dtype=np.int32)
indices = wp.array(indices_np, dtype=int)

# Create random rotation matrices (identity matrices for simplicity)
R_matrices_np = np.array([np.eye(3, dtype=np.float32).flatten() for _ in range(n_shapes)])
R_matrices = wp.array(R_matrices_np, dtype=wp.mat33)

# Create random translation vectors
T_vectors_np = np.random.rand(n_shapes, 3).astype(np.float32) * 10  # Random translation
T_vectors = wp.array(T_vectors_np, dtype=wp.vec3)

# Output array
output = wp.zeros(n_points, dtype=wp.vec3)

# Launch kernel
wp.launch(kernel=apply_transform, dim=n_points, inputs=[points, indices, R_matrices, T_vectors, output])

# Print results
print("Transformed points:\n", output.numpy())