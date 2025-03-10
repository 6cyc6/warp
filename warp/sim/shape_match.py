import warp as wp
import numpy as np

wp.init()


# === Warp Kernel: Compute Center of Mass (CoM) ===
@wp.kernel
def compute_center_of_mass(
    points: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    com_out: wp.array(dtype=wp.vec3),
    mass_sum_out: wp.array(dtype=float)
):
    tid = wp.tid()
    shape_idx = indices[tid]  # Get shape index

    wp.atomic_add(com_out, shape_idx, points[tid] * masses[tid])  # Weighted sum
    wp.atomic_add(mass_sum_out, shape_idx, masses[tid])  # Sum of masses


# === Warp Kernel: Compute Apq Matrices ===
@wp.kernel
def compute_Apq(
    p_points: wp.array(dtype=wp.vec3),
    q_points: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    p_com: wp.array(dtype=wp.vec3),
    q_com: wp.array(dtype=wp.vec3),
    Apq_out: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()
    shape_idx = indices[tid]

    # Compute p' and q' (relative to CoM)
    p_prime = p_points[tid] - p_com[shape_idx]
    q_prime = q_points[tid] - q_com[shape_idx]

    # Compute weighted outer product
    outer_product = wp.mat33(
        p_prime[0] * q_prime[0], p_prime[0] * q_prime[1], p_prime[0] * q_prime[2],
        p_prime[1] * q_prime[0], p_prime[1] * q_prime[1], p_prime[1] * q_prime[2],
        p_prime[2] * q_prime[0], p_prime[2] * q_prime[1], p_prime[2] * q_prime[2]
    ) * masses[tid]

    # Accumulate into Apq
    wp.atomic_add(Apq_out, shape_idx, outer_product)


# === Warp Kernel: Compute Rotation Matrix R using SVD ===
@wp.kernel
def compute_rotation(
        Apq: wp.array(dtype=wp.mat33),
        R_out: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()

    # Compute SVD: Apq = U * Sigma * V^T
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()

    wp.svd3(Apq[tid], U, sigma, V)  # SVD decomposition

    # Compute rotation: R = U * V^T
    R_out[tid] = U @ wp.transpose(V)


# === Warp Kernel: Compute Translation T ===
@wp.kernel
def compute_translation(
        p_com: wp.array(dtype=wp.vec3),
        q_com: wp.array(dtype=wp.vec3),
        T_out: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    T_out[tid] = q_com[tid] - p_com[tid]


# === Warp Kernel: Compute Deviation ===
@wp.kernel
def compute_deviation(
        p_points: wp.array(dtype=wp.vec3),
        indices: wp.array(dtype=int),
        R: wp.array(dtype=wp.mat33),
        T: wp.array(dtype=wp.vec3),
        deviation_out: wp.array(dtype=float)
):
    tid = wp.tid()
    shape_idx = indices[tid]

    # Apply transformation
    transformed_p = R[shape_idx] @ p_points[tid] + T[shape_idx]

    # Compute deviation
    deviation = wp.length(transformed_p - p_points[tid])
    deviation_out[tid] = deviation


# === Data Setup ===
n_points = 100
n_shapes = 2  # Suppose we have 5 different objects

# Random point positions (current and next time step)
p_points_np = np.random.rand(n_points, 3).astype(np.float32)
q_points_np = np.random.rand(n_points, 3).astype(np.float32)

p_points = wp.array(p_points_np, dtype=wp.vec3)
q_points = wp.array(q_points_np, dtype=wp.vec3)

# Random mass per point
masses_np = np.random.rand(n_points).astype(np.float32)
masses = wp.array(masses_np, dtype=float)

# Shape indices
indices_np = np.random.randint(0, n_shapes, size=n_points, dtype=np.int32)
indices = wp.array(indices_np, dtype=int)

# Output arrays
p_com = wp.zeros(n_shapes, dtype=wp.vec3)
q_com = wp.zeros(n_shapes, dtype=wp.vec3)
mass_sum = wp.zeros(n_shapes, dtype=float)
Apq = wp.zeros(n_shapes, dtype=wp.mat33)
R = wp.zeros(n_shapes, dtype=wp.mat33)
T = wp.zeros(n_shapes, dtype=wp.vec3)
deviation = wp.zeros(n_points, dtype=float)

# === Step 1: Compute Centers of Mass ===
wp.launch(kernel=compute_center_of_mass, dim=n_points, inputs=[p_points, masses, indices, p_com, mass_sum])
wp.launch(kernel=compute_center_of_mass, dim=n_points, inputs=[q_points, masses, indices, q_com, mass_sum])

# Normalize CoM (divide sum by mass)
p_com_np = p_com.numpy() / mass_sum.numpy()[:, None]
q_com_np = q_com.numpy() / mass_sum.numpy()[:, None]
p_com = wp.array(p_com_np, dtype=wp.vec3)
q_com = wp.array(q_com_np, dtype=wp.vec3)

# === Step 2: Compute Apq Matrices ===
wp.launch(kernel=compute_Apq, dim=n_points, inputs=[p_points, q_points, masses, indices, p_com, q_com, Apq])

# === Step 3: Compute Rotation using SVD ===
wp.launch(kernel=compute_rotation, dim=n_shapes, inputs=[Apq, R])

# === Step 4: Compute Translation ===
wp.launch(kernel=compute_translation, dim=n_shapes, inputs=[p_com, q_com, T])

# === Step 5: Compute Deviation ===
wp.launch(kernel=compute_deviation, dim=n_points, inputs=[p_points, indices, R, T, deviation])

# === Print Results ===
print("Rotation Matrices R:\n", R.numpy())
print("Translation Vectors T:\n", T.numpy())
print("Deviation Per Point:\n", deviation.numpy())
