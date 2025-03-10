import numpy as np
import warp as wp
from typing import List, Optional, Tuple
from utils import compute_center_of_mass, compute_Apq, compute_rotation, compute_deviation, compute_translation

np.random.seed(0)
pts = np.random.random((100, 3))
q_points = wp.array(pts, dtype=wp.vec3)
# mass = np.random.random(100) * 0.1
mass = np.ones(100) * 0.1

disp = np.ones(3)
# noise = np.random.random((100, 3)) * 0.001
noise = np.zeros((100, 3)) * 0.001
# pts_forwarded = pts + disp
rot_mat = np.array([[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0.0, 0.0, 1]])

particle_q = pts.tolist()
particle_mass = mass.tolist()

idx_np = np.arange(100)
idx_1 = idx_np[:30].tolist()
idx_2 = idx_np[30:50].tolist()
idx_3 = idx_np[50:100].tolist()

shape_match_indices = []
shape_match_pts = []
shape_match_coms = []
shape_match_ps = []
shape_match_rots = []
shape_match_ts = []

for idx in [idx_1, idx_2, idx_3]:
    shape_match_id = len(shape_match_coms)
    # add indices
    shape_match_indices.extend([shape_match_id] * len(idx))

    # get particle positions and masses
    pts = np.array(particle_q)[idx]
    ms = np.array(particle_mass)[idx]

    # compute center of mass and centered coordinates
    com = np.average(pts, axis=0, weights=ms)
    ps = pts - com
    shape_match_pts.extend(pts.tolist())
    shape_match_coms.append(com.tolist())
    shape_match_ps.extend(ps.tolist())

    # add rotation and translation
    shape_match_rots.append(np.eye(3).tolist())
    shape_match_ts.append([0.0, 0.0, 0.0])

shape_match_indices = wp.array(shape_match_indices, dtype=int)
shape_match_pts = wp.array(shape_match_pts, dtype=wp.vec3)
shape_match_coms = wp.array(shape_match_coms, dtype=wp.vec3)
shape_match_ps = wp.array(shape_match_ps, dtype=wp.vec3)
shape_match_rots = wp.array(shape_match_rots, dtype=wp.mat33f)
shape_match_ts = wp.array(shape_match_ts, dtype=wp.vec3)

pts_np = shape_match_ps.numpy()
pts_forwarded = (rot_mat @ pts_np.T).T + noise

p_points = wp.array(pts_forwarded, dtype=wp.vec3)
masses = wp.array(mass, dtype=float)

n_shapes = len(shape_match_coms)
n_points = 100
# Output arrays
q_com = shape_match_coms
p_com = wp.zeros(n_shapes, dtype=wp.vec3)

mass_sum = wp.zeros(n_shapes, dtype=float)
Apq = wp.zeros(n_shapes, dtype=wp.mat33)
R = wp.zeros(n_shapes, dtype=wp.mat33)
T = wp.zeros(n_shapes, dtype=wp.vec3)
deviation = wp.zeros(n_points, dtype=wp.vec3)

# compute com
wp.launch(kernel=compute_center_of_mass, dim=n_points, inputs=[p_points, masses, shape_match_indices, p_com, mass_sum])
p_com_np = p_com.numpy() / mass_sum.numpy()[:, None]
p_com = wp.array(p_com_np, dtype=wp.vec3)

# === Step 2: Compute Apq Matrices ===
wp.launch(kernel=compute_Apq, dim=n_points, inputs=[p_points, q_points, p_com, q_com, masses, shape_match_indices, Apq])

# === Step 3: Compute Rotation using SVD ===
wp.launch(kernel=compute_rotation, dim=n_shapes, inputs=[Apq, R])

# === Step 4: Compute Translation ===
# wp.launch(kernel=compute_translation, dim=n_shapes, inputs=[p_com, q_com, T])
T = wp.clone(p_com)

# === Step 5: Compute Deviation ===
wp.launch(kernel=compute_deviation, dim=n_points, inputs=[shape_match_ps, shape_match_indices, R, T, deviation])

# === Print Results ===
print("Rotation Matrices R:\n", R.numpy())
print("Translation Vectors T:\n", T.numpy())
print("Deviation Per Point:\n", deviation.numpy())

m = deviation.numpy()

print(pts_forwarded - m)