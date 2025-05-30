# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np

import warp as wp
from .model import Model
from warp import kernel


# @wp.kernel
# def solve_shape_match(
#     p: wp.array(dtype=wp.vec3),
#     q: wp.array(dtype=wp.vec3),
#     masses: wp.array(dtype=float),
#     weights: wp.array(dtype=float),
#     indices: wp.array(dtype=int),
# ):
#     tid = wp.tid()
#     shape_idx = indices[tid]
#
#     p_com = wp.vec3(0.0)
#     pt = p[tid] * weights[tid]
#     wp.atomic_add(p_com, shape_idx, pt)  # Weighted sum



@wp.kernel
def assign_deltas(
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_q_target: wp.array(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    delta[tid] = particle_q_target[tid] - particle_q_init[tid]


@wp.kernel
def velocity_damping(
    alpha: float,
    dq: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dq[tid] = dq[tid] * alpha


@wp.kernel
def update_particle_positions(
    delta: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    points[tid] += delta[tid]
    # wp.atomic_add(points, tid, delta[tid])


@wp.kernel
def compute_Apq3(
    p_points: wp.array(dtype=wp.vec3),
    p_com: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    weights: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    Apq_out: wp.array(dtype=wp.mat33),
    # R_out: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    shape_idx = indices[tid]

    pt = p_points[tid] * weights[tid]
    wp.atomic_add(p_com, shape_idx, pt)  # Weighted sum

    # Compute p' and q' (relative to CoM)
    q_prime = q[tid]

    if tid == 0 and tid == 1:
        p_prime = p_points[tid] - p_com[shape_idx]

    # Accumulate into Apq
    wp.atomic_add(Apq_out, shape_idx, wp.outer(p_prime, q_prime) * masses[tid])

    # # Compute SVD: Apq = U * Sigma * V^T
    # U = wp.mat33()
    # sigma = wp.vec3()
    # V = wp.mat33()
    #
    # if tid == 0 and tid == 1:
    #     wp.svd3(Apq_out[tid], U, sigma, V)  # SVD decomposition
    #
    # # Compute rotation: R = U * V^T
    # R_out[tid] = U @ wp.transpose(V)


@wp.kernel
def compute_Apq2(
    p_points: wp.array(dtype=wp.vec3),
    p_com: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    Apq_out: wp.array(dtype=wp.mat33)
):
    """
    Compute Apq:
    Apq = sum(mi * outer_product(p_points, q_points))
    """
    tid = wp.tid()
    shape_idx = indices[tid]

    # Compute p' and q' (relative to CoM)
    p_prime = p_points[tid] - p_com[shape_idx]
    q_prime = q[tid]

    # Compute weighted outer product
    # outer_product = wp.mat33(
    #     p_prime[0] * q_prime[0], p_prime[0] * q_prime[1], p_prime[0] * q_prime[2],
    #     p_prime[1] * q_prime[0], p_prime[1] * q_prime[1], p_prime[1] * q_prime[2],
    #     p_prime[2] * q_prime[0], p_prime[2] * q_prime[1], p_prime[2] * q_prime[2]
    # ) * masses[tid]

    # Accumulate into Apq
    wp.atomic_add(Apq_out, shape_idx, wp.outer(p_prime, q_prime) * masses[tid])


# === Warp Kernel: Compute Center of Mass (CoM) ===
@wp.kernel
def compute_center_of_mass(
    points: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    com_out: wp.array(dtype=wp.vec3),
    mass_sum_out: wp.array(dtype=float)
):
    """
    Compute the center of mass
    """
    tid = wp.tid()
    shape_idx = indices[tid]  # Get shape index

    wp.atomic_add(com_out, shape_idx, points[tid] * masses[tid])  # Weighted sum
    wp.atomic_add(mass_sum_out, shape_idx, masses[tid])  # Sum of masses


@wp.kernel
def compute_center_of_mass2(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    weights: wp.array(dtype=float),
    com_out: wp.array(dtype=wp.vec3),
):
    """
    Compute the center of mass
    """
    tid = wp.tid()
    shape_idx = indices[tid]

    pt = points[tid] * weights[tid]
    wp.atomic_add(com_out, shape_idx, pt)  # Weighted sum


# === Warp Kernel: Compute Apq Matrices ===
@wp.kernel
def compute_Apq(
    p_points: wp.array(dtype=wp.vec3),
    q_points: wp.array(dtype=wp.vec3),
    p_com: wp.array(dtype=wp.vec3),
    q_com: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    Apq_out: wp.array(dtype=wp.mat33)
):
    """
    Compute Apq:
    Apq = sum(mi * outer_product(p_points, q_points))
    """
    tid = wp.tid()
    shape_idx = indices[tid]

    # Compute p' and q' (relative to CoM)
    p_prime = p_points[tid] - p_com[shape_idx]
    q_prime = q_points[tid] - q_com[shape_idx]

    # Compute weighted outer product
    # outer_product = wp.mat33(
    #     p_prime[0] * q_prime[0], p_prime[0] * q_prime[1], p_prime[0] * q_prime[2],
    #     p_prime[1] * q_prime[0], p_prime[1] * q_prime[1], p_prime[1] * q_prime[2],
    #     p_prime[2] * q_prime[0], p_prime[2] * q_prime[1], p_prime[2] * q_prime[2]
    # ) * masses[tid]
    outer_product = wp.outer(p_prime, q_prime) * masses[tid]

    # Accumulate into Apq
    wp.atomic_add(Apq_out, shape_idx, outer_product)


# === Warp Kernel: Compute Rotation Matrix R using SVD ===
@wp.kernel
def compute_rotation(
    Apq: wp.array(dtype=wp.mat33),
    R_out: wp.array(dtype=wp.mat33)
):
    """
    Use SVD to compute rotation matrix of polar decomposition
    """
    tid = wp.tid()

    # Compute SVD: Apq = U * Sigma * V^T
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()

    # wp.svd3(Apq[tid], U, sigma, V)  # SVD decomposition
    #
    # # Compute rotation: R = U * V^T
    # R_out[tid] = U @ wp.transpose(V)

    wp.svd3(Apq[tid], U, sigma, V)  # Perform SVD decomposition

    # Compute M = V * U^T
    M = V @ wp.transpose(U)

    # Ensure determinant is +1 to avoid reflections
    correction = wp.mat33(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, wp.determinant(M))
    )

    R_out[tid] = V @ correction @ wp.transpose(U)


# === Warp Kernel: Compute Translation T ===
@wp.kernel
def compute_translation(
    p_com: wp.array(dtype=wp.vec3),
    q_com: wp.array(dtype=wp.vec3),
    T_out: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    # T_out[tid] = q_com[tid] - p_com[tid]
    T_out[tid] = p_com[tid]


# === Warp Kernel: Compute Deviation ===
@wp.kernel
def compute_goal(
    p_points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    R: wp.array(dtype=wp.mat33),
    T: wp.array(dtype=wp.vec3),
    goal: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    shape_idx = indices[tid]

    # Apply transformation
    transformed_p = R[shape_idx] @ p_points[tid] + T[shape_idx]

    # Compute goal
    goal[tid] = transformed_p


@wp.kernel
def compute_translation(
    p: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.vec3),
    dt: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dt[tid] = p[tid] - q[tid]


@wp.func
def compute_transformation(
    n_pts: int,
    n_shapes: int,
    shape_match_ps: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=wp.float32),
    shape_match_indices: wp.array(dtype=wp.int32),
    shape_match_w_pts: wp.array(dtype=wp.vec3),
    shape_match_coms: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3)
):

    p_com = wp.zeros(n_shapes, dtype=wp.vec3, device=shape_match_indices.device)
    dt = wp.zeros(n_shapes, dtype=wp.vec3, device=shape_match_indices.device)
    Apq = wp.zeros(n_shapes, dtype=wp.mat33, device=shape_match_indices.device)
    R = wp.zeros(n_shapes, dtype=wp.mat33, device=shape_match_indices.device)

    wp.launch(
        kernel=compute_center_of_mass2,
        dim=n_pts,
        # inputs=[particle_q, model.shape_match_indices, model.shape_match_w_pts],
        inputs=[p, shape_match_indices, shape_match_w_pts],
        outputs=[p_com],
    )

    wp.launch(
        kernel=compute_Apq2, dim=n_pts,
        inputs=[p, p_com, shape_match_ps,
                particle_mass, shape_match_indices],
        outputs=[Apq, ],
    )
    # Compute Rotation using SVD
    wp.launch(kernel=compute_rotation, dim=n_shapes, inputs=[Apq, ], outputs=[R])

    # wp.launch(kernel=compute_translation, dim=n_shapes, inputs=[p_com, shape_match_coms], outputs=[dt])

    return R, p_com

@wp.kernel
def transform_pts(
    pts: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int16),
    R: wp.array(dtype=wp.mat33),
    T: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    shape_idx = indices[tid]

    # Apply transformation
    pts[tid] = R[shape_idx] @ pts[tid] + T[shape_idx]


@wp.kernel
def compute_particle_f(
    gs_f: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int16),
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_idx = indices[tid]

    wp.atomic_add(particle_f, particle_idx, gs_f[tid])

# @wp.kernel
# def polar_decomposition(A: wp.array(dtype=wp.mat33), R: wp.array(dtype=wp.mat33)):
#     tid = wp.tid()
#
#     U = wp.mat33()
#     sigma = wp.vec3()
#     V = wp.mat33()
#
#     # SVD Decomposition
#     wp.svd3(A[tid], U, sigma, V)
#
#     Vt = wp.transpose(V)
#     Rt = wp.mul(U, Vt)
#
#     R[tid] = Rt
#
#
# @wp.func
# def calculate_rotation_matrix(A: wp.array(dtype=wp.mat33)):
#     R = wp.empty(len(A), dtype=wp.mat33)
#     wp.launch(kernel=polar_decomposition, dim=len(A), inputs=[A, R])
#     return R
#
#
# @wp.kernel
# def apply_transformation(R: wp.array(dtype=wp.mat33), T: wp.vec3, points: wp.array(dtype=wp.vec3), transformed_points: wp.array(dtype=wp.vec3)):
#     tid = wp.tid()
#     rotated_point = wp.mul(R[tid], points[tid])  # Apply rotation
#     transformed_points[tid] = rotated_point + T  # Apply translation
#
#
# @wp.kernel
# def sum_vec3(arr: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.vec3)):
#     tid = wp.tid()
#     wp.atomic_add(result, 0, arr[tid])
#
#
# @wp.kernel
# def subtract_mean(arr: wp.array(dtype=wp.vec3), mean: wp.vec3, result: wp.array(dtype=wp.vec3)):
#     tid = wp.tid()
#     result[tid] = arr[tid] - mean
#
#
# @wp.kernel
# def divide_vec3(result: wp.array(dtype=wp.vec3), count: int):
#     result[0] = result[0] / float(count)
#
#
# @wp.func
# def calculate_mean(arr: wp.array(dtype=wp.vec3)):
#     result = wp.zeros(1, dtype=wp.vec3)
#
#     # Sum array
#     wp.launch(kernel=sum_vec3, dim=len(arr), inputs=[arr, result])
#
#     # Divide by number of elements
#     wp.launch(kernel=divide_vec3, dim=1, inputs=[result, len(arr)])
#     return result
#
#
# @wp.kernel
# def outer_product_sum(arr1: wp.array(dtype=wp.vec3), arr2: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.mat33)):
#     tid = wp.tid()
#     outer = wp.outer(arr1[tid], arr2[tid])
#     wp.atomic_add(result, 0, outer)
#
#
# @wp.func
# def compute_transformation(
#     particle_x_init: wp.array(dtype=wp.vec3),
#     particle_x_rest: wp.array(dtype=wp.vec3),
#     particle_mass: wp.array(dtype=wp.vec3),
# ):
#     # compute center of mass
#     com_init = calculate_mean(particle_x_init)
#     com_rest = calculate_mean(particle_x_rest)
#
#     # Subtract means
#     q = wp.empty_like(particle_x_init)
#     p = wp.empty_like(particle_x_rest)
#     com_init_host = wp.vec3(*com_init.numpy()[0])
#     com_rest_host = wp.vec3(*com_rest.numpy()[0])
#     wp.launch(kernel=subtract_mean, dim=len(particle_x_init), inputs=[particle_x_init, com_init_host, q])
#     wp.launch(kernel=subtract_mean, dim=len(particle_x_rest), inputs=[particle_x_rest, com_rest_host, p])
#
#     # compute matrix A
#     mat_a = wp.zeros(1, dtype=wp.mat33)
#     wp.launch(kernel=outer_product_sum, dim=len(particle_x_init), inputs=[p, q, mat_a])
#
#     # polar decomposition
#     R = calculate_rotation_matrix(mat_a)
#
#     return R, com_rest


@wp.func
def velocity_at_point(qd: wp.spatial_vector, r: wp.vec3):
    """
    Returns the velocity of a point relative to the frame with the given spatial velocity.

    Args:
        qd (spatial_vector): The spatial velocity of the frame.
        r (vec3): The position of the point relative to the frame.

    Returns:
        vec3: The velocity of the point.
    """
    return wp.cross(wp.spatial_top(qd), r) + wp.spatial_bottom(qd)


@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    """
    Returns the twist around an axis.
    """

    # project imaginary part onto axis
    a = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(a, axis)
    a = proj * axis
    # if proj < 0.0:
    #     # ensure twist points in same direction as axis
    #     a = -a
    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


@wp.func
def quat_twist_angle(axis: wp.vec3, q: wp.quat):
    """
    Returns the angle of the twist around an axis.
    """
    return 2.0 * wp.acos(quat_twist(axis, q)[3])


@wp.func
def quat_decompose(q: wp.quat):
    """
    Decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x.
    """

    R = wp.matrix_from_cols(
        wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),
    )

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    sinp = -R[0, 2]
    if wp.abs(sinp) >= 1.0:
        theta = wp.HALF_PI * wp.sign(sinp)
    else:
        theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)


@wp.func
def quat_to_rpy(q: wp.quat):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = wp.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = wp.clamp(t2, -1.0, 1.0)
    pitch_y = wp.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = wp.atan2(t3, t4)

    return wp.vec3(roll_x, pitch_y, yaw_z)


@wp.func
def quat_to_euler(q: wp.quat, i: int, j: int, k: int) -> wp.vec3:
    """
    Convert a quaternion into Euler angles.

    :math:`i, j, k` are the indices in :math:`[0, 1, 2]` of the axes to use
    (:math:`i \\neq j, j \\neq k`).

    Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302

    Args:
        q (quat): The quaternion to convert
        i (int): The index of the first axis
        j (int): The index of the second axis
        k (int): The index of the third axis

    Returns:
        vec3: The Euler angles (in radians)
    """
    # i, j, k are actually assumed to follow 1-based indexing but
    # we want to be compatible with quat_from_euler
    i += 1
    j += 1
    k += 1
    not_proper = True
    if i == k:
        not_proper = False
        k = 6 - i - j  # because i + j + k = 1 + 2 + 3 = 6
    e = float((i - j) * (j - k) * (k - i)) / 2.0  # Levi-Civita symbol
    a = q[0]
    b = q[i]
    c = q[j]
    d = q[k] * e
    if not_proper:
        a -= q[j]
        b += q[k] * e
        c += q[0]
        d -= q[i]
    t2 = wp.acos(2.0 * (a * a + b * b) / (a * a + b * b + c * c + d * d) - 1.0)
    tp = wp.atan2(b, a)
    tm = wp.atan2(d, c)
    t1 = 0.0
    t3 = 0.0
    if wp.abs(t2) < 1e-6:
        t3 = 2.0 * tp - t1
    elif wp.abs(t2 - wp.HALF_PI) < 1e-6:
        t3 = 2.0 * tm + t1
    else:
        t1 = tp - tm
        t3 = tp + tm
    if not_proper:
        t2 -= wp.HALF_PI
        t3 *= e
    return wp.vec3(t1, t2, t3)


@wp.func
def quat_from_euler(e: wp.vec3, i: int, j: int, k: int) -> wp.quat:
    """
    Convert Euler angles to a quaternion.

    :math:`i, j, k` are the indices in :math:`[0, 1, 2]` of the axes in which the Euler angles are provided
    (:math:`i \\neq j, j \\neq k`), e.g. (0, 1, 2) for Euler sequence XYZ.

    Args:
        e (vec3): The Euler angles (in radians)
        i (int): The index of the first axis
        j (int): The index of the second axis
        k (int): The index of the third axis

    Returns:
        quat: The quaternion
    """
    # Half angles
    half_e = e / 2.0

    # Precompute sines and cosines of half angles
    cr = wp.cos(half_e[i])
    sr = wp.sin(half_e[i])
    cp = wp.cos(half_e[j])
    sp = wp.sin(half_e[j])
    cy = wp.cos(half_e[k])
    sy = wp.sin(half_e[k])

    # Components of the quaternion based on the rotation sequence
    return wp.quat(
        (cy * sr * cp - sy * cr * sp),
        (cy * cr * sp + sy * sr * cp),
        (sy * cr * cp - cy * sr * sp),
        (cy * cr * cp + sy * sr * sp),
    )


@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    # Frank & Park definition 3.20, pg 100

    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


@wp.func
def transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates).
    (Frank & Park, section 8.2.3, pg 290)
    """

    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.mul(wp.skew(p), R)

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def boltzmann(a: float, b: float, alpha: float):
    e1 = wp.exp(alpha * a)
    e2 = wp.exp(alpha * b)
    return (a * e1 + b * e2) / (e1 + e2)


@wp.func
def smooth_max(a: float, b: float, eps: float):
    d = a - b
    return 0.5 * (a + b + wp.sqrt(d * d + eps))


@wp.func
def smooth_min(a: float, b: float, eps: float):
    d = a - b
    return 0.5 * (a + b - wp.sqrt(d * d + eps))


@wp.func
def leaky_max(a: float, b: float):
    return smooth_max(a, b, 1e-5)


@wp.func
def leaky_min(a: float, b: float):
    return smooth_min(a, b, 1e-5)


@wp.func
def vec_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def vec_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def vec_leaky_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(leaky_min(a[0], b[0]), leaky_min(a[1], b[1]), leaky_min(a[2], b[2]))


@wp.func
def vec_leaky_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(leaky_max(a[0], b[0]), leaky_max(a[1], b[1]), leaky_max(a[2], b[2]))


@wp.func
def vec_abs(a: wp.vec3):
    return wp.vec3(wp.abs(a[0]), wp.abs(a[1]), wp.abs(a[2]))


def load_mesh(filename: str, method: str | None = None):
    """Load a 3D triangular surface mesh from a file.

    Args:
        filename (str): The path to the 3D model file (obj, and other formats
          supported by the different methods) to load.
        method (str): The method to use for loading the mesh (default ``None``).
          Can be either ``"trimesh"``, ``"meshio"``, ``"pcu"``, or ``"openmesh"``.
          If ``None``, every method is tried and the first successful mesh import
          where the number of vertices is greater than 0 is returned.

    Returns:
        Tuple of (mesh_points, mesh_indices), where mesh_points is a Nx3 numpy array of vertex positions (float32),
        and mesh_indices is a Mx3 numpy array of vertex indices (int32) for the triangular faces.
    """
    import os

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    def load_mesh_with_method(method):
        if method == "meshio":
            import meshio

            m = meshio.read(filename)
            mesh_points = np.array(m.points)
            mesh_indices = np.array(m.cells[0].data, dtype=np.int32)
        elif method == "openmesh":
            import openmesh

            m = openmesh.read_trimesh(filename)
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32)
        elif method == "pcu":
            import point_cloud_utils as pcu

            mesh_points, mesh_indices = pcu.load_mesh_vf(filename)
            mesh_indices = mesh_indices.flatten()
        else:
            import trimesh

            m = trimesh.load(filename)
            if hasattr(m, "geometry"):
                # multiple meshes are contained in a scene; combine to one mesh
                mesh_points = []
                mesh_indices = []
                index_offset = 0
                for geom in m.geometry.values():
                    vertices = np.array(geom.vertices, dtype=np.float32)
                    faces = np.array(geom.faces.flatten(), dtype=np.int32)
                    mesh_points.append(vertices)
                    mesh_indices.append(faces + index_offset)
                    index_offset += len(vertices)
                mesh_points = np.concatenate(mesh_points, axis=0)
                mesh_indices = np.concatenate(mesh_indices)
            else:
                # a single mesh
                mesh_points = np.array(m.vertices, dtype=np.float32)
                mesh_indices = np.array(m.faces.flatten(), dtype=np.int32)
        return mesh_points, mesh_indices

    if method is None:
        methods = ["trimesh", "meshio", "pcu", "openmesh"]
        for method in methods:
            try:
                mesh = load_mesh_with_method(method)
                if mesh is not None and len(mesh[0]) > 0:
                    return mesh
            except Exception:
                pass
        raise ValueError(f"Failed to load mesh using any of the methods: {methods}")
    else:
        mesh = load_mesh_with_method(method)
        if mesh is None or len(mesh[0]) == 0:
            raise ValueError(f"Failed to load mesh using method {method}")
        return mesh


def visualize_meshes(
    meshes: list[tuple[list, list]], num_cols=0, num_rows=0, titles=None, scale_axes=True, show_plot=True
):
    # render meshes in a grid with matplotlib
    import matplotlib.pyplot as plt

    if titles is None:
        titles = []

    num_cols = min(num_cols, len(meshes))
    num_rows = min(num_rows, len(meshes))
    if num_cols and not num_rows:
        num_rows = int(np.ceil(len(meshes) / num_cols))
    elif num_rows and not num_cols:
        num_cols = int(np.ceil(len(meshes) / num_rows))
    else:
        num_cols = len(meshes)
        num_rows = 1

    vertices = [np.array(v).reshape((-1, 3)) for v, _ in meshes]
    faces = [np.array(f, dtype=np.int32).reshape((-1, 3)) for _, f in meshes]
    if scale_axes:
        ranges = np.array([v.max(axis=0) - v.min(axis=0) for v in vertices])
        max_range = ranges.max()
        mid_points = np.array([v.max(axis=0) + v.min(axis=0) for v in vertices]) * 0.5

    fig = plt.figure(figsize=(12, 6))
    for i, (vertices, faces) in enumerate(meshes):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection="3d")
        if i < len(titles):
            ax.set_title(titles[i])
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, edgecolor="k")
        if scale_axes:
            mid = mid_points[i]
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    if show_plot:
        plt.show()
    return fig
