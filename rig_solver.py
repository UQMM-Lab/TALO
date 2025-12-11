import numpy as np
import matplotlib.pyplot as plt

def plot_axis_projections_multi_camera(
    Rs_all: np.ndarray,           # (T, C, 3, 3) Sequence of rotation matrices: R for each camera at each time (in the reference camera coordinate system)
    body_axis: int = 2,           # Choose which axis of the body frame: 0=x, 1=y, 2=z (optical axis)
    labels: list = None,          # Camera names, length=C
    title: str = None
):
    """
    Plot the components of "body-axis direction vectors" on world coordinate x/y/z axes over time.
    Does not depend on mean values; directly shows whether data is concentrated or has outliers/jumps.

    - For each camera c and frame t:
    v_{t,c} = Rs_all[t, c][:, body_axis] ∈ R^3
    - Plot three time series v_x(t), v_y(t), v_z(t) respectively (each subplot contains C curves).

    Returns:
        V: (T, C, 3) stacked direction components for subsequent statistics
    """
    assert Rs_all.ndim == 4 and Rs_all.shape[-2:] == (3, 3), "Rs_all must have shape (T, C, 3, 3)"
    T, C = Rs_all.shape[:2]
    assert body_axis in (0, 1, 2)

    # Extract the body_axis column of each R: camera's axis direction in world coordinates
    # V[t, c, :] = Rs_all[t, c][:, body_axis]
    V = Rs_all[..., :, body_axis]  # shape = (T, C, 3)

    # Plot three subplots: world-x, world-y, world-z components
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    world_labels = ['world-X component', 'world-Y component', 'world-Z component']
    x = np.arange(T)

    for k in range(3):
        ax = axs[k]
        for c in range(C):
            name = labels[c] if (labels is not None and c < len(labels)) else f"cam{c}"
            series = V[:, c, k]
            ax.plot(x, series, '-o', markersize=2.5, linewidth=1.0, label=f"{name} (μ={series.mean():.3f}, σ={series.std():.3f})")
        ax.set_ylabel(world_labels[k])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc='upper right', fontsize=8)

    axs[-1].set_xlabel("Frame index (t)")
    if title is None:
        axis_name = ['camera-X axis', 'camera-Y axis', 'camera-Z axis'][body_axis]
        title = f"Projection of {axis_name} onto world axes over time"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    # Console output of statistics for each camera (mean/std/max for each component)
    for c in range(C):
        name = labels[c] if (labels is not None and c < len(labels)) else f"cam{c}"
        stats = []
        for k, wl in enumerate(['x','y','z']):
            s = V[:, c, k]
            stats.append(f"{wl}: μ={s.mean():.3f}, σ={s.std():.3f}, maxΔ={np.max(np.abs(s - s.mean())):.3f}")
        print(f"{name:>8s} | " + " | ".join(stats))

    return V


def plot_axis_on_sphere(
    Rs_all: np.ndarray,           # (T, C, 3, 3)
    cam_index: int,               # Select a specific camera
    body_axis: int = 2,           # Body frame axis 0/1/2
    title: str = None
):
    """
    Project the direction vector of a specific axis of a camera onto the unit sphere and plot a 3D scatter.
    If the data is stable, the points should cluster in a small area on the sphere; dispersion/bi-modality/outliers will be evident.
    """
    assert Rs_all.ndim == 4 and Rs_all.shape[-2:] == (3, 3)
    T, C = Rs_all.shape[:2]
    assert 0 <= cam_index < C and body_axis in (0, 1, 2)

    V = Rs_all[:, cam_index, :, body_axis]  # (T, 3)
    # Normalize (theoretically should already be unit vectors, numerically safe)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(V[:,0], V[:,1], V[:,2], s=12, alpha=0.8)
    # Draw coordinate axes
    ax.quiver(0,0,0, 1,0,0, length=1.0, color='r', linewidth=1.5)
    ax.quiver(0,0,0, 0,1,0, length=1.0, color='g', linewidth=1.5)
    ax.quiver(0,0,0, 0,0,1, length=1.0, color='b', linewidth=1.5)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if title is None:
        axis_name = ['camera-X axis', 'camera-Y axis', 'camera-Z axis'][body_axis]
        title = f"Unit-sphere scatter of {axis_name} (cam{cam_index})"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return V


def _skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=np.float64)

def so3_exp(w):
    """Rodrigues: w in R^3 -> R in SO(3)"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = _skew(k)
    s, c = np.sin(theta), np.cos(theta)
    return np.eye(3) + s * K + (1 - c) * (K @ K)

def so3_log(R):
    """Matrix log: R in SO(3) -> w in R^3 (axis-angle)"""
    # clamp trace for numerical stability
    tr = np.trace(R)
    x = (tr - 1.0) * 0.5
    x = np.clip(x, -1.0, 1.0)
    theta = np.arccos(x)
    if theta < 1e-12:
        return np.zeros(3)
    # when theta ~ pi, use robust formula
    if np.pi - theta < 1e-6:
        # Extract axis from R - R^T
        w = np.array([R[2,1] - R[1,2],
                      R[0,2] - R[2,0],
                      R[1,0] - R[0,1]]) / (2.0)
        if np.linalg.norm(w) < 1e-12:
            # Fallback: from diagonal
            A = (R + np.eye(3)) / 2.0
            axis = np.sqrt(np.maximum(np.diag(A), 0.0))
            # pick largest component as axis direction
            i = np.argmax(axis)
            v = np.zeros(3); v[i] = 1.0
            return theta * v
        return theta * w / np.linalg.norm(w)
    # general case
    w = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]]) / (2.0 * np.sin(theta))
    return theta * w

def geodesic_angle(Ra, Rb):
    """Angle between Ra and Rb on SO(3)"""
    R = Ra.T @ Rb
    x = (np.trace(R) - 1.0) * 0.5
    return np.arccos(np.clip(x, -1.0, 1.0))


def chordal_L2_mean_rotation(Rs: np.ndarray) -> np.ndarray:
    """
    Linear (fast): sum rotation matrices and project back to SO(3) via SVD
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

# ---------- 2) Geodesic L2 (Karcher mean) ----------

def karcher_L2_mean_rotation(Rs: np.ndarray, iters: int = 30, tol: float = 1e-12) -> np.ndarray:
    """
    Karcher mean (geodesic L2) on SO(3)
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    # Use chordal result as initial value for stability
    R = chordal_L2_mean_rotation(Rs)
    for _ in range(iters):
        dsum = np.zeros(3)
        for Ri in Rs:
            dsum += so3_log(R.T @ Ri)
        d = dsum / len(Rs)
        n = np.linalg.norm(d)
        R = R @ so3_exp(d)
        if n < tol:
            break
    return R

# ---------- 3) Quaternion L2 (fast approximation) ----------

def _R_to_q(R):
    """Rotation matrix -> unit quaternion (w,x,y,z)"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-15)

def _q_to_R(q):
    """Unit quaternion (w,x,y,z) -> rotation matrix"""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)
    return R

def quaternion_L2_mean_rotation(Rs: np.ndarray) -> np.ndarray:
    """
    Quaternion L2 mean: align hemispheres, then do Euclidean average and normalize (fast, good approximation)
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    qs = np.stack([_R_to_q(R) for R in Rs], axis=0)  # (T,4)
    # Align signs to the hemisphere of the first quaternion
    ref = qs[0]
    dots = np.dot(qs, ref)
    qs[dots < 0] *= -1.0
    q_mean = np.mean(qs, axis=0)
    q_mean /= (np.linalg.norm(q_mean) + 1e-15)
    return _q_to_R(q_mean)

# ---------- 4) Geodesic L1 (Weiszfeld on SO(3), robust median) ----------

def geodesic_L1_mean_rotation(Rs: np.ndarray, iters: int = 60, tol: float = 1e-10, eps: float = 1e-8) -> np.ndarray:
    """
    Geodesic L1 median on SO(3) (Weiszfeld): more robust to outliers
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    # Use Karcher L2 result as initial value for stability
    R = karcher_L2_mean_rotation(Rs, iters=15)
    for _ in range(iters):
        num = np.zeros(3)
        den = 0.0
        for Ri in Rs:
            v = so3_log(R.T @ Ri)  # tangent error
            n = np.linalg.norm(v)
            w = 1.0 / max(n, eps)  # Weiszfeld weight
            num += w * v
            den += w
        step = num / max(den, eps)
        nstep = np.linalg.norm(step)
        R = R @ so3_exp(step)
        if nstep < tol:
            break
    return R

# ---------- 5) Truncated-L2 / IRLS (Truncated robust L2) ----------

def truncated_L2_mean_rotation(Rs: np.ndarray, tau_deg: float = 5.0, iters: int = 20, tol: float = 1e-10) -> np.ndarray:
    """
    Truncated L2 (approximate TLUD idea): downweight large-angle residuals; approximate L2 for small angles.
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    R = chordal_L2_mean_rotation(Rs)
    tau = np.deg2rad(tau_deg)
    for _ in range(iters):
        num = np.zeros(3)
        den = 0.0
        for Ri in Rs:
            v = so3_log(R.T @ Ri)
            a = np.linalg.norm(v)
            if a < 1e-12:
                w = 1.0
            else:
                # Truncated weight: w=1 if a<=tau; w=tau/a if a>tau
                w = min(1.0, tau / a)
            num += w * v
            den += w
        d = num / max(den, 1e-12)
        nd = np.linalg.norm(d)
        R = R @ so3_exp(d)
        if nd < tol:
            break
    return R


def cal_rig(cam2worlds: np.ndarray, vis=False) -> np.ndarray:
    if vis:
        plot_axis_projections_multi_camera(
            Rs_all=cam2worlds[:, :, :3, :3],
            )
        for i in range(cam2worlds.shape[1]):
            plot_axis_on_sphere(
                Rs_all=cam2worlds[:, :, :3, :3],
                cam_index=i,
            )
    """
    Input:
      cam2worlds: (T, C, 4, 4)
        - The pose of each camera c at each time t is expressed in the coordinate system of the "reference camera (index 0)" at that time
        - Therefore, cam2worlds[:, 0] should be the identity matrix (or very close to it)

    Output:
      rig_poses: (C, 4, 4)
        - The 0th camera is a 4x4 identity matrix
        - The other cameras are the "average poses" in the reference camera coordinate system (rotation chordal-L2 mean + translation Euclidean mean)
    """
    assert cam2worlds.ndim == 4 and cam2worlds.shape[-2:] == (4, 4), "cam2worlds shape must be (T, C, 4, 4)"
    T, C = cam2worlds.shape[:2]

    rig_poses = np.zeros((C, 4, 4), dtype=np.float64)
    rig_poses[:] = np.eye(4)

    # Optional: check if the reference camera is close to identity matrix
    # (not a hard constraint, just a warning)
    ref_poses = cam2worlds[:, 0]  # (T, 4, 4)
    ref_dev = np.linalg.norm(ref_poses - np.eye(4), axis=(1, 2)).mean()
    if ref_dev > 1e-6:
        print(f"[Warn] Average norm of deviation of cam2worlds[:,0] from identity matrix = {ref_dev:.3e}, please ensure the input is transformed to the reference camera coordinate system.")

    for c in range(C):
        if c == 0:
            rig_poses[c] = np.eye(4)
            continue

        Tc = cam2worlds[:, c]  # (T, 4, 4)
        Rc = Tc[:, :3, :3]     # (T, 3, 3)
        tc = Tc[:, :3, 3]      # (T, 3)

        # Rotation averaging (baseline: chordal L2)
        R_mean = chordal_L2_mean_rotation(Rc)

        


        # Translation averaging (all in reference camera coordinate system, so Euclidean average is fine)
        t_mean = np.mean(tc, axis=0)

        # Assemble 4x4
        rig_poses[c, :3, :3] = R_mean
        rig_poses[c, :3, 3]   = t_mean
        rig_poses[c, 3, 3]    = 1.0

    return rig_poses


