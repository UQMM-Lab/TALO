import numpy as np
import matplotlib.pyplot as plt

def plot_axis_projections_multi_camera(
    Rs_all: np.ndarray,           # (T, C, 3, 3) 旋转矩阵序列：每时刻、每相机的 R（参考相机坐标系下）
    body_axis: int = 2,           # 选择机体系的哪条轴：0=x, 1=y, 2=z(光轴)
    labels: list = None,          # 相机名字，长度=C
    title: str = None
):
    """
    将“机体系某条轴”的方向向量在世界坐标系 x/y/z 上的分量随时间绘制出来。
    不依赖均值，可直接看数据是否集中、是否有离群/突变。

    - 对于每个相机 c、每个时刻 t：
    v_{t,c} = Rs_all[t, c][:, body_axis] ∈ R^3
    - 分别绘制 v_x(t), v_y(t), v_z(t) 三条时间序列（每张子图包含 C 条曲线）。

    返回:
        V: (T, C, 3) 所有方向分量堆叠，便于后续统计
    """
    assert Rs_all.ndim == 4 and Rs_all.shape[-2:] == (3, 3), "Rs_all must have shape (T, C, 3, 3)"
    T, C = Rs_all.shape[:2]
    assert body_axis in (0, 1, 2)

    # 取每个 R 的第 body_axis 列：相机该轴在世界坐标下的方向向量
    # V[t, c, :] = Rs_all[t, c][:, body_axis]
    V = Rs_all[..., :, body_axis]  # shape = (T, C, 3)

    # 画三张子图：world-x, world-y, world-z 分量
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

    # 控制台输出每个相机的统计（各分量 mean/std/max）
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
    cam_index: int,               # 选择某一相机
    body_axis: int = 2,           # 机体系轴 0/1/2
    title: str = None
):
    """
    将某个相机的“机体系某条轴”的方向向量投到单位球上画 3D 散点。
    若数据稳定，点应聚集在球面上某一小区；分散/双峰/离群会一目了然。
    """
    assert Rs_all.ndim == 4 and Rs_all.shape[-2:] == (3, 3)
    T, C = Rs_all.shape[:2]
    assert 0 <= cam_index < C and body_axis in (0, 1, 2)

    V = Rs_all[:, cam_index, :, body_axis]  # (T, 3)
    # 归一化（理论上应已是单位向量，数值上稳妥）
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(V[:,0], V[:,1], V[:,2], s=12, alpha=0.8)
    # 画坐标轴
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

# ---------- 工具函数（SO(3) 与 so(3)） ----------

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

# ---------- 1) Chordal L2（基线，已在用） ----------

def chordal_L2_mean_rotation(Rs: np.ndarray) -> np.ndarray:
    """
    线性快：对旋转矩阵求和再极分解投影回 SO(3)
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
    在 SO(3) 上做 Karcher 均值（测地 L2）
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    # 用 chordal 结果作为初值更稳
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

# ---------- 3) Quaternion L2（快速近似） ----------

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
    四元数 L2 平均：对齐半球后做欧氏平均并归一化（快速、好用的近似）
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    qs = np.stack([_R_to_q(R) for R in Rs], axis=0)  # (T,4)
    # 对齐符号到第一个四元数的半球
    ref = qs[0]
    dots = np.dot(qs, ref)
    qs[dots < 0] *= -1.0
    q_mean = np.mean(qs, axis=0)
    q_mean /= (np.linalg.norm(q_mean) + 1e-15)
    return _q_to_R(q_mean)

# ---------- 4) Geodesic L1（Weiszfeld on SO(3)，鲁棒中位数） ----------

def geodesic_L1_mean_rotation(Rs: np.ndarray, iters: int = 60, tol: float = 1e-10, eps: float = 1e-8) -> np.ndarray:
    """
    SO(3) 上的几何中位数（Weiszfeld）：对离群更鲁棒
    """
    assert Rs.ndim == 3 and Rs.shape[1:] == (3, 3)
    # 用 Karcher L2 结果作初值更稳
    R = karcher_L2_mean_rotation(Rs, iters=15)
    for _ in range(iters):
        num = np.zeros(3)
        den = 0.0
        for Ri in Rs:
            v = so3_log(R.T @ Ri)  # tangent error
            n = np.linalg.norm(v)
            w = 1.0 / max(n, eps)  # Weiszfeld 权重
            num += w * v
            den += w
        step = num / max(den, eps)
        nstep = np.linalg.norm(step)
        R = R @ so3_exp(step)
        if nstep < tol:
            break
    return R

# ---------- 5) Truncated-L2 / IRLS（截断式鲁棒 L2） ----------

def truncated_L2_mean_rotation(Rs: np.ndarray, tau_deg: float = 5.0, iters: int = 20, tol: float = 1e-10) -> np.ndarray:
    """
    截断式 L2（近似 TLUD 思想）：大角度残差降权；小角度近似 L2。
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
                # 截断式权重：a<=tau 时 w=1；a>tau 时 w=tau/a
                w = min(1.0, tau / a)
            num += w * v
            den += w
        d = num / max(den, 1e-12)
        nd = np.linalg.norm(d)
        R = R @ so3_exp(d)
        if nd < tol:
            break
    return R


def average_rig(cam2worlds: np.ndarray, vis=False) -> np.ndarray:
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
    输入:
      cam2worlds: (T, C, 4, 4)
        - 表示在每个时刻 t，每个相机 c 的位姿都已经表达在“该时刻参考相机(索引0)”的坐标系下
        - 因而 cam2worlds[:, 0] 应该是单位矩阵（或非常接近）

    输出:
      rig_poses: (C, 4, 4)
        - 第 0 个相机为 4x4 单位阵
        - 其余相机为在参考相机坐标系下的“平均位姿”（旋转 chordal-L2 平均 + 平移欧氏均值）
    """
    assert cam2worlds.ndim == 4 and cam2worlds.shape[-2:] == (4, 4), "cam2worlds shape must be (T, C, 4, 4)"
    T, C = cam2worlds.shape[:2]

    rig_poses = np.zeros((C, 4, 4), dtype=np.float64)
    rig_poses[:] = np.eye(4)

    # 可选：检查参考相机是否接近单位阵
    # （不是硬约束，只做提示）
    ref_poses = cam2worlds[:, 0]  # (T, 4, 4)
    ref_dev = np.linalg.norm(ref_poses - np.eye(4), axis=(1, 2)).mean()
    if ref_dev > 1e-6:
        print(f"[Warn] cam2worlds[:,0] 偏离单位阵的平均范数 = {ref_dev:.3e}，请确认输入是否已转到参考相机坐标系。")

    for c in range(C):
        if c == 0:
            rig_poses[c] = np.eye(4)
            continue

        Tc = cam2worlds[:, c]  # (T, 4, 4)
        Rc = Tc[:, :3, :3]     # (T, 3, 3)
        tc = Tc[:, :3, 3]      # (T, 3)

        # 旋转平均（baseline：chordal L2）
        R_mean = chordal_L2_mean_rotation(Rc)

        


        # 平均平移（都在参考相机坐标系下，可以直接欧氏平均）
        t_mean = np.mean(tc, axis=0)

        # 组装 4x4
        rig_poses[c, :3, :3] = R_mean
        rig_poses[c, :3, 3]   = t_mean
        rig_poses[c, 3, 3]    = 1.0

    return rig_poses


# =============== 示例用法 ===============
if __name__ == "__main__":
    # 假设我们有 T=5, C=3 的示例数据
    T, C = 5, 3
    cam2worlds = np.zeros((T, C, 4, 4), dtype=np.float64)
    cam2worlds[:] = np.eye(4)

    # 人为构造：相机1/2 相对参考相机的恒定位姿 + 小噪声
    R1 = np.array([[ 0.0, -1.0, 0.0],
                   [ 1.0,  0.0, 0.0],
                   [ 0.0,  0.0, 1.0]])
    t1 = np.array([0.5, 0.1, 0.0])

    R2 = np.array([[ 0.8660254, 0.0, 0.5],
                   [ 0.0,       1.0, 0.0],
                   [-0.5,       0.0, 0.8660254]])
    t2 = np.array([-0.4, 0.2, 0.0])

    rng = np.random.default_rng(0)
    def add_small_noise_T(R, t, noise_rot_deg=0.3, noise_t=0.01):
        # 极小噪声：用轴角小扰动近似 + 平移高斯噪声
        theta = np.deg2rad(noise_rot_deg)
        axis = rng.normal(size=3)
        axis /= (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        Rn = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
        tn = t + rng.normal(scale=noise_t, size=3)
        T = np.eye(4)
        T[:3,:3] = Rn @ R
        T[:3, 3] = tn
        return T

    for t in range(T):
        # 参考相机 = 单位
        cam2worlds[t, 0] = np.eye(4)
        cam2worlds[t, 1] = add_small_noise_T(R1, t1)
        cam2worlds[t, 2] = add_small_noise_T(R2, t2)

    rig = average_rig(cam2worlds)
    np.set_printoptions(precision=4, suppress=True)
    print("Estimated rig (C x 4 x 4):\n", rig)
