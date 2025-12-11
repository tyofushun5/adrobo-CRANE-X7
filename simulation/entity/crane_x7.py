import os
import numpy as np
import genesis as gs


from simulation.entity.robot import Robot
from simulation.entity.table import Table
from simulation.entity.workspace import Workspace


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotate_vec(q, v):
    vq = np.concatenate([[0.0], v])
    return _quat_mul(_quat_mul(q, vq), _quat_conj(q))[1:]


class CraneX7(object):
    def __init__(self, scene = None, surface=None, num_envs = 1, root_fixed = True):
        super().__init__()
        self.scene = scene
        self.surface = surface
        self.num_envs = num_envs
        self.root_fixed = root_fixed

        self.crane_x7 = None

        self.max_delta = 0.01
        self.workspace_min = np.array([0.100, -0.160, 0.070], dtype=np.float64)
        self.workspace_max = np.array([0.340, 0.160, 0.300], dtype=np.float64)
        self.workspace_margin = 0.0
        self.table_z = 0.40

        self.default_ee_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        self._ee_cache = None
        self.max_joint_delta = np.array([1.2] * 7, dtype=np.float64)

        self.arm_joint_names = [
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
            "crane_x7_wrist_joint",
        ]

        self.gripper_joint_names = [
            "crane_x7_gripper_finger_a_joint",
            "crane_x7_gripper_finger_b_joint",
        ]

        self.ee_link_name = [
            "crane_x7_gripper_base_link",
        ]

        self.all_joint_names = [
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
            "crane_x7_wrist_joint",
            "crane_x7_gripper_finger_a_joint",
            "crane_x7_gripper_finger_b_joint",
        ]

        self.arm_dofs_idx = None
        self.gripper_joint_dofs_idx = None
        self.all_joint_dofs_idx = None
        self.urdf_path = os.path.join(repo_root, "crane_x7_description", "urdf", "crane_x7.urdf")

        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

        self.reset_qpos = np.array(
            [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, -np.pi / 2, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def create(self):

        morph = gs.morphs.URDF(
            file=self.urdf_path,
            decimate=True,
            decimate_face_num=2000,
            decimate_aggressiveness=5,
            convexify=True,
            visualization=True,
            collision=True,
            requires_jac_and_IK=True,
            fixed=self.root_fixed,
        )

        self.crane_x7 = self.scene.add_entity(
            morph=morph,
            material=None,
            surface=self.surfaces,
            visualize_contact=False,
            vis_mode="visual",
        )

        self.arm_dofs_idx = [self.crane_x7.get_joint(name).dof_idx_local for name in self.arm_joint_names]
        self.gripper_joint_dofs_idx = [self.crane_x7.get_joint(name).dof_idx_local for name in self.gripper_joint_names]
        self.all_joint_dofs_idx = self.arm_dofs_idx + self.gripper_joint_dofs_idx

        return self.crane_x7

    def set_gain(self):

        self.crane_x7.set_dofs_kp(
            kp=np.array([800, 800, 600, 600, 400, 400, 400, 50, 50]),
            dofs_idx_local=self.all_joint_dofs_idx,
        )

        self.crane_x7.set_dofs_kv(
            kv=np.array([80, 80, 60, 60, 40, 40, 40, 5, 5]),
            dofs_idx_local=self.all_joint_dofs_idx,
        )

        self.crane_x7.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.all_joint_dofs_idx,
        )

    def action(
            self,
            target,
            envs_idx=None,
            control_mode = "delta_xy",
            target_quat=None,
    ):
        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.r_[envs_idx]

        if control_mode in ("delta_xy", "delta_xyz"):
            delta = np.asarray(target, dtype=np.float64)
            if delta.ndim == 1:
                delta = delta.reshape(1, -1)
            if control_mode == "delta_xy" and delta.shape[1] != 2:
                raise ValueError(f"delta_xy expects shape (2,), got {delta.shape}")
            if control_mode == "delta_xyz" and delta.shape[1] != 3:
                raise ValueError(f"delta_xyz expects shape (3,), got {delta.shape}")

            base_p = np.zeros(3, dtype=np.float64)
            base_q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

            if self._ee_cache is None:
                mid_base = 0.5 * (self.workspace_min + self.workspace_max)
                mid_base[2] = self.table_z
                self._ee_cache = np.tile(mid_base, (self.num_envs, 1))

            delta = np.clip(delta, -1.0, 1.0) * self.max_delta
            targets = []
            ws_min_margin = self.workspace_min + self.workspace_margin
            ws_max_margin = self.workspace_max - self.workspace_margin
            for idx, d in enumerate(delta):
                env_idx = envs_idx[idx] if idx < len(envs_idx) else envs_idx[0]
                curr_pos_base = self._ee_cache[env_idx].copy()
                tgt = curr_pos_base.copy()
                if control_mode == "delta_xy":
                    tgt[:2] += d[:2]
                    tgt[2] = self.table_z
                else:
                    tgt[:3] += d[:3]
                tgt = np.clip(tgt, ws_min_margin, ws_max_margin)
                self._ee_cache[env_idx] = tgt
                tgt_world = base_p + _rotate_vec(base_q, tgt)
                targets.append(tgt_world)

            target = np.stack(targets, axis=0)
            if target.shape[0] == 1 and len(envs_idx) > 1:
                target = np.tile(target, (len(envs_idx), 1))
            return self.action(
                target=target[: len(envs_idx)],
                envs_idx=envs_idx,
                control_mode="ik",
                target_quat=np.tile(base_q, (len(envs_idx), 1)) if target_quat is None else target_quat,
                ee_link_name=self.ee_link_name,
            )

        target_pos = np.asarray(target, dtype=np.float64)
        target_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float64)

        if target_pos.ndim == 1:
            target_pos = target_pos.reshape(1, -1)

        if target_pos.shape[0] == 1 and len(envs_idx) > 1:
            target_pos = np.tile(target_pos, (len(envs_idx), 1))

        if target_quat is None:
            target_quat = np.tile(self.default_ee_quat, (target_pos.shape[0], 1))

        if target_quat.ndim == 1:
            target_quat = target_quat.reshape(1, -1)

        if target_quat.shape[0] == 1 and len(envs_idx) > 1:
            target_quat = np.tile(target_quat, (len(envs_idx), 1))

        target_pos = target_pos[: len(envs_idx)]
        target_quat = target_quat[: len(envs_idx)]
        ws_min_margin = self.workspace_min + self.workspace_margin
        ws_max_margin = self.workspace_max - self.workspace_margin
        target_pos = np.clip(target_pos, ws_min_margin, ws_max_margin)
        rot_mask = [True, True, True]
        ik_qpos = self.crane_x7.inverse_kinematics_multilink(
            links=[self.crane_x7.get_link(self.ee_link_name)],
            poss=[target_pos],
            quats=[target_quat],
            rot_mask=rot_mask,
            dofs_idx_local=self.arm_dofs_idx,
            envs_idx=envs_idx,
        )

        if hasattr(ik_qpos, "detach"):
            ik_qpos = ik_qpos.detach().cpu().numpy()
        else:
            ik_qpos = np.asarray(ik_qpos, dtype=np.float64)
        rest_arm = self.reset_qpos[:7]
        deltas = ik_qpos[:, :7] - rest_arm
        deltas = np.clip(deltas, -self.max_joint_delta, self.max_joint_delta)
        ik_qpos[:, :7] = rest_arm + deltas
        twist_idx = self.arm_dofs_idx[2]
        ik_qpos[:, twist_idx] = self.reset_qpos[2]
        fixed_idx = self.arm_dofs_idx[4]
        ik_qpos[:, fixed_idx] = self.reset_qpos[4]
        self.crane_x7.control_dofs_position(ik_qpos, self.all_joint_dofs_idx, envs_idx)

    def sample_workspace(
        self,
        grid=(12, 12, 12),
        rot_mask=None,
        target_quat=None,
        margin=0.0,
        save_path=None,
        return_points=True,
    ):
        rot_mask = [True, True, True] if rot_mask is None else rot_mask
        target_quat = self.default_ee_quat if target_quat is None else target_quat
        ee_link = self.crane_x7.get_link("crane_x7_gripper_base_link")

        ws_min = self.workspace_min + margin
        ws_max = self.workspace_max - margin
        xs = np.linspace(ws_min[0], ws_max[0], grid[0])
        ys = np.linspace(ws_min[1], ws_max[1], grid[1])
        zs = np.linspace(ws_min[2], ws_max[2], grid[2])

        reachable = []
        quat = np.asarray(target_quat, dtype=np.float64).reshape(1, -1)
        for x in xs:
            for y in ys:
                for z in zs:
                    pos = np.array([[x, y, z]], dtype=np.float64)
                    self.crane_x7.inverse_kinematics_multilink(
                        links=[ee_link],
                        poss=[pos],
                        quats=[quat],
                        rot_mask=rot_mask,
                        dofs_idx_local=self.arm_dofs_idx,
                    )
                    reachable.append((x, y, z))

        reachable = np.asarray(reachable, dtype=np.float64)
        if save_path is not None:
            np.save(save_path, reachable)
        return reachable if return_points else None

    def reset(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.r_[envs_idx]
        rest_qpos = np.tile(self.reset_qpos, (len(envs_idx), 1))

        self.crane_x7.set_dofs_position(
            rest_qpos,
            self.all_joint_dofs_idx,
            zero_velocity=True,
            envs_idx=envs_idx.tolist(),
        )

        midpoint = 0.5 * (self.workspace_min + self.workspace_max)
        midpoint[2] = self.table_z
        self._ee_cache = np.tile(midpoint, (self.num_envs, 1))

if __name__ == "__main__":
    os.environ.setdefault("TMPDIR", "/tmp")
    num_envs = 1
    mode = "delta_xy"
    draw_workspace_bounds = True
    fps = 60

    gs.init(
        seed=None,
        precision="32",
        debug=False,
        eps=1e-12,
        backend=gs.cpu,
        theme="dark",
        logger_verbose_time=False,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=150,
            tolerance=1e-6,
            contact_resolve_time=0.01,
            use_contact_island=False,
            use_hibernation=False,
        ),
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=35,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=True,
            shadow=True,
            background_color=(0.02, 0.04, 0.08),
            ambient_light=(0.12, 0.12, 0.12),
            lights=[
                {"type": "directional", "dir": (-0.6, -0.7, -1.0), "color": (1.0, 0.98, 0.95), "intensity": 3.0},
                {"type": "directional", "dir": (0.4, 0.1, -1.0), "color": (0.9, 0.95, 1.0), "intensity": 1.5},
            ],
            rendered_envs_idx=list(range(num_envs)),
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.9196429)))
    crane_x7 = CraneX7(scene, num_envs=num_envs, root_fixed=True)
    crane_x7.create()
    table = Table(scene)
    table.create()
    workspace = Workspace(scene)
    workspace.create()

    scene.build(n_envs=num_envs, env_spacing=(2.0, 3.0))

    crane_x7.set_gain()
    crane_x7.reset()


    settle_steps = 120
    for _ in range(settle_steps):
        scene.step()

    if mode in ("delta_xy", "delta_xyz"):
        num_steps = 10000
        rng = np.random.default_rng(0)
        for _ in range(num_steps):
            if mode == "delta_xy":
                action = rng.uniform(-1.0, 1.0, size=(2,))
            else:
                action = rng.uniform(-1.0, 1.0, size=(3,))
            crane_x7.action(
                target=action,
                control_mode=mode,
                target_quat=None,
            )
            scene.step()
    elif mode == "workspace_vis":
        while scene.viewer.is_alive():
            scene.step()
