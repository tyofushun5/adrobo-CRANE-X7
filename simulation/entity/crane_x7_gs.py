import os
import numpy as np
import genesis as gs


from simulation.entity.table import TABLE_HEIGHT, add_table


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

URDF_PATH = os.path.join(repo_root, "crane_x7_description", "urdf", "crane_x7_d435.urdf")


class CraneX7(object):
    def __init__(self, scene: gs.Scene = None, num_envs: int = 1, urdf_path=None, root_fixed: bool = True):
        super().__init__()
        self.crane_x7 = None
        self.num_envs = num_envs

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

        self.all_joint_names = self.arm_joint_names + self.gripper_joint_names

        self.arm_dofs_idx = None
        self.gripper_joint_dofs_idx = None
        self.all_joint_dofs_idx = None
        self.scene = scene
        self.urdf_path = urdf_path or URDF_PATH
        self.root_fixed = root_fixed
        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

        self.rest_qpos = np.array(
            [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0],
            dtype=np.float64,
        )

    def create(self, urdf_path=None, root_fixed=None):
        urdf_path = self.urdf_path if urdf_path is None else urdf_path
        root_fixed = self.root_fixed if root_fixed is None else root_fixed

        morph = gs.morphs.URDF(
            file=urdf_path,
            decimate=True,
            decimate_face_num=2000,
            decimate_aggressiveness=5,
            convexify=True,
            visualization=True,
            collision=True,
            requires_jac_and_IK=True,
            fixed=root_fixed,
        )

        self.crane_x7 = self.scene.add_entity(
            morph=morph,
            material=None,
            surface=None,
            visualize_contact=False,
            vis_mode="visual",
        )

        return self.crane_x7

    def set_gain(self):
        self.arm_dofs_idx = [self.crane_x7.get_joint(name).dof_idx_local for name in self.arm_joint_names]
        self.gripper_joint_dofs_idx = [self.crane_x7.get_joint(name).dof_idx_local for name in self.gripper_joint_names]
        self.all_joint_dofs_idx = self.arm_dofs_idx + self.gripper_joint_dofs_idx

        self.crane_x7.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            dofs_idx_local=self.all_joint_dofs_idx,
        )

        self.crane_x7.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
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
        control_mode: str = "joint",
        target_quat=None,
        ee_link_name: str = "crane_x7_gripper_base_link",
    ):
        if envs_idx is None:
            n_envs = getattr(self.scene, "n_envs", self.num_envs)
            envs_idx = list(range(n_envs))
        else:
            envs_idx = np.r_[envs_idx].tolist()

        if control_mode == "joint":
            target_qpos = np.asarray(target, dtype=np.float64)
            if target_qpos.ndim == 1:
                target_qpos = target_qpos.reshape(1, -1)
            if target_qpos.shape[0] == 1 and len(envs_idx) > 1:
                target_qpos = np.tile(target_qpos, (len(envs_idx), 1))
            target_qpos = target_qpos[: len(envs_idx)]
            self.crane_x7.control_dofs_position(target_qpos, self.all_joint_dofs_idx, envs_idx)
            return

        # IK mode
        target_pos = np.asarray(target, dtype=np.float64)
        target_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float64)

        if target_pos.ndim == 1:
            target_pos = target_pos.reshape(1, -1)
        if target_pos.shape[0] == 1 and len(envs_idx) > 1:
            target_pos = np.tile(target_pos, (len(envs_idx), 1))

        if target_quat is not None:
            if target_quat.ndim == 1:
                target_quat = target_quat.reshape(1, -1)
            if target_quat.shape[0] == 1 and len(envs_idx) > 1:
                target_quat = np.tile(target_quat, (len(envs_idx), 1))

        target_pos = target_pos[: len(envs_idx)]
        if target_quat is not None:
            target_quat = target_quat[: len(envs_idx)]

        ee_link = self.crane_x7.get_link(ee_link_name)
        ik_qpos = self.crane_x7.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=self.arm_dofs_idx,
            envs_idx=envs_idx,
        )
        self.crane_x7.control_dofs_position(ik_qpos, self.all_joint_dofs_idx, envs_idx)

    def init_pose(self, envs_idx=None):
        if envs_idx is None:
            n_envs = getattr(self.scene, "n_envs", self.num_envs)
            envs_idx = np.arange(n_envs)
        envs_idx = np.r_[envs_idx]

        rest_qpos = np.tile(self.rest_qpos, (len(envs_idx), 1))
        self.crane_x7.set_dofs_position(
            rest_qpos,
            self.all_joint_dofs_idx,
            zero_velocity=True,
            envs_idx=envs_idx.tolist(),
        )

if __name__ == "__main__":
    num_envs = 1
    mode = "ik"  # "ik" or "joint_demo"
    target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    show_viewer = True

    gs.init(
        seed = None,
        precision = '32',
        debug = False,
        eps = 1e-12,
        backend = gs.gpu,
        theme = 'dark',
        logger_verbose_time = False
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=150,
            tolerance=1e-6,
            contact_resolve_time=0.01,
            use_contact_island=False,
            use_hibernation=False
        ),
        show_viewer=show_viewer,
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

    plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -TABLE_HEIGHT)))
    table = add_table(scene)
    crane_x7 = CraneX7(scene, num_envs=num_envs, root_fixed=True)
    crane_x7.create()

    scene.build(n_envs=num_envs, env_spacing=(2.0, 3.0))
    crane_x7.set_gain()
    crane_x7.init_pose()

    if mode == "ik":
        # DayDreamer 実験用: ワークスペースをなめるIKスイープ
        traj = []
        # 前方 x-z グリッド
        for x in np.linspace(0.30, 0.45, 4):
            for z in np.linspace(0.12, 0.28, 4):
                traj.append((x, 0.0, z))
        # 左右スイープ（Y方向）
        for y in np.linspace(-0.10, 0.10, 5):
            traj.append((0.35, y, 0.20))
        # 上方へのアーク
        for theta in np.linspace(0.0, np.pi / 2, 6):
            x = 0.32 + 0.1 * np.cos(theta)
            z = 0.14 + 0.12 * np.sin(theta)
            traj.append((x, 0.0, z))

        for pos in traj:
            crane_x7.action(
                target=np.array(pos, dtype=np.float64),
                control_mode="ik",
                target_quat=target_quat,
            )
            for _ in range(90):
                scene.step()
    else:
        # 旧デモ: 関節空間の簡易コマンド
        dofs_idx = crane_x7.all_joint_dofs_idx
        robot = crane_x7.crane_x7
        for i in range(1250):
            if i == 0:
                robot.control_dofs_position(
                    np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), (num_envs, 1)),
                    dofs_idx,
                )
            elif i == 250:
                robot.control_dofs_position(
                    np.tile(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), (num_envs, 1)),
                    dofs_idx,
                )
            elif i == 500:
                robot.control_dofs_position(
                    np.tile(np.zeros(9, dtype=np.float64), (num_envs, 1)),
                    dofs_idx,
                )
            elif i == 750:
                # 最初の自由度を速度で制御し、残りを位置で制御
                robot.control_dofs_position(
                    np.tile(np.zeros(8, dtype=np.float64), (num_envs, 1)),
                    dofs_idx[1:],
                )
                robot.control_dofs_velocity(
                    np.tile(np.array([1.0], dtype=np.float64), (num_envs, 1)),
                    dofs_idx[:1],
                )
            elif i == 1000:
                robot.control_dofs_force(
                    np.tile(np.zeros(9, dtype=np.float64), (num_envs, 1)),
                    dofs_idx,
                )
            scene.step()
