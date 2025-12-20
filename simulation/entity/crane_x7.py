import os
import numpy as np
import genesis as gs

from simulation.entity.workspace import Workspace

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

class CraneX7(object):
    def __init__(self, scene = None, surface=None, num_envs = 1, control_mode = "discrete_xyz", root_fixed = True):
        super().__init__()
        self.scene = scene
        self.surface = surface
        self.num_envs = num_envs
        self.control_mode = control_mode
        self.root_fixed = root_fixed

        self.crane_x7 = None

        self.urdf_path = os.path.join(repo_root, "crane_x7_description", "urdf", "crane_x7.urdf")

        self.init_pos = [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, -np.pi / 2, 0.0, -0.0873,  -0.0873]
        self.reset_qpos = np.array(self.init_pos, dtype=np.float64)

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

        self.ee_link_name = "crane_x7_gripper_base_link"
        self.ee_link = None

        self.num_delta = 0.01
        self.table_z = 0.0

        self.default_ee_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        self.__ee_cache = None
        self.max_joint_delta = np.array([1.2] * 7, dtype=np.float64)
        self.is_open_gripper = True

        self.arm_dofs_idx = None
        self.gripper_joint_dofs_idx = None
        self.all_joint_dofs_idx = None

        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

        self.workspace = Workspace(self.scene)


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
            prioritize_urdf_material=True,
            fixed=self.root_fixed,
        )

        self.crane_x7 = self.scene.add_entity(
            morph=morph,
            material=None,
            surface=self.surface,
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

    def get_joint_positions(self, envs_idx=None, include_gripper=True):
        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.r_[envs_idx]

        dof_idx = self.all_joint_dofs_idx if include_gripper else self.arm_dofs_idx
        qpos = self.crane_x7.get_dofs_position(
            dofs_idx_local=dof_idx,
            envs_idx=envs_idx,
        )
        if hasattr(qpos, "detach"):
            qpos = qpos.detach().cpu().numpy()
        else:
            qpos = np.asarray(qpos)
        return qpos

    def action(self, action, envs_idx=None):
        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.r_[envs_idx]

        action = np.asarray(action, dtype=np.int64)
        if action.ndim == 0:
            action = action.reshape(1)

        delta = np.zeros((len(action), 3), dtype=np.float64)

        for i, a in enumerate(action):
            if a == 0:
                delta[i, 0] = +1
            elif a == 1:
                delta[i, 0] = -1
            elif a == 2:
                delta[i, 1] = +1
            elif a == 3:
                delta[i, 1] = -1
            elif a == 4:
                if self.is_open_gripper is False:
                    delta[i, 2] = +1
            elif a == 5:
                if self.is_open_gripper is False:
                    delta[i, 2] = -1
            elif a == 6:
                self.is_open_gripper = False
                target = np.tile(np.array([[-0.0873, -0.0873]], dtype=np.float64), (len(envs_idx), 1))
                self.crane_x7.control_dofs_position(target, self.gripper_joint_dofs_idx, envs_idx)
            elif a == 7:
                self.is_open_gripper = True
                target = np.tile(np.array([[1.57, 1.57]], dtype=np.float64), (len(envs_idx), 1))
                self.crane_x7.control_dofs_position(target, self.gripper_joint_dofs_idx, envs_idx)
            else:
                raise ValueError("invalid discrete action")


        delta = np.clip(delta, -1.0, 1.0) * self.num_delta

        base_pos = np.zeros(3, dtype=np.float64)
        base_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        if self.__ee_cache is None:
            mid_base = 0.5 * (self.workspace.workspace_min + self.workspace.workspace_max)
            mid_base[2] = self.table_z
            self.__ee_cache = np.tile(mid_base, (self.num_envs, 1))

        targets = []
        ws_min_margin = self.workspace.workspace_min + self.workspace.workspace_margin
        ws_max_margin = self.workspace.workspace_max - self.workspace.workspace_margin
        for idx, d in enumerate(delta):
            env_idx = envs_idx[idx]
            current_pos_base = self.__ee_cache[env_idx].copy()
            tgt = current_pos_base.copy()
            tgt[:3] += d[:3]
            tgt = np.clip(tgt, ws_min_margin, ws_max_margin)
            self.__ee_cache[env_idx] = tgt
            tgt_world = base_pos + self.rotate_vec(base_quat, tgt)
            targets.append(tgt_world)

        target_pos = np.stack(targets, axis=0)
        return self.ik(target_pos, envs_idx, base_quat)

    def ik(self, target, envs_idx, target_quat=None):
        target_pos = np.asarray(target, dtype=np.float64)
        target_quat = self.default_ee_quat if target_quat is None else np.asarray(target_quat, dtype=np.float64)

        if target_pos.ndim == 1:
            target_pos = target_pos.reshape(1, -1)
        if target_pos.shape[0] == 1 and len(envs_idx) > 1:
            target_pos = np.tile(target_pos, (len(envs_idx), 1))

        if target_quat.ndim == 1:
            target_quat = target_quat.reshape(1, -1)
        if target_quat.shape[0] == 1 and len(envs_idx) > 1:
            target_quat = np.tile(target_quat, (len(envs_idx), 1))

        target_pos = target_pos[: len(envs_idx)]
        target_quat = target_quat[: len(envs_idx)]
        ws_min_margin = self.workspace.workspace_min + self.workspace.workspace_margin
        ws_max_margin = self.workspace.workspace_max - self.workspace.workspace_margin
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
        grip_pos = 0.0 if self.is_open_gripper else 1.57
        ik_qpos[:, 7:] = grip_pos
        self.crane_x7.control_dofs_position(ik_qpos, self.all_joint_dofs_idx, envs_idx)

    def reset(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.r_[envs_idx]

        reset_qpos = np.tile(self.reset_qpos, (len(envs_idx), 1))

        self.crane_x7.set_dofs_position(
            reset_qpos,
            self.all_joint_dofs_idx,
            zero_velocity=True,
            envs_idx=envs_idx.tolist(),
        )

        midpoint = 0.5 * (self.workspace.workspace_min + self.workspace.workspace_max)
        midpoint[2] = self.table_z
        self.__ee_cache = np.tile(midpoint, (self.num_envs, 1))
        self.is_open_gripper = True
        target = np.tile(np.array([[ -0.0873,  -0.0873]], dtype=np.float64), (len(envs_idx), 1))
        self.crane_x7.control_dofs_position(target, self.gripper_joint_dofs_idx, envs_idx)

    @staticmethod
    def rotate_vec(q, v):
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        w, x, y, z = q
        q_vec = np.array([x, y, z], dtype=np.float64)
        uv = np.cross(q_vec, v)
        uuv = np.cross(q_vec, uv)
        return v + 2.0 * (w * uv + uuv)
