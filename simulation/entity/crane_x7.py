import sapien
import numpy as np
from pathlib import Path
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.controllers import PDJointPosMimicControllerConfig
from mani_skill.agents.controllers import PDEEPosControllerConfig
from mani_skill.agents.controllers import deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


BASE_DIR = Path(__file__).resolve().parent
URDF_PATH = BASE_DIR / "crane_x7_description" / "urdf" / "crane_x7_d435.urdf"


@register_agent()
class CraneX7(BaseAgent):
    uid = "CRANE-X7"
    urdf_path = str(URDF_PATH)

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, -np.pi / 2, np.pi / 2, 0.0, 0.0]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "crane_x7_shoulder_fixed_part_pan_joint",
        "crane_x7_shoulder_revolute_part_tilt_joint",
        "crane_x7_upper_arm_revolute_part_twist_joint",
        "crane_x7_upper_arm_revolute_part_rotate_joint",
        "crane_x7_lower_arm_fixed_part_joint",
        "crane_x7_lower_arm_revolute_part_joint",
        "crane_x7_wrist_joint",
    ]
    gripper_joint_names = [
        "crane_x7_gripper_finger_a_joint",
        "crane_x7_gripper_finger_b_joint",
    ]

    #素材について
    # urdf_config = dict(
    #     _materials=dict(
    #         gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
    #     ),
    #     link=dict(
    #         panda_leftfinger=dict(
    #             material="gripper", patch_radius=0.1, min_patch_radius=0.1
    #         ),
    #         panda_rightfinger=dict(
    #             material="gripper", patch_radius=0.1, min_patch_radius=0.1
    #         ),
    #     ),
    # )

    arm_stiffness = 10.0
    arm_damping = 10.0
    arm_force_limit = 5.0

    gripper_stiffness = 100.0
    gripper_damping = 20.0
    gripper_force_limit = 5.0

    # Anchor-relative offsets (base-link local frame).
    # EEは初回の基準位置（ベース座標系）からこの範囲にクリップされる。
    ee_workspace_offset_lower = np.array([0.0, 0.0, 0.0])
    ee_workspace_offset_upper = np.array([0.0, 0.0, 0.0])
    _ee_anchor_local = None

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            pos_lower=-0.01,
            pos_upper=0.01,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link="crane_x7_gripper_base_link",
            urdf_path=str(URDF_PATH),
            use_delta=True,
            use_target=False,
            interpolate=True,
            normalize_action=True,
            frame="root_translation"
        )

        arm_pd_ee_delta_pos_xy = PDEEPosControllerConfig(
            self.arm_joint_names,
            pos_lower=[-0.005, -0.005, 0.0],
            pos_upper=[0.005, 0.005, 0.0],
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link="crane_x7_gripper_base_link",
            urdf_path=str(URDF_PATH),
            use_delta=True,
            use_target=False,
            interpolate=True,
            normalize_action=True,
            frame="crane_x7_gripper_base_link",
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.04,
            upper=1.571,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={
                "crane_x7_gripper_finger_b_joint": dict(
                    joint="crane_x7_gripper_finger_a_joint",
                    multiplier=1.0,
                    offset=0.0,
                )
            },
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pos_xy_clip=dict(
                arm=arm_pd_ee_delta_pos_xy,
                gripper=gripper_pd_joint_pos,
            )
        )
        return deepcopy_dict(controller_configs)

    def set_action(self, action):
        """
        Intercept actions for the XY-only EE delta controller to clip workspace
        and zero-out Z displacement before delegating to the controller.
        """
        print("set_action", self.control_mode, "incoming shape", getattr(action, "shape", None))
        if self.control_mode in {"pd_ee_delta_pos_xy_clip", "pd_ee_delta_pos"}:
            import torch

            def _pose_to_np(pose):
                p = pose.p
                q = pose.q
                if hasattr(p, "detach"):
                    p = p.detach().cpu().numpy()
                else:
                    p = np.array(p)
                if hasattr(q, "detach"):
                    q = q.detach().cpu().numpy()
                else:
                    q = np.array(q)
                p = p.astype(np.float32)
                q = q.astype(np.float32)
                if p.ndim == 2 and p.shape[0] == 1:
                    p = p[0]
                if q.ndim == 2 and q.shape[0] == 1:
                    q = q[0]
                return p, q

            # Ensure batched shape (B, action_dim)
            action = torch.as_tensor(action, device=self.scene.device)
            if action.ndim == 1:
                action = action.unsqueeze(0)

            # Locate arm slice in the flattened CombinedController action
            arm_start, arm_end = self.controller.action_mapping["arm"]
            arm_action = action[:, arm_start:arm_end]
            arm_ctrl = self.controller.controllers["arm"]

            # Convert normalized [-1,1] action back to delta (meters)
            if getattr(arm_ctrl, "_normalize_action", False):
                low, high = arm_ctrl.action_space_low, arm_ctrl.action_space_high
                delta = 0.5 * (high + low) + 0.5 * (high - low) * arm_action
            else:
                delta = arm_action

            # Force Z to stay unchanged for XY-only controller
            if self.control_mode == "pd_ee_delta_pos_xy_clip":
                delta[..., 2] = 0.0

            # Clip resulting target position in base-link local frame
            # Gather base and EE poses (numpy)
            base_link = self.robot.links_map.get(
                "crane_x7_base_link", self.robot.get_links()[0]
            )
            base_p, base_q = _pose_to_np(base_link.pose)
            ee_p, ee_q = _pose_to_np(
                self.robot.links_map["crane_x7_gripper_base_link"].pose
            )
            base_pose = sapien.Pose(p=base_p, q=base_q)
            ee_pose = sapien.Pose(p=ee_p, q=ee_q)
            ee_pose_local = (base_pose.inv() * ee_pose).p.astype(np.float32)

            # Capture anchor in base-local frame on first use
            if self._ee_anchor_local is None:
                anchor_local_pose = base_pose.inv() * ee_pose
                self._ee_anchor_local = np.array(anchor_local_pose.p, dtype=np.float32)

            # Current delta as numpy (B,3)
            delta_np = delta.detach().cpu().numpy().astype(np.float32)

            # Anchor-relative target in base local, clamp, back to world
            lower_np = self._ee_anchor_local + self.ee_workspace_offset_lower
            upper_np = self._ee_anchor_local + self.ee_workspace_offset_upper
            clamped_deltas = []
            for d_local in delta_np:
                # 1) target in base local = current pose + delta_local
                target_local = ee_pose_local + d_local
                # 2) clamp to AABB in base frame around anchor
                target_local = np.clip(target_local, lower_np, upper_np)
                # 3) delta in base frame (controller expects root_translation frame)
                clamped_delta_np = target_local.astype(np.float32) - ee_pose_local
                clamped_deltas.append(clamped_delta_np)

            delta = torch.as_tensor(np.stack(clamped_deltas, axis=0), device=self.scene.device)

            # Re-normalize back to controller action space if needed
            if getattr(arm_ctrl, "_normalize_action", False):
                low, high = arm_ctrl.action_space_low, arm_ctrl.action_space_high
                scaled = (2 * delta - (high + low)) / (high - low)
            else:
                scaled = delta
            # Replace NaNs (e.g., invalid IK/clip) with zeros to avoid crashing
            scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Use the clamped delta (re-normalized) for the arm slice
            action[:, arm_start:arm_end] = scaled
            # Clamp full action (including gripper) to [-1, 1] to avoid runaway values
            action = torch.clamp(action, -1.0, 1.0)
            print("final action (clamped):", action)

        super().set_action(action)

    # @property
    # def _sensor_configs(self):
    #
    #     p = [0.0, 0.0445, 0.034]
    #     q = [np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25)]
    #
    #     return [
    #         CameraConfig(
    #             uid="hand_camera",
    #             pose=sapien.Pose(p=p, q=q),
    #             width=640,
    #             height=480,
    #             fov=np.deg2rad(69),
    #             near=0.01,
    #             far=10.0,
    #             mount=self.entity.links_map["crane_x7_gripper_base_link"],
    #         )
    #     ]

