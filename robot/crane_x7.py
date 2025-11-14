import sapien
import numpy as np
from pathlib import Path
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.controllers import PDJointPosMimicControllerConfig
from mani_skill.agents.controllers import PDEEPoseControllerConfig
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

    arm_stiffness = 80.0
    arm_damping = 8.0
    arm_force_limit = 40.0

    gripper_stiffness = 200.0
    gripper_damping = 20.0
    gripper_force_limit = 40.0

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

        arm_pd_ee_delta_pos = PDEEPoseControllerConfig(
            self.arm_joint_names,
            pos_lower=-0.05,
            pos_upper=0.05,
            rot_lower=0.0,
            rot_upper=0.0,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link="crane_x7_gripper_base_link",
            urdf_path=str(URDF_PATH),
            use_delta=True,
            use_target=False,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
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
            )
        )
        return deepcopy_dict(controller_configs)

    @property
    def _sensor_configs(self):

        p = [0.0, 0.0445, 0.034]
        q = [np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25)]

        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=p, q=q),
                width=640,
                height=480,
                fov=np.deg2rad(69),
                near=0.01,
                far=10.0,
                mount=self.robot.links_map["crane_x7_gripper_base_link"],
            )
        ]
