from pathlib import Path
import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import URDFLoader


loader = URDFLoader()
loader.set_scene(ManiSkillScene())

BASE_DIR = Path(__file__).resolve().parent
URDF_PATH = BASE_DIR / "crane_x7_description" / "urdf" / "crane_x7_d435.urdf"
robot = loader.load(str(URDF_PATH))
print(robot.active_joints_map.keys())
