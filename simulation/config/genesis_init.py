import genesis as gs


class GenesisConfig(object):

    def __init__(self,
                 num_envs = 1,
                 device='cpu',
                 seed=None,
                 precision = '32',
                 logging_level = 'warning',
                 show_viewer=False,
                 record=False,
                 video_path='videos/preview.mp4',
                 fps=60,
                 cam_res=(1280, 960),
                 cam_pos=(3.5, 0.0, 2.5),
                 cam_lookat=(0.0, 0.0, 0.5),
                 cam_fov=40
                 ):

        self.num_envs = num_envs
        self.device = device
        self.seed = seed
        self.precision = precision
        self.logging_level = logging_level
        self.show_viewer = show_viewer
        self.record = record
        self.video_path = video_path
        self.fps = fps
        self.cam_res = cam_res
        self.cam_pos = cam_pos
        self.cam_lookat = cam_lookat
        self.cam_fov = cam_fov

        self.scene = None

    def gs_init(self):

        device_str = str(self.device)
        try:
            gs.init(
                seed = self.seed,
                precision = self.precision,
                debug = False,
                eps = 1e-12,
                logging_level = self.logging_level,
                backend = gs.cpu if device_str == 'cpu' else gs.gpu,
                theme = 'dark',
                logger_verbose_time = False
            )
        except Exception as exc:
            if "already initialized" not in str(exc).lower():
                raise

        self.scene = gs.Scene(
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
            show_viewer=self.show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=self.cam_pos,
                camera_lookat=self.cam_lookat,
                camera_fov=self.cam_fov,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=False,
                shadow=True,
                background_color=(0.02, 0.04, 0.08),
                ambient_light=(0.12, 0.12, 0.12),
                lights=[
                    {"type": "directional", "dir": (-0.6, -0.7, -1.0), "color": (1.0, 0.98, 0.95), "intensity": 3.0},
                    {"type": "directional", "dir": (0.4, 0.1, -1.0), "color": (0.9, 0.95, 1.0), "intensity": 1.5},
                ],
                rendered_envs_idx=list(range(self.num_envs)),
            ),
            renderer=gs.renderers.Rasterizer(),
        )
