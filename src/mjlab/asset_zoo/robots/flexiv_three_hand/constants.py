"""Flexiv Rizon4S arm with myCobot parallel-jaw gripper (mesh visual + box collision)."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

FLEXIV_XML: Path = Path(
  "/home/yibocheng/compliance/assets/flexiv_rizon4s/flexiv_rizon4s.xml"
)
assert FLEXIV_XML.exists(), f"Flexiv XML not found: {FLEXIV_XML}"

GRIPPER_MESH_DIR: Path = Path(
  "/home/yibocheng/compliance/assets/mycobot_ros/mycobot_description/urdf/parallel_gripper"
)

# Scale factor: mesh vertices are in mm; _MESH_SCALE converts to meters AND enlarges.
# At 3×: gripper base ≈ 200×140×160 mm, max finger gap ≈ 45 mm, stroke ±21 mm per finger.
_MESH_SCALE = 2.0
_ATTACH_Z = 0.16  # m: link7 mesh extends ~154 mm; attach below it

# Finger stroke: original URDF 7 mm per finger, scaled up.
_JOINT_RANGE = 0.007 * _MESH_SCALE  # m per finger

# Finger collision box — mesh bounding box (mm) × _MESH_SCALE × 0.001:
#   x: [7.5, 24.5] mm → center 16 mm, half 8.5 mm
#   y: [-7.5, 13.5] mm → center 3 mm, half 10.5 mm
#   z: [-12, 48.5] mm → center 18.25 mm, half 30.25 mm
_S = _MESH_SCALE * 0.001
_FINGER_COL_CENTER = (0.016 * _MESH_SCALE, 0.003 * _MESH_SCALE, 0.01825 * _MESH_SCALE)
_FINGER_COL_HALF = (0.0085 * _MESH_SCALE, 0.0105 * _MESH_SCALE, 0.03025 * _MESH_SCALE)

# Grasp site near fingertips: when aligned with cube center (z=9mm above table),
# finger tips land at ~2mm above table. Finger tip is at 0.097m in gripper_base frame;
# cube center at table is ~0.009m world z; gripper_base then at 0.097+0.009=0.106m world z;
# grasp_site_z = 0.106 - 0.009 = 0.097 - (cube_z - clearance) ≈ 0.088m.
_GRASP_SITE_Z = 0.044 * _MESH_SCALE  # = 0.088 m at MESH_SCALE=2

GRIPPER_STIFFNESS = 200.0  # N/m
GRIPPER_DAMPING = 20.0  # N·s/m
GRIPPER_EFFORT_LIMIT = 10.0  # N


def get_spec() -> mujoco.MjSpec:
  """Load Flexiv arm and attach myCobot parallel-jaw gripper to link7."""
  spec = mujoco.MjSpec.from_file(str(FLEXIV_XML))

  # Register mesh assets (OBJ in mm units, scale mm→m and enlarge by _MESH_SCALE)
  for mesh_name in ("gripper_base", "gripper_left", "gripper_right"):
    m = spec.add_mesh()
    m.name = f"pg_{mesh_name}"
    m.file = str(GRIPPER_MESH_DIR / f"{mesh_name}.obj")
    m.scale = [_S, _S, _S]

  link7 = next(
    (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "link7"),
    None,
  )
  assert link7 is not None, "link7 not found in Flexiv spec"

  gripper_base = link7.add_body(name="gripper_base", pos=[0.0, 0.0, _ATTACH_Z])

  # Gripper base — visual mesh only, no collision
  base_vis = gripper_base.add_geom()
  base_vis.name = "gripper_base_vis"
  base_vis.type = mujoco.mjtGeom.mjGEOM_MESH
  base_vis.meshname = "pg_gripper_base"
  base_vis.contype = 0
  base_vis.conaffinity = 0

  grasp_site = gripper_base.add_site()
  grasp_site.name = "grasp_site"
  grasp_site.pos = [0.0, _FINGER_COL_CENTER[1], _GRASP_SITE_Z]
  grasp_site.size = [0.005, 0.0, 0.0]

  # ── Left finger (gripper_controller joint, slides in –X to close) ──────────
  left_body = gripper_base.add_body(name="left_finger")
  lj = left_body.add_joint()
  lj.name = "gripper_controller"
  lj.type = mujoco.mjtJoint.mjJNT_SLIDE
  lj.axis = [1, 0, 0]
  lj.range = [-_JOINT_RANGE, 0.0]

  lv = left_body.add_geom()
  lv.name = "left_finger_vis"
  lv.type = mujoco.mjtGeom.mjGEOM_MESH
  lv.meshname = "pg_gripper_left"
  lv.contype = 0
  lv.conaffinity = 0

  lc = left_body.add_geom()
  lc.name = "left_finger_col"
  lc.type = mujoco.mjtGeom.mjGEOM_BOX
  lc.pos = list(_FINGER_COL_CENTER)
  lc.size = list(_FINGER_COL_HALF)
  lc.rgba = [0.8, 0.5, 0.1, 1.0]

  # ── Right finger (mirrors left via equality, slides in +X to close) ─────────
  right_body = gripper_base.add_body(name="right_finger")
  rj = right_body.add_joint()
  rj.name = "gripper_base_to_gripper_left"
  rj.type = mujoco.mjtJoint.mjJNT_SLIDE
  rj.axis = [1, 0, 0]
  rj.range = [0.0, _JOINT_RANGE]  # positive range; equality drives it to -1×left_q

  rv = right_body.add_geom()
  rv.name = "right_finger_vis"
  rv.type = mujoco.mjtGeom.mjGEOM_MESH
  rv.meshname = "pg_gripper_right"
  rv.contype = 0
  rv.conaffinity = 0

  rc = right_body.add_geom()
  rc.name = "right_finger_col"
  rc.type = mujoco.mjtGeom.mjGEOM_BOX
  rc.pos = [-_FINGER_COL_CENTER[0], _FINGER_COL_CENTER[1], _FINGER_COL_CENTER[2]]
  rc.size = list(_FINGER_COL_HALF)
  rc.rgba = [0.8, 0.5, 0.1, 1.0]

  # Equality: right_q = 0 + (−1) × left_q  → mirrors closing motion
  eq = spec.add_equality()
  eq.type = mujoco.mjtEq.mjEQ_JOINT
  eq.name1 = "gripper_base_to_gripper_left"
  eq.name2 = "gripper_controller"
  eq.data[0] = 0.0
  eq.data[1] = -1.0

  return spec


ARM_ACTUATOR = XmlActuatorCfg(
  target_names_expr=(
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
  ),
)

GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("gripper_controller",),
  stiffness=GRIPPER_STIFFNESS,
  damping=GRIPPER_DAMPING,
  effort_limit=GRIPPER_EFFORT_LIMIT,
)

# q=0 → fully open (15 mm gap); equality handles right finger automatically
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 1.57,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
    "gripper_controller": 0.0,
    "gripper_base_to_gripper_left": 0.0,
  },
  joint_vel={".*": 0.0},
)

GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=("left_finger_col", "right_finger_col"),
  condim={
    "left_finger_col": 6,
    "right_finger_col": 6,
  },
  friction={
    "left_finger_col": (1.0, 5e-3, 5e-4),
    "right_finger_col": (1.0, 5e-3, 5e-4),
  },
  solref={
    "left_finger_col": (0.01, 1),
    "right_finger_col": (0.01, 1),
  },
  priority={
    "left_finger_col": 1,
    "right_finger_col": 1,
  },
  disable_other_geoms=True,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_ACTUATOR, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_flexiv_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


# Arm joints: 0.25 * effort_limit / stiffness (covers ~±25% of joint torque budget)
# Gripper: use _JOINT_RANGE directly so action=±1 maps to full open/close stroke
FLEXIV_ACTION_SCALE: dict[str, float] = {
  "joint1": 0.25 * 123 / 289,
  "joint2": 0.25 * 123 / 673,
  "joint3": 0.25 * 64 / 224,
  "joint4": 0.25 * 64 / 373,
  "joint5": 0.25 * 39 / 237,
  "joint6": 0.25 * 39 / 232,
  "joint7": 0.25 * 39 / 186,
  "gripper_controller": _JOINT_RANGE,  # action=-1 → full close, action=0 → open
}
