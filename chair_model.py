import bpy
import bmesh
import math
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.filepath = "D:\\PYTHON\\Первая-производная-гаусса\\chair_render.png"
scene.render.image_settings.file_format = 'PNG'

def create_material(name, color, roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat

mat_fabric = create_material("BlackFabric", (0.02, 0.02, 0.02), roughness=0.9, metallic=0.0)
mat_metal = create_material("BlackMetal", (0.01, 0.01, 0.01), roughness=0.3, metallic=0.9)

# === SEAT ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.45))
seat = bpy.context.active_object
seat.name = "Seat"
seat.scale = (0.45, 0.42, 0.05)
bpy.ops.object.transform_apply(scale=True)
seat.data.materials.append(mat_fabric)
bpy.ops.object.shade_smooth()

# === BACKREST ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.38, 0, 0.78))
back = bpy.context.active_object
back.name = "Backrest"
back.scale = (0.04, 0.40, 0.17)
bpy.ops.object.transform_apply(scale=True)
back.data.materials.append(mat_fabric)
bpy.ops.object.shade_smooth()

# Round the backrest top
bpy.ops.object.modifier_add(type='SUBSURF')
back.modifiers["Subdivision"].levels = 2
back.modifiers["Subdivision"].render_levels = 2

# === METAL FRAME - LEGS ===
leg_positions = [
    (0.30, 0.30, 0),    # front-right
    (0.30, -0.30, 0),   # front-left
    (-0.30, 0.30, 0),   # back-right
    (-0.30, -0.30, 0),  # back-left
]

for i, (x, y, dz) in enumerate(leg_positions):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.45,
                                         location=(x, y, 0.225 + dz),
                                         rotation=(0, 0, 0))
    leg = bpy.context.active_object
    leg.name = f"Leg_{i}"
    leg.data.materials.append(mat_metal)

# === CROSS BARS (front legs) ===
bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.60,
                                     location=(0.30, 0, 0.12),
                                     rotation=(0, math.pi/2, 0))
cross_front = bpy.context.active_object
cross_front.name = "CrossBar_Front"
cross_front.data.materials.append(mat_metal)

# === CROSS BARS (back legs) ===
bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.60,
                                     location=(-0.30, 0, 0.12),
                                     rotation=(0, math.pi/2, 0))
cross_back = bpy.context.active_object
cross_back.name = "CrossBar_Back"
cross_back.data.materials.append(mat_metal)

# === SIDE BARS (connect front to back) ===
for y_sign in [1, -1]:
    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.60,
                                         location=(0, y_sign * 0.30, 0.12),
                                         rotation=(math.pi/2, 0, 0))
    side = bpy.context.active_object
    side.name = f"SideBar_{y_sign}"
    side.data.materials.append(mat_metal)

# === ARMRESTS ===
for y_sign in [1, -1]:
    # vertical support from seat to armrest height
    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.25,
                                         location=(0.15, y_sign * 0.38, 0.575),
                                         rotation=(0, 0, 0))
    arm_support = bpy.context.active_object
    arm_support.name = f"ArmSupport_{y_sign}"
    arm_support.data.materials.append(mat_metal)

    # horizontal armrest bar
    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.35,
                                         location=(-0.05, y_sign * 0.38, 0.68),
                                         rotation=(0, math.pi/2, 0))
    arm_bar = bpy.context.active_object
    arm_bar.name = f"ArmBar_{y_sign}"
    arm_bar.data.materials.append(mat_metal)

# === BACK FRAME supports ===
for y_sign in [1, -1]:
    # vertical from seat back edge up to backrest
    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.35,
                                         location=(-0.30, y_sign * 0.30, 0.625),
                                         rotation=(0, 0, 0))
    back_support = bpy.context.active_object
    back_support.name = f"BackSupport_{y_sign}"
    back_support.data.materials.append(mat_metal)

# === BACK FRAME top bar ===
bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.60,
                                     location=(-0.38, 0, 0.90),
                                     rotation=(0, math.pi/2, 0))
back_top = bpy.context.active_object
back_top.name = "BackFrameTop"
back_top.data.materials.append(mat_metal)

# === GROUND PLANE ===
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
mat_ground = create_material("Ground", (0.15, 0.15, 0.15), roughness=0.8)
ground.data.materials.append(mat_ground)

# === CAMERA ===
bpy.ops.object.camera_add(location=(2.2, -1.8, 1.6),
                           rotation=(math.radians(65), 0, math.radians(48)))
cam = bpy.context.active_object
cam.name = "Camera"
scene.camera = cam

# === LIGHTS ===
bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
key = bpy.context.active_object
key.name = "KeyLight"
key.data.energy = 800
key.data.size = 2

bpy.ops.object.light_add(type='AREA', location=(-1.5, 2, 2.5))
fill = bpy.context.active_object
fill.name = "FillLight"
fill.data.energy = 300
fill.data.size = 3

bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy = 2

print("SCENE BUILT - rendering...")
bpy.ops.render.render(write_still=True)
print("DONE:", scene.render.filepath)
