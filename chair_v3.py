import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.filepath = "D:\\PYTHON\\Первая-производная-гаусса\\chair_v3.png"
scene.render.image_settings.file_format = 'PNG'

def mat(name, color, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes["Principled BSDF"]
    n.inputs[0].default_value = (*color, 1)
    n.inputs["Roughness"].default_value = rough
    n.inputs["Metallic"].default_value = metal
    return m

mat_fabric = mat("Fabric", (0.025, 0.025, 0.025), 0.88, 0.0)
mat_metal = mat("Metal", (0.01, 0.01, 0.01), 0.2, 0.95)
mat_floor = mat("Floor", (0.55, 0.55, 0.55), 0.85, 0.0)

def cyl(name, r, h, loc, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name
    o.data.materials.append(mat_metal); return o

def box(name, sx, sy, sz, loc, m):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object; o.name = name
    o.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(m); return o

SEAT_H = 0.42
SEAT_W = 0.42
SEAT_D = 0.40
SEAT_THICK = 0.04

# === SEAT ===
box("Seat", SEAT_W/2, SEAT_D/2, SEAT_THICK, (0, 0, SEAT_H), mat_fabric)

# === BACKREST ===
BACK_H = 0.17
BACK_W = 0.36
back_z = SEAT_H + SEAT_THICK/2 + BACK_H/2 + 0.02
back = box("Backrest", BACK_W/2, 0.02, BACK_H/2, (-SEAT_D/2 + 0.02, 0, back_z), mat_fabric)
bpy.ops.object.modifier_add(type='SUBSURF')
back.modifiers["Subdivision"].levels = 3
bpy.ops.object.shade_smooth()

# === LEGS ===
LEG_R = 0.013
FRONT_X = SEAT_W/2 - 0.04
BACK_X = -SEAT_W/2 + 0.04
SIDE_Y = SEAT_D/2 - 0.04

# Front legs - straight
cyl("LegFL", LEG_R, SEAT_H, (FRONT_X, SIDE_Y, SEAT_H/2))
cyl("LegFR", LEG_R, SEAT_H, (FRONT_X, -SIDE_Y, SEAT_H/2))

# Back legs - slight backward angle
cyl("LegBL", LEG_R, SEAT_H + 0.06, (BACK_X, SIDE_Y, SEAT_H/2 + 0.03), (0.08, 0, 0))
cyl("LegBR", LEG_R, SEAT_H + 0.06, (BACK_X, -SIDE_Y, SEAT_H/2 + 0.03), (0.08, 0, 0))

# === CROSS BAR front legs ===
cyl("CrossF", 0.009, SEAT_W - 0.08, (FRONT_X, 0, 0.10), (0, math.pi/2, 0))

# === SIDE BARS ===
cyl("SideR", 0.009, SEAT_D - 0.08, (0, SIDE_Y, 0.10), (math.pi/2, 0, 0))
cyl("SideL", 0.009, SEAT_D - 0.08, (0, -SIDE_Y, 0.10), (math.pi/2, 0, 0))

# === BACK FRAME supports (from seat up to backrest) ===
BACK_SUP_H = back_z - SEAT_H/2
cyl("BackSupR", 0.009, BACK_SUP_H, (-SEAT_D/2 + 0.02, SIDE_Y - 0.02, SEAT_H/2 + BACK_SUP_H/2))
cyl("BackSupL", 0.009, BACK_SUP_H, (-SEAT_D/2 + 0.02, -SIDE_Y + 0.02, SEAT_H/2 + BACK_SUP_H/2))

# === BACK TOP BAR ===
cyl("BackTop", 0.009, BACK_W - 0.04, (-SEAT_D/2 + 0.02, 0, back_z + BACK_H/2 + 0.01), (0, math.pi/2, 0))

# === ARMREST vertical supports ===
ARM_SUP_H = 0.20
ARM_Z = SEAT_H + ARM_SUP_H/2
cyl("ArmSupFR", 0.009, ARM_SUP_H, (FRONT_X - 0.04, SIDE_Y + 0.02, ARM_Z))
cyl("ArmSupFL", 0.009, ARM_SUP_H, (FRONT_X - 0.04, -SIDE_Y - 0.02, ARM_Z))

# === ARMREST horizontal bars ===
ARM_BAR_H = SEAT_H + ARM_SUP_H
cyl("ArmR", 0.009, SEAT_D - 0.06, (FRONT_X/2 - 0.06, SIDE_Y + 0.02, ARM_BAR_H), (0, math.pi/2, 0))
cyl("ArmL", 0.009, SEAT_D - 0.06, (FRONT_X/2 - 0.06, -SIDE_Y - 0.02, ARM_BAR_H), (0, math.pi/2, 0))

# === GROUND ===
bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
gnd = bpy.context.active_object; gnd.name = "Floor"
gnd.data.materials.append(mat_floor)

# === CAMERA ===
bpy.ops.object.camera_add(location=(1.5, -1.5, 1.2),
                           rotation=(math.radians(68), 0, math.radians(48)))
cam = bpy.context.active_object; scene.camera = cam

# === LIGHTS ===
bpy.ops.object.light_add(type='AREA', location=(1.8, -1.2, 2.8))
k = bpy.context.active_object; k.data.energy = 550; k.data.size = 2.5

bpy.ops.object.light_add(type='AREA', location=(-1.0, 1.5, 2.2))
f = bpy.context.active_object; f.data.energy = 220; f.data.size = 3

bpy.ops.object.light_add(type='SUN', location=(0.5, -0.5, 5))
s = bpy.context.active_object; s.data.energy = 1.2

print("RENDERING v3...")
bpy.ops.render.render(write_still=True)
print("DONE:", scene.render.filepath)
