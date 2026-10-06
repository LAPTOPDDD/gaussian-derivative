import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.filepath = "D:\\PYTHON\\Первая-производная-гаусса\\chair_v6.png"
scene.render.image_settings.file_format = 'PNG'

def mat(name, color, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes["Principled BSDF"]
    n.inputs[0].default_value = (*color, 1)
    n.inputs["Roughness"].default_value = rough
    n.inputs["Metallic"].default_value = metal
    return m

mat_fabric = mat("Fabric", (0.02, 0.02, 0.02), 0.9, 0.0)
mat_metal = mat("Metal", (0.01, 0.01, 0.01), 0.2, 0.95)
mat_floor = mat("Floor", (0.5, 0.5, 0.5), 0.85, 0.0)

def cyl(name, r, h, loc, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name
    o.data.materials.append(mat_metal); return o

# === DIMENSIONS ===
SEAT_H = 0.42        # seat top height from ground
SEAT_T = 0.04        # seat thickness
SEAT_BOT = SEAT_H - SEAT_T  # seat bottom = 0.38
SEAT_W = 0.42        # width (Y)
SEAT_D = 0.40        # depth (X)
BACK_H = 0.15        # backrest height
BACK_W = 0.34        # backrest width
LEG_R = 0.013
TUBE_R = 0.009

# Corner positions (X forward, Y right)
FX = SEAT_D/2 - 0.03   # front X
BX = -SEAT_D/2 + 0.03  # back X
RY = SEAT_W/2 - 0.03   # right Y
LY = -SEAT_W/2 + 0.03  # left Y

# ================================================
# SEAT CUSHION: bottom at SEAT_BOT, top at SEAT_H
# ================================================
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, SEAT_BOT + SEAT_T/2))
seat = bpy.context.active_object; seat.name = "Seat"
seat.scale = (SEAT_D/2, SEAT_W/2, SEAT_T/2)
bpy.ops.object.transform_apply(scale=True)
seat.data.materials.append(mat_fabric)

# ================================================
# BACKREST: starts at seat top, goes up
# ================================================
bpy.ops.mesh.primitive_cube_add(size=1, location=(BX + 0.01, 0, SEAT_H + BACK_H/2))
back = bpy.context.active_object; back.name = "Backrest"
back.scale = (0.02, BACK_W/2, BACK_H/2)
bpy.ops.object.transform_apply(scale=True)
back.data.materials.append(mat_fabric)
bpy.ops.object.modifier_add(type='SUBSURF')
back.modifiers["Subdivision"].levels = 3
bpy.ops.object.shade_smooth()

# ================================================
# 4 LEGS: ground (Z=0) to seat bottom (Z=SEAT_BOT)
# ================================================
cyl("Leg_FL", LEG_R, SEAT_BOT, (FX, RY, SEAT_BOT/2))
cyl("Leg_FR", LEG_R, SEAT_BOT, (FX, LY, SEAT_BOT/2))
cyl("Leg_BL", LEG_R, SEAT_BOT, (BX, RY, SEAT_BOT/2))
cyl("Leg_BR", LEG_R, SEAT_BOT, (BX, LY, SEAT_BOT/2))

# ================================================
# CROSS BAR between front legs at Z=0.10
# ================================================
cyl("CrossF", TUBE_R, SEAT_W - 0.06, (FX, 0, 0.10), (0, math.pi/2, 0))

# ================================================
# SIDE BARS at Z=0.10
# ================================================
cyl("SideR", TUBE_R, SEAT_D - 0.06, (0, RY, 0.10), (math.pi/2, 0, 0))
cyl("SideL", TUBE_R, SEAT_D - 0.06, (0, LY, 0.10), (math.pi/2, 0, 0))

# ================================================
# BACK FRAME: vertical from seat bottom to backrest top
# ================================================
SUP_H = SEAT_T + BACK_H + 0.02
SUP_Z = SEAT_BOT/2 + SUP_H/2
cyl("BackSup_R", TUBE_R, SUP_H, (BX, RY - 0.01, SUP_Z))
cyl("BackSup_L", TUBE_R, SUP_H, (BX, LY + 0.01, SUP_Z))

# Back top bar
cyl("BackTop", TUBE_R, BACK_W, (BX, 0, SEAT_H + BACK_H + 0.01), (0, math.pi/2, 0))

# ================================================
# ARMRESTS: vertical supports + horizontal bars
# ================================================
ARM_H = 0.20
cyl("ArmSup_FR", TUBE_R, ARM_H, (FX - 0.01, RY + 0.01, SEAT_H + ARM_H/2))
cyl("ArmSup_FL", TUBE_R, ARM_H, (FX - 0.01, LY - 0.01, SEAT_H + ARM_H/2))
cyl("ArmR", TUBE_R, SEAT_D - 0.06, (FX/2, RY + 0.01, SEAT_H + ARM_H), (0, math.pi/2, 0))
cyl("ArmL", TUBE_R, SEAT_D - 0.06, (FX/2, LY - 0.01, SEAT_H + ARM_H), (0, math.pi/2, 0))

# ================================================
# GROUND
# ================================================
bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
gnd = bpy.context.active_object; gnd.name = "Floor"; gnd.data.materials.append(mat_floor)

# ================================================
# CAMERA - looking slightly down at the chair
# ================================================
bpy.ops.object.camera_add(location=(1.3, -1.2, 1.0),
                           rotation=(math.radians(68), 0, math.radians(47)))
cam = bpy.context.active_object; scene.camera = cam

# ================================================
# LIGHTS
# ================================================
bpy.ops.object.light_add(type='AREA', location=(1.5, -1.0, 2.2))
k = bpy.context.active_object; k.data.energy = 500; k.data.size = 2
bpy.ops.object.light_add(type='AREA', location=(-0.8, 1.2, 1.8))
f = bpy.context.active_object; f.data.energy = 200; f.data.size = 3
bpy.ops.object.light_add(type='SUN', location=(0.5, -0.5, 5))
s = bpy.context.active_object; s.data.energy = 1.0

print("RENDERING v6...")
bpy.ops.render.render(write_still=True)
print("DONE:", scene.render.filepath)
