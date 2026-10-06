import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.filepath = "D:\\PYTHON\\Первая-производная-гаусса\\chair_v4.png"
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

# CHAIR DIMENSIONS (in meters)
SEAT_Z = 0.44       # seat height from ground
SEAT_W = 0.44       # seat width (Y axis)
SEAT_D = 0.42       # seat depth (X axis)
SEAT_T = 0.045      # seat thickness
BACK_H = 0.18       # backrest height
BACK_W = 0.36       # backrest width
LEG_R = 0.013       # leg tube radius
TUBE_R = 0.009      # crossbar tube radius

FRONT = SEAT_D/2 - 0.03
BACK = -SEAT_D/2 + 0.03
RIGHT = SEAT_W/2 - 0.03
LEFT = -SEAT_W/2 + 0.03

# === SEAT CUSHION ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, SEAT_Z))
seat = bpy.context.active_object; seat.name = "Seat"
seat.scale = (SEAT_D/2, SEAT_W/2, SEAT_T/2)
bpy.ops.object.transform_apply(scale=True)
seat.data.materials.append(mat_fabric)

# === BACKREST ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(-SEAT_D/2 + 0.01, 0, SEAT_Z + SEAT_T/2 + BACK_H/2 + 0.01))
back = bpy.context.active_object; back.name = "Backrest"
back.scale = (0.025, BACK_W/2, BACK_H/2)
bpy.ops.object.transform_apply(scale=True)
back.data.materials.append(mat_fabric)
bpy.ops.object.modifier_add(type='SUBSURF')
back.modifiers["Subdivision"].levels = 3
bpy.ops.object.shade_smooth()

# === 4 LEGS (from ground to seat bottom) ===
cyl("Leg_FL", LEG_R, SEAT_Z, (FRONT, RIGHT, SEAT_Z/2))
cyl("Leg_FR", LEG_R, SEAT_Z, (FRONT, LEFT, SEAT_Z/2))
cyl("Leg_BL", LEG_R, SEAT_Z + 0.08, (BACK, RIGHT, (SEAT_Z+0.08)/2), (0.1, 0, 0))
cyl("Leg_BR", LEG_R, SEAT_Z + 0.08, (BACK, LEFT, (SEAT_Z+0.08)/2), (0.1, 0, 0))

# === CROSS BAR between front legs ===
cyl("CrossF", TUBE_R, SEAT_W - 0.06, (FRONT, 0, 0.11), (0, math.pi/2, 0))

# === SIDE BARS ===
cyl("SideR", TUBE_R, SEAT_D - 0.06, (0, RIGHT, 0.11), (math.pi/2, 0, 0))
cyl("SideL", TUBE_R, SEAT_D - 0.06, (0, LEFT, 0.11), (math.pi/2, 0, 0))

# === BACK FRAME vertical supports ===
BACK_SUP_Z = SEAT_Z + BACK_H/2 + 0.01
cyl("BackSup_R", TUBE_R, BACK_H + SEAT_T, (-SEAT_D/2 + 0.03, RIGHT, SEAT_Z/2 + BACK_H/2 + 0.02))
cyl("BackSup_L", TUBE_R, BACK_H + SEAT_T, (-SEAT_D/2 + 0.03, LEFT, SEAT_Z/2 + BACK_H/2 + 0.02))

# === BACK TOP BAR ===
cyl("BackTop", TUBE_R, BACK_W - 0.04, (-SEAT_D/2 + 0.03, 0, SEAT_Z + BACK_H + SEAT_T/2 + 0.02), (0, math.pi/2, 0))

# === ARMRESTS vertical supports from front legs ===
ARM_H = 0.22
cyl("ArmSup_FR", TUBE_R, ARM_H, (FRONT - 0.02, RIGHT + 0.02, SEAT_Z + ARM_H/2))
cyl("ArmSup_FL", TUBE_R, ARM_H, (FRONT - 0.02, LEFT - 0.02, SEAT_Z + ARM_H/2))

# === ARMRESTS horizontal bars ===
cyl("ArmR", TUBE_R, SEAT_D - 0.08, (FRONT/2 - 0.04, RIGHT + 0.02, SEAT_Z + ARM_H), (0, math.pi/2, 0))
cyl("ArmL", TUBE_R, SEAT_D - 0.08, (FRONT/2 - 0.04, LEFT - 0.02, SEAT_Z + ARM_H), (0, math.pi/2, 0))

# === GROUND ===
bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
gnd = bpy.context.active_object; gnd.name = "Floor"; gnd.data.materials.append(mat_floor)

# === CAMERA ===
bpy.ops.object.camera_add(location=(1.4, -1.4, 1.0),
                           rotation=(math.radians(72), 0, math.radians(47)))
cam = bpy.context.active_object; scene.camera = cam

# === LIGHTS ===
bpy.ops.object.light_add(type='AREA', location=(1.5, -1.0, 2.5))
k = bpy.context.active_object; k.data.energy = 500; k.data.size = 2
bpy.ops.object.light_add(type='AREA', location=(-0.8, 1.2, 2.0))
f = bpy.context.active_object; f.data.energy = 200; f.data.size = 3
bpy.ops.object.light_add(type='SUN', location=(0.5, -0.5, 5))
s = bpy.context.active_object; s.data.energy = 1.0

print("RENDERING v4...")
bpy.ops.render.render(write_still=True)
print("DONE:", scene.render.filepath)
