import bpy
import math
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.filepath = "D:\\PYTHON\\Первая-производная-гаусса\\chair_v7.png"
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

def make_tube(name, points, radius=0.009):
    curve = bpy.data.curves.new(name=name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spline.points[i].co = (*p, 1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat_metal)
    return obj

# === CHAIR DIMENSIONS ===
SEAT_TOP = 0.42
SEAT_T = 0.04
SEAT_BOT = SEAT_TOP - SEAT_T
W = 0.42   # width
D = 0.40   # depth

FX = D/2 - 0.03
BX = -D/2 + 0.03
RY = W/2 - 0.03
LY = -W/2 + 0.03

# === SEAT CUSHION ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, SEAT_BOT + SEAT_T/2))
seat = bpy.context.active_object; seat.name = "Seat"
seat.scale = (D/2, W/2, SEAT_T/2)
bpy.ops.object.transform_apply(scale=True)
seat.data.materials.append(mat_fabric)

# === BACKREST ===
bpy.ops.mesh.primitive_cube_add(size=1, location=(BX + 0.01, 0, SEAT_TOP + 0.08))
back = bpy.context.active_object; back.name = "Backrest"
back.scale = (0.02, 0.17, 0.08)
bpy.ops.object.transform_apply(scale=True)
back.data.materials.append(mat_fabric)
bpy.ops.object.modifier_add(type='SUBSURF')
back.modifiers["Subdivision"].levels = 3
bpy.ops.object.shade_smooth()

# === METAL FRAME using curves (tubes) ===
TUBE = 0.009

# Front-left leg: ground to seat
make_tube("Leg_FL", [(FX, RY, 0), (FX, RY, SEAT_BOT)], TUBE)
# Front-right leg
make_tube("Leg_FR", [(FX, LY, 0), (FX, LY, SEAT_BOT)], TUBE)
# Back-left leg
make_tube("Leg_BL", [(BX, RY, 0), (BX, RY, SEAT_BOT)], TUBE)
# Back-right leg
make_tube("Leg_BR", [(BX, LY, 0), (BX, LY, SEAT_BOT)], TUBE)

# Cross bar front
make_tube("CrossF", [(FX, RY, 0.10), (FX, LY, 0.10)], TUBE)
# Side bars
make_tube("SideR", [(FX, RY, 0.10), (BX, RY, 0.10)], TUBE)
make_tube("SideL", [(FX, LY, 0.10), (BX, LY, 0.10)], TUBE)

# Back frame supports (seat to backrest top)
BT_TOP = SEAT_TOP + 0.17
make_tube("BackSup_R", [(BX, RY, SEAT_BOT), (BX, RY, BT_TOP)], TUBE)
make_tube("BackSup_L", [(BX, LY, SEAT_BOT), (BX, LY, BT_TOP)], TUBE)
# Back top bar
make_tube("BackTop", [(BX, RY, BT_TOP), (BX, LY, BT_TOP)], TUBE)

# Armrest verticals
ARM_H = 0.20
make_tube("ArmSup_FR", [(FX, RY, SEAT_TOP), (FX, RY, SEAT_TOP + ARM_H)], TUBE)
make_tube("ArmSup_FL", [(FX, LY, SEAT_TOP), (FX, LY, SEAT_TOP + ARM_H)], TUBE)
# Armrest horizontals
AT = SEAT_TOP + ARM_H
make_tube("ArmR", [(FX, RY, AT), (BX + 0.05, RY, AT)], TUBE)
make_tube("ArmL", [(FX, LY, AT), (BX + 0.05, LY, AT)], TUBE)

# === GROUND ===
bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
gnd = bpy.context.active_object; gnd.name = "Floor"; gnd.data.materials.append(mat_floor)

# === CAMERA ===
bpy.ops.object.camera_add(location=(1.2, -1.2, 0.9),
                           rotation=(math.radians(72), 0, math.radians(46)))
cam = bpy.context.active_object; scene.camera = cam

# === LIGHTS ===
bpy.ops.object.light_add(type='AREA', location=(1.5, -1.0, 2.2))
k = bpy.context.active_object; k.data.energy = 500; k.data.size = 2
bpy.ops.object.light_add(type='AREA', location=(-0.8, 1.2, 1.8))
f = bpy.context.active_object; f.data.energy = 200; f.data.size = 3
bpy.ops.object.light_add(type='SUN', location=(0.5, -0.5, 5))
s = bpy.context.active_object; s.data.energy = 1.0

print("RENDERING v7 (curves)...")
bpy.ops.render.render(write_still=True)
print("DONE:", scene.render.filepath)
