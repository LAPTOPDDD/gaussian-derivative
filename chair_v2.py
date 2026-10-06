import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.filepath = "D:\\PYTHON\\Первая-производная-гаусса\\chair_v2.png"
scene.render.image_settings.file_format = 'PNG'

def mat(name, color, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes["Principled BSDF"]
    n.inputs[0].default_value = (*color, 1)
    n.inputs["Roughness"].default_value = rough
    n.inputs["Metallic"].default_value = metal
    return m

mat_fabric = mat("Fabric", (0.03, 0.03, 0.03), 0.85, 0.0)
mat_metal = mat("Metal", (0.015, 0.015, 0.015), 0.25, 0.95)
mat_ground = mat("Floor", (0.6, 0.6, 0.6), 0.9, 0.0)

def cylinder(name, r, h, loc, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat_metal)
    return o

def cube(name, sx, sy, sz, loc, mat_use):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat_use)
    return o

# === SEAT cushion ===
seat = cube("Seat", 0.42, 0.40, 0.045, (0, 0, 0.44), mat_fabric)

# === BACKREST ===
back = cube("Backrest", 0.035, 0.36, 0.18, (-0.19, 0, 0.72), mat_fabric)
bpy.ops.object.modifier_add(type='SUBSURF')
back.modifiers["Subdivision"].levels = 3
bpy.ops.object.shade_smooth()

# === LEGS ===
# Front legs - straight down from seat front edge
cylinder("Leg_FL", 0.014, 0.44, (0.18, 0.18, 0.22))
cylinder("Leg_FR", 0.014, 0.44, (0.18, -0.18, 0.22))

# Back legs - angled slightly back, from seat back edge
cylinder("Leg_BL", 0.014, 0.50, (-0.18, 0.18, 0.25), (0.15, 0, 0))
cylinder("Leg_BR", 0.014, 0.50, (-0.18, -0.18, 0.25), (0.15, 0, 0))

# === CROSS BAR between front legs ===
cylinder("CrossFront", 0.010, 0.36, (0.18, 0, 0.10), (0, math.pi/2, 0))

# === SIDE BARS connect front to back ===
cylinder("SideR", 0.010, 0.36, (0, 0.18, 0.10), (math.pi/2, 0, 0))
cylinder("SideL", 0.010, 0.36, (0, -0.18, 0.10), (math.pi/2, 0, 0))

# === ARMREST vertical supports ===
cylinder("ArmSupR", 0.010, 0.20, (0.08, 0.22, 0.54))
cylinder("ArmSupL", 0.010, 0.20, (0.08, -0.22, 0.54))

# === ARMREST horizontal bars ===
cylinder("ArmR", 0.010, 0.25, (-0.04, 0.22, 0.62), (0, math.pi/2, 0))
cylinder("ArmL", 0.010, 0.25, (-0.04, -0.22, 0.62), (0, math.pi/2, 0))

# === BACK FRAME vertical supports ===
cylinder("BackSupR", 0.010, 0.30, (-0.19, 0.18, 0.58))
cylinder("BackSupL", 0.010, 0.30, (-0.19, -0.18, 0.58))

# === BACK FRAME top bar ===
cylinder("BackTop", 0.010, 0.36, (-0.19, 0, 0.88), (0, math.pi/2, 0))

# === GROUND ===
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
gnd = bpy.context.active_object
gnd.name = "Ground"
gnd.data.materials.append(mat_ground)

# === CAMERA - lower angle, more frontal ===
bpy.ops.object.camera_add(location=(1.8, -1.6, 1.3),
                           rotation=(math.radians(70), 0, math.radians(50)))
cam = bpy.context.active_object
scene.camera = cam

# === LIGHTS ===
bpy.ops.object.light_add(type='AREA', location=(2.0, -1.5, 3.0))
k = bpy.context.active_object
k.name = "Key"
k.data.energy = 600
k.data.size = 2.5

bpy.ops.object.light_add(type='AREA', location=(-1.0, 2.0, 2.5))
f = bpy.context.active_object
f.name = "Fill"
f.data.energy = 250
f.data.size = 3

bpy.ops.object.light_add(type='SUN', location=(1, -1, 5))
s = bpy.context.active_object
s.name = "Sun"
s.data.energy = 1.5

print("RENDERING v2...")
bpy.ops.render.render(write_still=True)
print("DONE:", scene.render.filepath)
