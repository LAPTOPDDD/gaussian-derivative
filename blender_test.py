import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
scene.render.resolution_x = 800
scene.render.resolution_y = 600
scene.render.resolution_percentage = 100
scene.render.filepath = "D:\\PYTHON\\Первая-производная-гаусса\\blender_test.png"
scene.render.image_settings.file_format = 'PNG'

bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
bpy.context.active_object.name = "Sphere"

bpy.ops.mesh.primitive_cube_add(size=0.5, location=(2, 0, 0))
bpy.context.active_object.name = "Cube"

bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=1.5, location=(-2, 0, 0))
bpy.context.active_object.name = "Cylinder"

mat_red = bpy.data.materials.new(name="Red")
mat_red.use_nodes = True
mat_red.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.1, 0.1, 1)
bpy.data.objects["Sphere"].data.materials.append(mat_red)

mat_green = bpy.data.materials.new(name="Green")
mat_green.use_nodes = True
mat_green.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.1, 0.8, 0.1, 1)
bpy.data.objects["Cube"].data.materials.append(mat_green)

mat_blue = bpy.data.materials.new(name="Blue")
mat_blue.use_nodes = True
mat_blue.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.1, 0.1, 0.8, 1)
bpy.data.objects["Cylinder"].data.materials.append(mat_blue)

bpy.ops.object.camera_add(location=(5, -5, 4), rotation=(math.radians(60), 0, math.radians(45)))
scene.camera = bpy.context.active_object

bpy.ops.object.light_add(type='AREA', location=(3, -3, 5))
light = bpy.context.active_object
light.data.energy = 500

bpy.ops.object.light_add(type='SUN', location=(-2, -2, 8))

bpy.ops.render.render(write_still=True)
print("RENDER COMPLETE:", scene.render.filepath)
