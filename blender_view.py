"""
See what Blender itself computes for a boolean, independently of trimesh's Blender interface.

trimesh.boolean(..., engine='blender') runs a Blender script that applies a boolean modifier and
exports the result. With Blender 4.x that apply step fails ("Modifiers cannot be applied to
multi-user data": the STL importer leaves each mesh with two users), the script stops before
exporting, and trimesh silently returns an empty mesh. blender_boolean() instead reads the
modifier's evaluated result, which is what Blender shows in its viewport, and saves the scene
(both inputs plus the result, with the modifier left unapplied) as a .blend file to open in Blender.
"""
import os
import subprocess
import tempfile

import trimesh
from trimesh.interfaces import blender

SCRIPT = r'''
import bpy, sys
mesh1_path, mesh2_path, result_path, blend_path, operation = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.wm.stl_import(filepath=mesh1_path)
mesh1 = bpy.context.selected_objects[0]; mesh1.name = 'mesh1'
bpy.ops.wm.stl_import(filepath=mesh2_path)
mesh2 = bpy.context.selected_objects[0]; mesh2.name = 'mesh2'
mod = mesh1.modifiers.new('boolean', 'BOOLEAN')
mod.object, mod.operation, mod.solver = mesh2, operation, 'EXACT'
mesh2.display_type = 'WIRE'

# The modifier's evaluated mesh, as shown in Blender's viewport
evaluated = mesh1.evaluated_get(bpy.context.evaluated_depsgraph_get())
result = bpy.data.objects.new('blender_result', bpy.data.meshes.new_from_object(evaluated))
bpy.context.collection.objects.link(result)
print('BLENDER_RESULT_VERTICES', len(result.data.vertices))

bpy.ops.object.select_all(action='DESELECT')
result.select_set(True)
bpy.ops.wm.stl_export(filepath=result_path, export_selected_objects=True)
if blend_path != 'none':
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
'''


def blender_boolean(mesh1, mesh2, operation='INTERSECT', blend_path=None, timeout=600):
    """(result mesh, Blender version) for mesh1 <operation> mesh2 as Blender computes it; the
    result is empty when Blender finds no overlap. Saves the scene to blend_path when given."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, name) for name in ('mesh1.stl', 'mesh2.stl', 'result.stl', 'script.py')]
        mesh1.export(paths[0])
        mesh2.export(paths[1])
        with open(paths[3], 'w') as f:
            f.write(SCRIPT)
        if blend_path:
            os.makedirs(os.path.dirname(blend_path), exist_ok=True)
        run = subprocess.run([blender._blender_executable, '--background', '--factory-startup', '--python', paths[3],
                              '--', paths[0], paths[1], paths[2], blend_path or 'none', operation],
                             capture_output=True, text=True, timeout=timeout)
        if 'BLENDER_RESULT_VERTICES' not in run.stdout:
            raise RuntimeError(f"Blender script failed:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}")
        n_vertices = int(run.stdout.split('BLENDER_RESULT_VERTICES')[1].split()[0])
        result = trimesh.load(paths[2], force='mesh') if n_vertices else trimesh.Trimesh()
    version = subprocess.run([blender._blender_executable, '--version'], capture_output=True, text=True).stdout.split('\n')[0]
    return result, version
