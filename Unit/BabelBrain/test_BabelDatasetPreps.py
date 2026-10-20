"""
Tests for BabelDatasetPreps.DoIntersect, which Step 1 uses to cut the skin mesh with the transducer
cone and the skin/bone/CSF meshes with the field-of-view box.

TestDoIntersectSynthetic
    Spheres, boxes and cones whose intersection volumes are known analytically, the mesh-repair
    paths (multi-body and open meshes). Runs in seconds.
TestDoIntersectRealData
    The charm meshes of the real datasets cut with a Step 1-like cone and box at each tested target.
    No truth meshes needed: the results are checked with the properties any correct intersection has
    (see mesh_checks.check_intersection).

Every test runs once per boolean engine DoIntersect supports: manifold (the default) and Blender
(bForceUseBlender=True, skipped when Blender isn't installed). Blender variants also report Blender's
own result for the inputs (blender_view.blender_boolean: the boolean modifier evaluated in Blender,
bypassing trimesh's Blender script) and save the scene as a .blend file in
PyTest_Reports/blender_scenes/ (real-data scenes are ~20 MB each, so only with
BABEL_PYTEST_SAVE_BLENDER_SCENES=1).

trimesh's Blender script is checked once on two overlapping spheres. If it returns an empty mesh
(it does with Blender 4.x: see blender_view.py), Blender tests that expect an intersection fail with
that reason instead of a misleading "Trajectory is outside headspace", after reporting what Blender
itself computes. Tests of non-overlapping inputs always run: an empty result is correct for them with
either engine.
"""
import logging
import os
import re
import subprocess

import numpy as np
import pytest
import trimesh
from trimesh.interfaces import blender

import BabelBrain.BabelDatasetPreps as bdp
from ConvMatTransform import ReadTrajectoryBrainsight
from Tests.helpers.blender_view import blender_boolean
from Tests.helpers.mesh_checks import check_intersection
from Tests.helpers.parameters import BABELBRAIN_ROOT, trajectory_file
from Tests.helpers.report_viewers import slider_viewer_entry

ENGINES = [pytest.param(False, id='manifold'),
           pytest.param(True, id='blender',
                        marks=pytest.mark.skipif(not blender.exists, reason="Blender is not installed"))]

BLENDER_SCENES_DIR = BABELBRAIN_ROOT / 'PyTest_Reports' / 'blender_scenes'
SAVE_REAL_DATA_SCENES = os.environ.get('BABEL_PYTEST_SAVE_BLENDER_SCENES') == '1'


@pytest.fixture(scope='module')
def blender_boolean_problem():
    """None when trimesh's Blender booleans work here, else why not (checked once, only if a Blender
    test needs it)."""
    a = trimesh.creation.icosphere(subdivisions=2, radius=10)
    b = trimesh.creation.icosphere(subdivisions=2, radius=8)
    b.apply_translation((12, 0, 0))
    try:
        result = trimesh.boolean.intersection((a, b), engine='blender')
    except Exception as error:
        return f"trimesh's Blender boolean raised {type(error).__name__}: {error}"
    if result.is_empty:
        version = subprocess.run([blender._blender_executable, '--version'], capture_output=True, text=True).stdout.split('\n')[0]
        return (f"trimesh's Blender boolean returned an empty mesh for two overlapping spheres ({version}, trimesh "
                f"{trimesh.__version__}), so DoIntersect(bForceUseBlender=True) can't work in this environment")
    return None


# ================================================================================================================================
# SHAPES
# ================================================================================================================================
def sphere(radius, center=(0, 0, 0), subdivisions=4):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(center)
    return mesh


def box(extents, center=(0, 0, 0)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def cone_from_apex(apex, direction, height, radius, sections=256):
    """Cone with its apex at `apex`, opening along `direction` (as Step 1 builds the transducer cone:
    apex at the target, base towards the transducer)."""
    mesh = trimesh.creation.cone(radius=radius, height=height, sections=sections)  # base z=0, apex z=height
    flip = np.diag([1.0, 1.0, -1.0, 1.0])
    flip[2, 3] = height                                                              # apex z=0, base z=height
    mesh.apply_transform(flip)
    z = np.asarray(direction, float) / np.linalg.norm(direction)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    pose[:3, 3] = apex
    mesh.apply_transform(pose)
    return mesh


def lens_volume(r1, r2, d):
    """Volume of the intersection of two spheres of radii r1, r2 with centres d apart."""
    return np.pi*(r1 + r2 - d)**2*(d**2 + 2*d*(r1 + r2) - 3*(r1 - r2)**2) / (12*d)


def show_intersection(request, plots, mesh1, mesh2, result=None, names=('Mesh 1', 'Mesh 2'), blender_view=None):
    """Report the inputs and, once there is one, the intersection: a viewer with the inputs (grey and
    blue) and the intersection (red) side by side with the intersection on its own, plus Blender's own
    result for Blender variants (blender_view = (result, .blend path or None)), one slider moving the
    camera around all of them. Called before DoIntersect too, so a failing test still shows its inputs."""
    overview = f"{names[0]} (grey) and {names[1]} (blue)" + (", intersection in red" if result is not None else "")
    panels = [(overview, plots['intersection_plot'](mesh1, mesh2, result))]
    if result is not None and not result.is_empty:
        panels.append(("Intersection (DoIntersect)", plots['mesh_plot']([result], color='red')))
    notes = []
    if blender_view is not None:
        blender_result, blend_path = blender_view
        if blender_result.is_empty:
            notes.append("Blender computes no intersection for these inputs.")
        else:
            panels.append(("Blender's own result (modifier evaluated in Blender)",
                           plots['mesh_plot']([blender_result], color='orange')))
        notes.append(f"Blender scene: {blend_path}" if blend_path else
                     "Blender scene not saved (set BABEL_PYTEST_SAVE_BLENDER_SCENES=1 to save real-data scenes)")
    request.node.screenshots = [slider_viewer_entry(panels, frame_label='View')]
    if notes:
        request.node.screenshots.append({'kind': 'html', 'html': '<div style="font-size:12px">' + '<br>'.join(notes) + '</div>'})


def _blender_view(request, mesh1, mesh2, save_scene):
    """Blender's own intersection of the inputs, and the .blend scene saved for this test (or None)."""
    blend_path = None
    if save_scene:
        blend_path = BLENDER_SCENES_DIR / (re.sub(r'[^0-9A-Za-z._-]+', '_', request.node.name) + '.blend')
    result, _ = blender_boolean(mesh1, mesh2, blend_path=str(blend_path) if blend_path else None)
    return result, blend_path


def run_do_intersect(request, plots, mesh1, mesh2, use_blender, names=('Mesh 1', 'Mesh 2'), save_scene=True):
    """DoIntersect(mesh1, mesh2) with the selected engine, reported with show_intersection. Inputs are
    copied because DoIntersect repairs them in place. Blender variants fail with the reason (after
    reporting Blender's own result) when trimesh's Blender booleans don't work here."""
    view = _blender_view(request, mesh1, mesh2, save_scene) if use_blender else None
    show_intersection(request, plots, mesh1, mesh2, names=names, blender_view=view)
    if use_blender:
        problem = request.getfixturevalue('blender_boolean_problem')
        if problem:
            computed = "no" if view[0].is_empty else f"a {abs(view[0].volume):.6g} mm^3"
            pytest.fail(f"{problem}. Blender itself computes {computed} intersection for these inputs (see report)")
    result = bdp.DoIntersect(mesh1.copy(), mesh2.copy(), bForceUseBlender=use_blender)
    show_intersection(request, plots, mesh1, mesh2, result, names, blender_view=view)
    return result


def expect_outside_headspace(request, plots, mesh1, mesh2, use_blender, names=('Mesh 1', 'Mesh 2'), save_scene=True):
    """Non-overlapping inputs: DoIntersect must raise "Trajectory is outside headspace" (correct for
    any engine, so this doesn't depend on trimesh's Blender booleans working)."""
    view = _blender_view(request, mesh1, mesh2, save_scene) if use_blender else None
    show_intersection(request, plots, mesh1, mesh2, names=names, blender_view=view)
    with pytest.raises(ValueError, match="Trajectory is outside headspace"):
        bdp.DoIntersect(mesh1.copy(), mesh2.copy(), bForceUseBlender=use_blender)


def assert_valid_intersection(result, mesh1, mesh2):
    problems, agreement = check_intersection(result, mesh1, mesh2)
    logging.info(f"Sample point agreement: {100*agreement:.1f}%")
    assert not problems, "Invalid intersection: " + "; ".join(problems)


# ================================================================================================================================
# SYNTHETIC
# ================================================================================================================================
@pytest.mark.step1
@pytest.mark.parametrize('use_blender', ENGINES)
class TestDoIntersectSynthetic:

    def test_overlapping_spheres(self, use_blender, request, get_pyvista_plot):
        r1, r2, d = 10.0, 8.0, 12.0
        a, b = sphere(r1), sphere(r2, center=(d, 0, 0))
        result = run_do_intersect(request, get_pyvista_plot, a, b, use_blender)

        assert_valid_intersection(result, a, b)
        # Icospheres are slightly smaller than true spheres, hence the 2% tolerance
        assert result.volume == pytest.approx(lens_volume(r1, r2, d), rel=0.02)

    def test_cone_from_sphere_centre(self, use_blender, request, get_pyvista_plot):
        """Like Step 1's skin cut: cone apex inside the head, base outside. The intersection is a
        spherical sector, (2/3) pi R^3 (1 - cos(half angle))."""
        radius, cone_height, cone_radius = 10.0, 20.0, 12.0
        a = sphere(radius)
        b = cone_from_apex((0, 0, 0), (0.3, -0.2, 1.0), cone_height, cone_radius)
        result = run_do_intersect(request, get_pyvista_plot, a, b, use_blender)

        assert_valid_intersection(result, a, b)
        half_angle = np.arctan(cone_radius / cone_height)
        assert result.volume == pytest.approx(2/3*np.pi*radius**3*(1 - np.cos(half_angle)), rel=0.02)

    def test_box_inside_sphere(self, use_blender, request, get_pyvista_plot):
        """Like the FOV box cut of a deep region: a fully contained box comes back unchanged."""
        a, b = sphere(20.0), box((10, 8, 6), center=(2, -1, 1))
        result = run_do_intersect(request, get_pyvista_plot, a, b, use_blender)

        assert_valid_intersection(result, a, b)
        assert result.volume == pytest.approx(b.volume, rel=1e-6)
        assert np.allclose(result.bounds, b.bounds, atol=1e-6)

    def test_box_cuts_sphere_in_half(self, use_blender, request, get_pyvista_plot):
        a, b = sphere(10.0), box((30, 30, 15), center=(0, 0, 7.5))
        result = run_do_intersect(request, get_pyvista_plot, a, b, use_blender)

        assert_valid_intersection(result, a, b)
        assert result.volume == pytest.approx(a.volume/2, rel=1e-3)

    def test_disjoint_meshes_raise(self, use_blender, request, get_pyvista_plot):
        """A trajectory outside the head gives an empty intersection, which must be reported."""
        a, b = sphere(10.0), box((5, 5, 5), center=(30, 0, 0))
        expect_outside_headspace(request, get_pyvista_plot, a, b, use_blender)

    def test_multi_body_mesh_is_repaired(self, use_blender, request, get_pyvista_plot):
        """Meshes with more than one body (e.g. a stray island in a charm surface) go through FixMesh
        first; the intersection must still be valid and keep the main body."""
        main = sphere(10.0)
        stray = sphere(1.0, center=(25, 0, 0), subdivisions=2)
        two_bodies = trimesh.util.concatenate([main, stray])
        assert two_bodies.body_count == 2
        b = box((30, 30, 15), center=(0, 0, 7.5))
        result = run_do_intersect(request, get_pyvista_plot, two_bodies, b, use_blender)

        assert_valid_intersection(result, main, b)
        assert result.volume == pytest.approx(main.volume/2, rel=0.02)

    def test_open_mesh_is_repaired(self, use_blender, request, get_pyvista_plot):
        """A mesh with a hole (not a closed volume, like ernie's charm meshes) is closed before
        intersecting."""
        holed = sphere(10.0)
        top = np.argsort(holed.triangles_center[:, 2])[-3:]  # remove three faces at the top
        holed.update_faces(np.setdiff1d(np.arange(len(holed.faces)), top))
        assert not holed.is_volume
        reference = sphere(10.0)
        b = box((30, 30, 15), center=(0, 0, 7.5))
        result = run_do_intersect(request, get_pyvista_plot, holed, b, use_blender)

        assert_valid_intersection(result, reference, b)
        assert result.volume == pytest.approx(reference.volume/2, rel=0.02)


# ================================================================================================================================
# REAL DATA
# ================================================================================================================================
# Step 1 defaults (GetSkullMaskFromSimbNIBSSTL): transducer focal length and aperture (mm)
FOCAL_LENGTH, TX_DIAMETER, ENLARGE_RADIUS = 135.0, 157.0, 1.05
CONE_RADIUS = TX_DIAMETER/2*ENLARGE_RADIUS
CONE_HEIGHT = np.sqrt(FOCAL_LENGTH**2 - CONE_RADIUS**2)

# Meshes Step 1 cuts with each shape
REAL_CASES = [('skin', 'cone'), ('skin', 'box'), ('bone', 'box'), ('csf', 'box')]


def step1_shapes(trajectory_path):
    """Cone and field-of-view box at the trajectory target, oriented like Step 1's (Brainsight m2 axis
    points from the target towards the transducer)."""
    pose = ReadTrajectoryBrainsight(str(trajectory_path))
    if pose.ndim == 3:
        pose = pose[:, :, 0]
    target, towards_tx = pose[:3, 3], pose[:3, 2]
    cone = cone_from_apex(target, towards_tx, CONE_HEIGHT, CONE_RADIUS)
    # Box covering the cone, from 20 mm beyond the target to the transducer
    extents = (2*CONE_RADIUS, 2*CONE_RADIUS, CONE_HEIGHT + 20)
    fov = trimesh.creation.box(extents=extents)
    frame = np.eye(4)
    frame[:3, :3] = pose[:3, :3]
    frame[:3, 3] = target + towards_tx*(CONE_HEIGHT - 20)/2
    fov.apply_transform(frame)
    return {'cone': cone, 'box': fov}


@pytest.mark.step1
@pytest.mark.slow  # ~1 min in total per engine, left out of the quick minor release set
@pytest.mark.parametrize('use_blender', ENGINES)
class TestDoIntersectRealData:

    @pytest.mark.parametrize('tissue,shape', REAL_CASES, ids=[f"{t}-{s}" for t, s in REAL_CASES])
    def test_do_intersect_real_mesh(self, use_blender, dataset, trajectory, tissue, shape, load_files, get_pyvista_plot, request):
        traj = trajectory_file('brainsight', dataset['id'], trajectory)
        if not traj.exists():
            pytest.skip(f"Trajectory file not found: {traj}")
        mesh = load_files([dataset['m2m_folder_path'] + f"{tissue}.stl"])[0]
        cutter = step1_shapes(traj)[shape]

        result = run_do_intersect(request, get_pyvista_plot, mesh, cutter, use_blender, names=(tissue, shape),
                                  save_scene=SAVE_REAL_DATA_SCENES)
        assert_valid_intersection(result, mesh, cutter)
        assert result.volume > 0.01*cutter.volume, "Intersection is suspiciously small"

    def test_do_intersect_invalid_case(self, use_blender, dataset, load_files, get_pyvista_plot, request):
        """A box placed outside the head must raise "Trajectory is outside headspace"."""
        skin_stl = load_files([dataset['m2m_folder_path'] + "skin.stl"])[0]

        skin_dims = np.diff(skin_stl.bounds, axis=0)[0]
        max_skin_dim = np.max(skin_dims)
        box_radius = max_skin_dim//6
        offset = 10
        box_transform = np.eye(4)
        box_transform[0, 3] = -max_skin_dim/2 - box_radius - offset
        box_transform[1, 3] = max_skin_dim/2 + box_radius + offset
        box_transform[2, 3] = max_skin_dim/2 + box_radius + offset
        box_stl = trimesh.creation.box(extents=[box_radius*2]*3, transform=box_transform)

        expect_outside_headspace(request, get_pyvista_plot, skin_stl, box_stl, use_blender, names=('skin', 'box'),
                                 save_scene=SAVE_REAL_DATA_SCENES)
