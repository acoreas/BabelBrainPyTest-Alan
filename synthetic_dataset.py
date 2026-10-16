"""
Small deterministic synthetic head used by the GPU unit tests (Unit/BabelBrain/GPUFunctions), so they
can run every spatial step, including the stress resolution, in a reasonable time.

The head is a set of nested ellipsoids (~40 mm across) on a 1 mm grid:
scalp (charm label 5), outer/inner table (7), diploe (8), CSF (3), grey matter (2) and white
matter (1).

The CT also has the kind of bone defects Step 1's binary closing (5 mm box) is there for, so the
closing and labeling tests have something to work on: small pores, a 1 mm suture crack and two 2 mm
foramina (smaller than the box, should be closed), an 8 mm burr hole (larger, should stay open) and
two small intracranial calcifications (separate bone components). The charm segmentation stays
smooth, as charm's would.

The dataset folder mirrors the real ones (see parameters.DATASETS):

    <SYNTHETIC_DATASET_DIR>/
        T1W.nii.gz                   int16 T1-weighted intensities
        CT.nii.gz                    int16 Hounsfield units
        synthetic_dataset.yaml       generator version and parameters
        m2m_synthetic/
            final_tissues.nii.gz     uint16 charm labels, shape (X, Y, Z, 1) like charm
            skin.stl                 watertight skin surface (world coordinates)
            charm_log.html           marker so the folder is treated as a charm output

The GPU tests also cache their CPU reference outputs in this folder. Changing GENERATOR_VERSION or
the parameters below regenerates the dataset and deletes those cached references.
"""
import os
import shutil

import nibabel
import numpy as np
import trimesh
import yaml
from skimage.measure import marching_cubes

GENERATOR_VERSION = 2
SHAPE = (40, 40, 40)                 # voxels, 1 mm isotropic
SEMI_AXES = (17.0, 19.0, 17.0)       # outer (skin) ellipsoid semi-axes, mm
SEED = 0

# (charm label, depth from the skin in mm at which the layer starts)
LAYERS = [(5, 0.0), (7, 2.0), (8, 3.0), (7, 4.5), (3, 5.5), (2, 6.5), (1, 8.5)]
T1_VALUES = {0: 0, 1: 850, 2: 560, 3: 160, 5: 650, 7: 110, 8: 330}
CT_VALUES = {0: -1000, 1: 28, 2: 38, 3: 8, 5: 40, 7: 1800, 8: 900}

# Bone defects in the CT, (direction from the head centre, radius mm); see _ct_labels
N_PORES = 40                                   # 1-2 voxel pores in the skull
FORAMINA = [((0.3, 0.8, 0.5), 1.0), ((-0.7, -0.2, 0.7), 1.0)]
BURR_HOLE = ((1.0, 0.0, 0.6), 4.0)
CALCIFICATIONS = [((0.3, 0.2, 0.2), 1.5), ((-0.25, -0.3, -0.1), 1.0)]  # (position as a fraction of the semi-axes, radius mm)


def _params():
    return {'generator_version': GENERATOR_VERSION, 'shape': list(SHAPE), 'semi_axes': list(SEMI_AXES),
            'seed': SEED, 'layers': [list(layer) for layer in LAYERS], 'n_pores': N_PORES}


def _labels():
    """Charm labels on the grid, plus the affine (origin at the head centre)."""
    affine = np.eye(4)
    affine[:3, 3] = -(np.array(SHAPE) - 1) / 2.0
    ijk = np.indices(SHAPE).reshape(3, -1).T
    xyz = ijk + affine[:3, 3]
    # Normalised ellipsoid radius: 1 on the skin. Depth below the skin is approximated along the
    # ray from the centre using the smallest semi-axis, which is enough for a layered phantom.
    rho = np.sqrt(np.sum((xyz / np.array(SEMI_AXES)) ** 2, axis=1))
    depth = (1.0 - rho) * min(SEMI_AXES)
    labels = np.zeros(len(xyz), np.uint16)
    for label, start in LAYERS:
        labels[depth >= start] = label
    return labels.reshape(SHAPE), affine


def _world_coordinates(affine):
    return np.indices(SHAPE).transpose(1, 2, 3, 0) + affine[:3, 3]


def _channel(xyz, direction, radius):
    """Voxels within `radius` of the ray from the head centre along `direction`."""
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    along = xyz @ u
    distance = np.linalg.norm(xyz - along[..., None] * u, axis=-1)
    return (along > 0) & (distance <= radius)


def _ct_labels(labels, affine, rng):
    """Copy of the charm labels with the CT-only bone defects described in the module docstring."""
    ct_labels = labels.copy()
    bone = np.isin(labels, (7, 8))
    xyz = _world_coordinates(affine)

    # Holes smaller than the 5 mm closing box: pores (soft tissue inside bone), a 1 mm sagittal
    # suture crack across the top of the skull and two 2 mm foramina
    pore_centres = np.argwhere(bone)[rng.choice(bone.sum(), N_PORES, replace=False)]
    for centre in pore_centres:
        size = rng.integers(1, 3, 3)
        ct_labels[tuple(slice(c, c + n) for c, n in zip(centre, size))] = 3
    suture = (np.abs(xyz[..., 0]) < 0.6) & (xyz[..., 2] > 4)
    for mask in [suture] + [_channel(xyz, d, r) for d, r in FORAMINA]:
        ct_labels[bone & mask] = 3

    # Hole larger than the closing box, filled with scalp so it stays open after closing
    ct_labels[bone & _channel(xyz, *BURR_HOLE)] = 5

    # Isolated bone inside the brain
    for position, radius in CALCIFICATIONS:
        centre = np.asarray(position) * np.asarray(SEMI_AXES)
        ct_labels[np.linalg.norm(xyz - centre, axis=-1) <= radius] = 7
    return ct_labels


def _intensities(labels, values, rng, noise_sd):
    out = np.zeros(labels.shape, np.float32)
    for label, value in values.items():
        out[labels == label] = value
    inside = labels > 0
    out[inside] += rng.normal(0, noise_sd, inside.sum())
    return np.round(out).astype(np.int16)


def _skin_mesh(labels, affine):
    """Watertight skin surface in world coordinates."""
    padded = np.pad(labels > 0, 1).astype(np.uint8)
    verts, faces, _, _ = marching_cubes(padded, level=0.5)
    verts = verts - 1 + affine[:3, 3]  # undo padding, voxel -> world (1 mm grid)
    mesh = trimesh.Trimesh(verts, faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _is_current(folder):
    stamp = os.path.join(folder, 'synthetic_dataset.yaml')
    if not os.path.isfile(stamp):
        return False
    with open(stamp) as f:
        return yaml.safe_load(f) == _params()


def ensure_synthetic_dataset(folder):
    """Generate the dataset in `folder` unless an identical one is already there."""
    if _is_current(folder):
        return
    if os.path.isdir(folder):
        shutil.rmtree(folder)  # also drops CPU references cached from a different dataset
    m2m = os.path.join(folder, 'm2m_synthetic')
    os.makedirs(m2m)

    rng = np.random.default_rng(SEED)
    labels, affine = _labels()
    nibabel.save(nibabel.Nifti1Image(_intensities(labels, T1_VALUES, rng, 20), affine),
                 os.path.join(folder, 'T1W.nii.gz'))
    nibabel.save(nibabel.Nifti1Image(_intensities(_ct_labels(labels, affine, rng), CT_VALUES, rng, 30), affine),
                 os.path.join(folder, 'CT.nii.gz'))
    nibabel.save(nibabel.Nifti1Image(labels[..., np.newaxis], affine), os.path.join(m2m, 'final_tissues.nii.gz'))
    mesh = _skin_mesh(labels, affine)
    assert mesh.is_watertight and mesh.body_count == 1, "synthetic skin mesh must be a single closed surface"
    mesh.export(os.path.join(m2m, 'skin.stl'))
    with open(os.path.join(m2m, 'charm_log.html'), 'w') as f:
        f.write('synthetic charm output\n')

    with open(os.path.join(folder, 'synthetic_dataset.yaml'), 'w') as f:
        yaml.safe_dump(_params(), f)
