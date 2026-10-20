"""
Plots for the html report. Every plot is a slider viewer (report_viewers.py): the slider moves the
slice position for volumes and the camera position for meshes. GUI screenshots are plain images
(image_to_base64).
"""
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is noninteractive
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pyvista as pv
from PIL import Image

from Tests.helpers.report_viewers import slider_viewer_entry, slider_viewer_html


def _webp_b64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='webp')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def image_to_base64(image_path):
    """PNG screenshot at `image_path` (a Path) as a base64 WebP string."""
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"File {image_path} does not exist.")
    buffer = BytesIO()
    Image.open(image_path).save(buffer, format="WEBP", quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ================================================================================================================================
# VOLUMES
# ================================================================================================================================
def mpl_plot(datas,axes_num=1,titles=None,color_map='viridis',colorbars=False,zero_based=None,n_positions=24):
    """Report entry with a slider over slice position: one row per axis (the first `axes_num`),
    one column per volume, each row showing the slice at the slider's position along that axis.
    color_map may be one map or one per volume; zero_based (one bool per volume) starts that
    volume's colour scale at 0 (e.g. absolute differences, so no difference is the lowest colour).
    Colour scales are fixed per volume so they don't change between slices."""
    data_num = len(datas)
    color_maps = color_map if isinstance(color_map,(list,tuple)) else [color_map]*data_num
    zero_based = zero_based or [False]*data_num
    datas = [np.asarray(d) for d in datas]
    limits = []
    for num, data in enumerate(datas):
        vmax = float(np.nanmax(data)) if data.size else 1.0
        vmin = 0.0 if zero_based[num] else (float(np.nanmin(data)) if data.size else 0.0)
        if vmax <= vmin:  # constant volume, e.g. no differences at all
            vmax = vmin + 1.0
        limits.append((vmin, vmax))

    def slice_of(data, axis, fraction):
        index = int(round(fraction*(data.shape[axis]-1)))
        image = np.take(data, index, axis=axis).astype(np.float64)
        return (np.rot90(image) if axis < 2 else image), index

    positions = np.linspace(0, 1, n_positions)
    fig, axs = plt.subplots(axes_num, data_num, figsize=(data_num * 2.5, axes_num * 2.5), squeeze=False, dpi=80)
    images, row_labels = [], []
    for axis in range(axes_num):
        row = []
        for num in range(data_num):
            image, _ = slice_of(datas[num], axis, positions[0])
            im = axs[axis,num].imshow(image, cmap=color_maps[num], vmin=limits[num][0], vmax=limits[num][1])
            if colorbars:
                plt.colorbar(im, ax=axs[axis,num], fraction=0.046)
            if titles is not None and axis == 0:
                axs[axis,num].set_title(titles[num], fontsize=10)
            row.append(im)
        images.append(row)
        # Widest label text up front so tight_layout leaves room for it
        row_labels.append(axs[axis,0].set_ylabel(f"axis {axis}, slice {datas[0].shape[axis]}/{datas[0].shape[axis]}"))
    plt.tight_layout()

    frames = []
    for fraction in positions:
        for axis in range(axes_num):
            for num in range(data_num):
                image, index = slice_of(datas[num], axis, fraction)
                images[axis][num].set_data(image)
                if num == 0:
                    row_labels[axis].set_text(f"axis {axis}, slice {index+1}/{datas[0].shape[axis]}")
        frames.append(_webp_b64(fig))
    plt.close(fig)

    return slider_viewer_entry([('', frames)], frame_label='Slice position', start=n_positions//2)


def gpu_vs_cpu_plot(plots,titles,cpu_output,gpu_output,color_map='gray'):
    """`plots` (inputs, CPU and GPU outputs) plus a |GPU - CPU| panel, all sliced together. The
    difference panel's title gives the number of differing voxels in the whole volume."""
    diff = np.abs(np.asarray(gpu_output,dtype=np.float64) - np.asarray(cpu_output,dtype=np.float64))
    n_diff = int(np.count_nonzero(diff))
    return mpl_plot(list(plots) + [diff],
                    axes_num=3,
                    titles=list(titles) + [f"|GPU - CPU|\n{n_diff} voxels differ"],
                    color_map=[color_map]*len(plots) + [plt.cm.jet],
                    colorbars=True,
                    zero_based=[False]*len(plots) + [True])


def volume_diff_viewer(vol1,vol2,diff_vol,title,dlabel,unit='',diff_unit=None,max_frames=24):
    """Html slider viewer of reference, test and difference slices along the first axis."""
    if diff_unit is None:
        diff_unit = f'Diff{dlabel}'
    nz = vol1.shape[0]
    n_frames = min(nz, max_frames)
    frame_indices = np.unique(np.linspace(0, nz - 1, n_frames).astype(int))

    vmin=min(np.nanmin(vol1),np.nanmin(vol2))
    vmax=max(np.nanmax(vol1),np.nanmax(vol2))
    finite_diff = diff_vol[np.isfinite(diff_vol)]
    dvmin = finite_diff.min() if finite_diff.size else 0
    dvmax = finite_diff.max() if finite_diff.size else 1

    # Degenerate/near-uniform ranges (e.g. a diff volume of near-identical floats)
    # make matplotlib's default formatter print full-precision floats, which are
    # long enough to get clipped off the saved image; cap tick labels to 3 sig figs.
    tick_formatter = mticker.FormatStrFormatter('%.3g')

    fig, axs = plt.subplots(1,3,figsize=(9,3),dpi=70)
    panels = ((vol1,'Reference',vmin,vmax,None,unit),
              (vol2,'Test',vmin,vmax,None,unit),
              (diff_vol,f'Diff{dlabel}',dvmin,dvmax,plt.cm.jet,diff_unit))
    images = []
    for ax, (vol, panel_title, lo, hi, cmap, cb_label) in zip(axs, panels):
        im = ax.imshow(vol[frame_indices[0]],vmin=lo,vmax=hi,cmap=cmap)
        ax.set_title(panel_title)
        cb = plt.colorbar(im,ax=ax,fraction=0.046)
        cb.ax.yaxis.set_major_formatter(tick_formatter)
        cb.set_label(cb_label)
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        images.append(im)
    suptitle=fig.suptitle(f'{title}\nSlice {frame_indices[0]+1}/{nz}')
    plt.tight_layout()

    frames_b64 = []
    for z in frame_indices:
        for im, vol in zip(images, (vol1, vol2, diff_vol)):
            im.set_data(vol[z])
        suptitle.set_text(f'{title}\nSlice {z+1}/{nz}')
        frames_b64.append(_webp_b64(fig))
    plt.close(fig)

    return slider_viewer_html([('', frames_b64)], frame_names=[f'Slice {z+1}/{nz}' for z in frame_indices],
                              start=len(frame_indices)//2, width=630)


# ================================================================================================================================
# MESHES
# ================================================================================================================================
# 3D plots of meshes and voxels. Every plot returns the frames of a camera orbit around its reference
# mesh; show them with report_viewers.slider_viewer_entry, where the slider moves the camera (several
# plots of the same mesh line up frame by frame).
def _plotter(size=500):
    plotter = pv.Plotter(window_size=(size, size),off_screen=True)
    plotter.background_color = 'white'
    return plotter


def _orbit_frames(plotter,reference_mesh,n_frames=36):
    """Frames (base64 WebP) of the camera making a full turn around the scene while sweeping from
    60 degrees above to 60 degrees below it, so every side can be reached with the slider. The
    camera is fitted to reference_mesh (one mesh or a list) rather than to the scene, so plots
    sharing it are framed identically and their frames line up."""
    plotter.show(auto_close=False)
    references = reference_mesh if isinstance(reference_mesh, (list, tuple)) else [reference_mesh]
    all_bounds = np.array([np.asarray(pv.wrap(m).bounds).reshape(3, 2) for m in references])
    bounds = np.stack([all_bounds[:, :, 0].min(axis=0), all_bounds[:, :, 1].max(axis=0)], axis=1)
    focal = bounds.mean(axis=1)
    radius = np.linalg.norm(bounds[:, 1] - bounds[:, 0]) / 2
    distance = 1.1 * radius / np.tan(np.radians(plotter.camera.view_angle) / 2)
    frames = []
    for i in range(n_frames):
        azimuth = 2*np.pi*i/n_frames
        elevation = np.radians(60)*np.sin(azimuth)
        direction = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
        plotter.camera.position = focal + distance*direction
        plotter.camera.focal_point = focal
        plotter.camera.up = (0, 0, 1)
        plotter.render()
        buffer = BytesIO()
        Image.fromarray(plotter.screenshot(return_img=True)).save(buffer,format='WEBP',quality=70)
        frames.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    plotter.close()
    return frames


def _add_points(plotter,points,max_points,**kwargs):
    step = max(1, len(points)//max_points) # Plot at most ~max_points points
    plotter.add_mesh(pv.PolyData(np.asarray(points[::step],dtype=np.float64)),**kwargs)


def intersection_plot(mesh1,mesh2,mesh3=None,title=''):
    """mesh1 (grey) and mesh2 (blue), with their intersection mesh3 in red when given. Framed to
    fit both inputs."""
    plotter = _plotter()
    plotter.add_mesh(pv.wrap(mesh1), opacity=0.2, color='lightgray', show_scalar_bar=False)
    plotter.add_mesh(pv.wrap(mesh2), opacity=0.2, color='steelblue', show_scalar_bar=False)
    if mesh3 is not None:
        plotter.add_mesh(pv.wrap(mesh3), opacity=0.6, color='red', show_scalar_bar=False)
    plotter.add_title(title, font_size=10)
    return _orbit_frames(plotter,[mesh1,mesh2])


def mesh_plot(meshes,title='',color=None):
    """Meshes framed to fit all of them (default colour, or `color` for every mesh)."""
    plotter = _plotter()
    for mesh in meshes:
        plotter.add_mesh(pv.wrap(mesh),opacity=0.5,color=color,show_scalar_bar=False)
    plotter.add_title(title, font_size=12)
    return _orbit_frames(plotter,list(meshes))


def voxel_plot(mesh,Points,title=''):
    plotter = _plotter()
    plotter.add_mesh(pv.wrap(mesh),opacity=0.5,show_scalar_bar=False)
    _add_points(plotter,Points,1000000,color='blue',opacity=0.1)
    plotter.add_title(title,font_size=12)
    return _orbit_frames(plotter,mesh)


def difference_plot(mesh,only_gpu_points,only_cpu_points,title=''):
    """Skin mesh with the voxels found by only one of the GPU (red) and CPU (blue) calculations."""
    plotter = _plotter()
    plotter.add_mesh(pv.wrap(mesh),opacity=0.2,color='lightgray',show_scalar_bar=False)
    for points, color, label in ((only_gpu_points,'red','GPU only'),(only_cpu_points,'blue','CPU only')):
        if len(points):
            _add_points(plotter,points,200000,color=color,point_size=4,label=f"{label} ({len(points)})")
    if len(only_gpu_points) or len(only_cpu_points):
        plotter.add_legend(loc='lower left',size=(0.4,0.1),bcolor='white')
    plotter.add_title(title,font_size=10)
    return _orbit_frames(plotter,mesh)


MESH_PLOTS = {'intersection_plot': intersection_plot,'mesh_plot': mesh_plot,'voxel_plot': voxel_plot,
              'difference_plot': difference_plot}
