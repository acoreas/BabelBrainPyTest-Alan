"""
Interactive image viewer for the pytest-html reports, and the html of a test's report entries.

Every test that puts images in a report (2D volume slices, 3D mesh views, h5 comparisons, pipeline
step screenshots) shows them through slider_viewer_html: a slider picks the frame (slice position,
camera position around the mesh, pipeline step, ...) and a Play/Pause button steps through them. All
panels of one viewer always show the same frame index.

pytest-html inserts report extras through innerHTML, so <script> tags never run; everything is
wired through inline on*="" attributes instead. The frames ride in a data-* attribute, are decoded
once up front and then drawn onto a <canvas> per panel, so changing frame never flickers (swapping
an <img> src to a new data URL makes the browser decode each frame as it is shown).
"""
import base64
import json
import uuid
from io import BytesIO

from PIL import Image


def frame_size(frame_b64):
    return Image.open(BytesIO(base64.b64decode(frame_b64))).size


def slider_viewer_html(panels, frame_label='Frame', frame_names=None, start=0, width=None,
                       autoplay=False, interval_ms=150):
    """HTML for side-by-side frame sequences driven by one slider.

    panels:      list of (caption, [base64 webp frames]); all with the same number of frames
    frame_label: text before the frame number in the label ("Slice 3/24")
    frame_names: optional list of names shown instead ("Step 2"), one per frame
    start:       frame shown first
    width:       display width (px) of each panel, default the frame width capped so the whole
                 viewer fits in ~1000 px
    autoplay:    start playing once the frames are decoded
    """
    n_frames = len(panels[0][1])
    assert all(len(frames) == n_frames for _, frames in panels), "all panels need the same number of frames"
    frame_w, frame_h = frame_size(panels[0][1][0])
    if width is None:
        width = min(frame_w, 1000 // len(panels))
    height = round(width * frame_h / frame_w)
    wid = f"viewer_{uuid.uuid4().hex[:8]}"
    frames_json = '[' + ','.join('[' + ','.join(f'"{b}"' for b in frames) + ']' for _, frames in panels) + ']'
    names = frame_names or [f"{frame_label} {i + 1}/{n_frames}" for i in range(n_frames)]
    names_json = json.dumps(names).replace("'", "&#39;")

    # show(v) draws frame v of every panel (only once all frames are decoded)
    show_js = (
        f"var c=document.getElementById('{wid}');"
        "function show(v){"
        "if(!c._ready){return;}"
        "for(var k=0;k<c._imgs.length;k++){"
        f"var cv=document.getElementById('{wid}_cv'+k);"
        "cv.getContext('2d').drawImage(c._imgs[k][v],0,0,cv.width,cv.height);}"
        f"document.getElementById('{wid}_label').textContent=c._names[v];}}"
    )
    slider_oninput = show_js + "show(parseInt(this.value,10));"
    play_onclick = (
        show_js +
        "if(!c._ready){return;}"
        f"var slider=document.getElementById('{wid}_slider');"
        "if(c._timer){clearInterval(c._timer);c._timer=null;this.textContent='Play';return;}"
        "this.textContent='Pause';"
        "c._timer=setInterval(function(){"
        "var v=(parseInt(slider.value,10)+1)%c._imgs[0].length;"
        "slider.value=v;show(v);"
        f"}},{interval_ms});"
    )
    # Runs once, when the hidden first frame has loaded: decode every frame, then draw the start frame
    init_onload = (
        show_js +
        "if(c._started){return;}"
        "c._started=true;"
        "var frames=JSON.parse(c.dataset.frames);"
        "c._names=JSON.parse(c.dataset.names);"
        "var pending=[];"
        "c._imgs=frames.map(function(panel){return panel.map(function(b){"
        "var im=new Image();im.src='data:image/webp;base64,'+b;"
        "pending.push(im.decode?im.decode():Promise.resolve());return im;});});"
        "Promise.all(pending).then(function(){"
        f"c._ready=true;show({start});"
        + (f"document.getElementById('{wid}_play').click();" if autoplay else "") +
        "});"
    )
    cells = ''.join(
        f"<td style='text-align:center;vertical-align:top'>"
        f"<canvas id='{wid}_cv{k}' width='{frame_w}' height='{frame_h}' style='width:{width}px;height:{height}px'></canvas>"
        + (f"<div style='font-size:12px'>{caption}</div>" if caption else '') +
        "</td>"
        for k, (caption, _) in enumerate(panels))
    return f"""
<div id='{wid}' data-frames='{frames_json}' data-names='{names_json}' style='text-align:center;margin:4px 0 12px 0'>
  <img src='data:image/webp;base64,{panels[0][1][0]}' style='display:none' onload="{init_onload}">
  <table style='margin:auto'><tr>{cells}</tr></table>
  <input type='range' id='{wid}_slider' min='0' max='{n_frames - 1}' value='{start}' style='width:{min(width * len(panels), 1000)}px'
    oninput="{slider_oninput}"><br>
  <span id='{wid}_label'>Loading...</span>
  <button type='button' id='{wid}_play' style='margin-left:8px' onclick="{play_onclick}">Play</button>
</div>
"""


def slider_viewer_entry(panels, **kwargs):
    """Report entry (for request.node.screenshots) holding a slider_viewer_html."""
    return {'kind': 'html', 'html': slider_viewer_html(panels, **kwargs)}


def report_entries_html(entries):
    """Html for a test's report entries (request.node.screenshots): html entries (viewers, tables)
    each get a row, consecutive images (dicts with 'image', or legacy base64 WebP strings) share one."""
    html_parts = []
    row_tags = ''

    def flush_row():
        nonlocal row_tags
        if row_tags:
            html_parts.append(f"<tr>{row_tags}</tr>")
            row_tags = ''

    for entry in entries:
        if isinstance(entry, dict):
            kind = entry.get('kind', 'image')
            if kind == 'html':
                flush_row()
                html_parts.append(f"<tr><td colspan='10'>{entry['html']}</td></tr>")
            else:
                caption = entry.get('caption')
                caption_html = f"<div style='text-align:center;font-size:12px'>{caption}</div>" if caption else ''
                width = entry.get('width', 500)
                mime = entry.get('mime', 'webp')
                row_tags += f"<td><img src='data:image/{mime};base64,{entry['image']}' width='{width}'>{caption_html}</td>"
        else:
            # Legacy entries: plain base64-encoded webp strings
            row_tags += "<td><img src='data:image/webp;base64,{}' width='500'></td>".format(entry)
    flush_row()
    return ''.join(html_parts)
