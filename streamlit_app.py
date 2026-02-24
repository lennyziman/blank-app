# streamlit_app.py
# Parking Layout Generator (MVP)
# - Upload satellite image (background)
# - Draw ONE boundary rectangle + optional obstacle rectangles
# - Draw ONE calibration line and input its real length (m)
# - Generates best layout via orientation sweep (90° stalls only)
#
# requirements.txt (recommended):
# streamlit==1.54.0
# streamlit-drawable-canvas-fix==0.9.8
# shapely==2.0.6
# pillow
# matplotlib
# numpy

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union

import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas


# -----------------------------
# Geometry helpers
# -----------------------------
def rect_from_fabric(obj: dict) -> Polygon:
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
    height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
    return box(left, top, left + width, top + height)


def line_from_fabric(obj: dict) -> Optional[LineString]:
    if obj.get("type") != "line":
        return None

    x1 = obj.get("x1", None)
    y1 = obj.get("y1", None)
    x2 = obj.get("x2", None)
    y2 = obj.get("y2", None)

    pts = obj.get("points", None)
    if (x1 is None or y1 is None or x2 is None or y2 is None) and isinstance(pts, list) and len(pts) >= 4:
        x1, y1, x2, y2 = pts[:4]

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))

    x1 = float(x1) + left
    y1 = float(y1) + top
    x2 = float(x2) + left
    y2 = float(y2) + top

    return LineString([(x1, y1), (x2, y2)])


def safe_buffer_negative(poly: Polygon, dist: float) -> Polygon:
    if dist <= 0:
        return poly
    out = poly.buffer(-dist)
    if out.is_empty:
        return Polygon()
    if out.geom_type == "MultiPolygon":
        out = max(out.geoms, key=lambda g: g.area)
    return out


# -----------------------------
# Layout generation
# -----------------------------
@dataclass
class LayoutParams:
    stall_w_m: float = 2.5
    stall_l_m: float = 5.0
    aisle_w_m: float = 6.0
    setback_m: float = 0.3
    angle_step_deg: int = 5


def generate_layout_90deg(
    buildable_px: Polygon,
    meters_per_px: float,
    params: LayoutParams,
    orientation_deg: float,
) -> List[Polygon]:
    if buildable_px.is_empty or buildable_px.area <= 1.0:
        return []

    stall_w_px = params.stall_w_m / meters_per_px
    stall_l_px = params.stall_l_m / meters_per_px
    aisle_w_px = params.aisle_w_m / meters_per_px

    cx, cy = buildable_px.centroid.x, buildable_px.centroid.y
    rot_build = shp_rotate(buildable_px, orientation_deg, origin=(cx, cy), use_radians=False)

    minx, miny, maxx, maxy = rot_build.bounds
    stalls: List[Polygon] = []

    band_h = stall_l_px + aisle_w_px + stall_l_px
    y = miny
    while y + band_h <= maxy + 1e-6:
        row1_y0 = y
        row1_y1 = y + stall_l_px
        row2_y0 = y + stall_l_px + aisle_w_px
        row2_y1 = row2_y0 + stall_l_px

        x = minx
        while x + stall_w_px <= maxx + 1e-6:
            r1 = box(x, row1_y0, x + stall_w_px, row1_y1)
            r2 = box(x, row2_y0, x + stall_w_px, row2_y1)

            if r1.within(rot_build):
                stalls.append(r1)
            if r2.within(rot_build):
                stalls.append(r2)

            x += stall_w_px

        y += band_h

    return [shp_rotate(s, -orientation_deg, origin=(cx, cy), use_radians=False) for s in stalls]


def best_layout_orientation_sweep(
    buildable_px: Polygon,
    meters_per_px: float,
    params: LayoutParams,
) -> Tuple[float, List[Polygon]]:
    best_angle = 0.0
    best_stalls: List[Polygon] = []
    step = max(1, int(params.angle_step_deg))

    for ang in range(0, 180, step):
        stalls = generate_layout_90deg(buildable_px, meters_per_px, params, orientation_deg=float(ang))
        if len(stalls) > len(best_stalls):
            best_stalls = stalls
            best_angle = float(ang)

    return best_angle, best_stalls


# -----------------------------
# Rendering
# -----------------------------
def render_result(
    img: Optional[Image.Image],
    boundary: Polygon,
    obstacles: List[Polygon],
    stalls: List[Polygon],
    title: str = "",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.set_title(title)

    if img is not None:
        ax.imshow(img)
        h, w = img.height, img.width
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    def draw_poly(p: Polygon, lw: float):
        if p.is_empty:
            return
        x, y = p.exterior.xy
        ax.plot(x, y, linewidth=lw)

    draw_poly(boundary, 3.0)
    for ob in obstacles:
        draw_poly(ob, 2.0)
    for s in stalls:
        draw_poly(s, 1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Parking Layout Generator (MVP)", layout="wide")
st.title("Parking Layout Generator (MVP) — Python + Streamlit")

# Initialize state
if "bg_img" not in st.session_state:
    st.session_state["bg_img"] = None
if "bg_name" not in st.session_state:
    st.session_state["bg_name"] = None

with st.sidebar:
    st.header("1) Parameters (meters)")
    stall_w_m = st.number_input("Stall width (m)", min_value=2.0, max_value=3.5, value=2.5, step=0.1)
    stall_l_m = st.number_input("Stall length (m)", min_value=4.0, max_value=7.0, value=5.0, step=0.1)
    aisle_w_m = st.number_input("Aisle width (m)", min_value=3.0, max_value=9.0, value=6.0, step=0.1)
    setback_m = st.number_input("Boundary setback (m)", min_value=0.0, max_value=3.0, value=0.3, step=0.1)
    angle_step_deg = st.number_input("Orientation sweep step (deg)", min_value=1, max_value=15, value=5, step=1)

    st.divider()
    st.header("2) Calibration")
    real_line_m = st.number_input(
        "Calibration line real length (m)",
        min_value=1.0,
        max_value=500.0,
        value=20.0,
        step=1.0,
        help="Draw ONE line on the canvas. Enter how many meters it represents.",
    )

params = LayoutParams(
    stall_w_m=float(stall_w_m),
    stall_l_m=float(stall_l_m),
    aisle_w_m=float(aisle_w_m),
    setback_m=float(setback_m),
    angle_step_deg=int(angle_step_deg),
)

col1, col2 = st.columns([1.0, 1.0], gap="large")


# ---- Left: upload + draw ----
with col1:
    st.subheader("A) Upload image (optional)")

    def _on_upload_change():
        up = st.session_state.get("uploader")
        if up is None:
            st.session_state["bg_img"] = None
            st.session_state["bg_name"] = None
            return

        img_local = Image.open(up).convert("RGBA")

        max_w = 1100
        scale = min(1.0, max_w / img_local.width)
        if scale < 1.0:
            img_local = img_local.resize(
                (int(img_local.width * scale), int(img_local.height * scale))
            )

        st.session_state["bg_img"] = img_local
        st.session_state["bg_name"] = up.name

    st.file_uploader(
        "Upload a satellite image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        key="uploader",
        on_change=_on_upload_change,
    )

    background_pil = st.session_state.get("bg_img", None)

    if background_pil is None:
        st.info("No image uploaded. A blank canvas will be used.")
        canvas_w, canvas_h = 900, 550
        canvas_key = "canvas_blank"
        background_for_canvas = None
    else:
        canvas_w, canvas_h = background_pil.width, background_pil.height
        canvas_key = f"canvas_{st.session_state.get('bg_name')}_{canvas_w}x{canvas_h}"
        # IMPORTANT: pass numpy array to the canvas background renderer
        background_for_canvas = np.array(background_pil)

    st.subheader("B) Draw boundary + obstacles (rectangles) and calibration line")
    drawing_mode = st.radio("Tool", ["rect", "line"], horizontal=True)
    stroke_width = 3 if drawing_mode == "rect" else 4

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_image=background_for_canvas,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode=drawing_mode,
        key=canvas_key,
    )


# ---- Right: generate ----
with col2:
    st.subheader("C) Generate layout")

    if st.button("Generate best layout", type="primary"):
        if canvas.json_data is None or "objects" not in canvas.json_data:
            st.error("No drawings found. Draw a boundary rectangle and a calibration line.")
            st.stop()

        objects = canvas.json_data["objects"]
        rects = [o for o in objects if o.get("type") == "rect"]
        lines = [o for o in objects if o.get("type") == "line"]

        if len(rects) < 1:
            st.error("Draw at least ONE rectangle (the boundary).")
            st.stop()
        if len(lines) < 1:
            st.error("Draw ONE calibration line.")
            st.stop()

        boundary_px = rect_from_fabric(rects[0])
        obstacles_px = [rect_from_fabric(r) for r in rects[1:]]

        cal_line = line_from_fabric(lines[0])
        if cal_line is None or cal_line.length <= 1e-6:
            st.error("Calibration line invalid. Draw a clear line.")
            st.stop()

        meters_per_px = float(real_line_m) / float(cal_line.length)
        if meters_per_px <= 0:
            st.error("Calibration failed (meters_per_px <= 0).")
            st.stop()

        setback_px = params.setback_m / meters_per_px
        buildable = safe_buffer_negative(boundary_px, setback_px)

        if obstacles_px:
            buildable = buildable.difference(unary_union(obstacles_px))

        if buildable.is_empty or buildable.area <= 1.0:
            st.error("Buildable area is empty. Reduce setback or redraw boundary/obstacles.")
            st.stop()

        best_ang, best_stalls = best_layout_orientation_sweep(buildable, meters_per_px, params)

        buildable_m2 = buildable.area * (meters_per_px ** 2)
        stalls_per_100m2 = (len(best_stalls) / buildable_m2) * 100.0 if buildable_m2 > 0 else 0.0

        st.markdown(
            f"""
            **Calibration:** {meters_per_px:.4f} m/px  
            **Best orientation:** {best_ang:.0f}°  
            **Stall count:** {len(best_stalls)}  
            **Buildable area:** {buildable_m2:.1f} m²  
            **Efficiency:** {stalls_per_100m2:.2f} stalls per 100 m²  
            """
        )

        fig = render_result(
            img=background_pil,
            boundary=boundary_px,
            obstacles=obstacles_px,
            stalls=best_stalls,
            title=f"Best layout (orientation {best_ang:.0f}°) — {len(best_stalls)} stalls",
        )
        st.pyplot(fig)

        export = {
            "meters_per_pixel": meters_per_px,
            "best_orientation_deg": best_ang,
            "stall_count": len(best_stalls),
            "stalls": [list(map(list, s.exterior.coords)) for s in best_stalls],
        }

        st.download_button(
            "Download stalls as JSON",
            data=json.dumps(export, indent=2),
            file_name="parking_layout.json",
            mime="application/json",
        )

st.divider()
st.caption(
    "MVP limitations: rectangles only, 90° stalls only, simple orientation sweep. "
    "Next upgrades: arbitrary polygon boundary, obstacle polygons, angled parking (45/60), accessibility rules, exports."
)
