"""
RaZui – YOLO Person Detection Overlay
======================================
REQUIREMENTS
    pip install ultralytics mss opencv-python pywin32 imgui[glfw] PyOpenGL

MODELS  (select in Settings tab – drop next to this script or set full path)
    yolov8n.pt  →  fastest  (~30+ fps on CPU after resize optimisation)
    yolov8s.pt  →  better accuracy, still fast
    yolov8m.pt  →  most accurate of the common ones
    yolov8l/x.pt → best accuracy, needs GPU for good fps
    Download:  github.com/ultralytics/assets/releases

GPU SUPPORT
    Install PyTorch with CUDA:  https://pytorch.org/get-started/locally/
    Then enable "Use GPU" in Settings.  Falls back to CPU if CUDA unavailable.

HOTKEYS
    INSERT   toggle menu
    F2       toggle debug window
"""

import os, sys, time, threading, colorsys
import win32api, win32con, win32gui
import glfw, imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import cv2, numpy as np, mss

# ════════════════════════════════════════════════════════
#  SCREEN SIZE
# ════════════════════════════════════════════════════════
if not glfw.init(): sys.exit("[RaZui] glfw init failed")
_vm = glfw.get_video_mode(glfw.get_primary_monitor())
SW, SH = _vm.size.width, _vm.size.height
glfw.terminate()

# ════════════════════════════════════════════════════════
#  LAYOUT BASE (scaled by DPI)
# ════════════════════════════════════════════════════════
_BASE_GW    = 310
_BASE_GH    = 310
_BASE_TITLE = 26
_BASE_TABS  = 24
_BASE_ROW   = 23
_BASE_FONT  = 13.0
INS_KEY     = 0x2D
F2_KEY      = 0x71

C_BG     = (0.078, 0.078, 0.102)
C_TITLE  = (0.048, 0.048, 0.065)
C_HOV    = (0.110, 0.110, 0.148)
C_BORDER = (0.26,  0.26,  0.34 )
C_TEXT   = (0.84,  0.86,  0.92 )
C_DIM    = (0.40,  0.42,  0.52 )
C_SW_OFF = (0.18,  0.18,  0.25 )

TABS = ["Aim Assist", "Detection", "Visuals", "Settings"]

# ════════════════════════════════════════════════════════
#  SHARED STATE
# ════════════════════════════════════════════════════════
class State:
    lock = threading.Lock()

    # YOLO
    enabled      = True
    confidence   = 0.45
    model_path   = "yolov8n.pt"
    detections   = []        # [(x1,y1,x2,y2,conf), ...] screen coords – raw from model
    smooth_dets  = []        # temporally smoothed detections for display / aim
    fps_det      = 0.0
    model_loaded = False
    model_error  = ""

    # Aim EMA smoothing – averages aim target across frames to kill jitter
    _aim_smooth_x: float = -1.0   # -1 = unset
    _aim_smooth_y: float = -1.0
    aim_smooth    = 0.25           # EMA alpha: lower = smoother/more lag (0.10–0.60)

    # Capture – centred square, half-side = fov_r pixels
    fov_r        = min(SW, SH) // 8  # ~90px radius default

    # Visuals
    show_boxes   = True
    show_conf    = True
    show_fov     = True      # white outline of capture square
    show_debug   = False
    debug_frame  = None

    # Settings
    watermark    = True
    chromatic    = True
    dpi_scale    = 100.0
    use_gpu      = False   # toggled in Settings; only active if CUDA is available

    # ── Placeholder / Aim Assist ──────────────────────────
    aim_enabled       = False   # master toggle for aim assist
    aim_hotkey        = 0x02    # VK code for aim hotkey (default: RMB)
    aim_hotkey_only   = False   # only assist when hotkey held
    aim_speed         = 8.0     # mouse drag step in pixels per tick
    aim_bone          = 1       # 0=Head  1=Chest  2=Body(center)
    _aim_binding      = False   # True while waiting for a key press
    _aim_bind_wait    = False   # True = "Set" was clicked, waiting for LMB release first

    # ── Clickbot ─────────────────────────────────────────
    click_enabled     = False   # master toggle for clickbot
    click_hotkey      = 0x02    # VK code (default: RMB)
    click_hotkey_only = False   # only click when hotkey held
    click_delay       = 100.0   # hold duration in ms
    _click_binding    = False   # True while waiting for a key press
    _click_bind_wait  = False   # True = "Set" clicked, waiting for LMB release
    _click_held       = False   # internal – currently holding LMB down

state = State()

# Model selector options
MODEL_OPTIONS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
_model_dd_open: bool = False

def cap_rect():
    """Return (x, y, w, h) of the centred capture square."""
    r = state.fov_r
    x = SW // 2 - r
    y = SH // 2 - r
    return max(0, x), max(0, y), max(4, r * 2), max(4, r * 2)

# ════════════════════════════════════════════════════════
#  DPI HELPERS
# ════════════════════════════════════════════════════════
def _dpi():    return max(0.75, min(1.50, state.dpi_scale / 100.0))
def GW():      return int(_BASE_GW    * _dpi())
def GH():      return int(_BASE_GH    * _dpi())
def TITLE():   return int(_BASE_TITLE * _dpi())
def TABS_H():  return int(_BASE_TABS  * _dpi())
def ROW_H():   return int(_BASE_ROW   * _dpi())

# ════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ════════════════════════════════════════════════════════
def u(r, g, b, a=1.0): return imgui.get_color_u32_rgba(r, g, b, a)
def ct(t, a=1.0):      return u(t[0], t[1], t[2], a)
def lerp(a, b, t):     return a + (b - a) * t
def clamp(v, a, b):    return max(a, min(b, v))

# Accent colour (HSV)
_acc_h: float = 0.60
_acc_s: float = 0.70
_acc_v: float = 0.95
_cp_drag: str  = ""
_accent_open:        bool  = False
_accent_just_opened: bool  = False
_cp_wx: float = -1.0
_cp_wy: float = -1.0

def _acc_rgb():
    return colorsys.hsv_to_rgb(_acc_h, _acc_s, _acc_v)

def _hline_fade(dl, x, y, w, col, alpha=0.55):
    if w <= 0: return
    r, g, b = col
    FADE = int(w * 0.30); mid0 = x + FADE; mid1 = x + w - FADE
    if mid1 > mid0:
        dl.add_line(mid0, y, mid1, y, u(r, g, b, alpha))
    for i in range(FADE):
        t = i / max(FADE, 1)
        dl.add_line(x + i,    y, x + i + 1,    y, u(r, g, b, alpha * t))
        dl.add_line(mid1 + i, y, mid1 + i + 1, y, u(r, g, b, alpha * (1.0 - t)))

# ════════════════════════════════════════════════════════
#  WIDGETS
# ════════════════════════════════════════════════════════
_sw_t:    dict = {}
_sl_drag: dict = {}
_dd_open: bool = False
DPI_OPTIONS = [75.0, 100.0, 125.0, 150.0]

def sw_anim(key, val):
    t = _sw_t.get(key, 1.0 if val else 0.0)
    t += ((1.0 if val else 0.0) - t) * 0.22
    _sw_t[key] = t
    return t

def draw_switch(dl, x, y, key, val):
    sc = _dpi()
    W = int(26*sc); H = int(11*sc); KW = int(8*sc); KH = H - int(3*sc)
    t = sw_anim(key, val); r, g, b = _acc_rgb()
    tr = lerp(C_SW_OFF[0], r, t); tg = lerp(C_SW_OFF[1], g, t); tb = lerp(C_SW_OFF[2], b, t)
    dl.add_rect_filled(x, y, x+W, y+H, u(tr, tg, tb, 1.0))
    dl.add_rect(x, y, x+W, y+H, u(1, 1, 1, 0.07 + t*0.04), thickness=0.8)
    kx = lerp(x+2, x+W-KW-2, t); ky = y + (H-KH)//2
    dl.add_rect_filled(kx+1, ky+1, kx+KW+1, ky+KH+1, u(0, 0, 0, 0.45))
    dl.add_rect_filled(kx,   ky,   kx+KW,   ky+KH,   u(0.94, 0.94, 1.0, 1.0))

def draw_slider(dl, x, y, w, key, val, lo=0.0, hi=200.0, fmt="%.0f"):
    sc = _dpi()
    BTN = int(10*sc); GAP = int(4*sc); TH = max(2, int(2*sc))
    ROW = int(14*sc); cy = y + ROW//2; ty = cy - TH//2
    tx0 = x + BTN + GAP; tx1 = x + w - BTN - GAP; tw = max(1, tx1 - tx0)
    mouse = imgui.get_mouse_pos(); mx, my = mouse[0], mouse[1]
    pressed = imgui.is_mouse_down(0); clicked = imgui.is_mouse_clicked(0)
    in_btn_l  = (x <= mx <= x+BTN)     and (y <= my <= y+ROW)
    in_btn_r  = (x+w-BTN <= mx <= x+w) and (y <= my <= y+ROW)
    in_track  = (tx0 <= mx <= tx1)     and (y <= my <= y+ROW)
    norm = clamp((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if _sl_drag.get(key, False):
        if pressed:
            norm = clamp((mx - tx0) / tw, 0.0, 1.0); val = lo + norm*(hi-lo)
        else: _sl_drag[key] = False
    elif in_track and clicked:
        _sl_drag[key] = True; norm = clamp((mx - tx0) / tw, 0.0, 1.0); val = lo + norm*(hi-lo)
    elif in_btn_l and clicked:
        val = clamp(val - (hi-lo)/20.0, lo, hi); norm = clamp((val-lo)/max(hi-lo,1e-6),0.0,1.0)
    elif in_btn_r and clicked:
        val = clamp(val + (hi-lo)/20.0, lo, hi); norm = clamp((val-lo)/max(hi-lo,1e-6),0.0,1.0)
    fx = tx0 + int(norm * tw); r, g, b = _acc_rgb()
    mc = u(r,g,b,1.0) if in_btn_l else ct(C_DIM, 1.0)
    pc = u(r,g,b,1.0) if in_btn_r else ct(C_DIM, 1.0)
    mw = imgui.calc_text_size("-").x; mh = imgui.calc_text_size("-").y
    dl.add_text(x + (BTN-mw)//2, cy - mh//2, mc, "-")
    pw = imgui.calc_text_size("+").x; ph = imgui.calc_text_size("+").y
    dl.add_text(x + w - BTN + (BTN-pw)//2, cy - ph//2, pc, "+")
    dl.add_rect_filled(tx0, ty, tx1, ty+TH, u(0.20, 0.20, 0.28, 1.0))
    if fx > tx0: dl.add_rect_filled(tx0, ty, fx, ty+TH, u(r, g, b, 1.0))
    vs = fmt % val; vsz = imgui.calc_text_size(vs)
    lx = clamp(fx - int(vsz.x/2), tx0, tx1 - int(vsz.x))
    label_y = cy - int(vsz.y/2)
    for ox, oy in ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)):
        dl.add_text(lx+ox, label_y+oy, u(0,0,0,0.90), vs)
    dl.add_text(lx, label_y, u(1,1,1,1.0), vs)
    return val

def draw_dpi_dropdown(dl, x, y, w):
    global _dd_open
    sc = _dpi(); H = int(18*sc); r, g, b = _acc_rgb()
    cur = state.dpi_scale; label = f"{int(cur)}%"
    mouse = imgui.get_mouse_pos()
    hov_hdr = (x <= mouse[0] <= x+w) and (y <= mouse[1] <= y+H)
    hdr_bg  = u(0.14,0.14,0.20,1.0) if hov_hdr else u(0.07,0.07,0.10,1.0)
    dl.add_rect_filled(x, y, x+w, y+H, hdr_bg)
    dl.add_rect(x, y, x+w, y+H, u(r,g,b, 0.90 if _dd_open else 0.55), thickness=1.0)
    lh = imgui.calc_text_size(label).y
    dl.add_text(x + int(7*sc), y + (H-lh)/2.0, u(r,g,b,1.0), label)
    arrow = "^" if _dd_open else "v"; aw = imgui.calc_text_size(arrow).x
    dl.add_text(x+w - aw - int(6*sc), y + (H-lh)/2.0, ct(C_DIM,1.0), arrow)
    if hov_hdr and imgui.is_mouse_clicked(0): _dd_open = not _dd_open
    if _dd_open:
        fdl = imgui.get_foreground_draw_list(); py = y + H
        panel_h = H * len(DPI_OPTIONS)
        fdl.add_rect_filled(x, py, x+w, py+panel_h, u(0.06,0.06,0.09,1.0))
        for opt in DPI_OPTIONS:
            opt_lbl = f"{int(opt)}%"; sel = abs(opt - cur) < 0.5
            hov_row = (x <= mouse[0] <= x+w) and (py <= mouse[1] <= py+H)
            if sel:      fdl.add_rect_filled(x, py, x+w, py+H, u(r*0.25,g*0.25,b*0.35,1.0))
            elif hov_row:fdl.add_rect_filled(x, py, x+w, py+H, u(0.13,0.13,0.19,1.0))
            fdl.add_line(x, py, x+w, py, u(0.20,0.20,0.28,1.0))
            if sel: fdl.add_rect_filled(x, py, x+int(2*sc), py+H, u(r,g,b,1.0))
            text_col = u(r,g,b,1.0) if sel else (ct(C_TEXT,1.0) if hov_row else ct(C_DIM,1.0))
            oh = imgui.calc_text_size(opt_lbl).y
            fdl.add_text(x+int(10*sc), py+(H-oh)/2.0, text_col, opt_lbl)
            if sel:
                ck = "+"; cw = imgui.calc_text_size(ck).x
                fdl.add_text(x+w-cw-int(7*sc), py+(H-oh)/2.0, u(r,g,b,1.0), ck)
            if hov_row and imgui.is_mouse_clicked(0):
                state.dpi_scale = opt; _dd_open = False
            py += H
        fdl.add_rect(x, y+H, x+w, py, u(r,g,b,0.70), thickness=1.0)

def draw_model_dropdown(dl, x, y, w):
    global _model_dd_open
    sc = _dpi(); H = int(18*sc); r, g, b = _acc_rgb()
    cur = state.model_path.strip()
    # show just the filename for display
    label = os.path.basename(cur) if cur else "select model"
    mouse = imgui.get_mouse_pos()
    hov_hdr = (x <= mouse[0] <= x+w) and (y <= mouse[1] <= y+H)
    hdr_bg  = u(0.14,0.14,0.20,1.0) if hov_hdr else u(0.07,0.07,0.10,1.0)
    dl.add_rect_filled(x, y, x+w, y+H, hdr_bg)
    dl.add_rect(x, y, x+w, y+H, u(r,g,b, 0.90 if _model_dd_open else 0.55), thickness=1.0)
    lh = imgui.calc_text_size(label).y
    dl.add_text(x + int(7*sc), y + (H-lh)/2.0, u(r,g,b,1.0), label)
    arrow = "^" if _model_dd_open else "v"; aw = imgui.calc_text_size(arrow).x
    dl.add_text(x+w - aw - int(6*sc), y + (H-lh)/2.0, ct(C_DIM,1.0), arrow)
    if hov_hdr and imgui.is_mouse_clicked(0): _model_dd_open = not _model_dd_open
    if _model_dd_open:
        fdl = imgui.get_foreground_draw_list(); py = y + H
        panel_h = H * len(MODEL_OPTIONS)
        fdl.add_rect_filled(x, py, x+w, py+panel_h, u(0.06,0.06,0.09,1.0))
        for opt in MODEL_OPTIONS:
            sel = (opt == os.path.basename(cur))
            hov_row = (x <= mouse[0] <= x+w) and (py <= mouse[1] <= py+H)
            if sel:       fdl.add_rect_filled(x, py, x+w, py+H, u(r*0.25,g*0.25,b*0.35,1.0))
            elif hov_row: fdl.add_rect_filled(x, py, x+w, py+H, u(0.13,0.13,0.19,1.0))
            fdl.add_line(x, py, x+w, py, u(0.20,0.20,0.28,1.0))
            if sel: fdl.add_rect_filled(x, py, x+int(2*sc), py+H, u(r,g,b,1.0))
            text_col = u(r,g,b,1.0) if sel else (ct(C_TEXT,1.0) if hov_row else ct(C_DIM,1.0))
            oh = imgui.calc_text_size(opt).y
            fdl.add_text(x+int(10*sc), py+(H-oh)/2.0, text_col, opt)
            if sel:
                ck = "+"; cw = imgui.calc_text_size(ck).x
                fdl.add_text(x+w-cw-int(7*sc), py+(H-oh)/2.0, u(r,g,b,1.0), ck)
            if hov_row and imgui.is_mouse_clicked(0):
                state.model_path = opt; _model_dd_open = False
            py += H
        fdl.add_rect(x, y+H, x+w, py, u(r,g,b,0.70), thickness=1.0)

def _draw_color_picker_at(dl, ox, oy, sc):
    global _acc_h, _acc_s, _acc_v, _cp_drag
    mouse = imgui.get_mouse_pos(); mx, my = mouse[0], mouse[1]
    pressed = imgui.is_mouse_down(0); clicked = imgui.is_mouse_clicked(0)
    SV_SIZE = int(80*sc); HUE_W = int(14*sc); GAP = int(6*sc); SW_W = int(20*sc)
    sv_x = ox; sv_y = oy; hue_x = ox + SV_SIZE + GAP; hue_y = oy
    GRID = 24; cell = SV_SIZE / GRID
    for xi in range(GRID):
        for yi in range(GRID):
            cr, cg, cb = colorsys.hsv_to_rgb(_acc_h, xi/(GRID-1), 1.0-yi/(GRID-1))
            dl.add_rect_filled(sv_x+int(xi*cell), sv_y+int(yi*cell),
                               sv_x+int((xi+1)*cell)+1, sv_y+int((yi+1)*cell)+1,
                               u(cr,cg,cb,1.0))
    dl.add_rect(sv_x, sv_y, sv_x+SV_SIZE, sv_y+SV_SIZE, ct(C_BORDER,1.0), thickness=1.0)
    cur_sx = sv_x+int(_acc_s*SV_SIZE); cur_sy = sv_y+int((1.0-_acc_v)*SV_SIZE)
    dl.add_circle_filled(cur_sx, cur_sy, int(4*sc), u(0,0,0,0.8))
    dl.add_circle_filled(cur_sx, cur_sy, int(3*sc), u(1,1,1,1.0))
    in_sv = (sv_x<=mx<=sv_x+SV_SIZE) and (sv_y<=my<=sv_y+SV_SIZE)
    if _cp_drag == "sv":
        if pressed:
            _acc_s = clamp((mx-sv_x)/SV_SIZE, 0.0, 1.0)
            _acc_v = 1.0 - clamp((my-sv_y)/SV_SIZE, 0.0, 1.0)
        else: _cp_drag = ""
    elif in_sv and clicked:
        _cp_drag = "sv"
        _acc_s = clamp((mx-sv_x)/SV_SIZE, 0.0, 1.0)
        _acc_v = 1.0 - clamp((my-sv_y)/SV_SIZE, 0.0, 1.0)
    H_STEPS = 64; seg_h = SV_SIZE / H_STEPS
    for i in range(H_STEPS):
        hr, hg, hb = colorsys.hsv_to_rgb(i/H_STEPS, 1.0, 1.0)
        hy0 = hue_y+int(i*seg_h); hy1 = hue_y+int((i+1)*seg_h)+1
        dl.add_rect_filled(hue_x, hy0, hue_x+HUE_W, hy1, u(hr,hg,hb,1.0))
    dl.add_rect(hue_x, hue_y, hue_x+HUE_W, hue_y+SV_SIZE, ct(C_BORDER,1.0), thickness=1.0)
    hcy = hue_y+int(_acc_h*SV_SIZE)
    dl.add_rect_filled(hue_x-1, hcy-1, hue_x+HUE_W+1, hcy+2, u(0,0,0,0.8))
    dl.add_rect_filled(hue_x,   hcy,   hue_x+HUE_W,   hcy+1, u(1,1,1,1.0))
    in_hue = (hue_x<=mx<=hue_x+HUE_W) and (hue_y<=my<=hue_y+SV_SIZE)
    if _cp_drag == "wheel":
        if pressed: _acc_h = clamp((my-hue_y)/SV_SIZE, 0.0, 1.0)
        else: _cp_drag = ""
    elif in_hue and clicked:
        _cp_drag = "wheel"; _acc_h = clamp((my-hue_y)/SV_SIZE, 0.0, 1.0)
    pr, pg, pb = colorsys.hsv_to_rgb(_acc_h, _acc_s, _acc_v)
    sw_x = hue_x + HUE_W + GAP
    dl.add_rect_filled(sw_x, sv_y, sw_x+SW_W, sv_y+SV_SIZE, u(pr,pg,pb,1.0))
    dl.add_rect(sw_x, sv_y, sw_x+SW_W, sv_y+SV_SIZE, ct(C_BORDER,1.0), thickness=1.0)
    return SV_SIZE

# ════════════════════════════════════════════════════════
#  WATERMARK
# ════════════════════════════════════════════════════════
def draw_watermark(dl, fps):
    if not state.watermark: return
    r, g, b = _acc_rgb(); SEP = "  |  "
    segments = [
        ("Ra", u(r,g,b,1.0)), ("Zui", ct(C_TEXT,1.0)),
        (SEP,  ct(C_DIM,1.0)), (f"{fps:.0f} fps", ct(C_TEXT,1.0)),
        (SEP,  ct(C_DIM,1.0)), ("undetected", ct(C_DIM,1.0)),
    ]
    total_w = sum(imgui.calc_text_size(t).x for t, _ in segments)
    lh = imgui.calc_text_size("RaZui").y; pad = 8
    W = int(total_w + pad*2); H = int(lh + pad*2)
    bx = SW - W - 14; by = 14
    dl.add_rect_filled(bx, by, bx+W, by+H, ct(C_TITLE, 0.96))
    dl.add_rect(bx, by, bx+W, by+H, ct(C_BORDER, 1.0), thickness=1.0)
    dl.add_line(bx, by+H-1, bx+W, by+H-1, u(r,g,b,0.80), 1.0)
    cx = bx + pad; ty = by + pad
    for txt, col in segments:
        dl.add_text(cx, ty, col, txt); cx += imgui.calc_text_size(txt).x

# ════════════════════════════════════════════════════════
#  DETECTION DRAW
# ════════════════════════════════════════════════════════
def draw_detections(dl):
    rx, ry, rw, rh = cap_rect()

    # white outline = FOV box
    if state.show_fov:
        dl.add_rect(rx, ry, rx+rw, ry+rh, u(1,1,1,0.65), thickness=1.0)

    if not state.show_boxes: return

    r, g, b = _acc_rgb()
    draw_list = state.smooth_dets if state.smooth_dets else state.detections
    for entry in draw_list:
        x1, y1, x2, y2, conf = entry[0], entry[1], entry[2], entry[3], entry[4]
        # Plain outline box, no fill, original size
        dl.add_rect(x1, y1, x2, y2, u(r, g, b, 0.90), thickness=1.5)
        if state.show_conf:
            bw = x2 - x1
            lbl   = "Player"
            lsz   = imgui.calc_text_size(lbl)
            lx    = x1 + (bw - lsz.x) / 2.0
            ly    = y1 - lsz.y - 2
            for ox, oy in ((-1,0),(1,0),(0,-1),(0,1)):
                dl.add_text(lx+ox, ly+oy, u(0,0,0,0.75), lbl)
            dl.add_text(lx, ly, u(r,g,b,1.0), lbl)
            pct   = f"{conf:.0%}"
            psz   = imgui.calc_text_size(pct)
            px_   = x2 - psz.x - 4
            py_   = y1 + 3
            for ox, oy in ((-1,0),(1,0),(0,-1),(0,1)):
                dl.add_text(px_+ox, py_+oy, u(0,0,0,0.75), pct)
            dl.add_text(px_, py_, u(r,g,b,0.85), pct)

# ════════════════════════════════════════════════════════
#  DEBUG WINDOW
# ════════════════════════════════════════════════════════
_debug_tex_id = None

def _update_debug_texture(frame_bgr):
    global _debug_tex_id
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if _debug_tex_id is None: _debug_tex_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, _debug_tex_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_rgb)
    return _debug_tex_id

def _draw_debug_window():
    if not state.show_debug: return
    with state.lock: frame = state.debug_frame
    DW, DH = 540, 340
    imgui.set_next_window_position(SW-DW-20, 60, imgui.ONCE)
    imgui.set_next_window_size(DW, DH)
    expanded, opened = imgui.begin("Debug – Capture View  [F2]",
                                   state.show_debug, imgui.WINDOW_NO_RESIZE)
    if not opened:
        state.show_debug = False
    else:
        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min((DW-16)/fw, 260/fh)
            tid = _update_debug_texture(cv2.resize(frame,(int(fw*scale),int(fh*scale))))
            imgui.image(tid, int(fw*scale), int(fh*scale))
        else:
            imgui.text_colored("Waiting for first frame…", 0.5,0.5,0.5,1.0)
        imgui.separator()
        rx, ry, rw, rh = cap_rect()
        imgui.text(f"Region: ({rx},{ry})  {rw}x{rh}   Conf: {state.confidence:.2f}   Found: {len(state.detections)}")
        imgui.text(f"Det FPS: {state.fps_det:.1f}   Model: {os.path.basename(state.model_path)}")
    imgui.end()

# ════════════════════════════════════════════════════════
#  DETECTOR THREAD
# ════════════════════════════════════════════════════════
# Cache GPU availability once at startup (CUDA / AMD ROCm)
_CUDA_AVAILABLE: bool = False
_GPU_DEVICE:     str  = "cpu"   # "cuda:0" (works for both NVIDIA CUDA and AMD ROCm)

def _check_gpu():
    global _CUDA_AVAILABLE, _GPU_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            _CUDA_AVAILABLE = True
            _GPU_DEVICE = "cuda:0"
            backend = "ROCm (AMD)" if (hasattr(torch.version, "hip") and torch.version.hip) else "CUDA (NVIDIA)"
            print(f"[RaZui] GPU detected via {backend}: {torch.cuda.get_device_name(0)}")
        else:
            _CUDA_AVAILABLE = False
            _GPU_DEVICE = "cpu"
            # Detect whether this is a CPU-only torch build (no CUDA or ROCm compiled in)
            has_cuda_build = bool(getattr(torch.version, "cuda", None))
            has_rocm_build = bool(getattr(torch.version, "hip",  None))
            if not has_cuda_build and not has_rocm_build:
                print("[RaZui] GPU unavailable – CPU-only PyTorch detected.")
                print("  To enable GPU, FORCE-reinstall the correct build:")
                print("  AMD:    pip install torch torchvision --force-reinstall "
                      "--index-url https://download.pytorch.org/whl/rocm6.1")
                print("  NVIDIA: pip install torch torchvision --force-reinstall "
                      "--index-url https://download.pytorch.org/whl/cu121")
            else:
                print(f"[RaZui] GPU build present (cuda={getattr(torch.version,'cuda',None)} "
                      f"hip={getattr(torch.version,'hip',None)}) but no device found – driver issue?")
    except Exception as e:
        _CUDA_AVAILABLE = False
        _GPU_DEVICE = "cpu"
        print(f"[RaZui] GPU check failed ({e}) – running on CPU")

_check_gpu()

def _cuda_ok(): return _CUDA_AVAILABLE

def _detector_loop():
    import torch
    from ultralytics import YOLO
    model = None; last_path = None; last_device = None; use_half = False; sct = mss.mss()
    frames = 0; t0 = time.time()
    INF_SZ = 320  # square input – fastest YOLO path, no internal padding/repad

    while True:
        if not state.enabled: time.sleep(0.05); continue

        want_device = _GPU_DEVICE if (state.use_gpu and _CUDA_AVAILABLE) else "cpu"
        mp = state.model_path.strip()

        # Reload model only when path or device changes
        if mp != last_path or want_device != last_device:
            try:
                model = YOLO(mp)
                # Move to device first, then try FP16 (some models/drivers don't support it)
                model.to(want_device)
                if want_device != "cpu":
                    try:
                        model.half()
                        use_half = True
                        print(f"[YOLO] FP16 enabled on GPU")
                    except Exception as he:
                        use_half = False
                        print(f"[YOLO] FP16 unavailable ({he}), using FP32 on GPU")
                else:
                    use_half = False
                last_path = mp; last_device = want_device
                state.model_loaded = True; state.model_error = ""
                print(f"[YOLO] loaded → {mp}  device={want_device}")
            except Exception as e:
                state.model_error = str(e); state.model_loaded = False; model = None
                print(f"[YOLO] load error: {e}")
                time.sleep(1.0); continue

        if model is None: time.sleep(0.1); continue

        rx, ry, rw, rh = cap_rect()
        monitor = {"left": rx, "top": ry, "width": rw, "height": rh}
        try: raw = sct.grab(monitor)
        except: time.sleep(0.05); continue

        # Zero-copy BGRA grab → BGR view
        frame_bgra = np.frombuffer(raw.bgra, dtype=np.uint8).reshape((rh, rw, 4))
        frame_bgr  = frame_bgra[:, :, :3]

        # Resize to exact INF_SZ square (avoids YOLO's internal letterbox repad)
        frame_inf = cv2.resize(frame_bgr, (INF_SZ, INF_SZ), interpolation=cv2.INTER_LINEAR)
        scale_x = rw / INF_SZ; scale_y = rh / INF_SZ

        # Always pass numpy frame – ultralytics handles dtype matching internally.
        # half= must match the model's actual weight dtype (set by model.half() above).
        try:
            with torch.inference_mode():
                results = model.predict(frame_inf, classes=[0], conf=state.confidence,
                                        verbose=False, imgsz=INF_SZ, half=use_half,
                                        augment=False, agnostic_nms=False)
        except Exception as e:
            state.model_error = str(e)
            print(f"[YOLO] predict error: {e}")
            time.sleep(0.1); continue

        dets = []; dbg = frame_bgr.copy() if state.show_debug else None
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist(); conf = float(box.conf[0])
                sx1 = int(x1 * scale_x) + rx; sy1 = int(y1 * scale_y) + ry
                sx2 = int(x2 * scale_x) + rx; sy2 = int(y2 * scale_y) + ry
                dets.append((sx1, sy1, sx2, sy2, conf))
                if dbg is not None:
                    cv2.rectangle(dbg, (int(x1*scale_x), int(y1*scale_y)),
                                       (int(x2*scale_x), int(y2*scale_y)), (80,200,120), 2)
                    cv2.putText(dbg, f"{conf:.2f}", (int(x1*scale_x)+3, int(y1*scale_y)+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,200,120), 1)
        with state.lock:
            state.detections = dets
            state.debug_frame = dbg
            # ── temporal smoothing ──────────────────────────────────────────
            # Each tracked entry: [x1,y1,x2,y2,conf, ttl]
            # New detections refresh TTL; unmatched ones decay and die.
            TTL_MAX  = 4    # frames a detection stays alive without a refresh
            IOU_MIN  = 0.25 # IoU threshold to match a new det to an existing track
            prev     = state.smooth_dets
            updated  = []

            def _iou(a, b):
                ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
                iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                inter = iw * ih
                if inter == 0: return 0.0
                ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
                return inter / max(1, ua + ub - inter)

            matched_prev = set()
            matched_new  = set()
            for ni, nd in enumerate(dets):
                best_iou = IOU_MIN; best_pi = -1
                for pi, pd in enumerate(prev):
                    if pi in matched_prev: continue
                    iou = _iou(nd, pd)
                    if iou > best_iou:
                        best_iou = iou; best_pi = pi
                if best_pi >= 0:
                    # blend box toward new detection
                    alpha = 0.55
                    ox1,oy1,ox2,oy2,_oc,_ttl = prev[best_pi]
                    nx1,ny1,nx2,ny2,nc = nd
                    bx1 = int(ox1 + alpha*(nx1-ox1)); by1 = int(oy1 + alpha*(ny1-oy1))
                    bx2 = int(ox2 + alpha*(nx2-ox2)); by2 = int(oy2 + alpha*(ny2-oy2))
                    updated.append([bx1,by1,bx2,by2, nc, TTL_MAX])
                    matched_prev.add(best_pi); matched_new.add(ni)
                else:
                    updated.append([nd[0],nd[1],nd[2],nd[3], nd[4], TTL_MAX])
                    matched_new.add(ni)
            # decay unmatched previous tracks
            for pi, pd in enumerate(prev):
                if pi not in matched_prev:
                    ttl = pd[5] - 1
                    if ttl > 0:
                        updated.append([pd[0],pd[1],pd[2],pd[3], pd[4], ttl])
            state.smooth_dets = updated
        frames += 1; now = time.time()
        if now - t0 >= 1.0:
            state.fps_det = frames / (now - t0); frames = 0; t0 = now

# ════════════════════════════════════════════════════════
#  FONT / STYLE
# ════════════════════════════════════════════════════════
def _font():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(here,"razui.ttf"),
              r"C:\Windows\Fonts\consola.ttf",
              r"C:\Windows\Fonts\lucon.ttf",
              r"C:\Windows\Fonts\cour.ttf",
              r"C:\Windows\Fonts\segoeui.ttf"]:
        if os.path.isfile(p): return p
    return None

def _style():
    st = imgui.get_style(); V = imgui.Vec4
    st.colors[imgui.COLOR_TEXT]              = V(*C_TEXT,1)
    st.colors[imgui.COLOR_TEXT_DISABLED]     = V(*C_DIM,1)
    st.colors[imgui.COLOR_WINDOW_BACKGROUND] = V(0.07,0.07,0.10,0.96)
    st.colors[imgui.COLOR_FRAME_BACKGROUND]  = V(0.12,0.12,0.17,1.0)
    st.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = V(0.18,0.18,0.26,1.0)
    st.colors[imgui.COLOR_SCROLLBAR_BACKGROUND] = V(.03,.03,.05,1)
    st.window_rounding=0; st.child_rounding=0; st.frame_rounding=0
    st.grab_rounding=0; st.popup_rounding=0; st.scrollbar_rounding=0
    st.window_border_size=0; st.frame_border_size=0
    st.item_spacing=imgui.Vec2(4,2); st.frame_padding=imgui.Vec2(4,2)
    st.scrollbar_size=4

# ════════════════════════════════════════════════════════
#  WIDGET ID
# ════════════════════════════════════════════════════════
_wid = 0
def _id(): global _wid; _wid+=1; return f"##w{_wid}"
def _rst(): global _wid; _wid=0

# ════════════════════════════════════════════════════════
#  MAIN GUI
# ════════════════════════════════════════════════════════
show_gui  = False
gui_alpha = 0.0
wx = (SW - _BASE_GW) // 2
wy = (SH - _BASE_GH) // 2
_drag = False; _dox = _doy = 0
active_tab = 0
_fps = 0.0

def _gui():
    global show_gui, gui_alpha, wx, wy, _drag, _dox, _doy, active_tab
    global _accent_open, _accent_just_opened, _cp_wx, _cp_wy, _cp_drag, _model_dd_open

    if gui_alpha < 0.01: return

    gw = GW(); th = TITLE(); tabh = TABS_H(); rowh = ROW_H(); sc = _dpi(); gh = GH()

    mouse = imgui.get_mouse_pos()
    in_tb = wx < mouse[0] < wx+gw and wy < mouse[1] < wy+th
    if in_tb and imgui.is_mouse_clicked(0):
        _drag=True; _dox=mouse[0]-wx; _doy=mouse[1]-wy
    if not imgui.is_mouse_down(0): _drag=False

    gh = GH()

    if _drag:
        wx = int(clamp(mouse[0]-_dox, 0, SW-GW()))
        wy = int(clamp(mouse[1]-_doy, 0, SH-gh))

    imgui.set_next_window_position(wx, wy)
    imgui.set_next_window_size(GW(), gh)
    imgui.push_style_var(imgui.STYLE_ALPHA, gui_alpha)
    imgui.begin("##razui",
        flags=(imgui.WINDOW_NO_TITLE_BAR|imgui.WINDOW_NO_RESIZE|
               imgui.WINDOW_NO_SCROLLBAR|imgui.WINDOW_NO_COLLAPSE|
               imgui.WINDOW_NO_BACKGROUND|imgui.WINDOW_NO_MOVE))

    _rst()
    dl = imgui.get_window_draw_list()
    x0, y0 = wx, wy

    # shadow
    for d in range(10, 0, -1):
        dl.add_rect_filled(x0-d, y0-d, x0+gw+d, y0+gh+d, u(0,0,0,.015*d))

    # panels
    dl.add_rect_filled(x0, y0,       x0+gw, y0+gh,        ct(C_BG,1.0))
    dl.add_rect_filled(x0, y0,       x0+gw, y0+th,        ct(C_TITLE,1.0))
    dl.add_rect_filled(x0, y0+th,    x0+gw, y0+th+tabh,   ct(C_TITLE,1.0))

    # chromatic top line
    if state.chromatic:
        STEPS = 128; seg_w = gw / STEPS
        for i in range(STEPS):
            sr, sg, sb = colorsys.hsv_to_rgb(i/STEPS, 0.90, 1.0)
            dl.add_rect_filled(x0+int(i*seg_w), y0,
                               x0+int((i+1)*seg_w)+1, y0+max(2,int(2*sc)),
                               u(sr,sg,sb,1.0))

    # borders
    dl.add_rect(x0, y0,        x0+gw, y0+gh,       ct(C_BORDER,1.0), thickness=1.0)
    dl.add_line(x0, y0+th,     x0+gw, y0+th,       ct(C_BORDER,1.0))
    dl.add_line(x0, y0+th+tabh,x0+gw, y0+th+tabh,  ct(C_BORDER,1.0))

    # title
    r, g, b = _acc_rgb()
    rw_ = imgui.calc_text_size("Ra").x; full = imgui.calc_text_size("RaZui")
    tx = x0 + (gw - full.x)/2.0; ty_ = y0 + (th - full.y)/2.0
    dl.add_text(tx,       ty_, u(r,g,b,1.0),   "Ra")
    dl.add_text(tx + rw_, ty_, ct(C_TEXT,1.0), "Zui")
    fps_s = f"{_fps:.0f} fps"; fw_ = imgui.calc_text_size(fps_s).x
    dl.add_text(x0+gw-fw_-int(8*sc), ty_, ct(C_DIM,1.0), fps_s)

    # tabs
    tw_each = gw / len(TABS)
    for i, name in enumerate(TABS):
        tx0 = x0+i*tw_each; ty0 = y0+th; sel = (i==active_tab)
        if sel:
            dl.add_rect_filled(tx0+1, ty0+1, tx0+tw_each-1, ty0+tabh, ct(C_HOV,1.0))
        tw_ = imgui.calc_text_size(name)
        dl.add_text(tx0+(tw_each-tw_.x)/2.0, ty0+(tabh-tw_.y)/2.0,
                    ct(C_TEXT,1.0) if sel else ct(C_DIM,1.0), name)
        if i < len(TABS)-1:
            dl.add_line(tx0+tw_each, ty0, tx0+tw_each, ty0+tabh, ct(C_BORDER,1.0))
        imgui.set_cursor_pos((i*tw_each, th))
        imgui.invisible_button(_id(), tw_each, tabh)
        if imgui.is_item_clicked(): active_tab = i

    # section header
    tab_name = TABS[active_tab]
    SHY = y0+th+tabh; SHH = int(20*sc)
    tw_ = imgui.calc_text_size(tab_name)
    lx = x0+(gw-tw_.x)/2.0; ly = SHY+(SHH-tw_.y)/2.0
    dl.add_text(lx, ly, ct(C_DIM,1.0), tab_name)
    mid = SHY+SHH/2.0; pad2=10; gap=7
    _hline_fade(dl, x0+pad2, mid, lx-(x0+pad2)-gap, C_BORDER, 0.60)
    _hline_fade(dl, lx+tw_.x+gap, mid, (x0+gw-pad2)-(lx+tw_.x+gap), C_BORDER, 0.60)

    ITEMS_Y0 = SHY + SHH
    SW_W = int(26*sc); SW_H = int(11*sc)
    SL_W = int(100*sc); PAD_R = int(8*sc)

    # ── row helpers ─────────────────────────────────────────
    def _row_bg(j):
        ry = ITEMS_Y0 + j*rowh
        imgui.set_cursor_pos((0, th+tabh+SHH+j*rowh))
        imgui.invisible_button(_id(), gw, rowh)
        hov = imgui.is_item_hovered(); clicked = imgui.is_item_clicked()
        if hov: dl.add_rect_filled(x0, ry, x0+gw, ry+rowh, ct(C_HOV,1.0))
        _hline_fade(dl, x0+6, ry+rowh-1, gw-12, C_BORDER, 0.35)
        return ry, clicked

    def _label(ry, text, on=True):
        lh = imgui.calc_text_size(text).y
        dl.add_text(x0+int(8*sc), ry+(rowh-lh)/2.0,
                    ct(C_TEXT,1.0) if on else ct(C_DIM,1.0), text)

    def _toggle_row(j, label, getter, setter):
        ry, clicked = _row_bg(j)
        val = getter(); _label(ry, label, val)
        sx = x0+gw-SW_W-PAD_R; sy = ry+(rowh-SW_H)/2.0
        draw_switch(dl, int(sx), int(sy), label, val)
        if clicked: setter(not val)

    def _slider_row(j, label, getter, setter, lo, hi, fmt="%.0f"):
        ry, _ = _row_bg(j); _label(ry, label)
        sx = x0+gw-SL_W-PAD_R; sy = ry+(rowh-int(14*sc))//2
        setter(draw_slider(dl, int(sx), int(sy), SL_W, label, float(getter()), lo, hi, fmt))

    # ══════════════════════════════════════════════════
    #  Placeholder tab  –  Aim Assist + Clickbot
    # ══════════════════════════════════════════════════
    if tab_name == "Aim Assist":
        j = 0
        sc = _dpi()
        r_, g_, b_ = _acc_rgb()
        BTN_W = int(56*sc); BTN_H = int(13*sc)
        mouse_ = imgui.get_mouse_pos()

        # ── helper: inline "Set" key-bind button ─────────
        def _bind_btn(ry_, is_binding, is_wait, vk, on_click_set):
            """Draws key label + Set button. Returns (new_is_binding, new_is_wait, new_vk)."""
            # key label on left
            lbl = "[ press key ]" if is_binding else vk_name(vk)
            lsz = imgui.calc_text_size(lbl)
            lc  = u(r_,g_,b_,1.0) if is_binding else ct(C_DIM, 0.85)
            dl.add_text(x0+int(8*sc), ry_+(rowh-lsz.y)/2.0, lc, lbl)
            # button on right
            bx_ = x0+gw-BTN_W-PAD_R; by__ = ry_+(rowh-BTN_H)//2
            bg_ = u(r_*0.40, g_*0.40, b_*0.55, 1.0) if is_binding else u(0.09,0.09,0.13,1.0)
            dl.add_rect_filled(bx_, by__, bx_+BTN_W, by__+BTN_H, bg_)
            dl.add_rect(bx_, by__, bx_+BTN_W, by__+BTN_H, u(r_,g_,b_, 0.80 if is_binding else 0.45), thickness=1.0)
            bl_ = "…" if is_binding else "Set"
            blsz_ = imgui.calc_text_size(bl_)
            dl.add_text(bx_+(BTN_W-blsz_.x)/2.0, by__+(BTN_H-blsz_.y)/2.0, ct(C_TEXT,1.0), bl_)
            in_btn_ = (bx_<=mouse_[0]<=bx_+BTN_W) and (by__<=mouse_[1]<=by__+BTN_H)
            # State machine: idle → wait_release → binding → captures key → idle
            if not is_binding and not is_wait:
                if in_btn_ and imgui.is_mouse_clicked(0):
                    is_wait = True          # clicked; wait for LMB to come back up
            elif is_wait:
                if not imgui.is_mouse_down(0):  # LMB released
                    is_binding = True
                    is_wait    = False
            elif is_binding:
                # scan for any key/button press
                for _vk in range(0x01, 0xDF):
                    if _vk in (INS_KEY, F2_KEY): continue  # skip menu/debug hotkeys
                    if bool(win32api.GetAsyncKeyState(_vk) & 0x8000):
                        vk         = _vk
                        is_binding = False
                        break
                # LMB (0x01) is allowed; the wait-for-release state above ensures
                # the same click that pressed "Set" doesn't instantly capture itself.
            return is_binding, is_wait, vk

        # ── bone selector helper ──────────────────────────
        BONE_LABELS = ["Head", "Chest", "Body"]
        def _bone_selector(ry_):
            SEG_W = int((gw - PAD_R*2 - int(8*sc) - int(60*sc)) / 3)
            ox = x0 + gw - PAD_R - SEG_W*3; oy_ = ry_ + (rowh - BTN_H)//2
            for bi, bl in enumerate(BONE_LABELS):
                bx_ = ox + bi*SEG_W; sel = (state.aim_bone == bi)
                bg_ = u(r_*0.30, g_*0.30, b_*0.45, 1.0) if sel else u(0.08,0.08,0.12,1.0)
                dl.add_rect_filled(bx_, oy_, bx_+SEG_W, oy_+BTN_H, bg_)
                border_a = 0.85 if sel else 0.30
                dl.add_rect(bx_, oy_, bx_+SEG_W, oy_+BTN_H, u(r_,g_,b_,border_a), thickness=1.0)
                bsz = imgui.calc_text_size(bl)
                tc_ = u(r_,g_,b_,1.0) if sel else ct(C_DIM,0.80)
                dl.add_text(bx_+(SEG_W-bsz.x)/2.0, oy_+(BTN_H-bsz.y)/2.0, tc_, bl)
                in_seg = (bx_<=mouse_[0]<=bx_+SEG_W) and (oy_<=mouse_[1]<=oy_+BTN_H)
                if in_seg and imgui.is_mouse_clicked(0):
                    state.aim_bone = bi

        # ════════ AIM ASSIST ═════════════════════════════
        _toggle_row(j, "Aim Assist",
                    lambda: state.aim_enabled,
                    lambda v: setattr(state, "aim_enabled", v)); j+=1

        if state.aim_enabled:
            # Speed slider
            _slider_row(j, "Speed", lambda: state.aim_speed,
                        lambda v: setattr(state, "aim_speed", v), 1.0, 50.0, "%.0f"); j+=1

            # Smoothing slider (EMA alpha ×100 for display)
            ry, _ = _row_bg(j); _label(ry, "Smooth")
            sx = x0+gw-SL_W-PAD_R; sy = ry+(rowh-int(14*sc))//2
            raw_s = draw_slider(dl, int(sx), int(sy), SL_W, "aim_smooth",
                                state.aim_smooth * 100.0, 5.0, 100.0, "%.0f")
            state.aim_smooth = raw_s / 100.0; j+=1

            # Bone target
            ry, _ = _row_bg(j); _label(ry, "Bone")
            _bone_selector(ry); j+=1

            # Hotkey Only toggle
            _toggle_row(j, "Hotkey Only",
                        lambda: state.aim_hotkey_only,
                        lambda v: setattr(state, "aim_hotkey_only", v)); j+=1

            if state.aim_hotkey_only:
                ry, _ = _row_bg(j)
                state._aim_binding, state._aim_bind_wait, state.aim_hotkey = \
                    _bind_btn(ry, state._aim_binding, state._aim_bind_wait, state.aim_hotkey, None)
                j+=1

        # ════════ CLICKBOT ════════════════════════════════
        _toggle_row(j, "Clickbot",
                    lambda: state.click_enabled,
                    lambda v: setattr(state, "click_enabled", v)); j+=1

        if state.click_enabled:
            # Hold delay slider
            _slider_row(j, "Hold ms", lambda: state.click_delay,
                        lambda v: setattr(state, "click_delay", v), 10.0, 500.0, "%.0f"); j+=1

            # Hotkey Only toggle
            _toggle_row(j, "Hotkey Only##cb",
                        lambda: state.click_hotkey_only,
                        lambda v: setattr(state, "click_hotkey_only", v)); j+=1

            if state.click_hotkey_only:
                ry, _ = _row_bg(j)
                state._click_binding, state._click_bind_wait, state.click_hotkey = \
                    _bind_btn(ry, state._click_binding, state._click_bind_wait, state.click_hotkey, None)
                j+=1

    # ══════════════════════════════════════════════════
    #  Detection tab
    # ══════════════════════════════════════════════════
    elif tab_name == "Detection":
        j = 0
        _toggle_row(j, "Detection",
                    lambda: state.enabled,
                    lambda v: setattr(state,"enabled",v)); j+=1

        # confidence
        ry, _ = _row_bg(j); _label(ry, "Confidence")
        sx=x0+gw-SL_W-PAD_R; sy=ry+(rowh-int(14*sc))//2
        state.confidence = round(draw_slider(dl,int(sx),int(sy),SL_W,
                                              "confidence",state.confidence,
                                              0.05,0.95,"%.2f"), 2); j+=1

        # FOV size – single slider, centred square
        max_fov = 300
        ry, _ = _row_bg(j); _label(ry, "FOV Size")
        sx=x0+gw-SL_W-PAD_R; sy=ry+(rowh-int(14*sc))//2
        state.fov_r = int(draw_slider(dl,int(sx),int(sy),SL_W,
                                      "fov_r",float(state.fov_r),
                                      32,max_fov,"%.0f")); j+=1

        # model status
        ry, _ = _row_bg(j)
        if state.model_loaded:
            s = f"Model: {os.path.basename(state.model_path)}"
            sc_ = u(0.30,0.85,0.45,1.0)
        elif state.model_error:
            s = "Model error – check console"; sc_ = u(0.90,0.35,0.35,1.0)
        else:
            s = "Loading model…"; sc_ = ct(C_DIM,1.0)
        lh = imgui.calc_text_size(s).y
        dl.add_text(x0+int(8*sc), ry+(rowh-lh)/2.0, sc_, s); j+=1

        # stats row
        ry, _ = _row_bg(j)
        if state.use_gpu and _CUDA_AVAILABLE:
            try:
                import torch as _t
                dev = "ROCm GPU" if (hasattr(_t.version, "hip") and _t.version.hip) else "CUDA GPU"
            except Exception:
                dev = "GPU"
        elif state.use_gpu and not _CUDA_AVAILABLE:
            dev = "CPU (no GPU)"
        else:
            dev = "CPU"
        info = f"Det {state.fps_det:.0f} fps  ·  {dev}  ·  {len(state.detections)} found"
        lh = imgui.calc_text_size(info).y
        dl.add_text(x0+int(8*sc), ry+(rowh-lh)/2.0, ct(C_DIM,1.0), info)

    # ══════════════════════════════════════════════════
    #  Visuals tab
    # ══════════════════════════════════════════════════
    elif tab_name == "Visuals":
        j = 0
        _toggle_row(j,"Show Boxes",
                    lambda: state.show_boxes,
                    lambda v: setattr(state,"show_boxes",v));   j+=1
        _toggle_row(j,"Show Confidence",
                    lambda: state.show_conf,
                    lambda v: setattr(state,"show_conf",v));    j+=1
        _toggle_row(j,"Show FOV Box",
                    lambda: state.show_fov,
                    lambda v: setattr(state,"show_fov",v));     j+=1
        _toggle_row(j,"Debug Window  [F2]",
                    lambda: state.show_debug,
                    lambda v: setattr(state,"show_debug",v));   j+=1

    # ══════════════════════════════════════════════════
    #  Settings tab
    # ══════════════════════════════════════════════════
    elif tab_name == "Settings":
        j = 0
        _toggle_row(j,"Watermark",
                    lambda: state.watermark,
                    lambda v: setattr(state,"watermark",v));    j+=1
        _toggle_row(j,"Chromatic Top",
                    lambda: state.chromatic,
                    lambda v: setattr(state,"chromatic",v));    j+=1

        # DPI Scale
        ry, _ = _row_bg(j); _label(ry, "DPI Scale")
        DD_W = int(80*sc); ddx = x0+gw-DD_W-PAD_R; ddy = ry+(rowh-int(18*sc))/2.0
        draw_dpi_dropdown(dl, int(ddx), int(ddy), DD_W); j+=1

        # Accent Colour
        ry, _ = _row_bg(j); _label(ry, "Accent Colour")
        pr_, pg_, pb_ = colorsys.hsv_to_rgb(_acc_h, _acc_s, _acc_v)
        SW_BTN=int(32*sc); SH_BTN=int(13*sc)
        bx_=x0+gw-SW_BTN-PAD_R; by_=ry+(rowh-SH_BTN)//2
        dl.add_rect_filled(bx_,by_,bx_+SW_BTN,by_+SH_BTN,u(pr_,pg_,pb_,1.0))
        border_col = u(1,1,1,0.55) if _accent_open else ct(C_BORDER,1.0)
        dl.add_rect(bx_,by_,bx_+SW_BTN,by_+SH_BTN,border_col,thickness=1.0)
        mouse_ = imgui.get_mouse_pos()
        in_btn = (bx_<=mouse_[0]<=bx_+SW_BTN) and (by_<=mouse_[1]<=by_+SH_BTN)
        if in_btn and imgui.is_mouse_clicked(0):
            _accent_open = not _accent_open
            if _accent_open:
                _accent_just_opened = True
                _cp_wx = float(bx_+SW_BTN//2); _cp_wy = float(by_+SH_BTN+int(4*sc))
            else:
                _cp_wx = -1.0; _cp_wy = -1.0
        j+=1

        # GPU toggle with (?) tooltip for CUDA install hint
        ry, clicked_gpu = _row_bg(j)
        _label(ry, "Use GPU")
        # draw (?) badge next to label
        lw_ = imgui.calc_text_size("Use GPU").x
        qmark = "(?)"
        qsz   = imgui.calc_text_size(qmark)
        qx    = x0 + int(8*sc) + lw_ + int(5*sc)
        qy    = ry + (rowh - qsz.y) / 2.0
        if _CUDA_AVAILABLE:
            q_col = u(0.30, 0.85, 0.45, 0.80)
            try:
                import torch as _t
                _backend = "ROCm (AMD)" if (hasattr(_t.version, "hip") and _t.version.hip) else "CUDA (NVIDIA)"
            except Exception:
                _backend = "GPU"
            tip_text = f"{_backend} ready  –  GPU inference active when toggled on"
        else:
            q_col = ct(C_DIM, 0.80)
            tip_text = ("No GPU PyTorch detected.\n"
                        "NVIDIA:  pip install torch torchvision\n"
                        "         --index-url https://download.pytorch.org/whl/cu121\n"
                        "AMD:     pip install torch torchvision\n"
                        "         --index-url https://download.pytorch.org/whl/rocm6.1")
        dl.add_text(qx, qy, q_col, qmark)
        # hover check for tooltip
        mouse_ = imgui.get_mouse_pos()
        in_q = (qx <= mouse_[0] <= qx + qsz.x) and (qy <= mouse_[1] <= qy + qsz.y)
        if in_q:
            fdl2 = imgui.get_foreground_draw_list()
            lines = tip_text.split("\n")
            line_h = imgui.calc_text_size("A").y
            tp_w = max(imgui.calc_text_size(l).x for l in lines) + int(16*sc)
            tp_h = line_h * len(lines) + int(10*sc)
            tp_x = int(clamp(mouse_[0] + 12, 4, SW - tp_w - 4))
            tp_y = int(clamp(mouse_[1] - tp_h // 2, 4, SH - tp_h - 4))
            fdl2.add_rect_filled(tp_x, tp_y, tp_x+tp_w, tp_y+tp_h, u(0.06,0.06,0.10,0.97))
            fdl2.add_rect(tp_x, tp_y, tp_x+tp_w, tp_y+tp_h, ct(C_BORDER,0.85), thickness=1.0)
            for li, line in enumerate(lines):
                fdl2.add_text(tp_x+int(8*sc), tp_y+int(5*sc)+li*line_h,
                              ct(C_TEXT,1.0) if li==0 else ct(C_DIM,0.85), line)
        sx = x0+gw-SW_W-PAD_R; sy = ry+(rowh-SW_H)/2.0
        draw_switch(dl, int(sx), int(sy), "use_gpu", state.use_gpu)
        if clicked_gpu:
            state.use_gpu = not state.use_gpu
        j+=1

        # Model selector dropdown
        ry, _ = _row_bg(j); _label(ry, "Model")
        DD_W = int(110*sc); ddx = x0+gw-DD_W-PAD_R; ddy = ry+(rowh-int(18*sc))/2.0
        draw_model_dropdown(dl, int(ddx), int(ddy), DD_W); j+=1

        # model hint
        ry, _ = _row_bg(j)
        tip = "yolov8n=fast  s/m=accurate  l/x=best"
        lh = imgui.calc_text_size(tip).y
        dl.add_text(x0+int(8*sc), ry+(rowh-lh)/2.0, ct(C_DIM,0.70), tip)

    # ── accent colour picker popout ──────────────────────
    if tab_name == "Settings" and _accent_open:
        PAD=int(8*sc); SV_SZ=int(80*sc); HUE_W=int(14*sc); GAP=int(6*sc); PRV_W=int(20*sc)
        pw = PAD+SV_SZ+GAP+HUE_W+GAP+PRV_W+PAD; ph = PAD+SV_SZ+PAD
        px = int(clamp(_cp_wx-pw//2, 4, SW-pw-4))
        py = int(clamp(_cp_wy, 4, SH-ph-4))
        fdl = imgui.get_foreground_draw_list()
        for d in range(6,0,-1):
            fdl.add_rect_filled(px-d,py-d,px+pw+d,py+ph+d,u(0,0,0,0.014*d))
        fdl.add_rect_filled(px,py,px+pw,py+ph,u(0.06,0.06,0.09,0.97))
        fdl.add_rect(px,py,px+pw,py+ph,ct(C_BORDER,0.85),thickness=1.0)
        _draw_color_picker_at(fdl, px+PAD, py+PAD, sc)
        mouse_ = imgui.get_mouse_pos()
        in_panel = (px<=mouse_[0]<=px+pw) and (py<=mouse_[1]<=py+ph)
        if _accent_just_opened:
            _accent_just_opened = False
        elif imgui.is_mouse_clicked(0) and not in_panel:
            _accent_open = False

    imgui.end()
    imgui.pop_style_var()

# ════════════════════════════════════════════════════════
#  VK NAME HELPER
# ════════════════════════════════════════════════════════
_VK_NAMES = {
    0x01:"LMB", 0x02:"RMB", 0x04:"MMB", 0x05:"X1", 0x06:"X2",
    0x08:"Back", 0x09:"Tab", 0x0D:"Enter", 0x10:"Shift", 0x11:"Ctrl",
    0x12:"Alt", 0x14:"CapsLk", 0x1B:"Esc", 0x20:"Space",
    0x70:"F1", 0x71:"F2", 0x72:"F3", 0x73:"F4", 0x74:"F5",
    0x75:"F6", 0x76:"F7", 0x77:"F8", 0x78:"F9", 0x79:"F10",
    0x7A:"F11", 0x7B:"F12",
}
for _c in range(0x30, 0x3A): _VK_NAMES[_c] = chr(_c)
for _c in range(0x41, 0x5B): _VK_NAMES[_c] = chr(_c)

def vk_name(vk):
    return _VK_NAMES.get(vk, f"0x{vk:02X}")

# ════════════════════════════════════════════════════════
#  HOTKEY THREAD
# ════════════════════════════════════════════════════════
_li_ins = False; _li_f2 = False

def _hk():
    global show_gui, _li_ins, _li_f2
    while True:
        p_ins = bool(win32api.GetAsyncKeyState(INS_KEY) & 0x8000)
        p_f2  = bool(win32api.GetAsyncKeyState(F2_KEY)  & 0x8000)
        if p_ins and not _li_ins: show_gui = not show_gui
        if p_f2  and not _li_f2:  state.show_debug = not state.show_debug
        _li_ins = p_ins; _li_f2 = p_f2
        time.sleep(0.010)

# ════════════════════════════════════════════════════════
#  AIM ASSIST THREAD
# ════════════════════════════════════════════════════════
def _aim_loop():
    """Smoothly drag the mouse toward the target bone of the closest detected person."""
    import ctypes
    while True:
        time.sleep(0.010)
        if not state.aim_enabled:
            # Reset smoothed position when aim is off so next enable starts fresh
            state._aim_smooth_x = -1.0
            state._aim_smooth_y = -1.0
            continue
        if state.aim_hotkey_only:
            held = bool(win32api.GetAsyncKeyState(state.aim_hotkey) & 0x8000)
            if not held:
                state._aim_smooth_x = -1.0
                state._aim_smooth_y = -1.0
                continue
        with state.lock:
            dets = list(state.smooth_dets) if state.smooth_dets else list(state.detections)
        if not dets:
            state._aim_smooth_x = -1.0
            state._aim_smooth_y = -1.0
            continue
        cx, cy = SW // 2, SH // 2
        best = None; best_d = float("inf")
        for entry in dets:
            x1, y1, x2, y2 = entry[0], entry[1], entry[2], entry[3]
            bx_c = (x1 + x2) // 2
            bh   = max(1, y2 - y1)
            if   state.aim_bone == 0: ty_ = y1 + int(bh * 0.10)  # Head
            elif state.aim_bone == 1: ty_ = y1 + int(bh * 0.25)  # Chest
            else:                     ty_ = y1 + int(bh * 0.50)  # Body
            d = ((bx_c - cx)**2 + (ty_ - cy)**2) ** 0.5
            if d < best_d:
                best_d = d; best = (bx_c, ty_)
        if best is None:
            continue
        raw_tx, raw_ty = best
        # EMA smoothing: blend raw target toward previous smoothed position
        alpha = clamp(state.aim_smooth, 0.05, 1.0)
        if state._aim_smooth_x < 0:
            state._aim_smooth_x = float(raw_tx)
            state._aim_smooth_y = float(raw_ty)
        else:
            state._aim_smooth_x += alpha * (raw_tx - state._aim_smooth_x)
            state._aim_smooth_y += alpha * (raw_ty - state._aim_smooth_y)
        tx = int(state._aim_smooth_x)
        ty = int(state._aim_smooth_y)
        dx = tx - cx; dy = ty - cy
        dist = (dx*dx + dy*dy) ** 0.5
        # Dead zone – don't nudge if already close enough
        if dist < 2.0:
            continue
        # Scale step proportionally to distance so it decelerates as it closes in.
        # Cap at aim_speed; never move more than the remaining distance.
        step = min(state.aim_speed, dist) * (dist / (dist + 12.0))
        step = max(0.5, step)
        mx = dx / dist * step
        my = dy / dist * step
        # Round toward zero so we never overshoot by a whole pixel
        imx = int(mx); imy = int(my)
        if imx == 0 and abs(mx) >= 0.5: imx = 1 if mx > 0 else -1
        if imy == 0 and abs(my) >= 0.5: imy = 1 if my > 0 else -1
        if imx != 0 or imy != 0:
            ctypes.windll.user32.mouse_event(0x0001, imx, imy, 0, 0)

# ════════════════════════════════════════════════════════
#  CLICKBOT THREAD
# ════════════════════════════════════════════════════════
def _click_loop():
    """Hold LMB when crosshair is over a detected person."""
    import ctypes
    LDOWN = 0x0002; LUP = 0x0004
    while True:
        time.sleep(0.010)
        if not state.click_enabled:
            if state._click_held:
                ctypes.windll.user32.mouse_event(LUP, 0, 0, 0, 0)
                state._click_held = False
            continue
        if state.click_hotkey_only:
            held = bool(win32api.GetAsyncKeyState(state.click_hotkey) & 0x8000)
            if not held:
                if state._click_held:
                    ctypes.windll.user32.mouse_event(LUP, 0, 0, 0, 0)
                    state._click_held = False
                continue
        cx, cy = SW // 2, SH // 2
        with state.lock:
            dets = list(state.smooth_dets) if state.smooth_dets else list(state.detections)
        on_target = any(entry[0] <= cx <= entry[2] and entry[1] <= cy <= entry[3] for entry in dets)
        if on_target and not state._click_held:
            ctypes.windll.user32.mouse_event(LDOWN, 0, 0, 0, 0)
            state._click_held = True
            time.sleep(state.click_delay / 1000.0)
        elif not on_target and state._click_held:
            ctypes.windll.user32.mouse_event(LUP, 0, 0, 0, 0)
            state._click_held = False

# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
def run():
    global show_gui, gui_alpha, _fps

    if not glfw.init(): return print("[RaZui] glfw init failed")
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES,      4)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)
    # NOTE: Do NOT set GLFW_DECORATED=FALSE before window creation.
    # On AMD, telling GLFW to skip decorations causes it to allocate a non-DWM-composited
    # surface – per-pixel alpha breaks and the overlay goes black.
    # Instead, strip decorations AFTER creation via SetWindowLong (same as malevolent).

    EXTEND_RIGHT = 15   # slight oversizing forces AMD's DWM into the correct composite path
    win = glfw.create_window(SW + EXTEND_RIGHT, SH, "RaZui", None, None)
    if not win: glfw.terminate(); return print("[RaZui] window failed")

    hwnd = glfw.get_win32_window(win)

    # Strip caption/border post-creation so DWM keeps its compositing surface
    ws = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    ws &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME)
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, ws)

    EX = (win32con.WS_EX_TOOLWINDOW|win32con.WS_EX_TRANSPARENT|win32con.WS_EX_LAYERED)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, EX)

    # Position at -2,-2 (same as malevolent) – keeps the window just off-screen edge
    # which on AMD's compositor prevents the black-fill fallback path
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, -2, -2,
                          SW + EXTEND_RIGHT, SH, win32con.SWP_NOACTIVATE)

    # DWM glass – belt-and-suspenders for any remaining compositor edge cases
    try:
        import ctypes
        class _MARGINS(ctypes.Structure):
            _fields_ = [("left",   ctypes.c_int), ("right",  ctypes.c_int),
                        ("top",    ctypes.c_int), ("bottom", ctypes.c_int)]
        ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(_MARGINS(-1,-1,-1,-1)))
        print("[RaZui] DWM glass enabled")
    except Exception as _e:
        print(f"[RaZui] DWM glass skipped: {_e}")

    glfw.make_context_current(win)
    imgui.create_context()
    io = imgui.get_io()
    fp = _font()
    if fp: io.fonts.add_font_from_file_ttf(fp, _BASE_FONT)
    else:  io.fonts.add_font_default()
    io.fonts.get_tex_data_as_rgba32()
    _style()

    impl = GlfwRenderer(win)
    threading.Thread(target=_detector_loop, daemon=True).start()
    threading.Thread(target=_hk,            daemon=True).start()
    threading.Thread(target=_aim_loop,      daemon=True).start()
    threading.Thread(target=_click_loop,    daemon=True).start()

    print("[RaZui] YOLO overlay ready")
    print("  INSERT  →  toggle menu")
    print("  F2      →  toggle debug window")
    print("  Place yolov8n.pt next to this script, or set path in Settings tab")

    frames = 0; t0 = time.time()

    while not glfw.window_should_close(win):
        glfw.poll_events(); impl.process_inputs()
        frames += 1; now = time.time()
        if now - t0 >= 1.0:
            _fps = frames / (now - t0); frames = 0; t0 = now

        tg = 1.0 if show_gui else 0.0
        gui_alpha += (tg - gui_alpha) * 0.18

        if show_gui or state.show_debug:
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, EX & ~win32con.WS_EX_TRANSPARENT)
        else:
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, EX | win32con.WS_EX_TRANSPARENT)

        imgui.new_frame()

        # shell layer – always rendered (covers full GL framebuffer incl. EXTEND_RIGHT strip)
        imgui.set_next_window_size(SW + EXTEND_RIGHT, SH)
        imgui.set_next_window_position(0, 0)
        imgui.begin("##shell",
            flags=(imgui.WINDOW_NO_TITLE_BAR|imgui.WINDOW_NO_RESIZE|
                   imgui.WINDOW_NO_SCROLLBAR|imgui.WINDOW_NO_COLLAPSE|
                   imgui.WINDOW_NO_BACKGROUND|imgui.WINDOW_NO_MOVE))
        sdl = imgui.get_window_draw_list()
        draw_detections(sdl)
        draw_watermark(sdl, _fps)
        imgui.end()

        _gui()
        _draw_debug_window()

        imgui.end_frame()
        gl.glClearColor(0,0,0,0); gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render(); impl.render(imgui.get_draw_data()); gl.glFlush()

    impl.shutdown(); glfw.terminate()

if __name__ == "__main__":
    run()
