"""
Verai – YOLO Person Detection Overlay
======================================
REQUIREMENTS
    pip install ultralytics mss opencv-python pywin32 imgui[glfw] PyOpenGL dxcam

    GPU (AMD – DirectML, recommended):
        pip install onnxruntime-directml
        (do NOT have plain onnxruntime installed at the same time)

    GPU (NVIDIA – CUDA):
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    GPU (AMD – ROCm PyTorch):
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

CAPTURE METHODS (auto-selected at startup)
    dxcam   → hardware / DXGI Desktop Duplication – reads directly from the GPU
               compositor buffer. Lowest latency, zero CPU copy.  Requires dxcam.
    mss     → software / GDI fallback if dxcam is not installed.

MODELS  (select in Settings tab – place in %LOCALAPPDATA%\veraiassets or set full path)
    yolov8n.pt  →  fastest  (~30+ fps on CPU after resize optimisation)
    yolov8s.pt  →  better accuracy, still fast
    yolov8m.pt  →  most accurate of the common ones
    yolov8l/x.pt → best accuracy, needs GPU for good fps
    Download:  github.com/ultralytics/assets/releases

GPU SUPPORT
    NVIDIA / AMD ROCm:  install PyTorch with CUDA or ROCm then enable "Use GPU" in Settings.
    AMD (recommended):  pip install onnxruntime-directml
                        (remove plain onnxruntime first if already installed)
                        Verai auto-detects DmlExecutionProvider and uses it when "Use GPU" is on.
                        On first run it exports your .pt model to ONNX – subsequent loads are instant.
    Falls back to CPU if no GPU backend is found.

HOTKEYS
    INSERT   toggle menu
    F2       toggle debug window
"""

import os, sys, time, threading, colorsys, math
import win32api, win32con, win32gui
import glfw, imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import cv2, numpy as np, mss

# ── DXGI / Hardware capture ──────────────────────────────
# dxcam uses the Windows Desktop Duplication API (DXGI) to grab frames
# directly from the GPU compositor – zero CPU copy, lowest latency.
# Falls back to mss (GDI/software) if dxcam is not installed.
try:
    import dxcam as _dxcam_mod
    _DXCAM_AVAILABLE = True
except ImportError:
    _DXCAM_AVAILABLE = False
    print("[Verai] dxcam not found – using mss (software capture).")
    print("  For hardware capture:  pip install dxcam")

# ════════════════════════════════════════════════════════
#  SCREEN SIZE
# ════════════════════════════════════════════════════════
if not glfw.init(): sys.exit("[Verai] glfw init failed")
_vm = glfw.get_video_mode(glfw.get_primary_monitor())
SW, SH = _vm.size.width, _vm.size.height
glfw.terminate()

# ════════════════════════════════════════════════════════
#  LAYOUT BASE (scaled by DPI)
# ════════════════════════════════════════════════════════
_BASE_GW    = 340
_BASE_GH    = 340
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

    # Aim PID controller state
    # PID replaces EMA: P=proportional (speed), D=derivative (damping, kills recoil jitter)
    # aim_smooth maps to overall PID gain: lower = slower/smoother, higher = snappier
    _pid_err_x:  float = 0.0   # previous error X for derivative term
    _pid_err_y:  float = 0.0   # previous error Y for derivative term
    _pid_active: bool  = False  # True = PID has a live target this session
    aim_smooth   = 0.50        # PID gain multiplier 0.1–1.0 (shown as 10–100 in UI)

    # Capture – centred square, half-side = fov_r pixels
    fov_r        = min(SW, SH) // 8  # ~90px radius default

    # Visuals
    show_boxes   = True
    show_conf    = True
    show_fov     = True      # white outline of capture square
    show_debug   = False
    debug_frame  = None
    # box_color: HSV for detection box / label colour (defaults to accent)
    box_color_h: float = -1.0   # -1 = follow accent
    box_color_s: float = 0.95
    box_color_v: float = 1.00
    # conf_color: HSV for confidence % text (defaults to accent)
    conf_color_h: float = -1.0   # -1 = follow accent
    conf_color_s: float = 0.95
    conf_color_v: float = 1.00

    # Settings
    watermark    = True
    chromatic    = True
    stream_proof = False   # hide overlay from OBS / Discord / NVIDIA capture via WDA_EXCLUDEFROMCAPTURE
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
    click_padding     = 0       # px inset from box edges – cursor must be this deep inside
    click_on_person   = False   # only trigger when cursor is inside the person silhouette ellipse
    _click_binding    = False   # True while waiting for a key press
    _click_bind_wait  = False   # True = "Set" clicked, waiting for LMB release
    _click_held       = False   # internal – currently holding LMB down

    # Visuals – extra
    show_outline      = False   # draw human silhouette outline around detected persons
    # outline_color: HSV for silhouette outline colour (independent, defaults to accent)
    outline_color_h: float = -1.0   # -1 = follow accent
    outline_color_s: float = 0.95
    outline_color_v: float = 1.00

state = State()

# Model selector options
# Base .pt options; engine/onnx files found next to the script are appended at startup
MODEL_OPTIONS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

def _refresh_model_options():
    """Scan veraiassets folder for .engine and .onnx files and add them to the dropdown."""
    found = []
    try:
        for f in sorted(os.listdir(_ASSETS_DIR)):
            if f.endswith(".engine") or f.endswith(".onnx"):
                if f not in MODEL_OPTIONS:
                    found.append(f)
    except Exception:
        pass
    for f in found:
        MODEL_OPTIONS.append(f)
    if found:
        print(f"[Verai] Found compiled models: {found}")

_refresh_model_options()
_model_dd_open: bool = False

# ════════════════════════════════════════════════════════
#  CONFIG SYSTEM  (.cfg save / load)
# ════════════════════════════════════════════════════════
_ASSETS_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "veraiassets")
os.makedirs(_ASSETS_DIR, exist_ok=True)
_CFG_DIR = _ASSETS_DIR
_cfg_options: list = []          # list of .cfg filenames found on disk
_cfg_dd_open: bool = False       # configs dropdown open state
_cfg_export_flash: float = 0.0   # timestamp of last export (for brief flash feedback)
_cfg_selected: str = ""          # currently selected/loaded config filename

def _scan_cfg_files():
    """Refresh the list of .cfg files next to the script."""
    global _cfg_options
    try:
        _cfg_options = sorted(
            f for f in os.listdir(_CFG_DIR) if f.endswith(".cfg")
        )
    except Exception:
        _cfg_options = []

_scan_cfg_files()

def _export_cfg(name: str = ""):
    """Write current settings to <name>.cfg (auto-names if empty)."""
    global _acc_h, _acc_s, _acc_v
    if not name:
        name = f"verai_{int(time.time())}"
    if not name.endswith(".cfg"):
        name += ".cfg"
    path = os.path.join(_CFG_DIR, name)
    lines = [
        "# Verai config – auto-generated\n",
        f"[visuals]\n",
        f"show_boxes      = {int(state.show_boxes)}\n",
        f"show_conf       = {int(state.show_conf)}\n",
        f"show_fov        = {int(state.show_fov)}\n",
        f"box_color_h     = {state.box_color_h:.6f}\n",
        f"box_color_s     = {state.box_color_s:.6f}\n",
        f"box_color_v     = {state.box_color_v:.6f}\n",
        f"conf_color_h    = {state.conf_color_h:.6f}\n",
        f"conf_color_s    = {state.conf_color_s:.6f}\n",
        f"conf_color_v    = {state.conf_color_v:.6f}\n",
        f"outline_color_h = {state.outline_color_h:.6f}\n",
        f"outline_color_s = {state.outline_color_s:.6f}\n",
        f"outline_color_v = {state.outline_color_v:.6f}\n",
        f"\n[accent]\n",
        f"accent_h        = {_acc_h:.6f}\n",
        f"accent_s        = {_acc_s:.6f}\n",
        f"accent_v        = {_acc_v:.6f}\n",
        f"\n[detection]\n",
        f"confidence      = {state.confidence:.4f}\n",
        f"fov_r           = {state.fov_r}\n",
        f"model_path      = {state.model_path}\n",
        f"\n[aim]\n",
        f"aim_enabled     = {int(state.aim_enabled)}\n",
        f"aim_smooth      = {state.aim_smooth:.4f}\n",
        f"aim_speed       = {state.aim_speed:.4f}\n",
        f"aim_bone        = {state.aim_bone}\n",
        f"aim_hotkey_only = {int(state.aim_hotkey_only)}\n",
        f"aim_hotkey      = {state.aim_hotkey}\n",
        f"\n[clickbot]\n",
        f"click_enabled     = {int(state.click_enabled)}\n",
        f"click_delay       = {state.click_delay:.4f}\n",
        f"click_hotkey_only = {int(state.click_hotkey_only)}\n",
        f"click_hotkey      = {state.click_hotkey}\n",
        f"click_padding     = {state.click_padding}\n",
        f"click_on_person   = {int(state.click_on_person)}\n",
        f"show_outline      = {int(state.show_outline)}\n",
        f"\n[settings]\n",
        f"watermark       = {int(state.watermark)}\n",
        f"chromatic       = {int(state.chromatic)}\n",
        f"stream_proof    = {int(state.stream_proof)}\n",
        f"dpi_scale       = {state.dpi_scale:.1f}\n",
        f"use_gpu         = {int(state.use_gpu)}\n",
    ]
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        print(f"[Verai] Config exported → {path}")
        _scan_cfg_files()
        return True
    except Exception as e:
        print(f"[Verai] Config export failed: {e}")
        return False

def _load_cfg(filename: str):
    """Load a .cfg file and apply settings to state + accent globals."""
    global _acc_h, _acc_s, _acc_v
    path = os.path.join(_CFG_DIR, filename)
    if not os.path.isfile(path):
        print(f"[Verai] Config not found: {path}")
        return
    kv: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    kv[k.strip()] = v.strip()
    except Exception as e:
        print(f"[Verai] Config read error: {e}"); return

    def _f(key, default): 
        try: return float(kv[key])
        except: return default
    def _i(key, default):
        try: return int(kv[key])
        except: return default
    def _b(key, default):
        try: return bool(int(kv[key]))
        except: return default
    def _s(key, default):
        return kv.get(key, default)

    # visuals
    state.show_boxes    = _b("show_boxes",   state.show_boxes)
    state.show_conf     = _b("show_conf",    state.show_conf)
    state.show_fov      = _b("show_fov",     state.show_fov)
    state.box_color_h   = _f("box_color_h",  state.box_color_h)
    state.box_color_s   = _f("box_color_s",  state.box_color_s)
    state.box_color_v   = _f("box_color_v",  state.box_color_v)
    state.conf_color_h  = _f("conf_color_h", state.conf_color_h)
    state.conf_color_s  = _f("conf_color_s", state.conf_color_s)
    state.conf_color_v  = _f("conf_color_v", state.conf_color_v)
    state.outline_color_h = _f("outline_color_h", state.outline_color_h)
    state.outline_color_s = _f("outline_color_s", state.outline_color_s)
    state.outline_color_v = _f("outline_color_v", state.outline_color_v)
    # accent
    _acc_h = _f("accent_h", _acc_h)
    _acc_s = _f("accent_s", _acc_s)
    _acc_v = _f("accent_v", _acc_v)
    # detection
    state.confidence    = _f("confidence",   state.confidence)
    state.fov_r         = _i("fov_r",        state.fov_r)
    state.model_path    = _s("model_path",   state.model_path)
    # aim
    state.aim_enabled     = _b("aim_enabled",     state.aim_enabled)
    state.aim_smooth      = _f("aim_smooth",       state.aim_smooth)
    state.aim_speed       = _f("aim_speed",        state.aim_speed)
    state.aim_bone        = _i("aim_bone",          state.aim_bone)
    state.aim_hotkey_only = _b("aim_hotkey_only",  state.aim_hotkey_only)
    state.aim_hotkey      = _i("aim_hotkey",        state.aim_hotkey)
    # clickbot
    state.click_enabled     = _b("click_enabled",     state.click_enabled)
    state.click_delay       = _f("click_delay",        state.click_delay)
    state.click_hotkey_only = _b("click_hotkey_only",  state.click_hotkey_only)
    state.click_hotkey      = _i("click_hotkey",       state.click_hotkey)
    state.click_padding     = _i("click_padding",      state.click_padding)
    state.click_on_person   = _b("click_on_person",    state.click_on_person)
    state.show_outline      = _b("show_outline",       state.show_outline)
    # settings
    state.watermark   = _b("watermark",    state.watermark)
    state.chromatic   = _b("chromatic",    state.chromatic)
    state.stream_proof= _b("stream_proof", state.stream_proof)
    state.dpi_scale   = _f("dpi_scale",    state.dpi_scale)
    state.use_gpu     = _b("use_gpu",    state.use_gpu)

    print(f"[Verai] Config loaded ← {path}")

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
_acc_h: float = 0.15   # bright yellow – high visibility default
_acc_s: float = 0.95
_acc_v: float = 1.00
_cp_drag: str  = ""
_accent_open:        bool  = False
_accent_just_opened: bool  = False
_cp_wx: float = -1.0
_cp_wy: float = -1.0

# per-picker open state for Visuals tab color pickers
_box_cp_open:        bool  = False
_box_cp_just_opened: bool  = False
_box_cp_wx: float = -1.0
_box_cp_wy: float = -1.0
_box_cp_drag: str = ""

_conf_cp_open:        bool  = False
_conf_cp_just_opened: bool  = False
_conf_cp_wx: float = -1.0
_conf_cp_wy: float = -1.0
_conf_cp_drag: str = ""

_outline_cp_open:        bool  = False
_outline_cp_just_opened: bool  = False
_outline_cp_wx: float = -1.0
_outline_cp_wy: float = -1.0
_outline_cp_drag: str = ""

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
_dd_click_consumed: bool = False   # True for one frame after a dropdown consumes a click
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
    global _dd_open, _dd_click_consumed
    sc = _dpi(); H = int(18*sc); r, g, b = _acc_rgb()
    cur = state.dpi_scale; label = f"{int(cur)}%"
    mouse = imgui.get_mouse_pos()
    clicked = imgui.is_mouse_clicked(0) and not _dd_click_consumed
    hov_hdr = (x <= mouse[0] <= x+w) and (y <= mouse[1] <= y+H)
    hdr_bg  = u(0.14,0.14,0.20,1.0) if hov_hdr else u(0.07,0.07,0.10,1.0)
    dl.add_rect_filled(x, y, x+w, y+H, hdr_bg)
    dl.add_rect(x, y, x+w, y+H, u(r,g,b, 0.90 if _dd_open else 0.55), thickness=1.0)
    lh = imgui.calc_text_size(label).y
    dl.add_text(x + int(7*sc), y + (H-lh)/2.0, u(r,g,b,1.0), label)
    arrow = "^" if _dd_open else "v"; aw = imgui.calc_text_size(arrow).x
    dl.add_text(x+w - aw - int(6*sc), y + (H-lh)/2.0, ct(C_DIM,1.0), arrow)
    if hov_hdr and clicked:
        _dd_open = not _dd_open
        _dd_click_consumed = True
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
            if hov_row and clicked:
                state.dpi_scale = opt; _dd_open = False
                _dd_click_consumed = True
            py += H
        fdl.add_rect(x, y+H, x+w, py, u(r,g,b,0.70), thickness=1.0)

def draw_model_dropdown(dl, x, y, w):
    global _model_dd_open, _dd_click_consumed
    sc = _dpi(); H = int(18*sc); r, g, b = _acc_rgb()
    cur = state.model_path.strip()
    # show just the filename for display
    label = os.path.basename(cur) if cur else "select model"
    mouse = imgui.get_mouse_pos()
    clicked = imgui.is_mouse_clicked(0) and not _dd_click_consumed
    hov_hdr = (x <= mouse[0] <= x+w) and (y <= mouse[1] <= y+H)
    hdr_bg  = u(0.14,0.14,0.20,1.0) if hov_hdr else u(0.07,0.07,0.10,1.0)
    dl.add_rect_filled(x, y, x+w, y+H, hdr_bg)
    dl.add_rect(x, y, x+w, y+H, u(r,g,b, 0.90 if _model_dd_open else 0.55), thickness=1.0)
    lh = imgui.calc_text_size(label).y
    dl.add_text(x + int(7*sc), y + (H-lh)/2.0, u(r,g,b,1.0), label)
    arrow = "^" if _model_dd_open else "v"; aw = imgui.calc_text_size(arrow).x
    dl.add_text(x+w - aw - int(6*sc), y + (H-lh)/2.0, ct(C_DIM,1.0), arrow)
    if hov_hdr and clicked:
        _model_dd_open = not _model_dd_open
        _dd_click_consumed = True
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
            if hov_row and clicked:
                state.model_path = os.path.join(_ASSETS_DIR, opt); _model_dd_open = False
                _dd_click_consumed = True
            py += H
        fdl.add_rect(x, y+H, x+w, py, u(r,g,b,0.70), thickness=1.0)

def draw_cfg_dropdown(dl, x, y, w):
    """Configs dropdown – lists .cfg files in the script folder."""
    global _cfg_dd_open, _dd_click_consumed, _cfg_selected
    sc = _dpi(); H = int(18*sc); r, g, b = _acc_rgb()
    opts = _cfg_options  # already refreshed on export / startup
    # Show the selected config name, or a placeholder if nothing selected yet
    if not opts:
        label = "-empty-"
    elif _cfg_selected and _cfg_selected in opts:
        label = _cfg_selected
    else:
        label = "select config"
    mouse = imgui.get_mouse_pos()
    clicked = imgui.is_mouse_clicked(0) and not _dd_click_consumed
    hov_hdr = (x <= mouse[0] <= x+w) and (y <= mouse[1] <= y+H)
    hdr_bg  = u(0.14,0.14,0.20,1.0) if hov_hdr else u(0.07,0.07,0.10,1.0)
    dl.add_rect_filled(x, y, x+w, y+H, hdr_bg)
    dl.add_rect(x, y, x+w, y+H, u(r,g,b, 0.90 if _cfg_dd_open else 0.55), thickness=1.0)
    lh = imgui.calc_text_size(label).y
    # Clip label text to fit inside dropdown width
    max_lbl_w = w - int(20*sc)
    disp_label = label
    while len(disp_label) > 1 and imgui.calc_text_size(disp_label).x > max_lbl_w:
        disp_label = disp_label[:-1]
    if disp_label != label: disp_label = disp_label[:-1] + "…"
    lbl_col = ct(C_DIM, 0.60) if not opts else u(r,g,b,1.0)
    dl.add_text(x + int(7*sc), y + (H-lh)/2.0, lbl_col, disp_label)
    arrow = "^" if _cfg_dd_open else "v"; aw = imgui.calc_text_size(arrow).x
    dl.add_text(x+w - aw - int(6*sc), y + (H-lh)/2.0, ct(C_DIM,1.0), arrow)
    if hov_hdr and clicked:
        if opts:  # only open if there is something to show
            _cfg_dd_open = not _cfg_dd_open
        _dd_click_consumed = True
    if _cfg_dd_open and opts:
        fdl = imgui.get_foreground_draw_list(); py = y + H
        panel_h = H * len(opts)
        fdl.add_rect_filled(x, py, x+w, py+panel_h, u(0.06,0.06,0.09,1.0))
        for opt in opts:
            hov_row = (x <= mouse[0] <= x+w) and (py <= mouse[1] <= py+H)
            if hov_row: fdl.add_rect_filled(x, py, x+w, py+H, u(0.13,0.13,0.19,1.0))
            fdl.add_line(x, py, x+w, py, u(0.20,0.20,0.28,1.0))
            oh = imgui.calc_text_size(opt).y
            # Clip option text
            disp_opt = opt
            max_opt_w = w - int(14*sc)
            while len(disp_opt) > 1 and imgui.calc_text_size(disp_opt).x > max_opt_w:
                disp_opt = disp_opt[:-1]
            if disp_opt != opt: disp_opt = disp_opt[:-1] + "…"
            fdl.add_text(x+int(10*sc), py+(H-oh)/2.0,
                         ct(C_TEXT,1.0) if hov_row else ct(C_DIM,1.0), disp_opt)
            if hov_row and clicked:
                _cfg_dd_open = False
                _dd_click_consumed = True
                _cfg_selected = opt
                _load_cfg(opt)
            py += H
        fdl.add_rect(x, y+H, x+w, py, u(r,g,b,0.70), thickness=1.0)

def _draw_color_picker_at(dl, ox, oy, sc, h_ref=None, s_ref=None, v_ref=None, drag_key="accent"):
    """Draw a colour picker at (ox,oy).
    h_ref/s_ref/v_ref are single-element lists [float] for in-place mutation.
    drag_key is a unique string prefix to namespace drag state per picker."""
    global _acc_h, _acc_s, _acc_v, _cp_drag
    # Default: edit the global accent colour
    if h_ref is None:
        h_ref = [_acc_h]; s_ref = [_acc_s]; v_ref = [_acc_v]
        def _commit():
            global _acc_h, _acc_s, _acc_v
            _acc_h = h_ref[0]; _acc_s = s_ref[0]; _acc_v = v_ref[0]
    else:
        def _commit(): pass  # caller owns the lists
    mouse = imgui.get_mouse_pos(); mx, my = mouse[0], mouse[1]
    pressed = imgui.is_mouse_down(0); clicked = imgui.is_mouse_clicked(0)
    SV_SIZE = int(80*sc); HUE_W = int(14*sc); GAP = int(6*sc); SW_W = int(20*sc)
    sv_x = ox; sv_y = oy; hue_x = ox + SV_SIZE + GAP; hue_y = oy
    GRID = 24; cell = SV_SIZE / GRID
    for xi in range(GRID):
        for yi in range(GRID):
            cr, cg, cb = colorsys.hsv_to_rgb(h_ref[0], xi/(GRID-1), 1.0-yi/(GRID-1))
            dl.add_rect_filled(sv_x+int(xi*cell), sv_y+int(yi*cell),
                               sv_x+int((xi+1)*cell)+1, sv_y+int((yi+1)*cell)+1,
                               u(cr,cg,cb,1.0))
    dl.add_rect(sv_x, sv_y, sv_x+SV_SIZE, sv_y+SV_SIZE, ct(C_BORDER,1.0), thickness=1.0)
    cur_sx = sv_x+int(s_ref[0]*SV_SIZE); cur_sy = sv_y+int((1.0-v_ref[0])*SV_SIZE)
    dl.add_circle_filled(cur_sx, cur_sy, int(4*sc), u(0,0,0,0.8))
    dl.add_circle_filled(cur_sx, cur_sy, int(3*sc), u(1,1,1,1.0))
    in_sv = (sv_x<=mx<=sv_x+SV_SIZE) and (sv_y<=my<=sv_y+SV_SIZE)
    sv_dk = drag_key + "_sv"
    if _cp_drag == sv_dk:
        if pressed:
            s_ref[0] = clamp((mx-sv_x)/SV_SIZE, 0.0, 1.0)
            v_ref[0] = 1.0 - clamp((my-sv_y)/SV_SIZE, 0.0, 1.0)
            _commit()
        else: _cp_drag = ""
    elif in_sv and clicked:
        _cp_drag = sv_dk
        s_ref[0] = clamp((mx-sv_x)/SV_SIZE, 0.0, 1.0)
        v_ref[0] = 1.0 - clamp((my-sv_y)/SV_SIZE, 0.0, 1.0)
        _commit()
    H_STEPS = 64; seg_h = SV_SIZE / H_STEPS
    for i in range(H_STEPS):
        hr, hg, hb = colorsys.hsv_to_rgb(i/H_STEPS, 1.0, 1.0)
        hy0 = hue_y+int(i*seg_h); hy1 = hue_y+int((i+1)*seg_h)+1
        dl.add_rect_filled(hue_x, hy0, hue_x+HUE_W, hy1, u(hr,hg,hb,1.0))
    dl.add_rect(hue_x, hue_y, hue_x+HUE_W, hue_y+SV_SIZE, ct(C_BORDER,1.0), thickness=1.0)
    hcy = hue_y+int(h_ref[0]*SV_SIZE)
    dl.add_rect_filled(hue_x-1, hcy-1, hue_x+HUE_W+1, hcy+2, u(0,0,0,0.8))
    dl.add_rect_filled(hue_x,   hcy,   hue_x+HUE_W,   hcy+1, u(1,1,1,1.0))
    in_hue = (hue_x<=mx<=hue_x+HUE_W) and (hue_y<=my<=hue_y+SV_SIZE)
    wh_dk = drag_key + "_wh"
    if _cp_drag == wh_dk:
        if pressed: h_ref[0] = clamp((my-hue_y)/SV_SIZE, 0.0, 1.0); _commit()
        else: _cp_drag = ""
    elif in_hue and clicked:
        _cp_drag = wh_dk; h_ref[0] = clamp((my-hue_y)/SV_SIZE, 0.0, 1.0); _commit()
    pr, pg, pb = colorsys.hsv_to_rgb(h_ref[0], s_ref[0], v_ref[0])
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
        ("Ver", ct(C_TEXT,1.0)), ("ai", u(r,g,b,1.0)),
        (SEP,  ct(C_DIM,1.0)), (f"{fps:.0f} fps", ct(C_TEXT,1.0)),
        (SEP,  ct(C_DIM,1.0)), ("undetected", ct(C_DIM,1.0)),
    ]
    total_w = sum(imgui.calc_text_size(t).x for t, _ in segments)
    lh = imgui.calc_text_size("Verai").y; pad = 8
    W = int(total_w + pad*2); H = int(lh + pad*2)
    bx = SW - W - 14; by = 14
    dl.add_rect_filled(bx, by, bx+W, by+H, ct(C_TITLE, 0.96))
    dl.add_rect(bx, by, bx+W, by+H, ct(C_BORDER, 1.0), thickness=1.0)
    dl.add_line(bx, by+1, bx+W, by+1, u(r,g,b,0.80), 1.0)
    cx = bx + pad; ty = by + pad
    for txt, col in segments:
        dl.add_text(cx, ty, col, txt); cx += imgui.calc_text_size(txt).x

# ════════════════════════════════════════════════════════
#  DETECTION DRAW
# ════════════════════════════════════════════════════════
def _draw_human_silhouette(dl, x1, y1, x2, y2, r, g, b, alpha=0.85):
    """Draw a segmented human body silhouette (head + torso + legs) fitted to a bounding box.
    Uses Claude's understanding of human proportions rather than a plain ellipse."""
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0

    # ── proportions (tuned to standard YOLO person bbox) ─────────
    # Head: top 15% of bbox, roughly round
    head_cy = y1 + bh * 0.09
    head_rx = bw * 0.16
    head_ry = bh * 0.10
    # Neck/shoulders: y1+17% → y1+30%, shoulder width ~55% of bbox
    shoulder_y = y1 + bh * 0.20
    shoulder_w = bw * 0.46
    # Waist: y1+50%, narrower at ~32% of bbox
    waist_y    = y1 + bh * 0.52
    waist_w    = bw * 0.28
    # Hips: y1+58%, wider than waist ~40%
    hip_y      = y1 + bh * 0.60
    hip_w      = bw * 0.38
    # Feet: bottom of bbox, hip width
    feet_y     = y2

    col = u(r, g, b, alpha)
    thick = 1.5

    # Head ellipse (SEGS segments)
    SEGS = 24
    head_pts = []
    for si in range(SEGS):
        ang = 2.0 * math.pi * si / SEGS
        head_pts.append((cx + head_rx * math.cos(ang), head_cy + head_ry * math.sin(ang)))
    for si in range(SEGS):
        ax, ay = head_pts[si]
        bx2_, by2_ = head_pts[(si + 1) % SEGS]
        dl.add_line(ax, ay, bx2_, by2_, col, thick)

    # Torso outline: shoulders → waist → hips as a polygon
    # Left side: shoulder → waist → hip
    # Right side mirrored
    # Use bezier-like stepped polyline for smooth curves
    torso_pts_left = [
        (cx - head_rx * 0.9,  head_cy + head_ry),        # base of neck, left
        (cx - shoulder_w,     shoulder_y),                # left shoulder
        (cx - waist_w,        waist_y),                   # left waist
        (cx - hip_w,          hip_y),                     # left hip
        (cx - hip_w * 0.55,   feet_y),                    # left ankle
    ]
    torso_pts_right = [
        (cx + head_rx * 0.9,  head_cy + head_ry),         # base of neck, right
        (cx + shoulder_w,     shoulder_y),                 # right shoulder
        (cx + waist_w,        waist_y),                    # right waist
        (cx + hip_w,          hip_y),                      # right hip
        (cx + hip_w * 0.55,   feet_y),                     # right ankle
    ]

    # Draw left and right body outline
    for pts in (torso_pts_left, torso_pts_right):
        for si in range(len(pts) - 1):
            ax, ay = pts[si]; bx2_, by2_ = pts[si + 1]
            dl.add_line(ax, ay, bx2_, by2_, col, thick)

    # Shoulder crossbar (top of torso)
    dl.add_line(torso_pts_left[0][0],  torso_pts_left[0][1],
                torso_pts_right[0][0], torso_pts_right[0][1], col, thick)
    # Foot crossbar
    dl.add_line(torso_pts_left[-1][0],  torso_pts_left[-1][1],
                torso_pts_right[-1][0], torso_pts_right[-1][1], col, thick)


def draw_detections(dl):
    rx, ry, rw, rh = cap_rect()

    # white outline = FOV box
    if state.show_fov:
        dl.add_rect(rx, ry, rx+rw, ry+rh, u(1,1,1,0.65), thickness=1.0)

    with state.lock:
        draw_list = list(state.detections)

    # ── Silhouette outline – independent of show_boxes ───────────
    if state.show_outline and draw_list:
        if state.outline_color_h >= 0.0:
            or_, og_, ob_ = colorsys.hsv_to_rgb(state.outline_color_h, state.outline_color_s, state.outline_color_v)
        else:
            or_, og_, ob_ = _acc_rgb()
        for entry in draw_list:
            x1, y1, x2, y2 = entry[0], entry[1], entry[2], entry[3]
            _draw_human_silhouette(dl, x1, y1, x2, y2, or_, og_, ob_, alpha=0.30)

    if not state.show_boxes:
        return

    # Box colour – use custom if set, else follow accent
    if state.box_color_h >= 0.0:
        br, bg, bb = colorsys.hsv_to_rgb(state.box_color_h, state.box_color_s, state.box_color_v)
    else:
        br, bg, bb = _acc_rgb()

    # Confidence colour – use custom if set, else follow accent
    if state.conf_color_h >= 0.0:
        cr2, cg2, cb2 = colorsys.hsv_to_rgb(state.conf_color_h, state.conf_color_s, state.conf_color_v)
    else:
        cr2, cg2, cb2 = _acc_rgb()

    for entry in draw_list:
        x1, y1, x2, y2, conf = entry[0], entry[1], entry[2], entry[3], entry[4]
        dl.add_rect(x1, y1, x2, y2, u(br, bg, bb, 0.90), thickness=1.5)

        if state.show_conf:
            bw = x2 - x1
            lbl   = "Player"
            lsz   = imgui.calc_text_size(lbl)
            lx    = x1 + (bw - lsz.x) / 2.0
            ly    = y1 - lsz.y - 2
            for ox, oy in ((-1,0),(1,0),(0,-1),(0,1)):
                dl.add_text(lx+ox, ly+oy, u(0,0,0,0.45), lbl)
            dl.add_text(lx, ly, u(br,bg,bb,0.55), lbl)
            pct   = f"{conf:.0%}"
            psz   = imgui.calc_text_size(pct)
            px_   = x2 + 4
            py_   = y1 + (y2 - y1) / 2.0 - psz.y / 2.0
            for ox, oy in ((-1,0),(1,0),(0,-1),(0,1)):
                dl.add_text(px_+ox, py_+oy, u(0,0,0,0.45), pct)
            dl.add_text(px_, py_, u(cr2,cg2,cb2,0.60), pct)

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
    # NO_FOCUS_ON_APPEARING + NO_BRING_TO_FRONT_ON_FOCUS keep the debug window
    # behind everything else so it never steals clicks or moves to the top of
    # the imgui window stack (which would make the whole overlay unclickable).
    expanded, opened = imgui.begin("Debug – Capture View  [F2]",
                                   state.show_debug,
                                   imgui.WINDOW_NO_RESIZE |
                                   imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                                   imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
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
# Cache GPU availability once at startup.
# Three possible paths:
#   "cuda"     – NVIDIA CUDA or AMD ROCm via PyTorch  (TensorRT export)
#   "directml" – AMD (and Intel) GPU via onnxruntime-directml  (ONNX export)
#   "cpu"      – no usable GPU
_CUDA_AVAILABLE:     bool = False
_DIRECTML_AVAILABLE: bool = False
_GPU_DEVICE:         str  = "cpu"   # "cuda:0" | "directml" | "cpu"
_GPU_BACKEND_NAME:   str  = "CPU"   # human-readable label shown in the UI tooltip

def _check_gpu():
    global _CUDA_AVAILABLE, _DIRECTML_AVAILABLE, _GPU_DEVICE, _GPU_BACKEND_NAME

    # ── 1. Try PyTorch CUDA / ROCm first (NVIDIA or AMD ROCm PyTorch build) ──
    try:
        import torch
        if torch.cuda.is_available():
            _CUDA_AVAILABLE = True
            _GPU_DEVICE     = "cuda:0"
            backend = "ROCm (AMD)" if (hasattr(torch.version, "hip") and torch.version.hip) else "CUDA (NVIDIA)"
            _GPU_BACKEND_NAME = backend
            print(f"[Verai] GPU detected via {backend}: {torch.cuda.get_device_name(0)}")
            return
        # PyTorch present but no CUDA/ROCm device – note it and fall through to DirectML
        has_cuda_build = bool(getattr(torch.version, "cuda", None))
        has_rocm_build = bool(getattr(torch.version, "hip",  None))
        if has_cuda_build or has_rocm_build:
            print(f"[Verai] PyTorch GPU build present (cuda={getattr(torch.version,'cuda',None)} "
                  f"hip={getattr(torch.version,'hip',None)}) but no device found – driver issue?")
        else:
            print("[Verai] CPU-only PyTorch. Checking for DirectML (AMD/Intel) …")
    except Exception as _torch_err:
        print(f"[Verai] PyTorch not available ({_torch_err}). Checking for DirectML …")

    # ── 2. Try onnxruntime-directml  (AMD RDNA / GCN, Intel Xe, any DX12 GPU) ──
    # Install with:  pip install onnxruntime-directml
    # (Do NOT have plain onnxruntime installed at the same time – they conflict.)
    try:
        import onnxruntime as _ort
        providers = _ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            _DIRECTML_AVAILABLE = True
            _GPU_DEVICE         = "directml"
            _GPU_BACKEND_NAME   = "DirectML (AMD/Intel)"
            print(f"[Verai] AMD/Intel GPU detected via DirectML (onnxruntime-directml).")
            print(f"  Available ORT providers: {providers}")
            return
        else:
            print(f"[Verai] onnxruntime found but DmlExecutionProvider missing. "
                  f"Providers: {providers}")
            print("  Install:  pip install onnxruntime-directml  "
                  "(remove plain onnxruntime first)")
    except ImportError:
        print("[Verai] onnxruntime-directml not installed.")
        print("  For AMD GPU acceleration:  pip install onnxruntime-directml")
        print("  For NVIDIA (ROCm-less):     pip install onnxruntime-gpu")

    # ── 3. Pure CPU fallback ──────────────────────────────────────────────────
    _CUDA_AVAILABLE     = False
    _DIRECTML_AVAILABLE = False
    _GPU_DEVICE         = "cpu"
    _GPU_BACKEND_NAME   = "CPU"
    print("[Verai] No GPU acceleration available – running on CPU.")

_check_gpu()

# Enable cuDNN auto-tuner for NVIDIA – finds fastest conv algorithm for fixed input size.
# Meaningless on CPU/DirectML; safe to always set.
if _CUDA_AVAILABLE:
    try:
        import torch as _torch_cudnn
        _torch_cudnn.backends.cudnn.benchmark = True
        _torch_cudnn.backends.cudnn.enabled   = True
        print("[Verai] cuDNN benchmark mode enabled (NVIDIA perf boost).")
    except Exception as _ce:
        print(f"[Verai] cuDNN benchmark skipped: {_ce}")

def _cuda_ok():     return _CUDA_AVAILABLE
def _directml_ok(): return _DIRECTML_AVAILABLE
def _any_gpu_ok():  return _CUDA_AVAILABLE or _DIRECTML_AVAILABLE

def _make_capturer():
    """Return a capturer dict with a unified .grab(rx, ry, rw, rh) → BGR ndarray interface."""
    if _DXCAM_AVAILABLE:
        try:
            cam = _dxcam_mod.create(output_color="BGR")
            print("[Verai] Hardware capture active (DXGI Desktop Duplication / GPU buffer).")
            def _grab(rx, ry, rw, rh):
                region = (rx, ry, rx + rw, ry + rh)
                frame = cam.grab(region=region)
                # dxcam returns None if the desktop hasn't changed – retry once
                if frame is None:
                    frame = cam.grab(region=region)
                return frame   # BGR ndarray or None
            return {"grab": _grab, "close": cam.release, "backend": "dxcam"}
        except Exception as e:
            print(f"[Verai] dxcam init failed ({e}) – falling back to mss.")

    # --- mss fallback (software / GDI) ---
    sct = mss.mss()
    print("[Verai] Software capture active (mss / GDI).")
    def _grab_mss(rx, ry, rw, rh):
        monitor = {"left": rx, "top": ry, "width": rw, "height": rh}
        raw = sct.grab(monitor)
        frame_bgra = np.frombuffer(raw.bgra, dtype=np.uint8).reshape((rh, rw, 4))
        return frame_bgra[:, :, :3]   # BGR view
    return {"grab": _grab_mss, "close": sct.close, "backend": "mss"}


def _detector_loop():
    from ultralytics import YOLO
    model        = None   # YOLO wrapper  OR  None when using DirectML ORT session
    ort_session  = None   # onnxruntime InferenceSession (DirectML path only)
    last_path    = None
    last_device  = None
    use_half     = False
    is_engine    = False   # True = .engine or Ultralytics-managed .onnx
    is_dml_onnx  = False   # True = raw ORT/DirectML session (AMD fast path)

    frames = 0; t0 = time.time()
    INF_SZ = 256  # square input – fastest YOLO path, no internal padding/repad
                  # 256 is ~35% faster than 320 on GPU with only marginal accuracy loss

    capturer = _make_capturer()
    _null_streak = 0          # consecutive None frames – triggers capturer reset on tab-in

    # ── DirectML ONNX inference helper ───────────────────────────────────────
    def _dml_predict(session, frame_bgr_inf, conf_thresh):
        """Run a raw ONNX session on the DirectML EP and return [(x1,y1,x2,y2,conf)]."""
        import onnxruntime as _ort
        # Prepare input: BGR → RGB, HWC → NCHW float32 [0,1]
        rgb   = cv2.cvtColor(frame_bgr_inf, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp   = np.expand_dims(rgb.transpose(2, 0, 1), 0)   # NCHW
        in_name  = session.get_inputs()[0].name
        outputs  = session.run(None, {in_name: inp})
        # YOLOv8 ONNX output: [1, 84, N] where first 4 rows = cx,cy,w,h, rest = class scores
        raw = outputs[0][0]                    # shape (84, N) or (N, 84)
        if raw.shape[0] == 84:                 # (84, N) – transpose
            raw = raw.T
        # raw is now (N, 84): [cx, cy, w, h, cls0..cls79]
        cx, cy, bw, bh = raw[:,0], raw[:,1], raw[:,2], raw[:,3]
        person_scores   = raw[:, 4]            # class 0 = person
        mask            = person_scores >= conf_thresh
        if not np.any(mask):
            return []
        cx = cx[mask]; cy = cy[mask]; bw_ = bw[mask]; bh_ = bh[mask]
        sc = person_scores[mask]
        x1 = cx - bw_ / 2; y1 = cy - bh_ / 2
        x2 = cx + bw_ / 2; y2 = cy + bh_ / 2
        return list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(), sc.tolist()))

    while True:
        if not state.enabled: time.sleep(0.05); continue

        want_device = _GPU_DEVICE if (state.use_gpu and _any_gpu_ok()) else "cpu"
        mp = state.model_path.strip()
        # If mp is a bare filename (no directory component), resolve it inside veraiassets
        if mp and not os.path.dirname(mp):
            mp = os.path.join(_ASSETS_DIR, mp)

        # Reload model only when path or device changes
        if mp != last_path or want_device != last_device:
            # Reset all model handles
            model = None; ort_session = None
            is_engine = False; is_dml_onnx = False; use_half = False

            try:
                load_path = mp

                # ════════════════════════════════════════════════════════════
                #  A) DirectML path  –  AMD / Intel GPU via onnxruntime-directml
                # ════════════════════════════════════════════════════════════
                if want_device == "directml" and mp.endswith(".pt"):
                    import onnxruntime as _ort

                    onnx_path     = mp.replace(".pt", f"_{INF_SZ}_dml.onnx")
                    onnx_raw_path = mp.replace(".pt", ".onnx")   # Ultralytics output name

                    if not os.path.isfile(onnx_path):
                        print(f"[YOLO/DML] ONNX model not found – exporting {mp} → {onnx_path} …")
                        print("[YOLO/DML] One-time export takes ~30 s. Subsequent loads are instant.")
                        try:
                            # ULTRALYTICS_AUTO_UPDATE=0 prevents Ultralytics from trying to
                            # pip-install plain onnxruntime during export, which causes a
                            # WinError 5 (Access Denied) because onnxruntime-directml already
                            # has its DLLs locked by the running ORT import above.
                            os.environ.setdefault("ULTRALYTICS_AUTO_UPDATE", "0")
                            _tmp = YOLO(mp)
                            _tmp.export(format="onnx", imgsz=INF_SZ, half=False,
                                        opset=17, simplify=True)
                            del _tmp
                            if os.path.isfile(onnx_raw_path) and onnx_raw_path != onnx_path:
                                os.rename(onnx_raw_path, onnx_path)
                                print(f"[YOLO/DML] Renamed {onnx_raw_path} → {onnx_path}")
                            if os.path.isfile(onnx_path):
                                print(f"[YOLO/DML] ONNX export complete → {onnx_path}")
                            else:
                                raise FileNotFoundError("ONNX file missing after export")
                        except Exception as _exp_err:
                            print(f"[YOLO/DML] ONNX export failed ({_exp_err}) – falling back to CPU .pt")
                            want_device = "cpu"
                            onnx_path   = None
                    else:
                        print(f"[YOLO/DML] ONNX model found → {onnx_path}")

                    if onnx_path and os.path.isfile(onnx_path):
                        # Build ORT session with DirectML as first provider
                        sess_opts = _ort.SessionOptions()
                        sess_opts.graph_optimization_level = (
                            _ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
                        sess_opts.enable_mem_pattern = False   # required for DirectML
                        # DirectML EP options: device_id 0 = first GPU
                        dml_ep_opts = {"device_id": 0}
                        ort_session = _ort.InferenceSession(
                            onnx_path,
                            sess_options=sess_opts,
                            providers=[("DmlExecutionProvider", dml_ep_opts),
                                       "CPUExecutionProvider"],
                        )
                        active_ep = ort_session.get_providers()[0]
                        print(f"[YOLO/DML] DirectML ORT session ready  "
                              f"(active EP: {active_ep})  model: {onnx_path}")
                        is_dml_onnx = True
                        last_path   = mp; last_device = want_device
                        state.model_loaded = True; state.model_error = ""
                        # Skip the YOLO-model block below
                        continue

                # ════════════════════════════════════════════════════════════
                #  B) CUDA / ROCm path  –  TensorRT auto-export (NVIDIA / AMD ROCm)
                # ════════════════════════════════════════════════════════════
                if want_device not in ("cpu", "directml") and mp.endswith(".pt"):
                    trt_path     = mp.replace(".pt", f"_{INF_SZ}.engine")
                    trt_raw_path = mp.replace(".pt", ".engine")
                    if not os.path.isfile(trt_path):
                        print(f"[YOLO] TensorRT engine not found – exporting to {trt_path} …")
                        print("[YOLO] This one-time export takes ~1-3 min. Future loads will be instant.")
                        try:
                            _tmp = YOLO(mp)
                            # workspace=6 gives the TRT builder more scratch RAM –
                            # helps on 8 GB cards (RTX 3070 etc.) find better layer fusion.
                            _tmp.export(format="engine", imgsz=INF_SZ, half=True, device=0,
                                        workspace=6)
                            del _tmp
                            if os.path.isfile(trt_raw_path) and trt_raw_path != trt_path:
                                os.rename(trt_raw_path, trt_path)
                                print(f"[YOLO] Renamed {trt_raw_path} → {trt_path}")
                            if os.path.isfile(trt_path):
                                print(f"[YOLO] TensorRT export complete → {trt_path}")
                            else:
                                print("[YOLO] TRT engine missing after export – falling back to .pt")
                                trt_path = None
                        except Exception as _trt_err:
                            print(f"[YOLO] TRT export failed ({_trt_err}) – falling back to .pt")
                            trt_path = None
                    else:
                        print(f"[YOLO] TensorRT engine found → {trt_path}")
                    if trt_path and os.path.isfile(trt_path):
                        load_path = trt_path

                # ════════════════════════════════════════════════════════════
                #  C) Standard YOLO / CPU path  (skipped if DML session is live)
                # ════════════════════════════════════════════════════════════
                if is_dml_onnx:
                    # DML session was just created above – don't also load a CPU YOLO model.
                    last_path = mp; last_device = want_device
                    state.model_loaded = True; state.model_error = ""
                    continue

                import torch as _torch_mod
                is_engine = load_path.endswith(".engine") or load_path.endswith(".onnx")
                model = YOLO(load_path, task="detect")
                if not is_engine:
                    model.to(want_device)
                if want_device not in ("cpu", "directml") and not is_engine:
                    use_half = True
                    print("[YOLO] autocast FP16 enabled on GPU")
                else:
                    use_half = False
                last_path = mp; last_device = want_device
                state.model_loaded = True; state.model_error = ""
                dev_label = ("GPU(TRT)" if is_engine else
                             ("GPU(CUDA)" if want_device != "cpu" else "CPU"))
                print(f"[YOLO] loaded → {load_path}  device={dev_label}")

                # Warm-up pass for NVIDIA CUDA: run a blank frame through the model
                # so cuDNN picks its fastest kernel before real inference starts.
                # Skipped for CPU, DirectML (no benefit) and .engine (TRT warms itself).
                if _CUDA_AVAILABLE and want_device != "cpu" and not is_engine and not is_dml_onnx:
                    try:
                        import torch as _tw
                        _dummy = np.zeros((INF_SZ, INF_SZ, 3), dtype=np.uint8)
                        with _tw.inference_mode():
                            model.predict(_dummy, classes=[0], conf=0.9, verbose=False,
                                          imgsz=INF_SZ, half=use_half, augment=False)
                        print("[YOLO] CUDA warm-up pass complete.")
                    except Exception as _wu_err:
                        print(f"[YOLO] Warm-up skipped: {_wu_err}")
            except Exception as e:
                state.model_error = str(e); state.model_loaded = False
                model = None; ort_session = None
                print(f"[YOLO] load error: {e}")
                time.sleep(1.0); continue

        if model is None and ort_session is None: time.sleep(0.1); continue

        rx, ry, rw, rh = cap_rect()
        try:
            frame_bgr = capturer["grab"](rx, ry, rw, rh)
        except Exception:
            time.sleep(0.05); continue
        if frame_bgr is None:
            _null_streak += 1
            if _null_streak >= 60:
                # dxcam can get stuck returning None after a tab-in / focus change.
                # Re-create the capturer to re-attach to the updated compositor surface.
                print("[Verai] Capture stalled – recreating capturer (tab-in recovery)…")
                try: capturer["close"]()
                except Exception: pass
                capturer = _make_capturer()
                _null_streak = 0
            time.sleep(0.005); continue   # dxcam: desktop unchanged, skip frame
        _null_streak = 0

        # Resize to exact INF_SZ square (avoids YOLO's internal letterbox repad)
        frame_inf = cv2.resize(frame_bgr, (INF_SZ, INF_SZ), interpolation=cv2.INTER_LINEAR)
        scale_x = rw / INF_SZ; scale_y = rh / INF_SZ

        # ── Inference dispatch ────────────────────────────────────────────────
        try:
            if is_dml_onnx and ort_session is not None:
                # DirectML / onnxruntime path – no torch needed at all
                raw_dets = _dml_predict(ort_session, frame_inf, state.confidence)
                # Fake a results-like object for the downstream box parsing.
                # Defined at local scope so _FakeBox is reachable from _FakeBoxes.__iter__.
                class _FakeBox:
                    def __init__(self, d):
                        self.xyxy = [np.array(d[:4])]
                        self.conf = [d[4]]
                class _FakeBoxes:
                    def __init__(self, dets):
                        self._dets = dets
                    def __iter__(self):
                        for d in self._dets:
                            yield _FakeBox(d)
                class _FakeResults:
                    def __init__(self, dets):
                        self.boxes = _FakeBoxes(dets)
                results = [_FakeResults(raw_dets)]

            elif is_engine:
                import torch as _torch_mod
                with _torch_mod.inference_mode():
                    results = model.predict(frame_inf, classes=[0], conf=state.confidence,
                                            verbose=False, imgsz=INF_SZ, half=True,
                                            augment=False, agnostic_nms=False, device=0)
            else:
                import torch as _torch_mod
                # NVIDIA CUDA: pass half=True directly to predict – lets Ultralytics
                # handle the tensor cast internally, which is faster than wrapping
                # in autocast on newer PyTorch builds.  CPU path: half must be False.
                _half_arg = use_half  # True only on CUDA, False on CPU/ROCm-CPU
                with _torch_mod.inference_mode():
                    results = model.predict(frame_inf, classes=[0], conf=state.confidence,
                                            verbose=False, imgsz=INF_SZ, half=_half_arg,
                                            augment=False, agnostic_nms=False)
        except Exception as e:
            state.model_error = str(e)
            print(f"[YOLO] predict error: {e}")
            time.sleep(0.1); continue

        dets = []; dbg = frame_bgr.copy()  # always keep last frame for debug window
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
            # Two-pass matching prevents ghost tracks when players move fast:
            #   Pass 1 – IoU match  (stationary / slow movement)
            #   Pass 2 – nearest-center fallback  (fast movement, IoU=0)
            # A prev track is matched at most once; ghosts cannot form.
            TTL_MAX   = 2     # frames a track survives without a match (low = no ghost trails)
            IOU_MIN   = 0.10  # low threshold – catch slow movers in pass 1
            DIST_MAX  = 160   # px between centers – max distance for fallback match
            VIS_ALPHA = 0.85  # high alpha = near-instant tracking, still filters pixel noise
            prev      = state.smooth_dets
            updated   = []

            def _cx(b): return (b[0]+b[2])/2.0
            def _cy(b): return (b[1]+b[3])/2.0

            def _iou(a, b):
                ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
                ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
                iw = max(0,ix2-ix1); ih = max(0,iy2-iy1)
                inter = iw*ih
                if inter == 0: return 0.0
                ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
                return inter / max(1, ua+ub-inter)

            def _blend(pi, nd):
                ox1,oy1,ox2,oy2,_oc,_ttl = prev[pi]
                nx1,ny1,nx2,ny2,nc = nd
                bx1 = int(ox1 + VIS_ALPHA*(nx1-ox1)); by1 = int(oy1 + VIS_ALPHA*(ny1-oy1))
                bx2 = int(ox2 + VIS_ALPHA*(nx2-ox2)); by2 = int(oy2 + VIS_ALPHA*(ny2-oy2))
                return [bx1,by1,bx2,by2, nc, TTL_MAX]

            matched_prev = set()
            matched_new  = set()

            # Pass 1: IoU matching
            for ni, nd in enumerate(dets):
                best_iou = IOU_MIN; best_pi = -1
                for pi, pd in enumerate(prev):
                    if pi in matched_prev: continue
                    iou = _iou(nd, pd)
                    if iou > best_iou: best_iou = iou; best_pi = pi
                if best_pi >= 0:
                    updated.append(_blend(best_pi, nd))
                    matched_prev.add(best_pi); matched_new.add(ni)

            # Pass 2: nearest-center fallback for unmatched detections
            for ni, nd in enumerate(dets):
                if ni in matched_new: continue
                best_d = DIST_MAX; best_pi = -1
                for pi, pd in enumerate(prev):
                    if pi in matched_prev: continue
                    d = ((_cx(nd)-_cx(pd))**2 + (_cy(nd)-_cy(pd))**2)**0.5
                    if d < best_d: best_d = d; best_pi = pi
                if best_pi >= 0:
                    updated.append(_blend(best_pi, nd))
                    matched_prev.add(best_pi); matched_new.add(ni)
                else:
                    # Truly new detection – no prev track nearby
                    updated.append([nd[0],nd[1],nd[2],nd[3], nd[4], TTL_MAX])
                    matched_new.add(ni)

            # Decay unmatched prev tracks – TTL_MAX=2 means they vanish in 2 frames max
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
    for p in [os.path.join(here,"verai.ttf"),
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
    global _dd_click_consumed
    global _box_cp_open, _box_cp_just_opened, _box_cp_wx, _box_cp_wy
    global _conf_cp_open, _conf_cp_just_opened, _conf_cp_wx, _conf_cp_wy
    global _outline_cp_open, _outline_cp_just_opened, _outline_cp_wx, _outline_cp_wy
    global _cfg_dd_open, _cfg_export_flash
    _dd_click_consumed = False  # reset per-frame dropdown click guard

    if gui_alpha < 0.01: return

    gw = GW(); th = TITLE(); tabh = TABS_H(); rowh = ROW_H(); sc = _dpi(); gh = GH()

    # ── Pre-consume clicks inside open color picker panels ─────────────────
    # The picker popups are drawn on the foreground draw list (visual only).
    # Without this guard, clicks inside the picker area fall through to the
    # invisible_button rows underneath and toggle unrelated settings.
    if imgui.is_mouse_clicked(0):
        def _cp_panel_rect(cp_wx, cp_wy):
            PAD=int(8*sc); SV_SZ=int(80*sc); HUE_W=int(14*sc); GAP=int(6*sc); PRV_W=int(20*sc)
            pw = PAD+SV_SZ+GAP+HUE_W+GAP+PRV_W+PAD; ph = PAD+SV_SZ+PAD
            px = int(clamp(cp_wx-pw//2, 4, SW-pw-4))
            py_ = int(clamp(cp_wy, 4, SH-ph-4))
            return px, py_, pw, ph
        _mx, _my = imgui.get_mouse_pos()
        for _open_flag, _cp_wx_val, _cp_wy_val in (
            (_accent_open,  _cp_wx,         _cp_wy),
            (_box_cp_open,  _box_cp_wx,     _box_cp_wy),
            (_conf_cp_open, _conf_cp_wx,    _conf_cp_wy),
            (_outline_cp_open, _outline_cp_wx, _outline_cp_wy),
        ):
            if _open_flag and _cp_wx_val >= 0:
                _px, _py, _pw, _ph = _cp_panel_rect(_cp_wx_val, _cp_wy_val)
                if _px <= _mx <= _px+_pw and _py <= _my <= _py+_ph:
                    _dd_click_consumed = True
                    break

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
    imgui.begin("##verai",
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
    ver_w = imgui.calc_text_size("Ver").x; full = imgui.calc_text_size("Verai")
    tx = x0 + (gw - full.x)/2.0; ty_ = y0 + (th - full.y)/2.0
    dl.add_text(tx,         ty_, ct(C_TEXT,1.0), "Ver")
    dl.add_text(tx + ver_w, ty_, u(r,g,b,1.0),   "ai")

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
        cur_y = th+tabh+SHH+j*rowh
        # Guard: only create the imgui widget if it fits in the window content area.
        # imgui asserts/crashes if set_cursor_pos goes past the clip rect.
        row_fits = (cur_y >= 0) and (cur_y + rowh) <= gh
        dd_active = _dd_open or _model_dd_open or _cfg_dd_open or _accent_open or _box_cp_open or _conf_cp_open or _outline_cp_open
        if row_fits:
            imgui.set_cursor_pos((0, cur_y))
            imgui.invisible_button(_id(), gw, rowh)
            hov     = imgui.is_item_hovered()
            clicked = imgui.is_item_clicked() and not _dd_click_consumed and not dd_active
        else:
            _id()  # consume id to keep numbering consistent
            hov     = False
            clicked = False
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
                        lambda v: setattr(state, "aim_speed", v), 1.0, 20.0, "%.0f"); j+=1

            # Smooth = PID gain: 10=very smooth/slow, 100=snappy (shown ×100 in UI)
            ry, _ = _row_bg(j); _label(ry, "Smooth")
            sx = x0+gw-SL_W-PAD_R; sy = ry+(rowh-int(14*sc))//2
            raw_s = draw_slider(dl, int(sx), int(sy), SL_W, "aim_smooth",
                                state.aim_smooth * 100.0, 10.0, 100.0, "%.0f")
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

            # Only on Person toggle – when on, uses ellipse hit-test instead of padded box
            _toggle_row(j, "Only on Person",
                        lambda: state.click_on_person,
                        lambda v: setattr(state, "click_on_person", v)); j+=1

            if not state.click_on_person:
                pass  # no padding slider

            # Hotkey Only toggle
            _toggle_row(j, "Hotkey Only",
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
        _toggle_row(j,"Show Outline",
                    lambda: state.show_outline,
                    lambda v: setattr(state,"show_outline",v)); j+=1
        _toggle_row(j,"Show FOV Box",
                    lambda: state.show_fov,
                    lambda v: setattr(state,"show_fov",v));     j+=1
        _toggle_row(j,"Debug Window  [F2]",
                    lambda: state.show_debug,
                    lambda v: setattr(state,"show_debug",v));   j+=1

        # ── Box Colour picker row ─────────────────────────
        ry, _ = _row_bg(j); _label(ry, "Box Colour")
        bh_use = state.box_color_h if state.box_color_h >= 0 else _acc_h
        bs_use = state.box_color_s if state.box_color_h >= 0 else _acc_s
        bv_use = state.box_color_v if state.box_color_h >= 0 else _acc_v
        bpr, bpg, bpb = colorsys.hsv_to_rgb(bh_use, bs_use, bv_use)
        SW_BTN=int(32*sc); SH_BTN=int(13*sc)
        bx_b=x0+gw-SW_BTN-PAD_R; by_b=ry+(rowh-SH_BTN)//2
        dl.add_rect_filled(bx_b,by_b,bx_b+SW_BTN,by_b+SH_BTN,u(bpr,bpg,bpb,1.0))
        b_border = u(1,1,1,0.55) if _box_cp_open else ct(C_BORDER,1.0)
        dl.add_rect(bx_b,by_b,bx_b+SW_BTN,by_b+SH_BTN,b_border,thickness=1.0)
        mouse_ = imgui.get_mouse_pos()
        in_btn_b = (bx_b<=mouse_[0]<=bx_b+SW_BTN) and (by_b<=mouse_[1]<=by_b+SH_BTN)
        if in_btn_b and imgui.is_mouse_clicked(0) and not _dd_click_consumed:
            _box_cp_open = not _box_cp_open
            if _box_cp_open:
                _box_cp_just_opened = True
                _box_cp_wx = float(bx_b+SW_BTN//2); _box_cp_wy = float(by_b+SH_BTN+int(4*sc))
                # initialise custom colour from current effective colour
                if state.box_color_h < 0:
                    state.box_color_h = _acc_h; state.box_color_s = _acc_s; state.box_color_v = _acc_v
            else:
                _box_cp_wx = -1.0; _box_cp_wy = -1.0
        j+=1

        # ── Confidence Colour picker row ──────────────────
        ry, _ = _row_bg(j); _label(ry, "Conf Colour")
        ch_use = state.conf_color_h if state.conf_color_h >= 0 else _acc_h
        cs_use = state.conf_color_s if state.conf_color_h >= 0 else _acc_s
        cv_use = state.conf_color_v if state.conf_color_h >= 0 else _acc_v
        cpr, cpg, cpb = colorsys.hsv_to_rgb(ch_use, cs_use, cv_use)
        bx_c2=x0+gw-SW_BTN-PAD_R; by_c2=ry+(rowh-SH_BTN)//2
        dl.add_rect_filled(bx_c2,by_c2,bx_c2+SW_BTN,by_c2+SH_BTN,u(cpr,cpg,cpb,1.0))
        c_border = u(1,1,1,0.55) if _conf_cp_open else ct(C_BORDER,1.0)
        dl.add_rect(bx_c2,by_c2,bx_c2+SW_BTN,by_c2+SH_BTN,c_border,thickness=1.0)
        in_btn_c = (bx_c2<=mouse_[0]<=bx_c2+SW_BTN) and (by_c2<=mouse_[1]<=by_c2+SH_BTN)
        if in_btn_c and imgui.is_mouse_clicked(0) and not _dd_click_consumed:
            _conf_cp_open = not _conf_cp_open
            if _conf_cp_open:
                _conf_cp_just_opened = True
                _conf_cp_wx = float(bx_c2+SW_BTN//2); _conf_cp_wy = float(by_c2+SH_BTN+int(4*sc))
                if state.conf_color_h < 0:
                    state.conf_color_h = _acc_h; state.conf_color_s = _acc_s; state.conf_color_v = _acc_v
            else:
                _conf_cp_wx = -1.0; _conf_cp_wy = -1.0
        j+=1

        # ── Outline Colour picker row ─────────────────────
        ry, _ = _row_bg(j); _label(ry, "Outline Colour")
        oh_use = state.outline_color_h if state.outline_color_h >= 0 else _acc_h
        os_use = state.outline_color_s if state.outline_color_h >= 0 else _acc_s
        ov_use = state.outline_color_v if state.outline_color_h >= 0 else _acc_v
        opr, opg, opb = colorsys.hsv_to_rgb(oh_use, os_use, ov_use)
        bx_o=x0+gw-SW_BTN-PAD_R; by_o=ry+(rowh-SH_BTN)//2
        dl.add_rect_filled(bx_o,by_o,bx_o+SW_BTN,by_o+SH_BTN,u(opr,opg,opb,1.0))
        o_border = u(1,1,1,0.55) if _outline_cp_open else ct(C_BORDER,1.0)
        dl.add_rect(bx_o,by_o,bx_o+SW_BTN,by_o+SH_BTN,o_border,thickness=1.0)
        in_btn_o = (bx_o<=mouse_[0]<=bx_o+SW_BTN) and (by_o<=mouse_[1]<=by_o+SH_BTN)
        if in_btn_o and imgui.is_mouse_clicked(0) and not _dd_click_consumed:
            _outline_cp_open = not _outline_cp_open
            if _outline_cp_open:
                _outline_cp_just_opened = True
                _outline_cp_wx = float(bx_o+SW_BTN//2); _outline_cp_wy = float(by_o+SH_BTN+int(4*sc))
                if state.outline_color_h < 0:
                    state.outline_color_h = _acc_h; state.outline_color_s = _acc_s; state.outline_color_v = _acc_v
            else:
                _outline_cp_wx = -1.0; _outline_cp_wy = -1.0
        j+=1

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
        _toggle_row(j,"Stream Proof",
                    lambda: state.stream_proof,
                    lambda v: setattr(state,"stream_proof",v)); j+=1

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
        if in_btn and imgui.is_mouse_clicked(0) and not _dd_click_consumed:
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
        elif _DIRECTML_AVAILABLE:
            q_col = u(0.30, 0.85, 0.45, 0.80)   # green – DirectML ready, same confidence as CUDA
            tip_text = ("DirectML (AMD/Intel) ready\n"
                        "ONNX model will be auto-exported on first GPU run.\n"
                        "Uses onnxruntime-directml for GPU acceleration.")
        else:
            q_col = ct(C_DIM, 0.80)
            tip_text = ("No GPU acceleration detected.\n"
                        "\n"
                        "AMD GPU (recommended):\n"
                        "  pip install onnxruntime-directml\n"
                        "  (remove plain onnxruntime first if installed)\n"
                        "\n"
                        "NVIDIA GPU (CUDA):\n"
                        "  pip install torch torchvision\n"
                        "       --index-url https://download.pytorch.org/whl/cu121\n"
                        "\n"
                        "AMD GPU (ROCm PyTorch):\n"
                        "  pip install torch torchvision\n"
                        "       --index-url https://download.pytorch.org/whl/rocm6.1")
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
        j+=1

        # ── Configs dropdown ──────────────────────────────
        ry, _ = _row_bg(j); _label(ry, "Configs")
        DD_W = int(130*sc); ddx = x0+gw-DD_W-PAD_R; ddy = ry+(rowh-int(18*sc))/2.0
        draw_cfg_dropdown(dl, int(ddx), int(ddy), DD_W); j+=1

        # ── Export button ─────────────────────────────────
        ry, _ = _row_bg(j)
        r_, g_, b_ = _acc_rgb()
        EBTN_W = int(60*sc); EBTN_H = int(13*sc)
        ebx = x0+gw-EBTN_W-PAD_R; eby = ry+(rowh-EBTN_H)//2
        now_t = time.time()
        flash = (now_t - _cfg_export_flash) < 1.2   # green flash for 1.2 s after export
        if flash:
            ebg = u(0.15, 0.55, 0.25, 1.0)
            ebc = u(0.30, 0.90, 0.45, 1.0)
            elbl = "Saved!"
        else:
            ebg = u(0.09, 0.09, 0.13, 1.0)
            ebc = u(r_, g_, b_, 0.55)
            elbl = "Export"
        mouse_ = imgui.get_mouse_pos()
        in_ebtn = (ebx<=mouse_[0]<=ebx+EBTN_W) and (eby<=mouse_[1]<=eby+EBTN_H)
        if in_ebtn and not flash:
            ebg = u(0.14, 0.14, 0.20, 1.0)
            ebc = u(r_, g_, b_, 0.90)
        dl.add_rect_filled(ebx, eby, ebx+EBTN_W, eby+EBTN_H, ebg)
        dl.add_rect(ebx, eby, ebx+EBTN_W, eby+EBTN_H, ebc, thickness=1.0)
        elsz = imgui.calc_text_size(elbl)
        dl.add_text(ebx+(EBTN_W-elsz.x)/2.0, eby+(EBTN_H-elsz.y)/2.0,
                    u(0.85,0.95,0.85,1.0) if flash else ct(C_TEXT,1.0), elbl)
        if in_ebtn and imgui.is_mouse_clicked(0) and not flash:
            if _export_cfg():
                _cfg_export_flash = time.time()
                _scan_cfg_files()
        _label(ry, "Export config")

    # ── accent colour picker (Settings tab) ──────────────────────────────────
    if tab_name == "Settings" and _accent_open:
        PAD=int(8*sc); SV_SZ=int(80*sc); HUE_W=int(14*sc); GAP=int(6*sc); PRV_W=int(20*sc)
        pw = PAD+SV_SZ+GAP+HUE_W+GAP+PRV_W+PAD; ph = PAD+SV_SZ+PAD
        px = int(clamp(_cp_wx-pw//2, 4, SW-pw-4))
        py_ = int(clamp(_cp_wy, 4, SH-ph-4))
        fdl = imgui.get_foreground_draw_list()
        for d in range(6,0,-1):
            fdl.add_rect_filled(px-d,py_-d,px+pw+d,py_+ph+d,u(0,0,0,0.014*d))
        fdl.add_rect_filled(px,py_,px+pw,py_+ph,u(0.06,0.06,0.09,0.97))
        fdl.add_rect(px,py_,px+pw,py_+ph,ct(C_BORDER,0.85),thickness=1.0)
        _draw_color_picker_at(fdl, px+PAD, py_+PAD, sc)  # accent – no h_ref, edits globals
        mouse_ = imgui.get_mouse_pos()
        in_panel = (px<=mouse_[0]<=px+pw) and (py_<=mouse_[1]<=py_+ph)
        if in_panel and imgui.is_mouse_clicked(0):
            _dd_click_consumed = True
        if _accent_just_opened:
            _accent_just_opened = False
        elif imgui.is_mouse_clicked(0) and not in_panel:
            _accent_open = False

    # ── box colour picker (Visuals tab) ──────────────────────────────────────
    if tab_name == "Visuals" and _box_cp_open:
        _bh = [state.box_color_h]; _bs = [state.box_color_s]; _bv = [state.box_color_v]
        PAD=int(8*sc); SV_SZ=int(80*sc); HUE_W=int(14*sc); GAP=int(6*sc); PRV_W=int(20*sc)
        pw = PAD+SV_SZ+GAP+HUE_W+GAP+PRV_W+PAD; ph = PAD+SV_SZ+PAD
        px = int(clamp(_box_cp_wx-pw//2, 4, SW-pw-4))
        py_ = int(clamp(_box_cp_wy, 4, SH-ph-4))
        fdl = imgui.get_foreground_draw_list()
        for d in range(6,0,-1):
            fdl.add_rect_filled(px-d,py_-d,px+pw+d,py_+ph+d,u(0,0,0,0.014*d))
        fdl.add_rect_filled(px,py_,px+pw,py_+ph,u(0.06,0.06,0.09,0.97))
        fdl.add_rect(px,py_,px+pw,py_+ph,ct(C_BORDER,0.85),thickness=1.0)
        _draw_color_picker_at(fdl, px+PAD, py_+PAD, sc,
                              h_ref=_bh, s_ref=_bs, v_ref=_bv, drag_key="box")
        state.box_color_h = _bh[0]; state.box_color_s = _bs[0]; state.box_color_v = _bv[0]
        mouse_ = imgui.get_mouse_pos()
        in_panel = (px<=mouse_[0]<=px+pw) and (py_<=mouse_[1]<=py_+ph)
        if in_panel and imgui.is_mouse_clicked(0):
            _dd_click_consumed = True
        if _box_cp_just_opened:
            _box_cp_just_opened = False
        elif imgui.is_mouse_clicked(0) and not in_panel:
            _box_cp_open = False

    # ── confidence colour picker (Visuals tab) ────────────────────────────────
    if tab_name == "Visuals" and _conf_cp_open:
        _ch = [state.conf_color_h]; _cs = [state.conf_color_s]; _cv = [state.conf_color_v]
        PAD=int(8*sc); SV_SZ=int(80*sc); HUE_W=int(14*sc); GAP=int(6*sc); PRV_W=int(20*sc)
        pw = PAD+SV_SZ+GAP+HUE_W+GAP+PRV_W+PAD; ph = PAD+SV_SZ+PAD
        px = int(clamp(_conf_cp_wx-pw//2, 4, SW-pw-4))
        py_ = int(clamp(_conf_cp_wy, 4, SH-ph-4))
        fdl = imgui.get_foreground_draw_list()
        for d in range(6,0,-1):
            fdl.add_rect_filled(px-d,py_-d,px+pw+d,py_+ph+d,u(0,0,0,0.014*d))
        fdl.add_rect_filled(px,py_,px+pw,py_+ph,u(0.06,0.06,0.09,0.97))
        fdl.add_rect(px,py_,px+pw,py_+ph,ct(C_BORDER,0.85),thickness=1.0)
        _draw_color_picker_at(fdl, px+PAD, py_+PAD, sc,
                              h_ref=_ch, s_ref=_cs, v_ref=_cv, drag_key="conf")
        state.conf_color_h = _ch[0]; state.conf_color_s = _cs[0]; state.conf_color_v = _cv[0]
        mouse_ = imgui.get_mouse_pos()
        in_panel = (px<=mouse_[0]<=px+pw) and (py_<=mouse_[1]<=py_+ph)
        if in_panel and imgui.is_mouse_clicked(0):
            _dd_click_consumed = True
        if _conf_cp_just_opened:
            _conf_cp_just_opened = False
        elif imgui.is_mouse_clicked(0) and not in_panel:
            _conf_cp_open = False

    # ── outline colour picker (Visuals tab) ──────────────────────────────────
    if tab_name == "Visuals" and _outline_cp_open:
        _oh = [state.outline_color_h]; _os = [state.outline_color_s]; _ov = [state.outline_color_v]
        PAD=int(8*sc); SV_SZ=int(80*sc); HUE_W=int(14*sc); GAP=int(6*sc); PRV_W=int(20*sc)
        pw = PAD+SV_SZ+GAP+HUE_W+GAP+PRV_W+PAD; ph = PAD+SV_SZ+PAD
        px = int(clamp(_outline_cp_wx-pw//2, 4, SW-pw-4))
        py_ = int(clamp(_outline_cp_wy, 4, SH-ph-4))
        fdl = imgui.get_foreground_draw_list()
        for d in range(6,0,-1):
            fdl.add_rect_filled(px-d,py_-d,px+pw+d,py_+ph+d,u(0,0,0,0.014*d))
        fdl.add_rect_filled(px,py_,px+pw,py_+ph,u(0.06,0.06,0.09,0.97))
        fdl.add_rect(px,py_,px+pw,py_+ph,ct(C_BORDER,0.85),thickness=1.0)
        _draw_color_picker_at(fdl, px+PAD, py_+PAD, sc,
                              h_ref=_oh, s_ref=_os, v_ref=_ov, drag_key="outline")
        state.outline_color_h = _oh[0]; state.outline_color_s = _os[0]; state.outline_color_v = _ov[0]
        mouse_ = imgui.get_mouse_pos()
        in_panel = (px<=mouse_[0]<=px+pw) and (py_<=mouse_[1]<=py_+ph)
        if in_panel and imgui.is_mouse_clicked(0):
            _dd_click_consumed = True
        if _outline_cp_just_opened:
            _outline_cp_just_opened = False
        elif imgui.is_mouse_clicked(0) and not in_panel:
            _outline_cp_open = False

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
#  AIM ASSIST THREAD  –  PID controller
# ════════════════════════════════════════════════════════
def _aim_loop():
    """
    PID-based aim assist with velocity prediction.

    Why PID over EMA:
      - P term  : proportional move toward target each tick
      - D term  : derivative – brakes when error is shrinking fast (prevents overshoot,
                  critically dampens the oscillation that happens during/after recoil)
      - No I term – integral winds up during recoil and fights the player; omitted.

    Velocity prediction:
      - Tracks target center movement between ticks (px/tick)
      - Leads the aim point by predicted_pos = current + velocity * LEAD_FRAMES
      - This fixes the "chasing the box" problem on fast-moving targets
      - Uses exponential smoothing on velocity to avoid jitter from detection noise

    KEY: aim thread reads raw detections only (zero-lag YOLO output).
         smooth_dets is intentionally left for the visual renderer only.

    aim_speed   : scales overall output (pixels/tick ceiling)
    aim_smooth  : 0.10–1.0 – PID gain; lower = softer/more human-like

    MOUSE DRIVER: Uses ctypes SendInput (MOUSEEVENTF_MOVE) for mouse movement.
    SendInput is the modern replacement for the deprecated mouse_event and sends
    input via the same Win32 path — no driver install required.
    """
    import ctypes, ctypes.wintypes

    # ── SendInput structure definitions ────────────────────────────────────
    # SendInput is the modern Win32 API for injecting input events.
    # MOUSEEVENTF_MOVE (0x0001) with dx/dy = relative move in mickeys.
    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx",          ctypes.c_long),
            ("dy",          ctypes.c_long),
            ("mouseData",   ctypes.c_ulong),
            ("dwFlags",     ctypes.c_ulong),
            ("time",        ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUT(ctypes.Structure):
        class _INPUT_UNION(ctypes.Union):
            _fields_ = [("mi", _MOUSEINPUT)]
        _anonymous_ = ("_u",)
        _fields_  = [("type", ctypes.c_ulong), ("_u", _INPUT_UNION)]

    _SendInput   = ctypes.windll.user32.SendInput
    _INPUT_MOUSE = 0
    _MOVE        = 0x0001   # MOUSEEVENTF_MOVE – relative

    def _move_mouse(dx: int, dy: int) -> None:
        """Inject a relative mouse move via SendInput."""
        inp        = _INPUT()
        inp.type   = _INPUT_MOUSE
        inp.mi.dx  = dx
        inp.mi.dy  = dy
        inp.mi.dwFlags = _MOVE
        _SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    # PID tuning constants (relative weights, scaled by aim_smooth at runtime)
    KP = 0.60   # proportional – main drive force
    KD = 0.22   # derivative   – damping / recoil stabiliser

    TICK        = 0.008   # ~125 Hz loop
    DZPX        = 1.5     # dead-zone radius in pixels – don't nudge if this close

    # Velocity prediction
    LEAD_FRAMES = 3.5     # how many ticks ahead to lead (tune: higher = more lead)
    VEL_ALPHA   = 0.40    # EMA on velocity (lower = smoother but slower to react)
    VEL_CAP     = 60.0    # max velocity magnitude in px/tick to accept (filters teleports)

    prev_err_x  = 0.0
    prev_err_y  = 0.0
    active      = False   # True once we've locked onto a target this session

    # Velocity tracking per target (keyed by rounded center to loosely track identity)
    prev_tx     = None    # previous raw target X
    prev_ty     = None    # previous raw target Y
    vel_x       = 0.0     # smoothed velocity X (px/tick)
    vel_y       = 0.0     # smoothed velocity Y (px/tick)

    def _reset():
        nonlocal prev_err_x, prev_err_y, active, prev_tx, prev_ty, vel_x, vel_y
        prev_err_x = 0.0; prev_err_y = 0.0; active = False
        prev_tx = None; prev_ty = None; vel_x = 0.0; vel_y = 0.0

    while True:
        time.sleep(TICK)

        if not state.aim_enabled:
            _reset(); continue

        if state.aim_hotkey_only:
            if not (win32api.GetAsyncKeyState(state.aim_hotkey) & 0x8000):
                _reset(); continue

        # ── ALWAYS use raw detections for aim (zero lag) ─────
        # smooth_dets has VIS_ALPHA baked-in lag – it makes the aim chase the box.
        # Prediction below compensates for YOLO's detection delay instead.
        with state.lock:
            dets = list(state.detections)

        if not dets:
            _reset(); continue

        cx, cy = SW // 2, SH // 2

        # ── pick closest target to crosshair ────────────────
        best = None; best_d = float("inf")
        for entry in dets:
            x1, y1, x2, y2 = entry[0], entry[1], entry[2], entry[3]
            bx_c = (x1 + x2) / 2.0
            bh   = max(1, y2 - y1)
            if   state.aim_bone == 0: ty_ = y1 + bh * 0.10   # Head
            elif state.aim_bone == 1: ty_ = y1 + bh * 0.25   # Chest
            else:                     ty_ = y1 + bh * 0.50   # Body
            d = ((bx_c - cx)**2 + (ty_ - cy)**2) ** 0.5
            if d < best_d:
                best_d = d; best = (bx_c, ty_)

        if best is None:
            _reset(); continue

        raw_tx, raw_ty = best

        # ── velocity estimation (EMA-smoothed) ───────────────
        if prev_tx is not None:
            raw_vx = raw_tx - prev_tx
            raw_vy = raw_ty - prev_ty
            # Cap to reject teleport/new-detection spikes
            raw_mag = (raw_vx**2 + raw_vy**2) ** 0.5
            if raw_mag > VEL_CAP:
                raw_vx = raw_vy = 0.0
            vel_x = vel_x + VEL_ALPHA * (raw_vx - vel_x)
            vel_y = vel_y + VEL_ALPHA * (raw_vy - vel_y)
        prev_tx = raw_tx; prev_ty = raw_ty

        # ── predicted aim point (lead the target) ────────────
        pred_tx = raw_tx + vel_x * LEAD_FRAMES
        pred_ty = raw_ty + vel_y * LEAD_FRAMES

        # ── PID on predicted position ────────────────────────
        err_x = pred_tx - cx
        err_y = pred_ty - cy

        dist = (err_x**2 + err_y**2) ** 0.5
        if dist < DZPX:
            # Inside dead-zone: keep prev_err for D continuity, don't move
            prev_err_x = err_x; prev_err_y = err_y
            active = True
            continue

        gain = clamp(state.aim_smooth, 0.10, 1.0)

        # First frame on a fresh target – skip D term to avoid kick
        if not active:
            prev_err_x = err_x; prev_err_y = err_y
            active = True

        d_x = err_x - prev_err_x
        d_y = err_y - prev_err_y

        out_x = (KP * err_x + KD * d_x) * gain
        out_y = (KP * err_y + KD * d_y) * gain

        prev_err_x = err_x
        prev_err_y = err_y

        # ── clamp to aim_speed ceiling ───────────────────────
        speed_cap = clamp(state.aim_speed, 1.0, 20.0)
        mag = (out_x**2 + out_y**2) ** 0.5
        if mag > speed_cap:
            scale = speed_cap / mag
            out_x *= scale; out_y *= scale

        # ── sub-pixel accumulation so slow speeds don't stall ─
        imx = int(out_x); imy = int(out_y)
        if imx == 0 and abs(out_x) >= 0.5: imx = 1 if out_x > 0 else -1
        if imy == 0 and abs(out_y) >= 0.5: imy = 1 if out_y > 0 else -1

        if imx != 0 or imy != 0:
            _move_mouse(imx, imy)

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
        if state.click_on_person:
            # Silhouette hit-test: uses same human body proportions as draw_detections.
            # Tests against head ellipse OR torso polygon to match the rendered outline.
            on_target = False
            for entry in dets:
                x1, y1, x2, y2 = entry[0], entry[1], entry[2], entry[3]
                bw = x2 - x1; bh = y2 - y1
                ecx = (x1 + x2) / 2.0
                # Head hit-test
                head_cy = y1 + bh * 0.09
                head_rx = bw * 0.16; head_ry = bh * 0.10
                if head_rx > 0 and head_ry > 0:
                    dhx = (cx - ecx) / head_rx; dhy = (cy - head_cy) / head_ry
                    if dhx*dhx + dhy*dhy <= 1.0:
                        on_target = True; break
                # Torso hit-test (bounding trapezoid between shoulders and hips)
                shoulder_y = y1 + bh * 0.20; shoulder_w = bw * 0.46
                hip_y      = y1 + bh * 0.60; hip_w      = bw * 0.38
                feet_y     = y2;              feet_w     = bw * 0.38 * 0.55
                if y1 + bh*0.20 <= cy <= y2:
                    # Interpolate width at cursor y
                    if cy <= hip_y:
                        t = (cy - shoulder_y) / max(1, hip_y - shoulder_y)
                        half_w = shoulder_w + (hip_w - shoulder_w) * t
                    else:
                        t = (cy - hip_y) / max(1, feet_y - hip_y)
                        half_w = hip_w + (feet_w - hip_w) * t
                    if abs(cx - ecx) <= half_w:
                        on_target = True; break
        else:
            pad = state.click_padding
            on_target = any(
                entry[0] + pad <= cx <= entry[2] - pad and
                entry[1] + pad <= cy <= entry[3] - pad
                for entry in dets
            )
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

    if not glfw.init(): return print("[Verai] glfw init failed")
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES,      4)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)
    # NOTE: Do NOT set GLFW_DECORATED=FALSE before window creation.
    # On AMD, telling GLFW to skip decorations causes it to allocate a non-DWM-composited
    # surface – per-pixel alpha breaks and the overlay goes black.
    # Instead, strip decorations AFTER creation via SetWindowLong (same as malevolent).

    EXTEND_RIGHT = 15   # slight oversizing forces AMD's DWM into the correct composite path
    win = glfw.create_window(SW + EXTEND_RIGHT, SH, "Verai", None, None)
    if not win: glfw.terminate(); return print("[Verai] window failed")

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
        print("[Verai] DWM glass enabled")
    except Exception as _e:
        print(f"[Verai] DWM glass skipped: {_e}")

    # ── Stream-proof setup ──────────────────────────────────────────────────
    # SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE = 0x11) causes the OS
    # compositor to replace the window with a black rect in any capture stream:
    #   OBS (Game/Window/Display capture), Discord screen-share,
    #   Windows Game Bar / Xbox DVR, NVIDIA ShadowPlay highlights.
    # Note: NVIDIA GeForce Experience "Instant Replay" still uses DXGI Desktop
    # Duplication which sees everything – WDA cannot block that path.
    import ctypes as _ct
    WDA_NONE               = 0x00
    WDA_EXCLUDEFROMCAPTURE = 0x11
    _stream_proof_state: list = [None]   # [last applied value] – avoids redundant calls

    def _apply_stream_proof(enabled: bool):
        affinity = WDA_EXCLUDEFROMCAPTURE if enabled else WDA_NONE
        if _stream_proof_state[0] != enabled:
            ok = _ct.windll.user32.SetWindowDisplayAffinity(_ct.c_void_p(hwnd), affinity)
            if ok:
                _stream_proof_state[0] = enabled
                print(f"[Verai] Stream Proof {'ON  (WDA_EXCLUDEFROMCAPTURE)' if enabled else 'OFF'}")
            else:
                err = _ct.windll.kernel32.GetLastError()
                print(f"[Verai] SetWindowDisplayAffinity failed (err={err})"
                      " – requires Windows 10 2004+ and a DWM-composited window")

    # Apply initial state (in case it was loaded from config)
    _apply_stream_proof(state.stream_proof)

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

    print("[Verai] YOLO overlay ready")
    print("  INSERT  →  toggle menu")
    print("  F2      →  toggle debug window")
    print(f"  Place model files in: {_ASSETS_DIR}")

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

        # Reactively apply / remove stream-proof capture exclusion
        _apply_stream_proof(state.stream_proof)

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

        try:
            _gui()
        except Exception as _gui_err:
            import traceback
            traceback.print_exc()
            print(f"[Verai] _gui crashed: {_gui_err}")
        _draw_debug_window()

        imgui.end_frame()
        gl.glClearColor(0,0,0,0); gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render(); impl.render(imgui.get_draw_data()); gl.glFlush()

    impl.shutdown(); glfw.terminate()

if __name__ == "__main__":
    run()
