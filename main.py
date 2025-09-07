import sys
import os
import json
import time
import math
import threading
import queue
import random
import traceback
import uuid
import io
import ctypes
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    PYSIDE = True
except:
    from PyQt5 import QtCore, QtGui, QtWidgets
    PYSIDE = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
try:
    import dxcam
except:
    dxcam = None
try:
    import mss
except:
    mss = None
try:
    from pynput import keyboard as pynput_keyboard
    from pynput import mouse as pynput_mouse
except:
    pynput_keyboard = None
    pynput_mouse = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
POOL_PATH = os.path.join(APP_DIR, "experience_pool.bin")
INDEX_PATH = os.path.join(APP_DIR, "experience_index.json")
KB_MODEL_PATH = os.path.join(APP_DIR, "keyboard_model.pt")
MS_MODEL_PATH = os.path.join(APP_DIR, "mouse_model.pt")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
POOL_LIMIT = 10 * 1024 * 1024 * 1024

ALLOWED_KEYS = []
for c in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    ALLOWED_KEYS.append(c)
for d in list("0123456789"):
    ALLOWED_KEYS.append(d)
ALLOWED_KEYS += ["UP","DOWN","LEFT","RIGHT","W","A","S","D","SPACE","SHIFT","CTRL","TAB","Q","E","R","F","Z","X","C","V","B","T","Y","G","H","J","K","L","U","I","O","P","N","M"]
ALLOWED_KEYS = list(dict.fromkeys([k.upper() for k in ALLOWED_KEYS]))

def torch_device_and_mem():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            free,total = torch.cuda.mem_get_info()
        except:
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            free = int(total*0.8)
        return torch.device("cuda"), free, total
    return torch.device("cpu"), 0, 0

def system_memory():
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint),
                ("dwMemoryLoad", ctypes.c_uint),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total = int(stat.ullTotalPhys)
        avail = int(stat.ullAvailPhys)
        used = total - avail
        return used, total
    except:
        return 0, 0

def adaptive_dims(mem_total_bytes):
    base = 0.5
    if mem_total_bytes<=0:
        return 0.5, 1
    g = mem_total_bytes/ (12*1024*1024*1024)
    width_mult = max(0.35, min(1.25, base + 0.6*g))
    depth_mult = 1 if g<0.4 else (2 if g<0.9 else 3)
    return width_mult, depth_mult

def adaptive_resolution(mem_total_bytes):
    if mem_total_bytes<=0:
        return 640,360
    g = mem_total_bytes/(12*1024*1024*1024)
    w = int(640 + 400*g)
    h = int(360 + 240*g)
    w = (w//32)*32
    h = (h//32)*32
    return max(320,w), max(180,h)

def adaptive_seq_len(mem_total_bytes):
    if mem_total_bytes<=0:
        return 2
    g = mem_total_bytes/(12*1024*1024*1024)
    return 3 if g>=0.8 else 2

def adaptive_batch_steps(mem_free_bytes, recent_latency_ms, ep_len, seq_len):
    cap = int(96 + 160*min(1.0, mem_free_bytes/(11*1024*1024*1024)))
    b = max(2, int((mem_free_bytes/(1024*1024*1024))*4))
    b = max(1, b//max(1,seq_len))
    if recent_latency_ms>60:
        b = max(1, b//2)
    steps = max(1, min(cap, ep_len//max(1,b)))
    return b, steps

def adaptive_lr(mem_total_bytes, loss_trend):
    scale = (mem_total_bytes/(12*1024*1024*1024)) if mem_total_bytes>0 else 0.5
    base = 1e-4 + 6e-4*scale
    if loss_trend>0.02:
        base *= 0.7
    if loss_trend<-0.02:
        base *= 1.3
    return max(1e-5, min(5e-3, base))

def adaptive_wd(mem_total_bytes, loss_trend):
    scale = (mem_total_bytes/(12*1024*1024*1024)) if mem_total_bytes>0 else 0.5
    wd = 5e-5 + 4e-4*scale
    if loss_trend>0.02:
        wd *= 1.2
    if loss_trend<-0.02:
        wd *= 0.8
    return max(1e-6, min(1e-2, wd))

def adaptive_dropout(mem_total_bytes):
    if mem_total_bytes<=0:
        return 0.1
    g = mem_total_bytes/(12*1024*1024*1024)
    return max(0.05, min(0.25, 0.18-0.08*g))

def now_ts():
    return int(time.time()*1000)

def atomic_write_json(path, obj):
    tmp = path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False)
    os.replace(tmp,path)

class ScreenGrabber(QtCore.QObject):
    frame_ready = QtCore.Signal(np.ndarray) if PYSIDE else QtCore.pyqtSignal(np.ndarray)
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
        self.running = False
        self.thread = None
        self.use_dx = False
        self.cam = None
        self.ms = None
        self.monitor = None
        self.aug_strength = 0.15
        self.max_fps = 60
        self.fps_smooth = deque(maxlen=60)
        self.lock = threading.Lock()
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop,daemon=True)
        self.thread.start()
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
        self.cam = None
        self.ms = None
    def set_target_size(self, wh):
        with self.lock:
            self.target_size = wh
    def set_max_fps(self, fps):
        self.max_fps = max(5, min(120,int(fps)))
    def set_aug_strength(self, v):
        self.aug_strength = max(0.0, min(0.6, float(v)))
    def random_jitter(self, img):
        a = self.aug_strength
        if a<=0:
            return img
        b = 1.0 + (random.random()*2-1)*a
        c = 1.0 + (random.random()*2-1)*a
        img = cv2.convertScaleAbs(img, alpha=c, beta=(b-1.0)*50)
        if random.random()<0.25*a:
            h,w = img.shape[:2]
            dh = int(h*a*0.18)
            dw = int(w*a*0.18)
            y0 = random.randint(0,dh) if dh>0 else 0
            x0 = random.randint(0,dw) if dw>0 else 0
            y1 = h - random.randint(0,dh) if dh>0 else h
            x1 = w - random.randint(0,dw) if dw>0 else w
            y0 = max(0,min(h-2,y0)); y1 = max(y0+2,min(h,y1))
            x0 = max(0,min(w-2,x0)); x1 = max(x0+2,min(w,x1))
            img = img[y0:y1, x0:x1]
            img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
        return img
    def init_backend(self):
        if dxcam is not None:
            try:
                self.cam = dxcam.create(output_idx=0, output_color="BGR")
                self.use_dx = True
                return
            except:
                self.cam = None
        if mss is not None:
            try:
                self.ms = mss.mss()
                mon = self.ms.monitors[1] if isinstance(self.ms.monitors,list) and len(self.ms.monitors)>1 else self.ms.monitors[0]
                self.monitor = {"top":mon["top"],"left":mon["left"],"width":mon["width"],"height":mon["height"]}
                self.use_dx = False
                return
            except:
                self.ms = None
    def grab_once(self):
        if self.use_dx and self.cam is not None:
            f = self.cam.grab()
            return f
        if self.ms is not None:
            raw = self.ms.grab(self.monitor)
            img = np.array(raw)[:,:,:3]
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            return img
        return None
    def loop(self):
        try:
            self.init_backend()
            last = time.time()
            period = 1.0/max(1,self.max_fps)
            while self.running:
                t0 = time.time()
                img = self.grab_once()
                if img is None:
                    time.sleep(0.05)
                    continue
                with self.lock:
                    target_w, target_h = self.target_size
                img = cv2.resize(img,(target_w,target_h),interpolation=cv2.INTER_AREA)
                img = self.random_jitter(img)
                self.frame_ready.emit(img)
                t1 = time.time()
                dt = t1 - t0
                if dt<period:
                    time.sleep(period-dt)
                self.fps_smooth.append(1.0/max(1e-6, time.time()-last))
                last = time.time()
                with self.lock:
                    period = 1.0/max(1,self.max_fps)
        except:
            pass

class InputIO(QtCore.QObject):
    action_ready = QtCore.Signal(dict) if PYSIDE else QtCore.pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.mode = "学习模式"
        self.key_listener = None
        self.mouse_listener = None
        self.kb_controller = pynput_keyboard.Controller() if pynput_keyboard else None
        self.ms_controller = pynput_mouse.Controller() if pynput_mouse else None
        self.lock = threading.Lock()
        self.pressed_keys = set()
        self.last_sent = {}
        self.min_press_ms = 50
        self.min_interval_ms = 40
        self.mouse_speed_limit = 1200
        self.mouse_accel = 0.6
        self.last_mouse_move = 0
        self.block_keys = set(["ESC"])
        self.stop_flag = False
        self.out_thread = None
        self.out_queue = queue.Queue(maxsize=32)
        self.last_pointer = None
        self.sample_rate_s = 0.02
        self.last_emit = 0
        self.scroll_buffer = 0.0
        self.debounce = {}
        self.smooth_move = [0.0,0.0]
    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
    def update_hparams(self, sample_rate_s=None, min_interval_ms=None, mouse_speed_limit=None, mouse_accel=None):
        if sample_rate_s is not None:
            self.sample_rate_s = float(max(0.005, min(0.1, sample_rate_s)))
        if min_interval_ms is not None:
            self.min_interval_ms = int(max(10, min(120, min_interval_ms)))
        if mouse_speed_limit is not None:
            self.mouse_speed_limit = int(max(200, min(3000, mouse_speed_limit)))
        if mouse_accel is not None:
            self.mouse_accel = float(max(0.1, min(0.95, mouse_accel)))
    def key_to_pynput(self, k):
        if pynput_keyboard is None:
            return None
        if k in ["UP","DOWN","LEFT","RIGHT"]:
            return getattr(pynput_keyboard.Key, k.lower())
        if k=="SPACE":
            return pynput_keyboard.Key.space
        if k=="SHIFT":
            return pynput_keyboard.Key.shift
        if k=="CTRL":
            return pynput_keyboard.Key.ctrl
        if k=="TAB":
            return pynput_keyboard.Key.tab
        if len(k)==1 and k.isalnum():
            return pynput_keyboard.KeyCode.from_char(k.lower())
        if len(k)==1:
            return pynput_keyboard.KeyCode.from_char(k)
        return None
    def on_press(self, key):
        try:
            kname = None
            if isinstance(key, pynput_keyboard.Key):
                if key==pynput_keyboard.Key.esc:
                    self.action_ready.emit({"esc":True})
                    return
                if key==pynput_keyboard.Key.space:
                    kname="SPACE"
                elif key==pynput_keyboard.Key.shift or key==pynput_keyboard.Key.shift_r:
                    kname="SHIFT"
                elif key==pynput_keyboard.Key.ctrl or key==pynput_keyboard.Key.ctrl_r:
                    kname="CTRL"
                elif key==pynput_keyboard.Key.up:
                    kname="UP"
                elif key==pynput_keyboard.Key.down:
                    kname="DOWN"
                elif key==pynput_keyboard.Key.left:
                    kname="LEFT"
                elif key==pynput_keyboard.Key.right:
                    kname="RIGHT"
                elif key==pynput_keyboard.Key.tab:
                    kname="TAB"
            else:
                ch = key.char
                if ch is not None:
                    if ch.isalpha():
                        kname = ch.upper()
                    elif ch.isdigit():
                        kname = ch
            if kname is not None and kname in ALLOWED_KEYS:
                self.pressed_keys.add(kname)
        except:
            pass
    def on_release(self, key):
        try:
            kname = None
            if isinstance(key, pynput_keyboard.Key):
                if key==pynput_keyboard.Key.space:
                    kname="SPACE"
                elif key==pynput_keyboard.Key.shift or key==pynput_keyboard.Key.shift_r:
                    kname="SHIFT"
                elif key==pynput_keyboard.Key.ctrl or key==pynput_keyboard.Key.ctrl_r:
                    kname="CTRL"
                elif key==pynput_keyboard.Key.up:
                    kname="UP"
                elif key==pynput_keyboard.Key.down:
                    kname="DOWN"
                elif key==pynput_keyboard.Key.left:
                    kname="LEFT"
                elif key==pynput_keyboard.Key.right:
                    kname="RIGHT"
                elif key==pynput_keyboard.Key.tab:
                    kname="TAB"
            else:
                ch = key.char
                if ch is not None:
                    if ch.isalpha():
                        kname = ch.upper()
                    elif ch.isdigit():
                        kname = ch
            if kname is not None and kname in ALLOWED_KEYS and kname in self.pressed_keys:
                self.pressed_keys.discard(kname)
        except:
            pass
    def on_move(self,x,y):
        now = now_ts()
        if self.last_pointer is None:
            self.last_pointer = (x,y)
        lx,ly = self.last_pointer
        dx = x-lx
        dy = y-ly
        self.last_pointer = (x,y)
        self.smooth_move[0] = self.smooth_move[0]*0.6 + dx*0.4
        self.smooth_move[1] = self.smooth_move[1]*0.6 + dy*0.4
        if time.time()-self.last_emit>self.sample_rate_s:
            self.action_ready.emit({"mouse":(self.smooth_move[0],self.smooth_move[1]),"time":now})
            self.last_emit = time.time()
    def on_click(self,x,y,button,pressed):
        btn = None
        if button==pynput_mouse.Button.left:
            btn="LEFT"
        elif button==pynput_mouse.Button.right:
            btn="RIGHT"
        elif button==pynput_mouse.Button.middle:
            btn="MIDDLE"
        if btn is None:
            return
        self.action_ready.emit({"click":(btn,pressed),"time":now_ts()})
    def on_scroll(self,x,y,dx,dy):
        self.action_ready.emit({"scroll":dy,"time":now_ts()})
    def start_listen(self):
        if self.mode!="学习模式":
            return
        if pynput_keyboard is None or pynput_mouse is None:
            return
        self.stop_flag = False
        self.key_listener = pynput_keyboard.Listener(on_press=self.on_press,on_release=self.on_release)
        self.mouse_listener = pynput_mouse.Listener(on_move=self.on_move,on_click=self.on_click,on_scroll=self.on_scroll)
        self.key_listener.start()
        self.mouse_listener.start()
    def stop_listen(self):
        self.stop_flag = True
        if self.key_listener:
            self.key_listener.stop()
            self.key_listener = None
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
    def start_output(self):
        if self.mode!="训练模式":
            return
        self.stop_flag = False
        self.out_thread = threading.Thread(target=self.output_loop,daemon=True)
        self.out_thread.start()
    def stop_output(self):
        if self.out_thread:
            self.stop_flag = True
            try:
                while not self.out_queue.empty():
                    self.out_queue.get_nowait()
            except:
                pass
            self.out_thread.join(timeout=1.5)
            self.out_thread=None
        if self.kb_controller is not None:
            for k in ALLOWED_KEYS:
                if k!="ESC":
                    self.safe_press(k,False)
        if self.ms_controller is not None and pynput_mouse is not None:
            try:
                self.ms_controller.release(pynput_mouse.Button.left)
            except:
                pass
            try:
                self.ms_controller.release(pynput_mouse.Button.right)
            except:
                pass
            try:
                self.ms_controller.release(pynput_mouse.Button.middle)
            except:
                pass
    def send_action(self, action):
        if self.mode!="训练模式":
            return
        try:
            self.out_queue.put_nowait(action)
        except:
            pass
    def throttle_ok(self, key):
        t = now_ts()
        last = self.last_sent.get(key,0)
        if t - last >= self.min_interval_ms:
            self.last_sent[key] = t
            return True
        return False
    def safe_press(self, keyname, press=True):
        if keyname=="ESC":
            return
        if self.kb_controller is None:
            return
        pk = self.key_to_pynput(keyname)
        if pk is None:
            return
        if press:
            try:
                self.kb_controller.press(pk)
            except:
                pass
        else:
            try:
                self.kb_controller.release(pk)
            except:
                pass
    def move_mouse_smooth(self, dx, dy):
        if self.ms_controller is None:
            return
        mag = math.hypot(dx,dy)
        if mag<=0:
            return
        lim = float(self.mouse_speed_limit)
        if mag>lim:
            scale = lim/mag
            dx*=scale
            dy*=scale
        steps = max(1, int(5 + 10*(mag/lim)))
        for i in range(steps):
            if self.stop_flag:
                break
            px = int(dx*(i+1)/steps)
            py = int(dy*(i+1)/steps)
            try:
                self.ms_controller.move(px,py)
            except:
                pass
            time.sleep(0.002*(1.0-self.mouse_accel)+0.0005)
    def output_loop(self):
        pressed = set()
        while not self.stop_flag:
            try:
                act = self.out_queue.get(timeout=0.05)
            except:
                continue
            keys = act.get("keys",[])
            clicks = act.get("clicks",{})
            move = act.get("move",(0,0))
            scroll = act.get("scroll",0.0)
            for k in list(pressed):
                if k not in keys:
                    self.safe_press(k,False)
                    pressed.discard(k)
            for k in keys:
                if k not in pressed and self.throttle_ok("k_"+k):
                    self.safe_press(k,True)
                    pressed.add(k)
            dx,dy = move
            if abs(dx)+abs(dy)>0 and self.throttle_ok("m_move"):
                self.move_mouse_smooth(dx,dy)
            for btn, val in clicks.items():
                if btn=="LEFT":
                    b = pynput_mouse.Button.left if pynput_mouse else None
                elif btn=="RIGHT":
                    b = pynput_mouse.Button.right if pynput_mouse else None
                else:
                    b = pynput_mouse.Button.middle if pynput_mouse else None
                if b is not None and self.ms_controller is not None and self.throttle_ok("m_"+btn):
                    try:
                        if val>0.5:
                            self.ms_controller.press(b)
                        else:
                            self.ms_controller.release(b)
                    except:
                        pass
            if abs(scroll)>0.1 and self.ms_controller is not None and self.throttle_ok("m_scroll"):
                try:
                    self.ms_controller.scroll(0, int(scroll))
                except:
                    pass

class ReplayBuffer:
    def __init__(self):
        self.index = {"entries":[], "write_pos":0, "file_size":0, "episodes":{}}
        self.file = None
        self.lock = threading.Lock()
        self.ensure_files()
        self.load_index()
        self.open_file()
    def ensure_files(self):
        if not os.path.exists(POOL_PATH):
            with open(POOL_PATH,"wb") as f:
                pass
        if not os.path.exists(INDEX_PATH):
            atomic_write_json(INDEX_PATH,self.index)
        if not os.path.exists(KB_MODEL_PATH):
            torch.save({}, KB_MODEL_PATH)
        if not os.path.exists(MS_MODEL_PATH):
            torch.save({}, MS_MODEL_PATH)
        if not os.path.exists(CONFIG_PATH):
            atomic_write_json(CONFIG_PATH,{"last_run":now_ts()})
    def open_file(self):
        try:
            self.file = open(POOL_PATH,"r+b")
        except:
            self.file = open(POOL_PATH,"w+b")
        self.file.seek(0,os.SEEK_END)
        sz = self.file.tell()
        self.index["file_size"] = min(sz, POOL_LIMIT)
    def close(self):
        try:
            if self.file:
                self.file.flush()
                os.fsync(self.file.fileno())
                self.file.close()
        except:
            pass
        self.file = None
    def load_index(self):
        try:
            with open(INDEX_PATH,"r",encoding="utf-8") as f:
                self.index = json.load(f)
        except:
            self.index = {"entries":[], "write_pos":0, "file_size":0, "episodes":{}}
    def save_index(self):
        atomic_write_json(INDEX_PATH,self.index)
    def compress_frame(self, frame, quality):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            ok, buf = cv2.imencode(".png", frame)
        return buf.tobytes()
    def calc_priority(self, action, label, rw):
        kc = len(action.get("keys",[]))
        mv = action.get("move",(0,0))
        mv_mag = min(1.0, (abs(mv[0])+abs(mv[1]))/800.0)
        clicks = action.get("clicks",{})
        cc = sum(1 for b,v in clicks.items() if v>0.5)
        base = 0.2 + 0.3*min(1.0,kc/4.0) + 0.2*mv_mag + 0.2*min(1.0,cc/2.0) + 0.1*max(0.0, float(rw) if rw is not None else 0.0)
        labw = {"赢":1.6,"输":1.3,"平局":1.1,"unlabeled":1.0,"跳过":0.9}.get(label,1.0)
        p = max(0.05, min(3.0, base*labw))
        return p
    def write_sample(self, frame, action, label, episode_id, reward_signal=None):
        with self.lock:
            if self.file is None:
                self.open_file()
            h,w = frame.shape[:2]
            q = 70 - int(min(40, (w*h)/ (1280*720) * 10))
            q = max(40, min(90,q))
            data = {"ts":now_ts(),"ep":episode_id,"label":label,"action":action,"rw":reward_signal,"shape":[h,w]}
            header = json.dumps(data,separators=(",",":")).encode("utf-8")
            head_len = len(header).to_bytes(4,"little")
            comp = self.compress_frame(frame,q)
            rec = head_len + header + comp
            L = len(rec)
            wp = self.index.get("write_pos",0)
            fs = self.index.get("file_size",0)
            if fs<POOL_LIMIT and wp==fs and fs+L<=POOL_LIMIT:
                self.file.seek(wp,os.SEEK_SET)
                self.file.write(rec)
                self.file.flush()
                os.fsync(self.file.fileno())
                p = self.calc_priority(action,label,reward_signal)
                self.index["entries"].append({"off":wp,"size":L,"ep":episode_id,"label":label,"ts":data["ts"],"p":p})
                self.index["write_pos"] = wp+L
                self.index["file_size"] = max(fs, wp+L)
            else:
                if wp+L>POOL_LIMIT:
                    wp = 0
                to_overwrite_start = wp
                to_overwrite_end = wp+L
                survivors = []
                for e in list(self.index["entries"]):
                    e_start = e["off"]
                    e_end = e["off"]+e["size"]
                    overlap = not (e_end<=to_overwrite_start or e_start>=to_overwrite_end)
                    if not overlap:
                        survivors.append(e)
                self.index["entries"] = survivors
                self.file.seek(wp,os.SEEK_SET)
                self.file.write(rec)
                self.file.flush()
                os.fsync(self.file.fileno())
                p = self.calc_priority(action,label,reward_signal)
                self.index["entries"].append({"off":wp,"size":L,"ep":episode_id,"label":label,"ts":data["ts"],"p":p})
                self.index["write_pos"] = wp+L
                self.index["file_size"] = max(self.index["file_size"], wp+L)
                if self.index["file_size"]>POOL_LIMIT:
                    self.index["file_size"]=POOL_LIMIT
            eps = self.index.get("episodes",{})
            lst = eps.get(episode_id,{"count":0,"labels":[]})
            lst["count"] = lst.get("count",0)+1
            lst["labels"].append(label)
            eps[episode_id]=lst
            self.index["episodes"]=eps
    def occupancy(self):
        with self.lock:
            return len(self.index["entries"]), self.index["file_size"], self.index["write_pos"]
    def _decode_entry(self, idx):
        e = self.index["entries"][idx]
        off = int(e.get("off", -1))
        size = int(e.get("size", -1))
        sz = self.index.get("file_size",0)
        if off<0 or size<=8 or off+size>sz:
            return None
        self.file.seek(off,os.SEEK_SET)
        hlen_b = self.file.read(4)
        if len(hlen_b)!=4:
            return None
        hlen = int.from_bytes(hlen_b,"little")
        if hlen<=0 or hlen> (1<<20) or 4+hlen>size:
            return None
        header = self.file.read(hlen)
        try:
            meta = json.loads(header.decode("utf-8"))
        except:
            return None
        rest = size-4-hlen
        comp = self.file.read(rest)
        if len(comp)!=rest:
            return None
        img = cv2.imdecode(np.frombuffer(comp,np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img, meta, e
    def sample_batch_seq(self, n, seq_len):
        with self.lock:
            if self.file is None:
                self.open_file()
            if not self.index["entries"]:
                return []
            total = len(self.index["entries"])
            weights = []
            now = now_ts()
            for e in self.index["entries"]:
                p = float(e.get("p",1.0))
                age_ms = max(1, now - int(e.get("ts",now)))
                age_w = 0.7 + 0.3*(1.0/(1.0+age_ms/60000.0))
                weights.append(max(1e-6, p*age_w))
            take = self.weighted_choice(weights, min(n,total))
            out = []
            bad_idx = set()
            for i in take:
                try:
                    dec = self._decode_entry(i)
                    if dec is None:
                        bad_idx.add(i)
                        continue
                    img, meta, e0 = dec
                    ep = e0.get("ep")
                    seq = [img]
                    metas = [meta]
                    j = i-1
                    while len(seq)<seq_len and j>=0:
                        ej = self.index["entries"][j]
                        if ej.get("ep")==ep:
                            dj = self._decode_entry(j)
                            if dj is not None:
                                seq.insert(0, dj[0])
                                metas.insert(0, dj[1])
                        j -= 1
                    while len(seq)<seq_len:
                        seq.insert(0, seq[0])
                        metas.insert(0, metas[0])
                    out.append((seq, metas[-1]))
                except:
                    bad_idx.add(i)
            if bad_idx:
                survivors = [e for j,e in enumerate(self.index["entries"]) if j not in bad_idx]
                self.index["entries"] = survivors
                if self.index["write_pos"]>self.index["file_size"]:
                    self.index["write_pos"]=self.index["file_size"]
                self.save_index()
            return out
    def weighted_choice(self, weights, k):
        total = sum(weights)
        if total<=0:
            idxs = list(range(len(weights)))
            random.shuffle(idxs)
            return idxs[:k]
        chosen = set()
        tries = 0
        while len(chosen)<min(k,len(weights)) and tries<k*10:
            tries+=1
            r = random.random()*total
            s = 0.0
            for i,w in enumerate(weights):
                s += w
                if r<=s:
                    chosen.add(i)
                    break
        if len(chosen)<k:
            remain = [i for i in range(len(weights)) if i not in chosen]
            random.shuffle(remain)
            chosen.update(remain[:k-len(chosen)])
        return list(chosen)
    def save_all(self):
        with self.lock:
            self.save_index()

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.dw = nn.Conv2d(in_ch,in_ch,3,stride=stride,padding=1,groups=in_ch,bias=False)
        self.pw = nn.Conv2d(in_ch,out_ch,1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self,x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

class VisionBackbone(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1, in_ch=3):
        super().__init__()
        c1 = int(16*width_mult)
        c2 = int(32*width_mult)
        c3 = int(64*width_mult)
        c4 = int(128*width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch,c1,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        blocks = []
        ch = c1
        for i in range(depth_mult):
            blocks.append(DepthwiseSeparableConv(ch,c2,2))
            ch = c2
            blocks.append(DepthwiseSeparableConv(ch,c2,1))
        for i in range(depth_mult):
            blocks.append(DepthwiseSeparableConv(ch,c3,2))
            ch = c3
            blocks.append(DepthwiseSeparableConv(ch,c3,1))
        for i in range(depth_mult):
            blocks.append(DepthwiseSeparableConv(ch,c4,2))
            ch = c4
            blocks.append(DepthwiseSeparableConv(ch,c4,1))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.AdaptiveAvgPool2d(1)
        self.out_ch = ch
    def forward(self,x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.flatten(1)
        return x

class KeyboardHead(nn.Module):
    def __init__(self, in_ch, num_keys, drop_p):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, in_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop_p)
        self.fc = nn.Linear(in_ch, num_keys)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc(x)

class MouseHead(nn.Module):
    def __init__(self,in_ch, drop_p):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, in_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop_p)
        self.fc = nn.Linear(in_ch, 6)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        o = self.fc(x)
        return o

def make_scaler_for_device(device):
    try:
        return torch.amp.GradScaler('cuda' if device.type=='cuda' else 'cpu', enabled=(device.type=='cuda'))
    except TypeError:
        return torch.amp.GradScaler(enabled=(device.type=='cuda'))

class EMA:
    def __init__(self, model, decay=0.996):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad = False
        self.decay = decay
    def update(self, model):
        with torch.no_grad():
            for ep, p in zip(self.m.parameters(), model.parameters()):
                ep.copy_(ep*self.decay + p*(1.0-self.decay))

class Learner:
    def __init__(self, device, mem_total):
        self.device = device
        self.mem_total = mem_total
        wm, dm = adaptive_dims(mem_total)
        self.seq_len = adaptive_seq_len(mem_total)
        dp = adaptive_dropout(mem_total)
        self.backbone_kb = VisionBackbone(wm,dm,in_ch=3*self.seq_len).to(device)
        self.backbone_ms = VisionBackbone(wm,dm,in_ch=3*self.seq_len).to(device)
        self.kb_head = KeyboardHead(self.backbone_kb.out_ch, len(ALLOWED_KEYS), dp).to(device)
        self.ms_head = MouseHead(self.backbone_ms.out_ch, dp).to(device)
        self.ema_kb_backbone = EMA(self.backbone_kb)
        self.ema_ms_backbone = EMA(self.backbone_ms)
        self.ema_kb_head = EMA(self.kb_head)
        self.ema_ms_head = EMA(self.ms_head)
        self.scaler = make_scaler_for_device(device)
        lr0 = adaptive_lr(mem_total, 0.0)
        wd0 = adaptive_wd(mem_total, 0.0)
        g = (mem_total/(12*1024*1024*1024)) if mem_total>0 else 0.5
        beta2 = max(0.92, min(0.99, 0.95+0.03*g))
        self.opt_kb = torch.optim.AdamW(list(self.backbone_kb.parameters())+list(self.kb_head.parameters()), lr=lr0, weight_decay=wd0, betas=(0.9,beta2))
        self.opt_ms = torch.optim.AdamW(list(self.backbone_ms.parameters())+list(self.ms_head.parameters()), lr=lr0, weight_decay=wd0, betas=(0.9,beta2))
        self.sched_kb = None
        self.sched_ms = None
        self.loss_hist = deque(maxlen=200)
        self.step_counter = 0
        self.lr_display = 0.0
        self.set_dynamic_lrs(mem_total, 0.0, 1)
    def set_dynamic_lrs(self, mem_total, loss_trend, steps):
        lr = adaptive_lr(mem_total, loss_trend)
        wd = adaptive_wd(mem_total, loss_trend)
        for g in self.opt_kb.param_groups:
            g['lr']=lr
            g['weight_decay']=wd
        for g in self.opt_ms.param_groups:
            g['lr']=lr
            g['weight_decay']=wd
        steps = max(1, steps)
        pct = float(max(0.05, min(0.3, 0.12 + 0.12*(-loss_trend))))
        self.sched_kb = torch.optim.lr_scheduler.OneCycleLR(self.opt_kb, max_lr=lr, total_steps=steps, pct_start=pct, anneal_strategy='cos')
        self.sched_ms = torch.optim.lr_scheduler.OneCycleLR(self.opt_ms, max_lr=lr, total_steps=steps, pct_start=pct, anneal_strategy='cos')
        try:
            lrs = self.sched_kb.get_last_lr()
            if lrs:
                self.lr_display = float(lrs[0])
        except:
            pass
    def preprocess_seq(self, seq_imgs):
        batch = []
        for imgs in seq_imgs:
            x = torch.from_numpy(np.stack(imgs,0)).to(self.device, non_blocking=True)
            x = x.permute(0,3,1,2).contiguous().float()/255.0
            s,c,h,w = x.shape
            x = x.view(1, s*c, h, w)
            batch.append(x)
        t = torch.cat(batch, dim=0)
        noise_std = float(max(0.002, min(0.02, 0.005*(1.0 + (len(self.loss_hist)/200.0)))))
        noise = torch.randn_like(t)*noise_std
        t = (t+noise).clamp(0.0,1.0)
        return t
    def focal_bce(self, logits, targets, alpha, gamma, class_balance=None):
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets>0.5, prob, 1.0-prob)
        w = ((alpha*targets) + ((1.0-alpha)*(1.0-targets))) * ((1.0-pt).clamp(0,1)**gamma)
        if class_balance is not None:
            w = w * class_balance
        return (w*bce).mean()
    def class_balance_from_batch(self, targets):
        p = targets.mean(0)
        cb = ((1.0 - p) / (p + 1e-4)).clamp(0.5, 5.0)
        return cb
    def mixup(self, x, kb_t, ms_t, trend):
        if x.size(0)<2:
            return x, kb_t, ms_t
        lam = float(np.clip(0.2 + 0.6*random.random() + (-trend)*0.2, 0.1, 0.95))
        idx = torch.randperm(x.size(0), device=x.device)
        x = x*lam + x[idx]*(1.0-lam)
        kb_t = kb_t*lam + kb_t[idx]*(1.0-lam)
        ms_t = ms_t*lam + ms_t[idx]*(1.0-lam)
        return x, kb_t, ms_t
    def optimize_once(self, batch_seq, reward_weight=1.0):
        if not batch_seq:
            return 0.0, self.lr_display
        imgs_seq = []
        metas = []
        for imgs, meta in batch_seq:
            if isinstance(imgs, list) and imgs and isinstance(imgs[0], np.ndarray):
                imgs_seq.append(imgs)
                metas.append(meta)
        if not imgs_seq:
            return 0.0, self.lr_display
        labels = []
        mss = []
        for m in metas:
            la = m.get("action",{})
            ks = la.get("keys",[])
            kv = np.zeros(len(ALLOWED_KEYS),dtype=np.float32)
            for k in ks:
                if k in ALLOWED_KEYS:
                    kv[ALLOWED_KEYS.index(k)]=1.0
            labels.append(kv)
            mdx, mdy = la.get("move",(0,0))
            cl = la.get("clicks",{})
            lp = cl.get("LEFT",0.0)
            rp = cl.get("RIGHT",0.0)
            mp = cl.get("MIDDLE",0.0)
            mss.append([mdx/600.0, mdy/600.0, float(lp), float(rp), float(mp), la.get("scroll",0.0)/10.0])
        kb_t = torch.from_numpy(np.stack(labels,0).astype(np.float32)).to(self.device, non_blocking=True)
        ms_t = torch.from_numpy(np.stack(mss,0).astype(np.float32)).to(self.device, non_blocking=True)
        x = self.preprocess_seq(imgs_seq)
        loss_val = 0.0
        trend = 0.0
        if len(self.loss_hist)>10:
            v = list(self.loss_hist)
            trend = (v[-1]-v[0])/max(1,len(v))
        alpha = float(torch.clamp((1.0 - kb_t.mean()).item(),0.1,0.9))
        gamma = float(torch.clamp(torch.tensor(1.1 + (1.2*max(0.0,min(1.0,-trend*20.0)))),1.0,2.6).item())
        cons_w = float(max(0.05, min(0.3, 0.12 + (0.2*(0.5 - trend)))))
        x, kb_t, ms_t = self.mixup(x, kb_t, ms_t, trend)
        clip_val = float(max(0.5, min(1.2, 1.2 - 0.5*min(1.0, (self.mem_total/(12*1024*1024*1024))))))
        if self.device.type=="cuda":
            try:
                with torch.autocast('cuda'):
                    feat_kb = self.backbone_kb(x)
                    logits = self.kb_head(feat_kb)
                    cb = self.class_balance_from_batch(kb_t)
                    loss_kb = self.focal_bce(logits, kb_t, alpha=alpha, gamma=gamma, class_balance=cb)*reward_weight
                    feat_ms = self.backbone_ms(x)
                    out_ms = self.ms_head(feat_ms)
                    loss_pos = F.huber_loss(out_ms[:,:2], ms_t[:,:2], reduction='mean')
                    loss_click = F.binary_cross_entropy_with_logits(out_ms[:,2:5], ms_t[:,2:5], reduction='mean')
                    loss_scroll = F.huber_loss(out_ms[:,5:6], ms_t[:,5:6], reduction='mean')
                    loss_ms = (loss_pos+loss_click+loss_scroll)*reward_weight
                    with torch.no_grad():
                        ft_kb_t = self.ema_kb_backbone.m(x)
                        log_t = self.ema_kb_head.m(ft_kb_t)
                        ft_ms_t = self.ema_ms_backbone.m(x)
                        out_t = self.ema_ms_head.m(ft_ms_t)
                    loss_cons_kb = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(log_t))
                    loss_cons_ms = F.mse_loss(out_ms, out_t)
                    loss_total = loss_kb + loss_ms + cons_w*(loss_cons_kb+loss_cons_ms)
                self.opt_kb.zero_grad(set_to_none=True)
                self.opt_ms.zero_grad(set_to_none=True)
                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.opt_kb)
                self.scaler.unscale_(self.opt_ms)
                torch.nn.utils.clip_grad_norm_(list(self.backbone_kb.parameters())+list(self.kb_head.parameters()), clip_val)
                torch.nn.utils.clip_grad_norm_(list(self.backbone_ms.parameters())+list(self.ms_head.parameters()), clip_val)
                self.scaler.step(self.opt_kb)
                self.scaler.step(self.opt_ms)
                self.scaler.update()
                self.ema_kb_backbone.update(self.backbone_kb)
                self.ema_kb_head.update(self.kb_head)
                self.ema_ms_backbone.update(self.backbone_ms)
                self.ema_ms_head.update(self.ms_head)
                loss_val = float(loss_total.detach().item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
                    return 0.0, self.lr_display
                else:
                    raise
        else:
            feat_kb = self.backbone_kb(x)
            logits = self.kb_head(feat_kb)
            cb = self.class_balance_from_batch(kb_t)
            loss_kb = self.focal_bce(logits, kb_t, alpha=alpha, gamma=gamma, class_balance=cb)*reward_weight
            feat_ms = self.backbone_ms(x)
            out_ms = self.ms_head(feat_ms)
            loss_pos = F.huber_loss(out_ms[:,:2], ms_t[:,:2], reduction='mean')
            loss_click = F.binary_cross_entropy_with_logits(out_ms[:,2:5], ms_t[:,2:5], reduction='mean')
            loss_scroll = F.huber_loss(out_ms[:,5:6], ms_t[:,5:6], reduction='mean')
            with torch.no_grad():
                ft_kb_t = self.ema_kb_backbone.m(x)
                log_t = self.ema_kb_head.m(ft_kb_t)
                ft_ms_t = self.ema_ms_backbone.m(x)
                out_t = self.ema_ms_head.m(ft_ms_t)
            loss_cons_kb = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(log_t))
            loss_cons_ms = F.mse_loss(out_ms, out_t)
            loss_total = loss_kb + (loss_pos+loss_click+loss_scroll)*reward_weight + cons_w*(loss_cons_kb+loss_cons_ms)
            self.opt_kb.zero_grad(set_to_none=True)
            self.opt_ms.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(list(self.backbone_kb.parameters())+list(self.kb_head.parameters()), clip_val)
            torch.nn.utils.clip_grad_norm_(list(self.backbone_ms.parameters())+list(self.ms_head.parameters()), clip_val)
            self.opt_kb.step()
            self.opt_ms.step()
            self.ema_kb_backbone.update(self.backbone_kb)
            self.ema_kb_head.update(self.kb_head)
            self.ema_ms_backbone.update(self.backbone_ms)
            self.ema_ms_head.update(self.ms_head)
            loss_val = float(loss_total.detach().item())
        if self.sched_kb is not None:
            try:
                self.sched_kb.step()
                self.sched_ms.step()
                lrs = self.sched_kb.get_last_lr()
                if lrs:
                    self.lr_display = float(lrs[0])
            except:
                pass
        self.loss_hist.append(loss_val)
        self.step_counter += 1
        return loss_val, self.lr_display
    def infer(self, frames_seq, temp_k=1.0, temp_m=1.0):
        x = self.preprocess_seq([frames_seq])
        if self.device.type=="cuda":
            with torch.no_grad(), torch.autocast('cuda'):
                fk = self.ema_kb_backbone.m(x)
                logits = self.ema_kb_head.m(fk)[0]
                fm = self.ema_ms_backbone.m(x)
                out = self.ema_ms_head.m(fm)[0]
        else:
            with torch.no_grad():
                fk = self.ema_kb_backbone.m(x)
                logits = self.ema_kb_head.m(fk)[0]
                fm = self.ema_ms_backbone.m(x)
                out = self.ema_ms_head.m(fm)[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        keys = self.postprocess_keys(probs, temp_k)
        ms = out.detach().cpu().numpy()
        dx = float(np.tanh(ms[0]*temp_m)*600.0)
        dy = float(np.tanh(ms[1]*temp_m)*600.0)
        clicks = {"LEFT": float(torch.sigmoid(torch.tensor(ms[2]))), "RIGHT": float(torch.sigmoid(torch.tensor(ms[3]))), "MIDDLE": float(torch.sigmoid(torch.tensor(ms[4])))}
        scroll = float(np.tanh(ms[5]*temp_m)*10.0)
        return {"keys":keys,"move":(dx,dy),"clicks":clicks,"scroll":scroll}
    def postprocess_keys(self, probs, temp):
        p = probs**(1.0/max(1e-3,temp))
        chosen = []
        thr = min(0.9, max(0.2, float(np.median(p)+0.15)))
        order = np.argsort(-p)
        used = set()
        for idx in order:
            if p[idx]<thr:
                break
            k = ALLOWED_KEYS[idx]
            if k=="ESC":
                continue
            if k in ["UP","DOWN"]:
                if ("UP" in used) and k=="DOWN":
                    continue
                if ("DOWN" in used) and k=="UP":
                    continue
            if k in ["LEFT","RIGHT"]:
                if ("LEFT" in used) and k=="RIGHT":
                    continue
                if ("RIGHT" in used) and k=="LEFT":
                    continue
            chosen.append(k)
            used.add(k)
            if len(chosen)>8:
                break
        if "W" in chosen and "S" in chosen:
            if probs[ALLOWED_KEYS.index("W")]>=probs[ALLOWED_KEYS.index("S")]:
                chosen.remove("S")
            else:
                chosen.remove("W")
        if "A" in chosen and "D" in chosen:
            if probs[ALLOWED_KEYS.index("A")]>=probs[ALLOWED_KEYS.index("D")]:
                chosen.remove("D")
            else:
                chosen.remove("A")
        return chosen
    def save_models(self):
        kb_tmp = KB_MODEL_PATH+".tmp"
        ms_tmp = MS_MODEL_PATH+".tmp"
        torch.save({"backbone":self.backbone_kb.state_dict(),"head":self.kb_head.state_dict(),"ema_backbone":self.ema_kb_backbone.m.state_dict(),"ema_head":self.ema_kb_head.m.state_dict(),"seq_len":self.seq_len}, kb_tmp)
        torch.save({"backbone":self.backbone_ms.state_dict(),"head":self.ms_head.state_dict(),"ema_backbone":self.ema_ms_backbone.m.state_dict(),"ema_head":self.ema_ms_head.m.state_dict(),"seq_len":self.seq_len}, ms_tmp)
        os.replace(kb_tmp,KB_MODEL_PATH)
        os.replace(ms_tmp,MS_MODEL_PATH)
    def load_models(self):
        if os.path.exists(KB_MODEL_PATH):
            try:
                d = torch.load(KB_MODEL_PATH,map_location=self.device)
                if "backbone" in d:
                    self.backbone_kb.load_state_dict(d["backbone"],strict=False)
                if "head" in d:
                    self.kb_head.load_state_dict(d["head"],strict=False)
                if "ema_backbone" in d:
                    self.ema_kb_backbone.m.load_state_dict(d["ema_backbone"],strict=False)
                if "ema_head" in d:
                    self.ema_kb_head.m.load_state_dict(d["ema_head"],strict=False)
                else:
                    self.ema_kb_backbone.update(self.backbone_kb)
                    self.ema_kb_head.update(self.kb_head)
                if "seq_len" in d and int(d["seq_len"])>0:
                    self.seq_len = int(d["seq_len"])
            except:
                pass
        if os.path.exists(MS_MODEL_PATH):
            try:
                d = torch.load(MS_MODEL_PATH,map_location=self.device)
                if "backbone" in d:
                    self.backbone_ms.load_state_dict(d["backbone"],strict=False)
                if "head" in d:
                    self.ms_head.load_state_dict(d["head"],strict=False)
                if "ema_backbone" in d:
                    self.ema_ms_backbone.m.load_state_dict(d["ema_backbone"],strict=False)
                if "ema_head" in d:
                    self.ema_ms_head.m.load_state_dict(d["ema_head"],strict=False)
                else:
                    self.ema_ms_backbone.update(self.backbone_ms)
                    self.ema_ms_head.update(self.ms_head)
                if "seq_len" in d and int(d["seq_len"])>0:
                    self.seq_len = int(d["seq_len"])
            except:
                pass

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_facecolor("#111111")
        self.fig.patch.set_facecolor("#111111")
        self.line_reward, = self.ax1.plot([],[],linewidth=1.5,color="#2d8cff")
        self.line_loss, = self.ax1.plot([],[],linewidth=1.5,color="#ff6b6b")
        self.ax1.tick_params(colors="#bbbbbb")
        for s in ['bottom','top','left','right']:
            self.ax1.spines[s].set_color("#444444")
        self.ax1.set_title("Reward/Loss",color="#dddddd")
    def update_data(self, rewards, losses):
        xr = list(range(len(rewards)))
        xl = list(range(len(losses)))
        self.line_reward.set_data(xr, rewards)
        self.line_loss.set_data(xl, losses)
        if rewards or losses:
            ymin = min([-1.0]+rewards+losses)
            ymax = max([1.0]+rewards+losses)
        else:
            ymin, ymax = -1.0, 1.0
        if ymin==ymax:
            ymax=ymin+1.0
        self.ax1.set_ylim(ymin-0.1, ymax+0.1)
        xmax = max(10,len(xr),len(xl))
        self.ax1.set_xlim(0, xmax)
        self.draw_idle()

class AppWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("通用游戏AI")
        self.setMinimumSize(1000,700)
        self.setStyleSheet("""
            QWidget { background-color: #0f0f10; color: #dddddd; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
            QComboBox, QPushButton, QLabel, QTextEdit { background-color: #161617; border-radius: 8px; padding: 6px; }
            QComboBox::drop-down { border: none; }
            QPushButton { background-color: #1e1e20; }
            QPushButton:hover { background-color: #27272a; }
            QProgressBar { background-color:#1e1e20; border: 1px solid #2a2a2d; border-radius:8px; text-align:center; }
            QProgressBar::chunk { background-color:#2d8cff; border-radius:8px; }
            QGroupBox { border: 1px solid #2a2a2d; border-radius: 12px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 3px; }
        """)
        self.device, self.mem_free, self.mem_total = torch_device_and_mem()
        self.width_mult, self.depth_mult = adaptive_dims(self.mem_total)
        self.target_w, self.target_h = adaptive_resolution(self.mem_total)
        self.temp_k = 1.0
        self.temp_m = 1.0
        self.mode = "学习模式"
        self.state = "闲置"
        self.optimizing = False
        self.running = False
        self.episode_id = None
        self.current_episode = []
        self.recent_rewards = deque(maxlen=100)
        self.recent_losses = deque(maxlen=200)
        self.winloss = deque(maxlen=50)
        self.infer_latency_ms = 0.0
        self.sample_rate_hz = 0.0
        self.preview_img = None
        self.frame_buf = deque(maxlen=adaptive_seq_len(self.mem_total))
        self.grabber = ScreenGrabber((self.target_w,self.target_h))
        self.grabber.frame_ready.connect(self.on_frame)
        self.io = InputIO()
        self.io.action_ready.connect(self.on_user_action)
        self.replay = ReplayBuffer()
        self.learner = Learner(self.device, self.mem_total)
        self.learner.load_models()
        self.frame_buf = deque(maxlen=self.learner.seq_len)
        self.action_queue = queue.Queue(maxsize=8)
        self.infer_thread = None
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.refresh_status)
        self.status_timer.start(200)
        self.training_thread = None
        self.training_lock = threading.Lock()
        self.log_queue = deque()
        self.last_pool_wrap = 0
        self.click_threshold = 0.8
        self.last_learn_rec = 0.0
        self.build_ui()
        self.install_global_esc()
        self.append_log("程序已启动")
        self.check_files()
        self.bootstrap_stats_from_index()
        self.set_state("闲置")
    def build_ui(self):
        top_bar = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["训练模式","学习模式"])
        self.mode_combo.setCurrentText("学习模式")
        self.start_btn = QtWidgets.QPushButton("开始")
        self.stop_btn = QtWidgets.QPushButton("停止")
        self.opt_btn = QtWidgets.QPushButton("优化AI")
        self.stop_btn.setEnabled(False)
        top_layout.addWidget(self.mode_combo)
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.stop_btn)
        top_layout.addWidget(self.opt_btn)
        top_layout.addStretch(1)
        self.status_light = QtWidgets.QFrame()
        self.status_light.setFixedSize(14,14)
        self.status_light.setStyleSheet("QFrame { background-color:#ffaa00; border-radius:7px; }")
        self.status_label = QtWidgets.QLabel("闲置")
        top_layout.addWidget(self.status_light)
        top_layout.addWidget(self.status_label)
        center = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setMinimumSize(320,180)
        self.preview_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.preview_label)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        info_group = QtWidgets.QGroupBox("统计")
        form = QtWidgets.QFormLayout(info_group)
        self.lbl_recent_rewards = QtWidgets.QLabel("-")
        self.lbl_winrate = QtWidgets.QLabel("-")
        self.lbl_lr = QtWidgets.QLabel("-")
        self.lbl_loss = QtWidgets.QLabel("-")
        self.lbl_latency = QtWidgets.QLabel("-")
        form.addRow("最近奖励均值", self.lbl_recent_rewards)
        form.addRow("胜负比", self.lbl_winrate)
        form.addRow("当前学习率", self.lbl_lr)
        form.addRow("损失", self.lbl_loss)
        form.addRow("推理延迟", self.lbl_latency)
        right_layout.addWidget(info_group)
        center.addWidget(left)
        center.addWidget(right)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.canvas = MplCanvas()
        bottom.addWidget(self.canvas)
        self.log_edit = QtWidgets.QTextEdit()
        self.log_edit.setReadOnly(True)
        bottom.addWidget(self.log_edit)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(top_bar)
        inner = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        inner.addWidget(center)
        inner.addWidget(bottom)
        splitter.addWidget(inner)
        cw = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(cw)
        layout.addWidget(splitter)
        self.setCentralWidget(cw)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.opt_btn.clicked.connect(self.on_optimize_clicked)
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
    def set_state(self, state):
        self.state = state
        if state=="闲置":
            color = "#ffaa00"
        elif state=="学习模式":
            color = "#2d8cff"
        elif state=="训练模式":
            color = "#10c24a"
        else:
            color = "#a070ff"
        self.status_light.setStyleSheet(f"QFrame {{ background-color:{color}; border-radius:7px; }}")
        if state=="闲置":
            self.start_btn.setEnabled(True and not self.optimizing)
            self.stop_btn.setEnabled(False)
            self.opt_btn.setEnabled(True and not self.optimizing)
        elif state in ["学习模式","训练模式"]:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.opt_btn.setEnabled(False)
        elif state=="AI优化中":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.opt_btn.setEnabled(False)
        self.status_label.setText(state)
    def install_global_esc(self):
        if pynput_keyboard is None:
            return
        def on_press(k):
            try:
                if k==pynput_keyboard.Key.esc:
                    QtCore.QMetaObject.invokeMethod(self, "open_end_dialog", QtCore.Qt.QueuedConnection)
            except:
                pass
        t = threading.Thread(target=lambda: pynput_keyboard.Listener(on_press=on_press).run(),daemon=True)
        t.start()
    @QtCore.Slot()
    def open_end_dialog(self):
        if not self.running and not self.current_episode:
            return
        try:
            self.on_stop()
        except:
            pass
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("结束标记")
        v = QtWidgets.QVBoxLayout(dlg)
        group = QtWidgets.QButtonGroup(dlg)
        rb1 = QtWidgets.QRadioButton("赢")
        rb2 = QtWidgets.QRadioButton("输")
        rb3 = QtWidgets.QRadioButton("平局")
        rb4 = QtWidgets.QRadioButton("跳过")
        group.addButton(rb1,1)
        group.addButton(rb2,2)
        group.addButton(rb3,3)
        group.addButton(rb4,4)
        rb3.setChecked(True)
        v.addWidget(rb1); v.addWidget(rb2); v.addWidget(rb3); v.addWidget(rb4)
        btn = QtWidgets.QPushButton("确定")
        v.addWidget(btn)
        def ok():
            id_ = group.checkedId()
            label = {1:"赢",2:"输",3:"平局",4:"跳过"}.get(id_,"跳过")
            end_episode_finalize(label)
            dlg.accept()
        btn.clicked.connect(ok)
        dlg.setModal(True)
        dlg.exec_() if not PYSIDE else dlg.exec()
    def check_files(self):
        self.replay.ensure_files()
        self.append_log("文件已检查")
    def append_log(self, text):
        t = time.strftime("%H:%M:%S")
        self.log_edit.append(f"[{t}] {text}")
    def thread_log(self, text):
        self.log_queue.append(text)
    def on_mode_change(self, text):
        if self.running:
            self.on_stop()
        self.mode = text
        self.io.set_mode(text)
        if self.running:
            self.set_state(self.mode)
    def on_start(self):
        if self.running or self.optimizing:
            return
        self.running = True
        self.episode_id = str(uuid.uuid4())
        self.current_episode = []
        self.append_log(f"开始 {self.mode}")
        self.frame_buf.clear()
        self.grabber.start()
        if self.mode=="学习模式":
            self.io.set_mode("学习模式")
            self.io.start_listen()
        else:
            self.io.set_mode("训练模式")
            self.io.start_output()
        self.infer_thread = threading.Thread(target=self.infer_loop,daemon=True)
        self.infer_thread.start()
        self.set_state(self.mode)
    def on_stop(self):
        self.running = False
        try:
            self.grabber.stop()
        except:
            pass
        try:
            self.io.stop_listen()
        except:
            pass
        try:
            self.io.stop_output()
        except:
            pass
        try:
            if self.infer_thread:
                self.infer_thread.join(timeout=1.5)
        except:
            pass
        self.infer_thread = None
        try:
            self.replay.close()
        except:
            pass
        self.set_state("闲置")
        self.append_log("已停止")
    def on_optimize_clicked(self):
        if self.running or self.optimizing:
            return
        self.thread_log("开始优化")
        self.optimizing = True
        self.set_state("AI优化中")
        self.start_optimize_thread(None, None)
    def on_frame(self, frame):
        self.preview_img = frame
        self.frame_buf.append(frame)
        try:
            self.action_queue.put_nowait(("frame",frame))
        except:
            pass
        if self.mode=="学习模式" and self.running:
            now = time.time()
            if now - self.last_learn_rec >= max(0.01, self.io.sample_rate_s):
                act = {"keys":list(self.io.pressed_keys), "move":(0.0,0.0), "clicks":{}, "scroll":0.0}
                self.current_episode.append((frame.copy(), {"action":act}))
                try:
                    self.replay.write_sample(frame, act, "unlabeled", self.episode_id, None)
                except:
                    pass
                self.last_learn_rec = now
        self.update_preview(frame)
    def on_user_action(self, data):
        if not self.running:
            return
        if data.get("esc"):
            return
        if self.mode=="学习模式":
            f = self.preview_img
            if f is not None:
                act = {"keys":list(self.io.pressed_keys), "move":data.get("mouse",(0,0)), "clicks":{}, "scroll":data.get("scroll",0.0)}
                if "click" in data:
                    btn,prs = data["click"]
                    act["clicks"][btn]=1.0 if prs else 0.0
                self.current_episode.append((f.copy(), {"action":act}))
                try:
                    self.replay.write_sample(f, act, "unlabeled", self.episode_id, None)
                except:
                    pass
    def infer_loop(self):
        last = time.time()
        while self.running:
            try:
                item = self.action_queue.get(timeout=0.1)
            except:
                continue
            if item[0]!="frame":
                continue
            t0 = time.time()
            if self.mode=="训练模式" and len(self.frame_buf)>=self.learner.seq_len:
                seq = list(self.frame_buf)[-self.learner.seq_len:]
                res = self.learner.infer(seq, self.temp_k, self.temp_m)
                self.apply_nms_and_send(res)
                self.current_episode.append((seq[-1], {"action":res}))
            t1 = time.time()
            self.infer_latency_ms = (t1-t0)*1000.0
            dt = t1 - last
            if dt>0:
                self.sample_rate_hz = 1.0/dt
            last = t1
            if len(self.recent_rewards)>=10 and len(self.recent_losses)>=10:
                r = np.mean(self.recent_rewards)
                ltrend = np.mean(np.diff(list(self.recent_losses))) if len(self.recent_losses)>2 else 0.0
                self.temp_k = max(0.5, min(1.5, 1.0 + (-r*0.3) + (-ltrend*0.1)))
                self.temp_m = max(0.5, min(1.5, 1.0 + (-r*0.2)))
    def apply_nms_and_send(self, act):
        keys = act["keys"]
        clicks = act["clicks"]
        move = act["move"]
        if "UP" in keys and "DOWN" in keys:
            if random.random()<0.5:
                keys.remove("DOWN")
            else:
                keys.remove("UP")
        if "LEFT" in keys and "RIGHT" in keys:
            if random.random()<0.5:
                keys.remove("RIGHT")
            else:
                keys.remove("LEFT")
        for k in list(keys):
            if k=="ESC":
                keys.remove(k)
        thr = float(self.click_threshold)
        for b in ["LEFT","RIGHT","MIDDLE"]:
            clicks[b] = 1.0 if clicks.get(b,0.0)>thr else 0.0
        act["keys"]=keys
        act["clicks"]=clicks
        self.io.send_action(act)
    def end_episode(self, label):
        eps = self.current_episode
        self.current_episode = []
        eid = self.episode_id
        self.episode_id = str(uuid.uuid4())
        rw = {"赢":1.0,"输":-1.0,"平局":0.1,"跳过":0.0}.get(label,0.0)
        self.recent_rewards.append(rw)
        if label in ["赢","输"]:
            self.winloss.append(1 if label=="赢" else 0)
        for f, meta in eps:
            try:
                lab = "unlabeled" if label=="跳过" else label
                self.replay.write_sample(f, meta.get("action",{}), lab, eid, rw if label!="跳过" else None)
            except:
                pass
        return eps, rw
    def start_optimize_thread(self, eps=None, rw=None):
        if self.training_thread and self.training_thread.is_alive():
            self.thread_log("优化进行中，已排队")
            return
        def worker():
            try:
                occ, fs, wp = self.replay.occupancy()
                if occ<=0 and not eps:
                    self.thread_log("经验池为空，跳过优化")
                    return
                try:
                    free_now,total_now = torch.cuda.mem_get_info() if self.device.type=="cuda" else (0,0)
                except:
                    free_now = self.mem_free if self.mem_free>0 else (6*1024*1024*1024)
                mem_free_now = free_now if free_now>0 else (6*1024*1024*1024)
                ep_len = len(eps) if eps else min(occ, 640)
                b, steps = adaptive_batch_steps(mem_free_now, self.infer_latency_ms, max(1, ep_len), self.learner.seq_len)
                b_curr = max(1,b)
                ltrend = 0.0
                if len(self.learner.loss_hist)>10:
                    v = list(self.learner.loss_hist)
                    ltrend = (v[-1]-v[0])/max(1,len(v))
                self.learner.set_dynamic_lrs(self.mem_total, ltrend, max(1,steps))
                for i in range(max(1,steps)):
                    sub = []
                    try:
                        if eps and len(eps)>0:
                            L = len(eps)
                            for j in range(b_curr):
                                idx = (i*b_curr+j) % L
                                seq = []
                                for k in range(self.learner.seq_len):
                                    ii = max(0, idx - (self.learner.seq_len-1-k))
                                    seq.append(eps[ii][0])
                                sub.append((seq, {"action":eps[idx][1].get("action",{})}))
                        else:
                            batch_items = self.replay.sample_batch_seq(b_curr, self.learner.seq_len)
                            for imgs_seq, meta in batch_items:
                                sub.append((imgs_seq, meta))
                    except:
                        sub = []
                    if not sub:
                        if b_curr>1:
                            b_curr = max(1, b_curr//2)
                        continue
                    try:
                        loss, lr_now = self.learner.optimize_once(sub, reward_weight=max(0.1, abs(rw)) if rw is not None else 1.0)
                        if loss>0:
                            self.recent_losses.append(loss)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            try:
                                torch.cuda.empty_cache()
                            except:
                                pass
                            b_curr = max(1, b_curr//2)
                            self.thread_log(f"显存不足，自动降批次至 {b_curr}")
                            continue
                        else:
                            raise
                self.learner.save_models()
                self.replay.save_all()
                try:
                    cfg = {"last_run":now_ts(),"last_lr":(self.learner.sched_kb.get_last_lr()[0] if self.learner.sched_kb is not None else self.learner.lr_display),"device":str(self.device),"wm":self.width_mult,"dm":self.depth_mult,"seq_len":self.learner.seq_len}
                    atomic_write_json(CONFIG_PATH,cfg)
                except:
                    pass
                self.thread_log("优化完成")
            except:
                self.thread_log("优化异常")
            finally:
                self.optimizing = False
        self.training_thread = threading.Thread(target=worker,daemon=True)
        self.training_thread.start()
    def refresh_status(self):
        while self.log_queue:
            self.append_log(self.log_queue.popleft())
        if self.optimizing:
            if self.state!="AI优化中":
                self.set_state("AI优化中")
        else:
            if self.running and self.state!=self.mode:
                self.set_state(self.mode)
            if (not self.running) and self.state!="闲置":
                self.set_state("闲置")
        occ, fs, wp = self.replay.occupancy()
        occ_gb = (fs/POOL_LIMIT)*100 if POOL_LIMIT>0 else 0
        if self.device.type=="cuda":
            try:
                free,total = torch.cuda.mem_get_info()
                used = total - free
                self.mem_free = free
                self.mem_total = total
                vramtxt = f"{used//(1024*1024)}MB/{total//(1024*1024)}MB"
            except:
                vramtxt = "N/A"
        else:
            vramtxt = "N/A"
        rused, rtot = system_memory()
        ramtxt = f"{rused//(1024*1024)}MB/{rtot//(1024*1024)}MB" if rtot>0 else "N/A"
        fps = np.mean(self.grabber.fps_smooth) if self.grabber.fps_smooth else 0.0
        devtxt = "GPU" if self.device.type=="cuda" else "CPU"
        self.status_label.setText(f"{self.state} | 帧率:{fps:.1f} | 采样:{self.sample_rate_hz:.1f}Hz | 设备:{devtxt} | 内存:{ramtxt} | 显存:{vramtxt} | 经验池:{occ_gb:.1f}%")
        fallback_rmean, fallback_wr = self.compute_persistent_metrics(30)
        rmean = np.mean(self.recent_rewards) if self.recent_rewards else fallback_rmean
        self.lbl_recent_rewards.setText(f"{rmean:.3f}")
        wr = (np.mean(self.winloss) if self.winloss else fallback_wr)*100.0
        self.lbl_winrate.setText(f"{wr:.1f}%")
        try:
            self.lbl_lr.setText(f"{self.learner.sched_kb.get_last_lr()[0]:.6f}")
        except:
            self.lbl_lr.setText(f"{self.learner.lr_display:.6f}")
        l = self.recent_losses[-1] if self.recent_losses else 0.0
        self.lbl_loss.setText(f"{l:.5f}")
        self.lbl_latency.setText(f"{self.infer_latency_ms:.1f}ms")
        self.canvas.update_data(list(self.recent_rewards) if self.recent_rewards else [fallback_rmean], list(self.recent_losses))
        target_w, target_h = adaptive_resolution(self.mem_total)
        if abs(target_w-self.target_w)>31 or abs(target_h-self.target_h)>31:
            self.target_w, self.target_h = target_w, target_h
            self.grabber.set_target_size((self.target_w,self.target_h))
        fps_target = 90 if self.device.type=="cuda" else 60
        if self.infer_latency_ms>60:
            fps_target = max(30, fps_target-20)
        self.grabber.set_max_fps(fps_target)
        loss_trend = 0.0
        if len(self.learner.loss_hist)>10:
            v = list(self.learner.loss_hist)
            loss_trend = (v[-1]-v[0])/max(1,len(v))
        aug = 0.05 + 0.25*max(0.0, min(1.0, 0.5 - loss_trend))
        if self.mode=="学习模式":
            aug *= 0.5
        self.grabber.set_aug_strength(aug)
        dyn_sr = max(0.008, min(0.06, (0.02 if self.device.type=="cuda" else 0.03)*(1.0 + (self.infer_latency_ms-20.0)/100.0)))
        self.io.update_hparams(sample_rate_s=dyn_sr, min_interval_ms=int(dyn_sr*1000*0.75), mouse_speed_limit=1200 if self.device.type=="cuda" else 900, mouse_accel=0.6 if self.device.type=="cuda" else 0.5)
        self.click_threshold = max(0.6, min(0.9, 0.8 + (self.infer_latency_ms-25.0)/200.0))
        if wp<self.last_pool_wrap:
            self.append_log("经验池裁剪")
        self.last_pool_wrap = wp
        if self.training_thread and (not self.training_thread.is_alive()) and self.state=="AI优化中":
            self.set_state("闲置")
    def update_preview(self, frame):
        h,w = frame.shape[:2]
        qimg = QtGui.QImage(frame.data, w, h, 3*w, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.width(), self.preview_label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)
    def closeEvent(self, e):
        try:
            self.on_stop()
        except:
            pass
        try:
            self.learner.save_models()
            self.replay.save_all()
        except:
            pass
        e.accept()
    def extract_recent_episode_labels(self, max_n):
        labs = []
        seen = set()
        entries = list(self.replay.index.get("entries",[]))
        for e in reversed(entries):
            ep = e.get("ep")
            lb = e.get("label","unlabeled")
            if ep in seen:
                continue
            if lb!="unlabeled":
                labs.append(lb)
                seen.add(ep)
                if len(labs)>=max_n:
                    break
        return list(reversed(labs))
    def bootstrap_stats_from_index(self):
        labs = self.extract_recent_episode_labels(50)
        for lb in labs:
            rw = {"赢":1.0,"输":-1.0,"平局":0.1,"跳过":0.0}.get(lb,0.0)
            self.recent_rewards.append(rw)
            if lb in ["赢","输"]:
                self.winloss.append(1 if lb=="赢" else 0)
    def compute_persistent_metrics(self, max_n):
        labs = self.extract_recent_episode_labels(max_n)
        if not labs:
            return 0.0, 0.0
        vals = [ {"赢":1.0,"输":-1.0,"平局":0.1,"跳过":0.0}.get(lb,0.0) for lb in labs ]
        rmean = float(np.mean(vals)) if vals else 0.0
        wins = sum(1 for lb in labs if lb=="赢")
        losses = sum(1 for lb in labs if lb=="输")
        denom = wins+losses
        wr = (wins/denom) if denom>0 else 0.0
        return rmean, wr

def start_training_mode():
    if QtWidgets.QApplication.instance() and isinstance(QtWidgets.QApplication.instance().activeWindow(), AppWindow):
        w = QtWidgets.QApplication.instance().activeWindow()
        w.mode_combo.setCurrentText("训练模式")
        w.on_start()

def start_learning_mode():
    if QtWidgets.QApplication.instance() and isinstance(QtWidgets.QApplication.instance().activeWindow(), AppWindow):
        w = QtWidgets.QApplication.instance().activeWindow()
        w.mode_combo.setCurrentText("学习模式")
        w.on_start()

def end_episode_finalize(result_label):
    if QtWidgets.QApplication.instance() and isinstance(QtWidgets.QApplication.instance().activeWindow(), AppWindow):
        w = QtWidgets.QApplication.instance().activeWindow()
        w.end_episode(result_label)
        if not w.running:
            w.set_state("闲置")

def save_all():
    if QtWidgets.QApplication.instance() and isinstance(QtWidgets.QApplication.instance().activeWindow(), AppWindow):
        w = QtWidgets.QApplication.instance().activeWindow()
        w.learner.save_models()
        w.replay.save_all()
        w.append_log("已保存")

def load_all():
    if QtWidgets.QApplication.instance() and isinstance(QtWidgets.QApplication.instance().activeWindow(), AppWindow):
        w = QtWidgets.QApplication.instance().activeWindow()
        w.learner.load_models()
        w.append_log("已加载")

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_() if not PYSIDE else app.exec())

if __name__ == '__main__':
    main()

