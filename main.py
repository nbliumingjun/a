import os,sys,ctypes,ctypes.wintypes,threading,time,uuid,json,random,collections,hashlib,importlib,glob,shutil,math,zlib
from pathlib import Path
os.environ["QT_ENABLE_HIGHDPI_SCALING"]="1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"]="1"
os.environ.setdefault("QT_LOGGING_RULES","qt.qpa.window=false")
try:
    import importlib.metadata as _im
except:
    _im=None
CSIDL_DESKTOPDIRECTORY=0x0010
_SHGetFolderPath=ctypes.windll.shell32.SHGetFolderPathW
_SHGetFolderPath.argtypes=[ctypes.wintypes.HWND,ctypes.c_int,ctypes.wintypes.HANDLE,ctypes.wintypes.DWORD,ctypes.wintypes.LPWSTR]
def _desktop_path():
    buf=ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    if _SHGetFolderPath(0,CSIDL_DESKTOPDIRECTORY,0,0,buf)==0:
        return buf.value
    return str(Path.home()/"Desktop")
home=_desktop_path()
base_dir=os.path.join(home,"AAA")
exp_dir=os.path.join(base_dir,"experience")
models_dir=os.path.join(base_dir,"models")
logs_dir=os.path.join(base_dir,"logs")
cfg_path=os.path.join(base_dir,"config.json")
deps_path=os.path.join(base_dir,"deps.json")
meta_path=os.path.join(models_dir,"model_meta.json")
cfg_defaults={"idle_timeout":10,"screenshot_min_fps":1,"screenshot_max_fps":120,"ui_scale":1.0,"frame_png_compress":3,"save_change_thresh":6.0,"max_disk_gb":10.0,"pre_post_K":3,"block_margin_px":6,"preview_on":True}
os.makedirs(base_dir,exist_ok=True)
os.makedirs(exp_dir,exist_ok=True)
os.makedirs(models_dir,exist_ok=True)
os.makedirs(logs_dir,exist_ok=True)
def ensure_config():
    if not os.path.exists(cfg_path):
        with open(cfg_path,"w",encoding="utf-8") as f:
            json.dump(cfg_defaults,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
ensure_config()
def _msgbox(t,m):
    try:
        ctypes.windll.user32.MessageBoxW(0,str(m),str(t),0)
    except:
        pass
def _ver(name,modname=None):
    try:
        import importlib.metadata as im
        return im.version(modname or name)
    except:
        try:
            import pkg_resources as pr
            return pr.get_distribution(modname or name).version
        except:
            return ""
def deps_check():
    core=["numpy","cv2","psutil","mss","PySide6","win32gui","win32con","win32api","win32process"]
    miss=[]
    for n in core:
        if importlib.util.find_spec(n) is None:
            miss.append(n)
    if miss:
        _msgbox("依赖缺失","核心依赖缺失:\n"+("\n".join(miss)))
        sys.exit(1)
    d={"numpy":_ver("numpy"),"opencv-python":_ver("opencv-python","opencv-python"),"psutil":_ver("psutil"),"mss":_ver("mss"),"PySide6":_ver("PySide6"),"pywin32":_ver("pywin32","pywin32")}
    try:
        d["torch"]=_ver("torch")
    except:
        pass
    try:
        d["scikit-learn"]=_ver("scikit-learn","scikit-learn")
    except:
        pass
    try:
        d["pynvml"]=_ver("pynvml")
    except:
        pass
    with open(deps_path,"w",encoding="utf-8") as f:
        json.dump(d,f,ensure_ascii=False,indent=2)
        f.flush()
        os.fsync(f.fileno())
deps_check()
import numpy as np
import cv2
import psutil
import mss as mss_mod
import win32gui,win32con,win32api,win32process
try:
    import pynvml as _nv
    _nv.nvmlInit()
except:
    _nv=None
def _gpu_util_mem():
    try:
        if _nv is None:
            return None,None
        h=_nv.nvmlDeviceGetHandleByIndex(0)
        u=_nv.nvmlDeviceGetUtilizationRates(h)
        m=_nv.nvmlDeviceGetMemoryInfo(h)
        gu=int(u.gpu)
        gp=float(m.used*100.0/max(1,m.total))
        return gu,gp
    except:
        return None,None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except:
    torch=None
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QComboBox,QLabel,QCheckBox,QMessageBox,QProgressBar
from PySide6.QtCore import QTimer,Qt,Signal,QObject,QSize
from PySide6.QtGui import QImage,QPixmap
log_path=os.path.join(logs_dir,"app.log")
def log(s):
    try:
        with open(log_path,"a",encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")
    except:
        pass
if not os.path.exists(models_dir) or len(os.listdir(models_dir))==0:
    _msgbox("模型提示","models 目录不存在或为空")
def today_dir():
    d=time.strftime("%Y%m%d")
    td=os.path.join(exp_dir,d)
    os.makedirs(td,exist_ok=True)
    os.makedirs(os.path.join(td,"frames"),exist_ok=True)
    return td
def sanitize_name(t):
    import re
    x=re.sub(r"[^a-zA-Z0-9._ -]","_",t)[:80]
    if len(x)==0:
        x="default"
    return x
def model_path_for(title):
    return os.path.join(models_dir,f"model_{sanitize_name(title)}.npz")
def default_model_path():
    return os.path.join(models_dir,"model_default.npz")
def _dpi_for_hwnd(hwnd):
    try:
        gdfw=getattr(ctypes.windll.user32,"GetDpiForWindow")
        if gdfw:
            return int(gdfw(hwnd)),int(gdfw(hwnd))
    except:
        pass
    try:
        hdc=ctypes.windll.user32.GetDC(hwnd or 0)
        LOGPIXELSX=88
        LOGPIXELSY=90
        x=ctypes.windll.gdi32.GetDeviceCaps(hdc,LOGPIXELSX)
        y=ctypes.windll.gdi32.GetDeviceCaps(hdc,LOGPIXELSY)
        ctypes.windll.user32.ReleaseDC(hwnd or 0,hdc)
        return int(x),int(y)
    except:
        return 96,96
class JSONLWriter:
    def __init__(self,path):
        self.path=path
        self.f=open(self.path,"a",encoding="utf-8")
        self.lock=threading.Lock()
    @staticmethod
    def _crc_of(obj):
        if isinstance(obj,dict) and "crc" in obj:
            o=dict(obj)
            del o["crc"]
        else:
            o=obj
        s=json.dumps(o,ensure_ascii=False,separators=(",",":"))
        return int(zlib.adler32(s.encode("utf-8"))&0xFFFFFFFF)
    @staticmethod
    def repair(path):
        if not os.path.exists(path):
            return
        try:
            okoff=0
            off=0
            with open(path,"rb") as f:
                for line in f:
                    try:
                        off+=len(line)
                        s=line.decode("utf-8")
                        obj=json.loads(s)
                        c=obj.get("crc",None)
                        if c is not None and int(c)!=JSONLWriter._crc_of(obj):
                            break
                        okoff=off
                    except:
                        break
            with open(path,"rb+") as f:
                f.truncate(okoff)
        except:
            pass
    def append(self,obj):
        try:
            if isinstance(obj,dict):
                o=dict(obj)
                o["crc"]=JSONLWriter._crc_of(o)
            else:
                o=obj
            line=json.dumps(o,ensure_ascii=False,separators=(",",":"))+"\n"
            for _ in range(3):
                try:
                    with self.lock:
                        self.f.write(line)
                        self.f.flush()
                        os.fsync(self.f.fileno())
                    return True
                except:
                    time.sleep(0.02)
            return False
        except:
            return False
    def close(self):
        try:
            with self.lock:
                try:
                    self.f.flush()
                except:
                    pass
                try:
                    os.fsync(self.f.fileno())
                except:
                    pass
                self.f.close()
        except:
            pass
class DiskManager:
    @staticmethod
    def total_bytes():
        t=0
        for r,_,fs in os.walk(exp_dir):
            for n in fs:
                try:
                    t+=os.path.getsize(os.path.join(r,n))
                except:
                    pass
        return t
    @staticmethod
    def enforce(max_gb):
        cap=max_gb*1024*1024*1024
        t=DiskManager.total_bytes()
        if t<=cap:
            return
        subs=sorted([p for p in glob.glob(os.path.join(exp_dir,"*")) if os.path.isdir(p)])
        for d in subs:
            try:
                shutil.rmtree(d,ignore_errors=True)
            except:
                pass
            t=DiskManager.total_bytes()
            if t<=cap*0.85:
                break
class ClickModel:
    def __init__(self):
        self.W1=None
        self.b1=None
        self.W2=None
        self.b2=None
        self.W3=None
        self.b3=None
        self.mean=None
        self.std=None
        self.in_size=32*32
    def save(self,path):
        np.savez_compressed(path,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,W3=self.W3,b3=self.b3,mean=self.mean,std=self.std,in_size=self.in_size)
    def load(self,path):
        d=np.load(path,allow_pickle=True)
        self.W1=d["W1"]
        self.b1=d["b1"]
        self.W2=d["W2"]
        self.b2=d["b2"]
        self.mean=d["mean"]
        self.std=d["std"]
        self.in_size=int(d["in_size"])
        self.W3=d["W3"] if "W3" in d else np.random.randn(self.W2.shape[1],1).astype(np.float32)*0.05
        self.b3=d["b3"] if "b3" in d else np.zeros((1,),np.float32)
    def patch(self,img,pt,sz):
        x,y=pt
        x=max(0,min(img.shape[1]-1,int(x)))
        y=max(0,min(img.shape[0]-1,int(y)))
        hs=sz//2
        x1=x-hs
        y1=y-hs
        x2=x1+sz
        y2=y1+sz
        if x1<0 or y1<0 or x2>img.shape[1] or y2>img.shape[0]:
            dx1=max(0,-x1)
            dy1=max(0,-y1)
            dx2=max(0,x2-img.shape[1])
            dy2=max(0,y2-img.shape[0])
            x1=max(0,x1)
            y1=max(0,y1)
            x2=min(img.shape[1],x2)
            y2=min(img.shape[0],y2)
            crop=img[y1:y2,x1:x2]
            pad=cv2.copyMakeBorder(crop,dy1,dy2,dx1,dx2,cv2.BORDER_REFLECT_101)
        else:
            pad=img[y1:y2,x1:x2]
        g=cv2.cvtColor(pad,cv2.COLOR_BGR2GRAY)
        g=cv2.resize(g,(sz,sz))
        return g
    def score_points(self,img,cands,patch=32):
        if self.W1 is None:
            return None
        X=[]
        for (x,y) in cands:
            X.append(self.patch(img,(x,y),patch).reshape(-1)/255.0)
        X=np.array(X,dtype=np.float32)
        X=(X-self.mean)/self.std
        h1=np.maximum(0,X@self.W1+self.b1)
        h2=np.maximum(0,h1@self.W2+self.b2)
        o=h2@self.W3+self.b3
        p=1/(1+np.exp(-o))
        return p.ravel()
    def heatmap(self,img,patch=32,stride=16):
        if self.W1 is None:
            return None
        h,w=img.shape[:2]
        xs=list(range(patch//2,w-patch//2,stride))
        ys=list(range(patch//2,h-patch//2,stride))
        pts=[]
        for yy in ys:
            for xx in xs:
                pts.append((xx,yy))
        sc=self.score_points(img,pts,patch)
        if sc is None or len(sc)==0:
            return None
        m=np.zeros((h,w),dtype=np.float32)
        k=0
        for _y in ys:
            for _x in xs:
                v=sc[k]
                k+=1
                m[_y,_x]=v
        m=cv2.GaussianBlur(m,(0,0),3)
        mx=m.max()
        if mx>0:
            m/=mx
        return m
    def fit_from_logs(self,events,frames,max_samples=80000,neg_ratio=4,patch=32,where_title=None):
        X=[]
        Y=[]
        for e in reversed(events[-max_samples:]):
            if e.get("source")!="user":
                continue
            if e.get("type") not in ["left","right","middle"]:
                continue
            if where_title and e.get("window_title")!=where_title:
                continue
            fid=e.get("frame_id")
            fr=frames.get(fid)
            if not fr:
                continue
            img=cv2.imdecode(np.fromfile(fr["path"],dtype=np.uint8),cv2.IMREAD_COLOR)
            if img is None:
                continue
            pos=[]
            if e.get("ins_press")==1:
                pos.append((int(e.get("press_lx")),int(e.get("press_ly"))))
            if e.get("ins_release")==1:
                pos.append((int(e.get("release_lx")),int(e.get("release_ly"))))
            mv=e.get("moves") or []
            for it in mv:
                if isinstance(it,(list,tuple)) and len(it)>=4 and int(it[3])==1:
                    pos.append((int(it[1]),int(it[2])))
            used=set()
            pset=[]
            for (x,y) in pos:
                x=max(0,min(img.shape[1]-1,x))
                y=max(0,min(img.shape[0]-1,y))
                if (x,y) in used:
                    continue
                used.add((x,y))
                pset.append((x,y))
            for (x,y) in pset:
                ph=self.patch(img,(x,y),patch)
                if ph is None:
                    continue
                X.append(ph)
                Y.append(1)
                for _ in range(neg_ratio):
                    for _k in range(8):
                        nx=random.randint(0,img.shape[1]-1)
                        ny=random.randint(0,img.shape[0]-1)
                        ok=True
                        for (px,py) in pset:
                            if abs(px-nx)+abs(py-ny)<24:
                                ok=False
                                break
                        if ok:
                            break
                    phn=self.patch(img,(nx,ny),patch)
                    if phn is not None:
                        X.append(phn)
                        Y.append(0)
        if len(X)<50:
            return False,(0,0)
        X=np.array(X,dtype=np.float32).reshape(len(X),-1)/255.0
        self.mean=X.mean(0)
        self.std=X.std(0)+1e-6
        Xn=(X-self.mean)/self.std
        ok=False
        try:
            from sklearn.neural_network import MLPClassifier
            clf=MLPClassifier(hidden_layer_sizes=(384,192),max_iter=400,learning_rate_init=0.0006,solver="adam",random_state=0,verbose=False)
            clf.fit(Xn,Y)
            W1=clf.coefs_[0]
            b1=clf.intercepts_[0]
            W2=clf.coefs_[1]
            b2=clf.intercepts_[1]
            W3=clf.coefs_[2]
            b3=clf.intercepts_[2]
            self.W1=W1.astype(np.float32)
            self.b1=b1.astype(np.float32)
            self.W2=W2.astype(np.float32)
            self.b2=b2.astype(np.float32)
            self.W3=W3.astype(np.float32)
            self.b3=b3.astype(np.float32)
            ok=True
        except:
            pass
        if not ok:
            n_in=Xn.shape[1]
            n_h1=384
            n_h2=192
            W1=np.random.randn(n_in,n_h1).astype(np.float32)*0.05
            b1=np.zeros((n_h1,),np.float32)
            W2=np.random.randn(n_h1,n_h2).astype(np.float32)*0.05
            b2=np.zeros((n_h2,),np.float32)
            W3=np.random.randn(n_h2,1).astype(np.float32)*0.05
            b3=np.zeros((1,),np.float32)
            lr=0.0006
            for _ in range(260):
                bs=256
                for i in range(0,len(Xn),bs):
                    xb=Xn[i:i+bs]
                    yb=np.array(Y[i:i+bs],dtype=np.float32).reshape(-1,1)
                    h1=np.maximum(0,xb@W1+b1)
                    h2=np.maximum(0,h1@W2+b2)
                    o=h2@W3+b3
                    p=1/(1+np.exp(-o))
                    g=(p-yb)
                    gW3=h2.T@g/len(xb)
                    gb3=g.mean(0)
                    dh2=g@W3.T
                    dh2[h2<=0]=0
                    gW2=h1.T@dh2/len(xb)
                    gb2=dh2.mean(0)
                    dh1=dh2@W2.T
                    dh1[h1<=0]=0
                    gW1=xb.T@dh1/len(xb)
                    gb1=dh1.mean(0)
                    W3-=lr*gW3
                    b3-=lr*gb3
                    W2-=lr*gW2
                    b2-=lr*gb2
                    W1-=lr*gW1
                    b1-=lr*gb1
            self.W1=W1
            self.b1=b1
            self.W2=W2
            self.b2=b2
            self.W3=W3
            self.b3=b3
        return True,(len(X),int(np.sum(Y)))
class Foreground:
    GA_ROOT=2
    DWMWA_CLOAKED=14
    @staticmethod
    def client_rect(hwnd):
        r=win32gui.GetClientRect(hwnd)
        tl=win32gui.ClientToScreen(hwnd,(0,0))
        return (tl[0],tl[1],tl[0]+r[2],tl[1]+r[3])
    @staticmethod
    def ensure_front(hwnd):
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd,win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.02)
            return win32gui.GetForegroundWindow()==hwnd
        except:
            return False
    @staticmethod
    def visible(hwnd):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return False
            if win32gui.IsIconic(hwnd):
                return False
            try:
                cloaked=ctypes.c_int(0)
                ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(hwnd),Foreground.DWMWA_CLOAKED,ctypes.byref(cloaked),ctypes.sizeof(cloaked))
                if int(cloaked.value)!=0:
                    return False
            except:
                pass
            return True
        except:
            return False
    @staticmethod
    def top(hwnd):
        try:
            return ctypes.windll.user32.GetAncestor(ctypes.wintypes.HWND(hwnd),Foreground.GA_ROOT)
        except:
            return hwnd
    @staticmethod
    def occluded(hwnd,rect):
        try:
            if not Foreground.visible(hwnd):
                return True
            x1,y1,x2,y2=rect
            if x2<=x1 or y2<=y1:
                return True
            pts=[]
            for rx in [0.08,0.5,0.92]:
                for ry in [0.08,0.5,0.92]:
                    px=int(x1+(x2-x1)*rx)
                    py=int(y1+(y2-y1)*ry)
                    pts.append((px,py))
            th=Foreground.top(hwnd)
            for (px,py) in pts:
                hw=ctypes.windll.user32.WindowFromPoint(ctypes.wintypes.POINT(px,py))
                if hw==0:
                    return True
                rt=Foreground.top(hw)
                if rt!=th:
                    return True
            return False
        except:
            return True
    @staticmethod
    def ready(hwnd,rect):
        try:
            if not Foreground.visible(hwnd):
                return False
            if Foreground.occluded(hwnd,rect):
                return False
            return True
        except:
            return False
    @staticmethod
    def is_foreground(hwnd):
        try:
            return win32gui.GetForegroundWindow()==hwnd
        except:
            return False
class ModelMeta:
    @staticmethod
    def read():
        try:
            with open(meta_path,"r",encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"version":0,"history":[]}
    @staticmethod
    def write(d):
        with open(meta_path,"w",encoding="utf-8") as f:
            json.dump(d,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
    @staticmethod
    def update(files,train_range):
        m=ModelMeta.read()
        v=int(m.get("version",0))+1
        now=time.strftime("%Y-%m-%d %H:%M:%S")
        metas=[]
        for fp in files:
            if not fp or not os.path.exists(fp):
                continue
            h=hashlib.sha256()
            with open(fp,"rb") as f:
                for chunk in iter(lambda:f.read(1<<20),b""):
                    h.update(chunk)
            metas.append({"file":os.path.basename(fp),"sha256":h.hexdigest(),"size":os.path.getsize(fp)})
        m["version"]=v
        m["updated_at"]=now
        m["last_train_range"]=train_range
        m.setdefault("history",[]).append({"version":v,"time":now,"files":metas,"range":train_range})
        ModelMeta.write(m)
class State(QObject):
    signal_status=Signal(str)
    signal_mode=Signal(str)
    signal_window=Signal(str)
    signal_ai=Signal(str)
    signal_preview=Signal(object)
    signal_counts=Signal(int)
    signal_recsize=Signal(str)
    signal_modelver=Signal(str)
    signal_tip=Signal(str)
    signal_optprog=Signal(int,str)
    MARK=0x22ACE777
    def __init__(self):
        super().__init__()
        self.mode="learning"
        self.last_user_activity=time.time()
        self.selected_hwnd=None
        self.selected_title=""
        self.session_id=str(uuid.uuid4())
        self.running=True
        self.training_thread=None
        self.learning_thread=None
        self.optimizing=False
        self.interrupt_ai=False
        self.rect=(0,0,0,0)
        self.client_rect=(0,0,0,0)
        self.fps=8.0
        self.prev_imgs=collections.deque(maxlen=8)
        self.prev_img=None
        self._last_f_ts=0.0
        self.cfg=self.load_cfg()
        self.min_fps=self.cfg.get("screenshot_min_fps",1)
        self.max_fps=self.cfg.get("screenshot_max_fps",120)
        self.ui_scale=self.cfg.get("ui_scale",1.0)
        self.png_comp=int(self.cfg.get("frame_png_compress",3))
        self.save_change_thresh=float(self.cfg.get("save_change_thresh",6.0))
        self.disk_cap_gb=float(self.cfg.get("max_disk_gb",10.0))
        self.pre_post_K=int(self.cfg.get("pre_post_K",3))
        self.block_margin_px=int(self.cfg.get("block_margin_px",6))
        self.idle_timeout=int(self.cfg.get("idle_timeout",10))
        self.preview_on=bool(self.cfg.get("preview_on",True))
        self.stop_event=threading.Event()
        self.events_path=None
        self.frames_path=None
        self.events_writer=None
        self.frames_writer=None
        self.event_count=0
        self.model_click=ClickModel()
        self.model_loaded=False
        self.model_ver_str=""
        self.sal_map=None
        self.last_kbd_log_ts=0.0
        self.day_tag=None
        self._init_day_files()
    def load_cfg(self):
        data={}
        try:
            with open(cfg_path,"r",encoding="utf-8") as f:
                data=json.load(f)
        except:
            data={}
        merged=dict(cfg_defaults)
        for k,v in data.items():
            if v is not None:
                merged[k]=v
        if merged!=data:
            with open(cfg_path,"w",encoding="utf-8") as f:
                json.dump(merged,f,ensure_ascii=False,indent=2)
                f.flush()
                os.fsync(f.fileno())
        return merged
    def save_cfg(self,k,v):
        self.cfg[k]=v
        with open(cfg_path,"w",encoding="utf-8") as f:
            json.dump(self.cfg,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
    def _init_day_files(self):
        day=time.strftime("%Y%m%d")
        if self.day_tag==day and self.events_writer and self.frames_writer:
            return
        d=today_dir()
        self.day_tag=day
        ep=os.path.join(d,"events.jsonl")
        fp=os.path.join(d,"frames.jsonl")
        JSONLWriter.repair(ep)
        JSONLWriter.repair(fp)
        if self.events_writer:
            self.events_writer.close()
        if self.frames_writer:
            self.frames_writer.close()
        self.events_path=ep
        self.frames_path=fp
        self.events_writer=JSONLWriter(self.events_path)
        self.frames_writer=JSONLWriter(self.frames_path)
        self.event_count=self._load_event_count()
        self.signal_counts.emit(self.event_count)
    def _load_event_count(self):
        try:
            if not os.path.exists(self.events_path):
                return 0
            c=0
            with open(self.events_path,"r",encoding="utf-8") as f:
                for _ in f:
                    c+=1
            return c
        except:
            return 0
    def set_window(self,hwnd,title):
        self.selected_hwnd=hwnd
        self.selected_title=title
        self.signal_window.emit(title)
        self.load_model_hot()
        log(f"绑定窗口:{title}")
    def title_model_path(self):
        if self.selected_title:
            return model_path_for(self.selected_title)
        return default_model_path()
    def model_exists(self):
        ps=[self.title_model_path(),default_model_path()]
        for p in ps:
            if os.path.exists(p):
                return True
        return False
    def load_model_hot(self):
        try:
            p=self.title_model_path()
            if os.path.exists(p):
                self.model_click.load(p)
                self.model_loaded=True
            elif os.path.exists(default_model_path()):
                self.model_click.load(default_model_path())
                self.model_loaded=True
            else:
                self.model_click=ClickModel()
                self.model_loaded=False
        except:
            self.model_click=ClickModel()
            self.model_loaded=False
        mv=ModelMeta.read().get("version",0)
        self.model_ver_str=f"v{mv}"
        self.signal_modelver.emit(self.model_ver_str)
    def get_rects(self):
        if not self.selected_hwnd:
            return (0,0,0,0),(0,0,0,0)
        try:
            wr=win32gui.GetWindowRect(self.selected_hwnd)
            tl=win32gui.ClientToScreen(self.selected_hwnd,(0,0))
            cr=win32gui.GetClientRect(self.selected_hwnd)
            return wr,(tl[0],tl[1],tl[0]+cr[2],tl[1]+cr[3])
        except:
            return (0,0,0,0),(0,0,0,0)
    def _mss_for_thread(self):
        if not hasattr(self,"thread_mss"):
            self.thread_mss=threading.local()
        if not hasattr(self.thread_mss,"inst"):
            self.thread_mss.inst=mss_mod.mss()
        return self.thread_mss.inst
    def capture_client(self):
        x1,y1,x2,y2=self.client_rect
        w=max(1,x2-x1)
        h=max(1,y2-y1)
        if w<=0 or h<=0:
            return None
        try:
            shot=self._mss_for_thread().grab({"left":x1,"top":y1,"width":w,"height":h})
            img=np.array(shot)
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            return img
        except:
            return None
    def adapt_fps(self,prev,curr):
        if prev is None or curr is None:
            return
        try:
            a=cv2.resize(prev,(160,160))
            b=cv2.resize(curr,(160,160))
            d=cv2.absdiff(a,b)
            v=float(np.mean(d))
            g1=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
            g2=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
            flow=cv2.calcOpticalFlowFarneback(g1,g2,None,0.5,3,15,3,5,1.2,0)
            mag=np.linalg.norm(flow,axis=2).mean()
            s=v*0.4+mag*50.0
            cpu=psutil.cpu_percent(interval=None)
            mem=psutil.virtual_memory().percent
            t=self.fps
            gu,gm=_gpu_util_mem()
            if gu is not None and gm is not None:
                if gu>92 or gm>92:
                    t=max(self.min_fps,t-5.0)
                elif gu>80 or gm>85:
                    t=max(self.min_fps,t-3.0)
                elif gu<50 and gm<60 and s>10 and cpu<85 and mem<85:
                    t=min(self.max_fps,t+2.0)
            if s>30 and cpu<85 and mem<85:
                t=min(self.max_fps,t+6.0)
            elif s>15 and cpu<90 and mem<90:
                t=min(self.max_fps,t+3.0)
            if cpu>92 or mem>92:
                t=max(self.min_fps,t-4.0)
            elif s<4:
                t=max(self.min_fps,t-2.0)
            self.fps=t
        except:
            pass
    def clamp_abs(self,x,y):
        x1,y1,x2,y2=self.client_rect
        x=int(max(x1+self.block_margin_px,min(x2-1-self.block_margin_px,x)))
        y=int(max(y1+self.block_margin_px,min(y2-1-self.block_margin_px,y)))
        return x,y
    def _scale2560(self,x,y,cw,ch):
        sx=int(max(0,min(2560,round(x*2560.0/max(1,cw)))))
        sy=int(max(0,min(1600,round(y*1600.0/max(1,ch)))))
        return sx,sy
    def record_frame(self,img,mode,force=False):
        try:
            self._init_day_files()
            now=time.time()
            save=False
            if self.prev_img is None:
                save=True
            else:
                a=cv2.resize(self.prev_img,(128,128))
                b=cv2.resize(img,(128,128))
                d=cv2.absdiff(a,b)
                v=float(np.mean(d))
                if v>=self.save_change_thresh or force or (now-self._last_f_ts)>2.0:
                    save=True
            if not save:
                return None
            fid=str(int(now*1000))
            base=os.path.join(exp_dir,self.day_tag,"frames")
            path=os.path.join(base,f"{fid}.png")
            while os.path.exists(path):
                fid=str(int(fid)+1)
                path=os.path.join(base,f"{fid}.png")
            resized=cv2.resize(img,(2560,1600))
            ok,buf=cv2.imencode(".png",resized,[int(cv2.IMWRITE_PNG_COMPRESSION),self.png_comp])
            if not ok:
                return None
            with open(path,"wb") as f:
                buf.tofile(f)
                f.flush()
                os.fsync(f.fileno())
            dx,dy=_dpi_for_hwnd(self.selected_hwnd if self.selected_hwnd else 0)
            src="user" if mode=="learning" else "ai"
            rec={"id":fid,"ts":now,"source":src,"mode":mode,"window_title":self.selected_title,"rect":[0,0,2560,1600],"dpi":[int(dx),int(dy)],"path":path,"filename":f"{fid}.png","w":2560,"h":1600,"session_id":self.session_id}
            self.frames_writer.append(rec)
            self.prev_img=img.copy()
            self._last_f_ts=now
            return fid
        except:
            return None
    def record_op(self,source,typ,press_t,px,py,release_t,rx,ry,moves,frame_id,ins_press,ins_release,clip_ids=None):
        try:
            self._init_day_files()
            cx,cy=self.client_rect[0],self.client_rect[1]
            cw=max(1,self.client_rect[2]-self.client_rect[0])
            ch=max(1,self.client_rect[3]-self.client_rect[1])
            plx,ply=self._scale2560(px-cx,py-cy,cw,ch)
            rlx,rly=self._scale2560(rx-cx,ry-cy,cw,ch)
            mm=[(float(t),int(self._scale2560(x-cx,y-cy,cw,ch)[0]),int(self._scale2560(x-cx,y-cy,cw,ch)[1]),int(ins)) for (t,x,y,ins) in moves] if moves else []
            obj={"id":str(uuid.uuid4()),"source":source,"type":typ,"press_t":press_t,"press_x":px,"press_y":py,"press_lx":plx,"press_ly":ply,"release_t":release_t,"release_x":rx,"release_y":ry,"release_lx":rlx,"release_ly":rly,"moves":mm,"window_title":self.selected_title,"rect":[0,0,2560,1600],"frame_id":frame_id,"ins_press":int(ins_press),"ins_release":int(ins_release),"clip_ids":clip_ids or [],"session_id":self.session_id}
            self.events_writer.append(obj)
            self.event_count+=1
            self.signal_counts.emit(self.event_count)
        except:
            pass
    def record_kbd(self,ts):
        try:
            if ts-self.last_kbd_log_ts<0.3:
                return
            self._init_day_files()
            obj={"id":str(uuid.uuid4()),"source":"user","type":"kbd","ts":float(ts),"window_title":self.selected_title,"rect":[0,0,2560,1600],"session_id":self.session_id}
            self.events_writer.append(obj)
            self.event_count+=1
            self.signal_counts.emit(self.event_count)
            self.last_kbd_log_ts=ts
        except:
            pass
    def heat_from_events(self,w,h):
        try:
            self._init_day_files()
            p=self.events_path
            heat=np.zeros((h,w),dtype=np.float32)
            if not os.path.exists(p):
                return None
            with open(p,"r",encoding="utf-8") as f:
                for line in f:
                    try:
                        e=json.loads(line)
                        if e.get("source")!="user":
                            continue
                        if e.get("window_title")!=self.selected_title:
                            continue
                        if e.get("type") not in ["left","right","middle"]:
                            continue
                        x=int(e.get("press_lx",0))
                        y=int(e.get("press_ly",0))
                        x=max(0,min(w-1,x))
                        y=max(0,min(h-1,y))
                        heat[y,x]+=1.0
                    except:
                        pass
            heat=cv2.GaussianBlur(heat,(0,0),7)
            m=np.max(heat)
            if m>0:
                heat/=m
            return heat
        except:
            return None
    def spectral_saliency(self,img):
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        s=160
        gg=cv2.resize(g,(s,s))
        f=np.fft.fft2(gg)
        a=np.abs(f)
        p=np.angle(f)
        la=np.log(a+1e-6)
        bl=cv2.GaussianBlur(la,(0,0),3,3)
        sr=np.exp(la-bl)
        nf=sr*np.exp(1j*p)
        r=np.fft.ifft2(nf)
        m=np.abs(r)
        m=cv2.GaussianBlur(m,(0,0),3,3)
        m=(m-m.min())/(m.max()-m.min()+1e-6)
        sal=cv2.resize(m,(img.shape[1],img.shape[0]))
        return sal.astype(np.float32)
    def salient_points(self,img,kmax=400):
        sal=self.spectral_saliency(img)
        self.sal_map=sal
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        k=cv2.goodFeaturesToTrack(g,300,0.01,6)
        pts=[]
        if k is not None:
            for p in k:
                x,y=p.ravel()
                s=float(sal[int(max(0,min(img.shape[0]-1,y))),int(max(0,min(img.shape[1]-1,x)))])
                pts.append((int(x),int(y),0.6+0.6*s))
        thr=np.percentile(sal,95)
        ys,xs=np.where(sal>=thr)
        for x,y in zip(xs,ys):
            pts.append((int(x),int(y),1.0))
        if len(pts)>kmax:
            idx=np.random.choice(len(pts),kmax,replace=False)
            pts=[pts[i] for i in idx]
        return pts
    def local_maxima(self,hm,top=80,dist=14):
        if hm is None:
            return []
        h,w=hm.shape
        pts=[]
        m=hm.copy()
        for _ in range(top):
            idx=np.argmax(m)
            v=m.flat[idx]
            if v<=0:
                break
            y=idx//w
            x=idx%w
            pts.append((x,y,float(v)))
            x1=max(0,x-dist)
            x2=min(w,x+dist+1)
            y1=max(0,y-dist)
            y2=min(h,y+dist+1)
            m[y1:y2,x1:x2]=0
        return pts
    def ai_decide(self,img):
        h,w=img.shape[:2]
        rsz=cv2.resize(img,(2560,1600))
        hm=self.model_click.heatmap(rsz)
        uh=self.heat_from_events(2560,1600)
        sal=self.spectral_saliency(rsz)
        mot=None
        if self.prev_img is not None:
            a=cv2.resize(self.prev_img,(2560,1600))
            b=rsz
            d=cv2.absdiff(a,b)
            g=cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)
            mot=cv2.GaussianBlur(g,(0,0),3)
            mx=float(mot.max())
            if mx>0:
                mot=mot/mx
            else:
                mot=np.zeros_like(sal)
        if hm is None:
            hm=np.zeros_like(sal)
        if uh is None:
            uh=np.zeros_like(sal)
        if mot is None:
            mot=np.zeros_like(sal)
        comb=(hm*1.6+uh*1.0+sal*0.8+mot*0.6)
        mx=float(comb.max())
        if mx>0:
            comb=comb/mx
        pts=self.local_maxima(comb,top=90,dist=16)
        if len(pts)==0:
            spts=self.salient_points(rsz)
            if len(spts)==0:
                return ("idle",None,None,None)
            px,py,_=spts[0]
            return ("click",(px,py),"left",None)
        scores=np.array([p[2] for p in pts],dtype=np.float32)
        idx=int(np.argmax(scores))
        px,py,_=pts[idx]
        return ("click",(px,py),"left",None)
    def periodic(self):
        self._init_day_files()
        DiskManager.enforce(self.disk_cap_gb)
        self.signal_recsize.emit(self._rec_size_str())
    def _rec_size_str(self):
        try:
            s=DiskManager.total_bytes()
            if s<1024:
                return f"{s}B"
            if s<1024*1024:
                return f"{s//1024}KB"
            if s<1024*1024*1024:
                return f"{s//(1024*1024)}MB"
            return f"{s/(1024*1024*1024):.2f}GB"
        except:
            return "0B"
class MouseInjector:
    def __init__(self,st:State):
        self.st=st
        self.MOUSEEVENTF_MOVE=0x0001
        self.MOUSEEVENTF_LEFTDOWN=0x0002
        self.MOUSEEVENTF_LEFTUP=0x0004
        self.MOUSEEVENTF_RIGHTDOWN=0x0008
        self.MOUSEEVENTF_RIGHTUP=0x0010
        self.MOUSEEVENTF_MIDDLEDOWN=0x0020
        self.MOUSEEVENTF_MIDDLEUP=0x0040
        self.MOUSEEVENTF_WHEEL=0x0800
        self.MOUSEEVENTF_ABSOLUTE=0x8000
        self.MOUSEEVENTF_VIRTUALDESK=0x4000
        self.user32=ctypes.windll.user32
    def _vdesk_metrics(self):
        x=win32api.GetSystemMetrics(76)
        y=win32api.GetSystemMetrics(77)
        w=win32api.GetSystemMetrics(78)
        h=win32api.GetSystemMetrics(79)
        return x,y,w,h
    def _norm(self,x,y):
        vx,vy,vw,vh=self._vdesk_metrics()
        X=int((x-vx)*65535//max(1,vw-1))
        Y=int((y-vy)*65535//max(1,vh-1))
        return X,Y
    def _send(self,dx,dy,data,flags):
        class MOUSEINPUT(ctypes.Structure):
            _fields_=[('dx',ctypes.c_long),('dy',ctypes.c_long),('mouseData',ctypes.c_ulong),('dwFlags',ctypes.c_ulong),('time',ctypes.c_ulong),('dwExtraInfo',ctypes.c_void_p)]
        class INPUT(ctypes.Structure):
            _fields_=[('type',ctypes.c_ulong),('mi',MOUSEINPUT)]
        ex=ctypes.c_void_p(self.st.MARK)
        mi=MOUSEINPUT(dx,dy,data,flags,0,ex)
        inp=INPUT(0,mi)
        self.user32.SendInput(1,ctypes.pointer(inp),ctypes.sizeof(inp))
    def move_abs(self,x,y):
        X,Y=self._norm(x,y)
        self._send(X,Y,0,self.MOUSEEVENTF_MOVE|self.MOUSEEVENTF_ABSOLUTE|self.MOUSEEVENTF_VIRTUALDESK)
    def wheel(self,amt):
        self._send(0,0,ctypes.c_ulong(int(amt)),self.MOUSEEVENTF_WHEEL)
    def down(self,btn):
        f=self.MOUSEEVENTF_LEFTDOWN if btn=='left' else (self.MOUSEEVENTF_RIGHTDOWN if btn=='right' else self.MOUSEEVENTF_MIDDLEDOWN)
        self._send(0,0,0,f)
    def up(self,btn):
        f=self.MOUSEEVENTF_LEFTUP if btn=='left' else (self.MOUSEEVENTF_RIGHTUP if btn=='right' else self.MOUSEEVENTF_MIDDLEUP)
        self._send(0,0,0,f)
    def move_path(self,pts,step_delay):
        for (x,y) in pts:
            if self.st.interrupt_ai:
                try:
                    self.up('left')
                    self.up('right')
                    self.up('middle')
                except:
                    pass
                return False
            self.move_abs(x,y)
            time.sleep(step_delay)
        return True
class LowLevelHook(threading.Thread):
    def __init__(self,st:State):
        super().__init__(daemon=True)
        self.st=st
        self.user32=ctypes.windll.user32
        self.kernel32=ctypes.windll.kernel32
        self.WH_MOUSE_LL=14
        self.WH_KEYBOARD_LL=13
        self.WM_MOUSEMOVE=0x0200
        self.WM_LBUTTONDOWN=0x0201
        self.WM_LBUTTONUP=0x0202
        self.WM_RBUTTONDOWN=0x0204
        self.WM_RBUTTONUP=0x0205
        self.WM_MBUTTONDOWN=0x0207
        self.WM_MBUTTONUP=0x0208
        self.WM_MOUSEWHEEL=0x020A
        self.LLMHF_INJECTED=0x00000001
        self.LLMHF_LOWER_IL_INJECTED=0x00000002
        self.hMouse=None
        self.hKey=None
        self.mouse_pressed={}
        self.enabled=True
        self.tid=0
    def within(self,x,y):
        r=self.st.client_rect
        return x>=r[0] and x<=r[2] and y>=r[1] and y<=r[3]
    def stop(self):
        try:
            self.enabled=False
            if self.tid!=0:
                self.user32.PostThreadMessageW(self.tid,0x0012,0,0)
        except:
            pass
    def run(self):
        CMPFUNC_MOUSE=ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.wintypes.WPARAM,ctypes.wintypes.LPARAM)
        CMPFUNC_KEY=ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.wintypes.WPARAM,ctypes.wintypes.LPARAM)
        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_=[("pt",ctypes.wintypes.POINT),("mouseData",ctypes.wintypes.DWORD),("flags",ctypes.wintypes.DWORD),("time",ctypes.wintypes.DWORD),("dwExtraInfo",ctypes.c_void_p)]
        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_=[("vkCode",ctypes.wintypes.DWORD),("scanCode",ctypes.wintypes.DWORD),("flags",ctypes.wintypes.DWORD),("time",ctypes.wintypes.DWORD),("dwExtraInfo",ctypes.c_void_p)]
        def low_mouse(nCode,wParam,lParam):
            if nCode>=0:
                ms=ctypes.cast(lParam,ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                x=ms.pt.x
                y=ms.pt.y
                flags=ms.flags
                ex=int(ms.dwExtraInfo or 0)
                injected=((flags & self.LLMHF_INJECTED)!=0) or ((flags & self.LLMHF_LOWER_IL_INJECTED)!=0) or (ex==self.st.MARK)
                if not injected:
                    self.st.last_user_activity=time.time()
                    if self.st.mode=="training":
                        self.st.interrupt_ai=True
                        self.st.signal_mode.emit("学习")
                if wParam==self.WM_MOUSEMOVE:
                    for k,v in list(self.mouse_pressed.items()):
                        if k in self.mouse_pressed:
                            inside=1 if self.within(x,y) else 0
                            v["moves"].append((time.time(),x,y,inside))
                else:
                    btn=None
                    pressed=None
                    if wParam==self.WM_LBUTTONDOWN:
                        btn="left"
                        pressed=True
                    elif wParam==self.WM_LBUTTONUP:
                        btn="left"
                        pressed=False
                    elif wParam==self.WM_RBUTTONDOWN:
                        btn="right"
                        pressed=True
                    elif wParam==self.WM_RBUTTONUP:
                        btn="right"
                        pressed=False
                    elif wParam==self.WM_MBUTTONDOWN:
                        btn="middle"
                        pressed=True
                    elif wParam==self.WM_MBUTTONUP:
                        btn="middle"
                        pressed=False
                    elif wParam==self.WM_MOUSEWHEEL:
                        if not injected and self.within(x,y) and self.st.mode=="training":
                            delta=ctypes.c_short(ms.mouseData>>16).value
                            img=self.st.capture_client()
                            rid=self.st.record_frame(img,"training",force=True)
                            self.st.record_op("user","wheel",time.time(),x,y,time.time(),x,y,[("scroll",0,int(delta),1)],rid,1,1,[rid] if rid else [])
                        return self.user32.CallNextHookEx(self.hMouse,nCode,wParam,lParam)
                    if btn is not None and pressed is not None and not injected:
                        if pressed:
                            if self.st.mode=="training":
                                self.st.interrupt_ai=True
                                self.st.signal_mode.emit("学习")
                            if self.within(x,y):
                                self.mouse_pressed[btn]={"t":time.time(),"x":x,"y":y,"moves":[(time.time(),x,y,1)],"pre":[]}
                                for im in list(self.st.prev_imgs)[-self.st.pre_post_K:]:
                                    rid=self.st.record_frame(im,"learning",force=True)
                                    if rid:
                                        self.mouse_pressed[btn]["pre"].append(rid)
                        else:
                            if btn in self.mouse_pressed:
                                d=self.mouse_pressed[btn]
                                pre=d.get("pre",[])
                                inside_release=1 if self.within(x,y) else 0
                                has_inside=inside_release==1 or any(m[3]==1 for m in d["moves"])
                                if has_inside:
                                    img=self.st.capture_client()
                                    rid=self.st.record_frame(img,"learning",force=True)
                                    if rid:
                                        pre.append(rid)
                                    ip=1 if self.within(d["x"],d["y"]) else 0
                                    ir=1 if inside_release==1 else 0
                                    self.st.record_op("user",btn,d["t"],d["x"],d["y"],time.time(),x,y,d["moves"],rid,ip,ir,pre)
                                if btn in self.mouse_pressed:
                                    del self.mouse_pressed[btn]
                return self.user32.CallNextHookEx(self.hMouse,nCode,wParam,lParam)
            return self.user32.CallNextHookEx(self.hMouse,nCode,wParam,lParam)
        def low_key(nCode,wParam,lParam):
            if nCode>=0:
                ks=ctypes.cast(lParam,ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                flags=ks.flags
                ex=int(ks.dwExtraInfo or 0)
                injected=((flags & 0x00000010)!=0) or (ex==self.st.MARK)
                if not injected:
                    t=time.time()
                    self.st.last_user_activity=t
                    self.st.record_kbd(t)
                    if self.st.mode=="training":
                        self.st.interrupt_ai=True
                        self.st.signal_mode.emit("学习")
            return self.user32.CallNextHookEx(self.hKey,nCode,wParam,lParam)
        self.mouse_cb=CMPFUNC_MOUSE(low_mouse)
        self.key_cb=CMPFUNC_KEY(low_key)
        self.hMouse=self.user32.SetWindowsHookExW(self.WH_MOUSE_LL,self.mouse_cb,self.kernel32.GetModuleHandleW(None),0)
        self.hKey=self.user32.SetWindowsHookExW(self.WH_KEYBOARD_LL,self.key_cb,self.kernel32.GetModuleHandleW(None),0)
        msg=ctypes.wintypes.MSG()
        self.tid=self.kernel32.GetCurrentThreadId()
        while self.enabled:
            b=self.user32.GetMessageW(ctypes.byref(msg),0,0,0)
            if b==0 or b==-1:
                break
            self.user32.TranslateMessage(ctypes.byref(msg))
            self.user32.DispatchMessageW(ctypes.byref(msg))
        if self.hMouse:
            self.user32.UnhookWindowsHookEx(self.hMouse)
        if self.hKey:
            self.user32.UnhookWindowsHookEx(self.hKey)
class LearningThread(threading.Thread):
    def __init__(self,st,ui):
        super().__init__(daemon=True)
        self.st=st
        self.ui=ui
    def run(self):
        self.st.prev_imgs.clear()
        prev=None
        while (self.st.mode=="learning") and (not self.st.stop_event.is_set()):
            wr,cr=self.st.get_rects()
            self.st.rect=wr
            self.st.client_rect=cr
            if cr==(0,0,0,0) or not self.st.selected_hwnd or not Foreground.ready(self.st.selected_hwnd,cr):
                self.st.signal_tip.emit("目标窗口不可见或被遮挡，暂停采集")
                time.sleep(0.12)
                continue
            img=self.st.capture_client()
            if img is None:
                time.sleep(0.06)
                continue
            fid=self.st.record_frame(img,"learning",force=False)
            self.st.adapt_fps(prev,img)
            self.st.prev_img=img
            self.st.prev_imgs.append(img)
            prev=img
            if self.ui.preview_enabled():
                h,w=img.shape[:2]
                qi=QImage(img.data,w,h,3*w,QImage.Format_BGR888)
                self.st.signal_preview.emit(QPixmap.fromImage(qi))
            self.st.periodic()
            time.sleep(max(1.0/self.st.fps,0.01))
class TrainerThread(threading.Thread):
    def __init__(self,st,ui):
        super().__init__(daemon=True)
        self.st=st
        self.ui=ui
        self.inj=MouseInjector(st)
    def run(self):
        self.st.interrupt_ai=False
        prev=None
        cool_t=0
        while (self.st.mode=="training") and (not self.st.stop_event.is_set()):
            if self.st.interrupt_ai:
                break
            wr,cr=self.st.get_rects()
            self.st.rect=wr
            self.st.client_rect=cr
            if cr==(0,0,0,0) or not self.st.selected_hwnd or not Foreground.ready(self.st.selected_hwnd,cr):
                self.st.signal_tip.emit("自动暂停：窗口不可见或被遮挡")
                time.sleep(0.10)
                continue
            if not Foreground.ensure_front(self.st.selected_hwnd):
                time.sleep(0.06)
                continue
            if Foreground.occluded(self.st.selected_hwnd,cr):
                self.st.signal_tip.emit("自动暂停：窗口被遮挡")
                time.sleep(0.10)
                continue
            img=self.st.capture_client()
            if img is None:
                time.sleep(0.06)
                continue
            fid=self.st.record_frame(img,"training",force=False)
            self.st.adapt_fps(prev,img)
            self.st.prev_img=img
            self.st.prev_imgs.append(img)
            act=self.st.ai_decide(img)
            prev=img
            if self.ui.preview_enabled():
                h,w=img.shape[:2]
                qi=QImage(img.data,w,h,3*w,QImage.Format_BGR888)
                self.st.signal_preview.emit(QPixmap.fromImage(qi))
            if act[0]=="idle":
                time.sleep(max(1.0/self.st.fps,0.01))
                continue
            if time.time()<cool_t:
                time.sleep(max(1.0/self.st.fps,0.01))
                continue
            if act[0]=="click":
                x,y=act[1]
                bx=self.st.client_rect[0]+int(round(x*max(1,(self.st.client_rect[2]-self.st.client_rect[0]))/2560.0))
                by=self.st.client_rect[1]+int(round(y*max(1,(self.st.client_rect[3]-self.st.client_rect[1]))/1600.0))
                bx,by=self.st.clamp_abs(bx,by)
                btn=act[2]
                try:
                    if self.st.interrupt_ai:
                        break
                    steps=max(6,int(self.st.fps*0.4))
                    path=[(int(bx),int(by)) for _ in range(steps)]
                    t0=time.time()
                    self.inj.move_path(path,step_delay=max(0.005,0.25/self.st.fps))
                    if self.st.interrupt_ai:
                        break
                    self.inj.down(btn)
                    mv=[(time.time(),bx,by,1)]
                    time.sleep(0.02)
                    self.inj.up(btn)
                    t1=time.time()
                    self.st.record_op("ai",btn,t0,bx,by,t1,bx,by,mv,fid,1,1,[])
                except:
                    pass
                cool_t=time.time()+max(0.15,0.8/self.st.fps)
            time.sleep(max(1.0/self.st.fps,0.01))
class WindowSelector:
    def __init__(self):
        self.map={}
    def refresh(self):
        self.map={}
        def cb(hwnd,extra):
            if win32gui.IsWindowVisible(hwnd):
                title=win32gui.GetWindowText(hwnd)
                if title and len(title.strip())>0:
                    try:
                        tid,pid=win32process.GetWindowThreadProcessId(hwnd)
                    except:
                        pid=0
                    name=str(pid)
                    try:
                        p=psutil.Process(pid)
                        name=p.name()
                    except:
                        pass
                    key=f"{name} | {title} | {hwnd}"
                    self.map[key]=hwnd
            return True
        win32gui.EnumWindows(cb,None)
        return self.map
class OptimizerThread(threading.Thread):
    def __init__(self,st,ui,cb,cancel_flag):
        super().__init__(daemon=True)
        self.st=st
        self.ui=ui
        self.cb=cb
        self.cancel_flag=cancel_flag
    def run(self):
        self.st.optimizing=True
        try:
            all_days=sorted([p for p in glob.glob(os.path.join(exp_dir,"*")) if os.path.isdir(p)])
            frames={}
            events=[]
            tsmin=None
            tsmax=None
            total=max(1,len(all_days))
            done=0
            for d in all_days:
                if self.cancel_flag.is_set():
                    self._done(False)
                    return
                fp=os.path.join(d,"frames.jsonl")
                ep=os.path.join(d,"events.jsonl")
                if os.path.exists(fp):
                    JSONLWriter.repair(fp)
                    with open(fp,"r",encoding="utf-8") as f:
                        for line in f:
                            try:
                                r=json.loads(line)
                                frames[r["id"]]=r
                                tsmin=r["ts"] if tsmin is None else min(tsmin,r["ts"])
                                tsmax=r["ts"] if tsmax is None else max(tsmax,r["ts"])
                            except:
                                pass
                if os.path.exists(ep):
                    JSONLWriter.repair(ep)
                    with open(ep,"r",encoding="utf-8") as f:
                        for line in f:
                            try:
                                e=json.loads(line)
                                events.append(e)
                            except:
                                pass
                done+=1
                p=int(5+40*done/total)
                self.st.signal_optprog.emit(p,"读取数据")
            if self.cancel_flag.is_set():
                self._done(False)
                return
            cm=ClickModel()
            ok1,stat1=cm.fit_from_logs(events,frames,where_title=self.st.selected_title)
            if self.cancel_flag.is_set():
                self._done(False)
                return
            if not ok1:
                self.st.signal_optprog.emit(100,"训练失败")
                self._done(False)
                return
            self.st.signal_optprog.emit(85,"训练点击模型")
            p=self.st.title_model_path()
            npz_tmp=p+".tmp"
            try:
                cm.save(npz_tmp)
                shutil.move(npz_tmp,p)
            except:
                pass
            self.st.load_model_hot()
            ModelMeta.update([self.st.title_model_path()],{"start":tsmin,"end":tsmax})
            self.st.signal_optprog.emit(100,"完成")
            self._done(True)
        except:
            self._done(False)
    def _done(self,ok):
        self.st.optimizing=False
        self.cb(ok)
class Main(QMainWindow):
    finish_opt=Signal(bool)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能学习与训练")
        self.state=State()
        self.selector=WindowSelector()
        self.hook=LowLevelHook(self.state)
        self.hook.start()
        top=QWidget()
        mid=QWidget()
        bot=QWidget()
        self.cmb=QComboBox()
        self.btn_toggle=QPushButton("停止" if self.state.running else "开始")
        self.btn_opt=QPushButton("优化")
        self.btn_ui=QPushButton("UI识别")
        self.chk_preview=QCheckBox("预览")
        self.chk_preview.setChecked(self.state.preview_on)
        self.lbl_mode=QLabel("学习")
        self.lbl_fps=QLabel("FPS:0")
        self.lbl_rec=QLabel("0B")
        self.lbl_ver=QLabel("v0")
        self.lbl_tip=QLabel("就绪")
        self.progress=QProgressBar()
        self.progress.setRange(0,100)
        self.progress.setValue(0)
        self.preview_label=QLabel()
        self.preview_label.setFixedSize(QSize(640,400))
        lt=QHBoxLayout(top)
        lt.addWidget(self.cmb,1)
        lt.addWidget(self.btn_toggle)
        lt.addWidget(self.btn_opt)
        lt.addWidget(self.btn_ui)
        lt.addWidget(self.chk_preview)
        lm=QVBoxLayout(mid)
        lm.addWidget(self.preview_label,1)
        lh=QHBoxLayout()
        lh.addWidget(self.lbl_mode)
        lh.addStretch(1)
        lh.addWidget(self.lbl_fps)
        lm.addLayout(lh)
        lb=QHBoxLayout(bot)
        lb.addWidget(QLabel("经验池:"))
        lb.addWidget(self.lbl_rec)
        lb.addStretch(1)
        lb.addWidget(QLabel("模型:"))
        lb.addWidget(self.lbl_ver)
        lb.addStretch(1)
        lb.addWidget(self.lbl_tip,2)
        lb.addWidget(self.progress,2)
        root=QVBoxLayout()
        root.addWidget(top)
        root.addWidget(mid,1)
        root.addWidget(bot)
        box=QWidget()
        box.setLayout(root)
        self.setCentralWidget(box)
        self.btn_toggle.clicked.connect(self.on_toggle)
        self.btn_opt.clicked.connect(self.on_opt)
        self.btn_ui.clicked.connect(self.on_ui)
        self.chk_preview.stateChanged.connect(self.on_preview_changed)
        self.cmb.currentIndexChanged.connect(self.on_sel_changed)
        self.state.signal_preview.connect(self.on_preview)
        self.state.signal_mode.connect(self.on_mode)
        self.state.signal_window.connect(self.on_window)
        self.state.signal_recsize.connect(self.on_recsize)
        self.state.signal_modelver.connect(self.on_modelver)
        self.state.signal_tip.connect(self.on_tip)
        self.state.signal_optprog.connect(self.on_optprog)
        self.refresh_timer=QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_windows)
        self.refresh_timer.start(2000)
        self.mode_timer=QTimer(self)
        self.mode_timer.timeout.connect(self.on_tick_mode)
        self.mode_timer.start(200)
        self.fps_timer=QTimer(self)
        self.fps_timer.timeout.connect(self.on_tick_fps)
        self.fps_timer.start(500)
        self.learning_thread=None
        self.training_thread=None
        self._optim_flag=threading.Event()
        self.refresh_windows()
        self.set_mode("learning")
    def preview_enabled(self):
        return self.chk_preview.isChecked()
    def on_preview_changed(self,_):
        self.state.save_cfg("preview_on",self.chk_preview.isChecked())
    def on_preview(self,px):
        if not self.preview_enabled():
            return
        try:
            self.preview_label.setPixmap(px.scaled(self.preview_label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        except:
            pass
    def on_mode(self,s):
        self.lbl_mode.setText(s)
    def on_window(self,s):
        pass
    def on_recsize(self,s):
        self.lbl_rec.setText(s)
    def on_modelver(self,s):
        self.lbl_ver.setText(s)
    def on_tip(self,s):
        self.lbl_tip.setText(s)
    def on_optprog(self,val,txt):
        self.progress.setValue(int(val))
        self.lbl_tip.setText(str(txt))
    def refresh_windows(self):
        m=self.selector.refresh()
        keys=sorted(list(m.keys()))
        curr=self.cmb.currentText()
        self.cmb.blockSignals(True)
        self.cmb.clear()
        self.cmb.addItems(keys)
        self.cmb.blockSignals(False)
        if curr in keys:
            self.cmb.setCurrentText(curr)
        if self.state.selected_hwnd is None and keys:
            self.cmb.setCurrentIndex(0)
            self.on_sel_changed(0)
    def on_sel_changed(self,_):
        key=self.cmb.currentText()
        hwnd=self.selector.map.get(key)
        if hwnd:
            self.state.set_window(hwnd,key)
    def set_mode(self,mode):
        if mode==self.state.mode and self.state.running:
            return
        self.state.stop_event.set()
        time.sleep(0.05)
        self.state.stop_event.clear()
        self.state.mode=mode
        if mode=="learning":
            self.lbl_mode.setText("学习")
            self.state.signal_mode.emit("学习")
            self.learning_thread=LearningThread(self.state,self)
            self.learning_thread.start()
        elif mode=="training":
            if not self.state.model_exists():
                QMessageBox.information(self,"提示","模型不可用，仍处于学习模式")
                self.state.mode="learning"
                self.lbl_mode.setText("学习")
                self.learning_thread=LearningThread(self.state,self)
                self.learning_thread.start()
                return
            self.lbl_mode.setText("训练")
            self.state.signal_mode.emit("训练")
            self.training_thread=TrainerThread(self.state,self)
            self.training_thread.start()
    def on_tick_mode(self):
        if not self.state.running or self.state.optimizing:
            return
        t=time.time()-self.state.last_user_activity
        if self.state.mode=="learning":
            if t>=self.state.idle_timeout and self.state.selected_hwnd:
                self.set_mode("training")
        else:
            if self.state.interrupt_ai:
                self.state.interrupt_ai=False
                self.set_mode("learning")
    def on_tick_fps(self):
        self.lbl_fps.setText(f"FPS:{int(self.state.fps)}")
    def on_toggle(self):
        if self.state.running:
            self.state.running=False
            self.state.stop_event.set()
            self.btn_toggle.setText("开始")
            self.lbl_tip.setText("已停止")
        else:
            self.state.running=True
            self.state.stop_event.clear()
            self.btn_toggle.setText("停止")
            self.lbl_tip.setText("运行中")
            self.set_mode("learning")
    def on_opt(self):
        if self.state.optimizing:
            return
        self.state.stop_event.set()
        self.state.running=False
        self.btn_toggle.setText("开始")
        self.btn_opt.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_mode.setText("优化中")
        self.state.signal_mode.emit("优化中")
        self._optim_flag.clear()
        th=OptimizerThread(self.state,self,self._on_opt_done,self._optim_flag)
        th.start()
    def _on_opt_done(self,ok):
        QMessageBox.information(self,"提示","优化完成" if ok else "优化失败")
        self.btn_opt.setEnabled(True)
        self.state.running=True
        self.set_mode("learning")
    def on_ui(self):
        if not self.state.selected_hwnd:
            QMessageBox.information(self,"提示","未选择窗口")
            return
        img=self.state.capture_client()
        if img is None:
            QMessageBox.information(self,"提示","无法获取窗口画面")
            return
        out=cv2.resize(img,(1280,800))
        hm=self.state.heat_from_events(2560,1600)
        if hm is not None:
            hmn=cv2.resize((hm*255).astype(np.uint8),(1280,800))
            hmc=cv2.applyColorMap(hmn,cv2.COLORMAP_JET)
            out=cv2.addWeighted(out,0.65,hmc,0.35,0)
        pts=self.state.salient_points(cv2.resize(img,(2560,1600)))
        for i,(x,y,_) in enumerate(pts[:100]):
            xx=int(x*1280/2560)
            yy=int(y*800/1600)
            cv2.circle(out,(xx,yy),3,(0,255,0),-1)
        d=os.path.join(exp_dir,self.state.day_tag,"ui_summary.json")
        zones=[]
        clicks=0
        tm0=None
        tm1=None
        if hm is not None:
            thr=np.percentile(hm[hm>0],90) if np.any(hm>0) else 0.0
            mask=(hm>=thr).astype(np.uint8)*255
            cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            evs=[]
            try:
                with open(self.state.events_path,"r",encoding="utf-8") as f:
                    for line in f:
                        try:
                            e=json.loads(line)
                            if e.get("source")!="user":
                                continue
                            if e.get("window_title")!=self.state.selected_title:
                                continue
                            if e.get("type") not in ["left","right","middle"]:
                                continue
                            evs.append(e)
                        except:
                            pass
            except:
                pass
            for c in cnts:
                x,y,w,h=cv2.boundingRect(c)
                if w*h<50:
                    continue
                cx0=float(x)/2560.0
                cy0=float(y)/1600.0
                cx1=float(x+w)/2560.0
                cy1=float(y+h)/1600.0
                hot=float(np.mean(hm[y:y+h,x:x+w]))
                cnt=0
                tmin=None
                tmax=None
                for e in evs:
                    px=int(e.get("press_lx",0))
                    py=int(e.get("press_ly",0))
                    if px>=x and px<x+w and py>=y and py<y+h:
                        cnt+=1
                        tt=e.get("press_t") or e.get("ts") or 0
                        tmin=tt if tmin is None else min(tmin,tt)
                        tmax=tt if tmax is None else max(tmax,tt)
                clicks+=cnt
                if tm0 is None or (tmin is not None and tmin<tm0):
                    tm0=tmin
                if tm1 is None or (tmax is not None and tmax>tm1):
                    tm1=tmax
                zones.append({"bbox_abs":[int(x),int(y),int(x+w),int(y+h)],"bbox_rel":[cx0,cy0,cx1,cy1],"hot":hot,"clicks":int(cnt)})
                cv2.rectangle(out,(int(x*1280/2560),int(y*800/1600)),(int((x+w)*1280/2560),int((y+h)*800/1600)),(0,255,255),1)
        with open(d,"w",encoding="utf-8") as f:
            json.dump({"window":self.state.selected_title,"zones":zones,"time":time.time(),"total_clicks":int(clicks),"time_span":[tm0,tm1]},f,ensure_ascii=False,indent=2)
        ok,buf=cv2.imencode(".png",out,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
        if ok:
            outdir=os.path.join(exp_dir,self.state.day_tag)
            p=os.path.join(outdir,f"ui_{int(time.time())}.png")
            with open(p,"wb") as f:
                buf.tofile(f)
        QMessageBox.information(self,"提示","UI识别完成")
def main():
    app=QApplication(sys.argv)
    w=Main()
    w.resize(900,640)
    w.show()
    sys.exit(app.exec())
if __name__=="__main__":
    main()
