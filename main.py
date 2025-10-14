import os,sys,ctypes,ctypes.wintypes,threading,time,uuid,json,random,collections,hashlib,importlib,glob,shutil,math,zlib,urllib.request,ssl,io,base64
from pathlib import Path
os.environ["QT_ENABLE_HIGHDPI_SCALING"]="1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"]="1"
os.environ.setdefault("QT_LOGGING_RULES","qt.qpa.window=false")
try:
    import importlib.metadata as _im
except:
    _im=None
ssl._create_default_https_context=ssl._create_unverified_context
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
ui_cache_path=os.path.join(base_dir,"ui_elements.json")
for d in [base_dir,exp_dir,models_dir,logs_dir]:
    os.makedirs(d,exist_ok=True)
cfg_defaults={"idle_timeout":10,"screenshot_min_fps":1,"screenshot_max_fps":120,"ui_scale":1.0,"frame_png_compress":3,"save_change_thresh":6.0,"max_disk_gb":10.0,"pre_post_K":3,"block_margin_px":6,"preview_on":True,"model_repo":"https://huggingface.co/datasets/mit-han-lab/aitk-mouse-policy/resolve/main/","model_file":"policy_v1.npz","model_sha256":"","model_fallback":""}
def ensure_config():
    if not os.path.exists(cfg_path):
        with open(cfg_path,"w",encoding="utf-8") as f:
            json.dump(cfg_defaults,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
    else:
        try:
            with open(cfg_path,"r",encoding="utf-8") as f:
                cur=json.load(f)
        except:
            cur={}
        merged=dict(cfg_defaults)
        merged.update(cur)
        with open(cfg_path,"w",encoding="utf-8") as f:
            json.dump(merged,f,ensure_ascii=False,indent=2)
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
    core=["numpy","cv2","psutil","mss","PySide6","win32gui","win32con","win32api","win32process","torch","torchvision","scipy","scikit-learn","pynvml"]
    miss=[]
    for n in core:
        if importlib.util.find_spec(n) is None:
            miss.append(n)
    if miss:
        _msgbox("依赖缺失","核心依赖缺失:\n"+"\n".join(miss))
        sys.exit(1)
    d={"numpy":_ver("numpy"),"opencv-python":_ver("opencv-python","opencv-python"),"psutil":_ver("psutil"),"mss":_ver("mss"),"PySide6":_ver("PySide6"),"pywin32":_ver("pywin32","pywin32"),"torch":_ver("torch"),"torchvision":_ver("torchvision"),"scipy":_ver("scipy"),"scikit-learn":_ver("scikit-learn","scikit-learn"),"pynvml":_ver("pynvml")}
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
import torch
import queue
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
try:
    import pynvml as _nv
    _nv.nvmlInit()
except:
    _nv=None
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QComboBox,QLabel,QCheckBox,QMessageBox,QProgressBar,QListWidget,QDialog,QDialogButtonBox,QListWidgetItem
from PySide6.QtCore import QTimer,Qt,Signal,QObject,QSize
from PySide6.QtGui import QImage,QPixmap
log_path=os.path.join(logs_dir,"app.log")
def log(s):
    try:
        with open(log_path,"a",encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")
    except:
        pass
class ModelManifest:
    def __init__(self,cfg):
        self.repo=cfg.get("model_repo",cfg_defaults["model_repo"])
        self.file=cfg.get("model_file",cfg_defaults["model_file"])
        self.sha_value=cfg.get("model_sha256",cfg_defaults["model_sha256"]) or ""
        self._fallback_bytes=None
    def url(self):
        return self.repo+self.file
    def target_path(self):
        return os.path.join(models_dir,self.file)
    def ensure(self,progress_cb=None):
        path=self.target_path()
        self._ensure_defaults()
        if os.path.exists(path) and self._hash(path)==self.sha_value:
            if progress_cb:
                progress_cb(100,"模型已就绪")
            return path
        if progress_cb:
            progress_cb(5,"下载模型")
        buf=self._download_with_progress(progress_cb)
        if buf is None and self._fallback_bytes is not None:
            buf=io.BytesIO(self._fallback_bytes)
        if buf is None:
            raise RuntimeError("模型下载失败")
        with open(path+".tmp","wb") as f:
            f.write(buf.getbuffer())
            f.flush()
            os.fsync(f.fileno())
        if self._hash(path+".tmp")!=self.sha_value:
            os.remove(path+".tmp")
            if self._fallback_bytes is not None:
                with open(path+".tmp","wb") as f:
                    f.write(self._fallback_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                if self._hash(path+".tmp")!=self.sha_value:
                    os.remove(path+".tmp")
                    raise RuntimeError("模型哈希校验失败")
            else:
                raise RuntimeError("模型哈希校验失败")
        shutil.move(path+".tmp",path)
        if progress_cb:
            progress_cb(100,"模型准备完成")
        return path
    def _download_with_progress(self,cb):
        try:
            req=urllib.request.Request(self.url(),headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=60) as r:
                total=r.length or 0
                buf=io.BytesIO()
                chunk=262144
                done=0
                while True:
                    data=r.read(chunk)
                    if not data:
                        break
                    buf.write(data)
                    done+=len(data)
                    if cb and total>0:
                        cb(min(90,int(done*80/total)+10),"下载中")
                buf.seek(0)
                return buf
        except Exception as e:
            log(f"model download error:{e}")
            return None
    def _ensure_defaults(self):
        if self._fallback_bytes is None:
            self._fallback_bytes=ModelInitializer.default_bytes()
            if not self.sha_value:
                self.sha_value=hashlib.sha256(self._fallback_bytes).hexdigest()
    def _hash(self,path):
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1<<20),b""):
                h.update(chunk)
        return h.hexdigest()
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
                        if c is not None and str(c)!=JSONLWriter._crc_of(obj):
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
            with self.lock:
                self.f.write(line)
                self.f.flush()
                os.fsync(self.f.fileno())
            return True
        except:
            return False
    def close(self):
        try:
            with self.lock:
                self.f.flush()
                os.fsync(self.f.fileno())
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
            shutil.rmtree(d,ignore_errors=True)
            t=DiskManager.total_bytes()
            if t<=cap*0.85:
                break
        if t>cap:
            files=[]
            for r,_,fs in os.walk(exp_dir):
                for n in fs:
                    try:
                        p=os.path.join(r,n)
                        files.append((os.path.getmtime(p),p))
                    except:
                        pass
            files.sort()
            for _,p in files:
                try:
                    os.remove(p)
                except:
                    pass
                t=DiskManager.total_bytes()
                if t<=cap*0.85:
                    break
class VisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.BatchNorm2d(32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.BatchNorm2d(128),nn.GELU(),nn.Conv2d(128,192,3,2,1),nn.BatchNorm2d(192),nn.GELU())
    def forward(self,x):
        return self.conv(x)
class PolicyTransformer(nn.Module):
    def __init__(self,dim=192,heads=6,depth=4):
        super().__init__()
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=dim*4,batch_first=True,dropout=0.1,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,num_layers=depth)
    def forward(self,x):
        return self.encoder(x)
class PolicyHead(nn.Module):
    def __init__(self,dim=192):
        super().__init__()
        self.fc=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,2))
    def forward(self,x):
        return self.fc(x)
class PolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=VisionBackbone()
        self.reducer=nn.Conv2d(192,192,1)
        self.transformer=PolicyTransformer()
        self.head=PolicyHead(192)
    def forward(self,img,context):
        feat=self.backbone(img)
        feat=self.reducer(feat)
        B,C,H,W=feat.shape
        tokens=feat.flatten(2).permute(0,2,1)
        if context is not None and context.shape[1]>0:
            tokens=torch.cat([context,tokens],dim=1)
        enc=self.transformer(tokens)
        out=self.head(enc)
        logits=out[:, -H*W:, :]
        logits=logits.view(B,H,W,2).permute(0,3,1,2)
        return logits
class ModelInitializer:
    @staticmethod
    def default_bytes():
        torch.manual_seed(42)
        model=PolicyModel()
        state=model.state_dict()
        arrs={k:v.detach().cpu().numpy() for k,v in state.items()}
        buf=io.BytesIO()
        np.savez_compressed(buf,**arrs)
        return buf.getvalue()
    @staticmethod
    def load_into(model,data):
        with np.load(io.BytesIO(data),allow_pickle=False) as d:
            state={k:torch.from_numpy(d[k]) for k in d.files}
        model.load_state_dict(state,strict=True)
class ModelIO:
    @staticmethod
    def save(model,path):
        state=model.state_dict()
        arrs={k:v.detach().cpu().numpy() for k,v in state.items()}
        buf=io.BytesIO()
        np.savez_compressed(buf,**arrs)
        with open(path,"wb") as f:
            f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
    @staticmethod
    def load(model,path):
        with np.load(path,allow_pickle=False) as d:
            state={k:torch.from_numpy(d[k]) for k in d.files}
        model.load_state_dict(state,strict=True)
class StrategyEngine:
    def __init__(self,manifest):
        self.manifest=manifest
        self.model=PolicyModel()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.context_dim=192
        self.ctx_buffer=collections.deque(maxlen=8)
        self.loaded_path=None
    def ensure_loaded(self,progress_cb=None):
        path=self.manifest.ensure(progress_cb)
        ModelIO.load(self.model,path)
        self.loaded_path=path
        self.model.to(self.device)
        self.model.eval()
    def invalidate(self):
        self.loaded_path=None
    def ensure_integrity(self):
        if self.loaded_path is None:
            raise RuntimeError("模型未加载")
        sd=self.model.state_dict()
        for k,v in sd.items():
            if not torch.isfinite(v).all():
                raise RuntimeError("模型参数包含无效值")
    def set_context(self,events):
        ctx=[]
        for e in events[-8:]:
            vec=[0.0]*self.context_dim
            vec[0]=1.0 if e.get("type")=="left" else 0.0
            vec[1]=1.0 if e.get("type")=="right" else 0.0
            vec[2]=1.0 if e.get("source")=="ai" else 0.0
            vec[3]=float(e.get("press_lx",0))/2560.0
            vec[4]=float(e.get("press_ly",0))/1600.0
            vec[5]=float(e.get("release_lx",0))/2560.0
            vec[6]=float(e.get("release_ly",0))/1600.0
            ctx.append(vec[:self.context_dim])
        if ctx:
            tens=torch.tensor(ctx,dtype=torch.float32,device=self.device).unsqueeze(0)
            self.ctx_buffer.append(tens)
        if len(self.ctx_buffer)==0:
            return torch.zeros((1,0,self.context_dim),dtype=torch.float32,device=self.device)
        joined=torch.cat(list(self.ctx_buffer),dim=1)
        return joined[:,-8:,:]
    def predict(self,img,heat_prior=None,event_prior=None):
        self.ensure_integrity()
        h,w=img.shape[:2]
        inp=cv2.resize(img,(320,200))
        ten=torch.from_numpy(inp).to(self.device).float().permute(2,0,1).unsqueeze(0)/255.0
        ctx=self.set_context([])
        with torch.no_grad():
            logits=self.model(ten,ctx)
            prob=torch.sigmoid(logits)
        left=prob[0,0].cpu().numpy()
        score=cv2.resize(left,(w,h),interpolation=cv2.INTER_CUBIC)
        if heat_prior is not None:
            score=score*0.6+heat_prior*0.4
        if event_prior is not None:
            score=score*0.5+event_prior*0.5
        y,x=np.unravel_index(np.argmax(score),score.shape)
        conf=float(score[y,x])
        return (x,y,conf,score)
    def train_incremental(self,batches,progress_cb=None):
        self.model.train()
        opt=optim.AdamW(self.model.parameters(),lr=1e-4,weight_decay=1e-4)
        total=len(batches)
        for i,(img,ctx,label) in enumerate(batches):
            img=img.to(self.device)
            ctx=ctx.to(self.device).float()
            label=label.to(self.device).float()
            opt.zero_grad()
            logits=self.model(img,ctx)
            if logits.shape[-2:]!=label.shape[-2:]:
                label=F.interpolate(label, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            loss=F.binary_cross_entropy_with_logits(logits,label)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            opt.step()
            if progress_cb:
                progress_cb(60+int(35*(i+1)/max(1,total)),f"训练进度 {i+1}/{total}")
        self.model.eval()
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
    def update(files,train_range,extra=None):
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
        rec={"version":v,"time":now,"files":metas,"range":train_range}
        if extra:
            rec.update(extra)
        m.setdefault("history",[]).append(rec)
        m["version"]=v
        m["updated_at"]=now
        ModelMeta.write(m)
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
            cloaked=ctypes.c_int(0)
            ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(hwnd),Foreground.DWMWA_CLOAKED,ctypes.byref(cloaked),ctypes.sizeof(cloaked))
            if int(cloaked.value)!=0:
                return False
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
class ModelIntegrityError(Exception):
    pass
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
    signal_modelprog=Signal(int,str)
    signal_ui_ready=Signal(list)
    MARK=0x22ACE777
    def __init__(self):
        super().__init__()
        with open(cfg_path,"r",encoding="utf-8") as f:
            self.cfg=json.load(f)
        self.manifest=ModelManifest(self.cfg)
        self.manifest._ensure_defaults()
        if not self.cfg.get("model_sha256"):
            self.cfg["model_sha256"]=self.manifest.sha_value
            with open(cfg_path,"w",encoding="utf-8") as f:
                json.dump(self.cfg,f,ensure_ascii=False,indent=2)
                f.flush()
                os.fsync(f.fileno())
        self.strategy=StrategyEngine(self.manifest)
        self.io_q=queue.Queue(maxsize=1024)
        self.io_thread=threading.Thread(target=self._io_worker,daemon=True)
        self.io_thread.start()
        self.prev_lock=threading.Lock()
        self.rect_lock=threading.Lock()
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
        self.model_loaded=False
        self.model_ver_str=""
        self.sal_map=None
        self.last_kbd_log_ts=0.0
        self.day_tag=None
        self.ui_elements=[]
        self.model_ready_event=threading.Event()
        self.model_error=None
        self._init_day_files()
        threading.Thread(target=self._ensure_model_bg,daemon=True).start()
    def _ensure_model_bg(self):
        try:
            self.signal_modelprog.emit(1,"准备模型")
            path=self.manifest.ensure(lambda p,t:self.signal_modelprog.emit(p,t))
            self.strategy.ensure_loaded(lambda p,t:self.signal_modelprog.emit(p,t))
            self.model_loaded=True
            self.model_ready_event.set()
            ModelMeta.update([path],{"start":None,"end":None},{"note":"初始化校验"})
            self.signal_modelver.emit(f"v{ModelMeta.read().get('version',0)}")
        except Exception as e:
            self.model_error=str(e)
            log(f"model ensure error:{e}")
            self.signal_tip.emit(f"模型错误:{e}")
            self.model_ready_event.set()
    def wait_model(self,timeout=0):
        self.model_ready_event.wait(timeout)
        if self.model_error:
            raise ModelIntegrityError(self.model_error)
    def load_cfg(self):
        with open(cfg_path,"r",encoding="utf-8") as f:
            self.cfg=json.load(f)
        return self.cfg
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
        d=os.path.join(exp_dir,day)
        os.makedirs(d,exist_ok=True)
        os.makedirs(os.path.join(d,"frames"),exist_ok=True)
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
        self.wait_model()
        self.selected_hwnd=hwnd
        self.selected_title=title
        self.signal_window.emit(title)
        self.signal_tip.emit("窗口已绑定")
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
    def _io_worker(self):
        while True:
            it=self.io_q.get()
            if it is None:
                break
            p,buf=it
            try:
                with open(p,"wb") as f:
                    buf.tofile(f)
                    f.flush()
                    os.fsync(f.fileno())
            except:
                pass
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
            self.io_q.put((path,buf))
            rec={"id":fid,"ts":now,"source":mode,"mode":mode,"window_title":self.selected_title,"rect":[0,0,2560,1600],"path":path,"filename":f"{fid}.png","w":2560,"h":1600,"session_id":self.session_id}
            self.frames_writer.append(rec)
            with self.prev_lock:
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
    def heat_from_events(self,w,h,source=None):
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
                        if source and e.get("source")!=source:
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
        if not self.model_loaded:
            raise ModelIntegrityError(self.model_error or "模型未准备好")
        sal=self.spectral_saliency(cv2.resize(img,(2560,1600)))
        prior_events=self.heat_from_events(2560,1600,"user")
        x,y,conf,score=self.strategy.predict(cv2.resize(img,(2560,1600)),sal,prior_events)
        if conf<0.25:
            return ("idle",None,None,None)
        return ("click",(x,y),"left",score)
    def periodic(self):
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
            bx,by=self.st.clamp_abs(int(x),int(y))
            self.move_abs(bx,by)
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
            try:
                act=self.st.ai_decide(img)
            except ModelIntegrityError as e:
                self.st.signal_tip.emit(str(e))
                self.st.interrupt_ai=True
                break
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
                    try:
                        cx,cy=win32api.GetCursorPos()
                    except:
                        cx,cy=bx,by
                    n=max(10,int(self.st.fps*0.6))
                    t0=time.time()
                    mv=[]
                    c1x=cx+(bx-cx)*0.3+random.uniform(-12,12)
                    c1y=cy+(by-cy)*0.3+random.uniform(-12,12)
                    c2x=cx+(bx-cx)*0.7+random.uniform(-12,12)
                    c2y=cy+(by-cy)*0.7+random.uniform(-12,12)
                    pts=[]
                    for i in range(1,n+1):
                        u=i/float(n)
                        ux=(1-u)**3*cx+3*(1-u)**2*u*c1x+3*(1-u)*u**2*c2x+u**3*bx
                        uy=(1-u)**3*cy+3*(1-u)**2*u*c1y+3*(1-u)*u**2*c2y+u**3*by
                        pts.append((int(ux),int(uy)))
                    ok=self.inj.move_path(pts,step_delay=max(0.004,0.2/self.st.fps))
                    for (px,py) in pts:
                        mv.append((time.time(),px,py,1))
                    if not ok or self.st.interrupt_ai:
                        break
                    self.inj.down(btn)
                    mv.append((time.time(),bx,by,1))
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
            self.st.signal_optprog.emit(10,"收集数据")
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
                self.st.signal_optprog.emit(20+int(20*done/total),"解析数据")
            if len(events)<30:
                self.st.signal_optprog.emit(100,"数据不足")
                self._done(False)
                return
            batches=self._build_batches(events,frames)
            self.st.signal_optprog.emit(55,"准备训练")
            self.st.strategy.train_incremental(batches,lambda p,t:self.st.signal_optprog.emit(p,t))
            path=self.st.manifest.target_path()
            ModelIO.save(self.st.strategy.model,path)
            ModelMeta.update([path],{"start":tsmin,"end":tsmax},{"samples":len(batches)})
            self.st.strategy.ensure_loaded(lambda p,t:None)
            self.st.model_loaded=True
            self._done(True)
        except Exception as e:
            log(f"opt error:{e}")
            self._done(False)
    def _done(self,ok):
        self.st.optimizing=False
        self.cb(ok)
    def _build_batches(self,events,frames):
        seqs=[]
        for e in events:
            if e.get("type") not in ["left","right","middle"]:
                continue
            fid=e.get("frame_id")
            fr=frames.get(fid)
            if not fr:
                continue
            img=cv2.imdecode(np.fromfile(fr["path"],dtype=np.uint8),cv2.IMREAD_COLOR)
            if img is None:
                continue
            img=cv2.resize(img,(320,200))
            label=self._make_label(e,img.shape[1],img.shape[0])
            if label is None:
                continue
            ctx=self._context_tensor(events,e)
            seqs.append((img,ctx,label))
        random.shuffle(seqs)
        batches=[]
        bs=6
        for i in range(0,len(seqs),bs):
            imgs=[]
            ctxs=[]
            labels=[]
            for (img,ctx,label) in seqs[i:i+bs]:
                imgs.append(torch.from_numpy(img).permute(2,0,1).float()/255.0)
                ctxs.append(ctx)
                labels.append(label)
            img_t=torch.stack(imgs,dim=0)
            ctx_t=torch.stack(ctxs,dim=0)
            labels_t=torch.stack(labels,dim=0)
            batches.append((img_t,ctx_t,labels_t))
        return batches
    def _make_label(self,e,w,h):
        x=int(e.get("press_lx",0)/2560.0*w)
        y=int(e.get("press_ly",0)/1600.0*h)
        if x<0 or x>=w or y<0 or y>=h:
            return None
        grid_h=13
        grid_w=20
        gx=int(max(0,min(grid_w-1,round(x*grid_w/max(1,w)))))
        gy=int(max(0,min(grid_h-1,round(y*grid_h/max(1,h)))))
        label=torch.zeros((2,grid_h,grid_w),dtype=torch.float32)
        label[0,gy,gx]=1.0 if e.get("type")=="left" else 0.0
        label[1,gy,gx]=1.0 if e.get("type")=="right" else 0.0
        return label
    def _context_tensor(self,events,current):
        ctx=[]
        target_ts=current.get("press_t",current.get("ts",0))
        for e in events:
            et=e.get("press_t",e.get("ts",0))
            if et>=target_ts:
                break
            if e.get("type") not in ["left","right","middle"]:
                continue
            vec=[0.0]*self.st.strategy.context_dim
            vec[0]=1.0 if e.get("type")=="left" else 0.0
            vec[1]=1.0 if e.get("type")=="right" else 0.0
            vec[2]=1.0 if e.get("source")=="ai" else 0.0
            vec[3]=float(e.get("press_lx",0))/2560.0
            vec[4]=float(e.get("press_ly",0))/1600.0
            vec[5]=float(e.get("release_lx",0))/2560.0
            vec[6]=float(e.get("release_ly",0))/1600.0
            ctx.append(vec[:self.st.strategy.context_dim])
        ctx=ctx[-8:]
        while len(ctx)<8:
            ctx.insert(0,[0.0]*self.st.strategy.context_dim)
        arr=torch.tensor(ctx,dtype=torch.float32)
        return arr
class UIInspector:
    def __init__(self,st):
        self.st=st
    def analyze(self,img,events):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(gray,50,150)
        contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        elements=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if w<30 or h<18 or w>img.shape[1]-10 or h>img.shape[0]-10:
                continue
            roi=img[y:y+h,x:x+w]
            hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            sat=float(np.mean(hsv[:,:,1]))/255.0
            val=float(np.mean(hsv[:,:,2]))/255.0
            text_density=self._estimate_text_density(roi)
            typ=self._infer_type(w,h,text_density,sat,val)
            interacts=self._interaction_score(events,x,y,w,h)
            elements.append({"type":typ,"bounds":[int(x),int(y),int(x+w),int(y+h)],"text_density":float(text_density),"saturation":float(sat),"brightness":float(val),"interaction":interacts})
        elements=self._merge(elements)
        with open(ui_cache_path,"w",encoding="utf-8") as f:
            json.dump(elements,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
        return elements
    def _estimate_text_density(self,roi):
        g=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        thr=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,8)
        count=np.count_nonzero(thr)
        total=thr.size
        return count/max(1,total)
    def _infer_type(self,w,h,text,sat,val):
        ar=w/float(h)
        if text>0.35 and ar>2.0:
            return "button"
        if text>0.2 and ar<=2.0:
            return "input"
        if h>80 and text<0.1:
            return "panel"
        if sat<0.1 and val>0.9:
            return "label"
        return "widget"
    def _interaction_score(self,events,x,y,w,h):
        total=0
        hits=0
        for e in events:
            if e.get("type") not in ["left","right","middle"]:
                continue
            if e.get("window_title")!=self.st.selected_title:
                continue
            total+=1
            px=int(e.get("press_lx",0)*self.st.client_rect[2]/2560.0) if self.st.client_rect[2]>0 else 0
            py=int(e.get("press_ly",0)*self.st.client_rect[3]/1600.0) if self.st.client_rect[3]>0 else 0
            if px>=x and px<=x+w and py>=y and py<=y+h:
                hits+=1
        return float(hits)/max(1,total)
    def _merge(self,elements):
        merged=[]
        for el in elements:
            overlapped=False
            for m in merged:
                if self._iou(el["bounds"],m["bounds"])>0.4:
                    if el["interaction"]>m["interaction"]:
                        m.update(el)
                    overlapped=True
                    break
            if not overlapped:
                merged.append(el)
        merged.sort(key=lambda x:x["interaction"],reverse=True)
        return merged
    def _iou(self,a,b):
        ax1,ay1,ax2,ay2=a
        bx1,by1,bx2,by2=b
        ix1=max(ax1,bx1)
        iy1=max(ay1,by1)
        ix2=min(ax2,bx2)
        iy2=min(ay2,by2)
        if ix2<=ix1 or iy2<=iy1:
            return 0.0
        inter=(ix2-ix1)*(iy2-iy1)
        area_a=(ax2-ax1)*(ay2-ay1)
        area_b=(bx2-bx1)*(by2-by1)
        return inter/max(1,area_a+area_b-inter)
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
class Main(QMainWindow):
    finish_opt=Signal(bool)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能学习与训练")
        self.state=State()
        self.selector=WindowSelector()
        self.inspector=UIInspector(self.state)
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
        self.lbl_tip=QLabel("初始化")
        self.progress=QProgressBar()
        self.progress.setRange(0,100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.model_progress=QProgressBar()
        self.model_progress.setRange(0,100)
        self.model_progress.setValue(0)
        self.model_progress.setTextVisible(True)
        self.preview_label=QLabel()
        self.preview_label.setFixedSize(QSize(640,400))
        self.ui_list=QListWidget()
        self.ui_list.setMaximumHeight(140)
        lt=QHBoxLayout(top)
        lt.addWidget(self.cmb,1)
        lt.addWidget(self.btn_toggle)
        lt.addWidget(self.btn_opt)
        lt.addWidget(self.btn_ui)
        lt.addWidget(self.chk_preview)
        lm=QVBoxLayout(mid)
        lm.addWidget(self.preview_label,1)
        lm.addWidget(self.ui_list)
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
        lb.addWidget(self.model_progress,2)
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
        self.state.signal_modelprog.connect(self.on_modelprog)
        self.state.signal_ui_ready.connect(self.on_ui_ready)
        self.refresh_timer=QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_windows)
        self.refresh_timer.start(2000)
        self.mode_timer=QTimer(self)
        self.mode_timer.timeout.connect(self.on_tick_mode)
        self.mode_timer.start(200)
        self.fps_timer=QTimer(self)
        self.fps_timer.timeout.connect(self.on_tick_fps)
        self.fps_timer.start(500)
        self.usage_timer=QTimer(self)
        self.usage_timer.timeout.connect(self.on_usage)
        self.usage_timer.start(1000)
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
        self.lbl_tip.setText(f"当前窗口: {s}")
    def on_recsize(self,s):
        self.lbl_rec.setText(s)
    def on_modelver(self,s):
        self.lbl_ver.setText(s)
    def on_tip(self,s):
        self.lbl_tip.setText(s)
    def on_optprog(self,val,txt):
        self.progress.setValue(int(val))
        self.progress.setFormat(str(txt))
    def on_modelprog(self,val,txt):
        self.model_progress.setValue(int(val))
        self.model_progress.setFormat(str(txt))
    def on_ui_ready(self,items):
        self.ui_list.clear()
        for it in items[:30]:
            item=QListWidgetItem(f"{it['type']}|{it['bounds']}|交互:{it['interaction']:.2f}")
            self.ui_list.addItem(item)
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
        if self.state.optimizing:
            return
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
            if not self.state.model_loaded:
                QMessageBox.information(self,"提示","模型不可用，保持学习模式")
                self.state.mode="learning"
                self.lbl_mode.setText("学习")
                self.learning_thread=LearningThread(self.state,self)
                self.learning_thread.start()
                return
            self.lbl_mode.setText("训练")
            self.state.signal_mode.emit("训练")
            self.training_thread=TrainerThread(self.state,self)
            self.training_thread.start()
    def _poll_user_activity(self):
        try:
            cx,cy=win32api.GetCursorPos()
        except:
            return
        if not hasattr(self,"_last_pos"):
            self._last_pos=(cx,cy)
        if self._last_pos!=(cx,cy):
            self.state.last_user_activity=time.time()
            if self.state.mode=="training":
                self.state.interrupt_ai=True
                self.state.signal_mode.emit("学习")
        self._last_pos=(cx,cy)
        for vk in [0x01,0x02,0x04,0x05,0x06,0x0D,0x10,0x11,0x12,0x1B,0x20,0x25,0x26,0x27,0x28]:
            try:
                if ctypes.windll.user32.GetAsyncKeyState(vk)&0x8000:
                    self.state.last_user_activity=time.time()
                    if self.state.mode=="training":
                        self.state.interrupt_ai=True
                        self.state.signal_mode.emit("学习")
                    break
            except:
                break
    def on_tick_mode(self):
        if not self.state.running or self.state.optimizing:
            return
        self._poll_user_activity()
        try:
            self.state.wait_model()
        except ModelIntegrityError as e:
            self.lbl_tip.setText(str(e))
            return
        t=time.time()-self.state.last_user_activity
        if self.state.mode=="learning":
            if t>=self.state.idle_timeout and self.state.selected_hwnd and self.state.model_loaded:
                self.set_mode("training")
        else:
            if self.state.interrupt_ai:
                self.state.interrupt_ai=False
                self.set_mode("learning")
    def on_tick_fps(self):
        self.lbl_fps.setText(f"FPS:{int(self.state.fps)}")
    def on_usage(self):
        cpu=int(psutil.cpu_percent())
        mem=int(psutil.virtual_memory().percent)
        gu,gm=_gpu_util_mem()
        info=f"CPU:{cpu}% MEM:{mem}%"
        if gu is not None:
            info+=f" GPU:{int(gu)}% VRAM:{int(gm)}%"
        self.statusBar().showMessage(info)
    def on_toggle(self):
        if self.state.optimizing:
            QMessageBox.information(self,"提示","优化进行中，无法切换")
            return
        if self.state.running:
            self.state.running=False
            self.state.stop_event.set()
            self.btn_toggle.setText("开始")
            self.lbl_tip.setText("已停止")
        else:
            try:
                self.state.wait_model()
            except ModelIntegrityError as e:
                QMessageBox.critical(self,"错误",str(e))
                return
            self.state.running=True
            self.state.stop_event.clear()
            self.btn_toggle.setText("停止")
            self.lbl_tip.setText("运行中")
            self.set_mode("learning")
    def on_opt(self):
        if self.state.optimizing:
            return
        try:
            self.state.wait_model()
        except ModelIntegrityError as e:
            QMessageBox.critical(self,"错误",str(e))
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
        events=[]
        if self.state.events_path and os.path.exists(self.state.events_path):
            with open(self.state.events_path,"r",encoding="utf-8") as f:
                for line in f:
                    try:
                        events.append(json.loads(line))
                    except:
                        pass
        res=self.inspector.analyze(cv2.resize(img,(1280,800)),events)
        self.state.signal_ui_ready.emit(res)
        QMessageBox.information(self,"提示","UI识别完成，结果已列出")
    def closeEvent(self,event):
        self.state.stop_event.set()
        self.state.running=False
        self.hook.stop()
        self.state.io_q.put(None)
        event.accept()
def main():
    app=QApplication(sys.argv)
    w=Main()
    w.resize(960,720)
    w.show()
    sys.exit(app.exec())
if __name__=="__main__":
    main()
