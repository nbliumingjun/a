import os,sys,ctypes,ctypes.wintypes,threading,time,uuid,json,random,collections,re,hashlib,importlib,importlib.util,glob,shutil,math,zlib,urllib.request,ssl,io,base64,subprocess
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
data_cache_path=os.path.join(base_dir,"window_data.json")
schema_path=os.path.join(base_dir,"ui_schema.json")
for d in [base_dir,exp_dir,models_dir,logs_dir]:
    os.makedirs(d,exist_ok=True)
log_path=os.path.join(logs_dir,"app.log")
def log(s):
    try:
        with open(log_path,"a",encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")
    except:
        pass
cfg_defaults={"idle_timeout":10,"screenshot_min_fps":1,"screenshot_max_fps":120,"ui_scale":1.0,"frame_png_compress":3,"save_change_thresh":6.0,"max_disk_gb":10.0,"pre_post_K":3,"block_margin_px":6,"preview_on":True,"model_repo":"https://huggingface.co/datasets/mit-han-lab/aitk-mouse-policy/resolve/main/","model_file":"policy_v1.npz","model_sha256":"","model_fallback":"","model_sha256_backup":"","ui_preferences":{"__default__":"higher"},"hyperparam_state":{},"data_preferences":{"__default__":"higher"}}
default_ui_labels=["button","input","panel","label","widget","menu","icon","ability","joystick","skill","toggle","minimap"]
schema_defaults={"labels":default_ui_labels,"max_items":48}
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
        prefs=dict(merged.get("ui_preferences")) if isinstance(merged.get("ui_preferences"),dict) else {}
        if "ui_value_orientation" in merged:
            ori=str(merged.get("ui_value_orientation") or "")
            prefs=dict(prefs)
            prefs["__default__"]="lower" if ori.lower().startswith("low") else "higher"
            del merged["ui_value_orientation"]
        if "__default__" not in prefs:
            prefs["__default__"]=cfg_defaults["ui_preferences"]["__default__"]
        if "control" not in prefs:
            prefs["control"]=prefs["__default__"]
        merged["ui_preferences"]=prefs
        data_prefs=dict(merged.get("data_preferences")) if isinstance(merged.get("data_preferences"),dict) else {}
        if "__default__" not in data_prefs:
            data_prefs["__default__"]=cfg_defaults["data_preferences"]["__default__"]
        merged["data_preferences"]=data_prefs
        with open(cfg_path,"w",encoding="utf-8") as f:
            json.dump(merged,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
ensure_config()
def ensure_schema():
    data=None
    if os.path.exists(schema_path):
        try:
            with open(schema_path,"r",encoding="utf-8") as f:
                data=json.load(f)
        except:
            data=None
    if not isinstance(data,dict):
        data=dict(schema_defaults)
    else:
        merged=dict(schema_defaults)
        labs=[lab for lab in data.get("labels",[]) if isinstance(lab,str) and lab]
        if labs:
            merged["labels"]=labs
        max_items=int(data.get("max_items",merged.get("max_items",48)))
        merged["max_items"]=max(4,min(128,max_items))
        data=merged
    with open(schema_path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
        f.flush()
        os.fsync(f.fileno())
ensure_schema()
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
    core=[("numpy","numpy"),("cv2","opencv-python"),("psutil","psutil"),("mss","mss"),("PySide6","PySide6"),("win32gui","pywin32"),("win32con","pywin32"),("win32api","pywin32"),("win32process","pywin32"),("torch","torch"),("torchvision","torchvision"),("scipy","scipy"),("sklearn","scikit-learn"),("pynvml","pynvml"),("comtypes","comtypes")]
    attempt=0
    max_attempts=3
    while True:
        miss=[]
        for mod,pkg in core:
            if importlib.util.find_spec(mod) is None:
                miss.append((mod,pkg))
        if not miss:
            break
        attempt+=1
        pkg_names=sorted(set(pkg for _,pkg in miss))
        _msgbox("依赖准备","检测到缺失依赖:\n"+"\n".join(pkg_names)+"\n系统将自动安装")
        cmd=[sys.executable,"-m","pip","install"]+pkg_names
        try:
            print("installing dependency bundle:"+" ".join(pkg_names))
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            if proc.stdout is not None:
                for line in proc.stdout:
                    msg=line.rstrip()
                    if msg:
                        log("pip:"+msg)
            code=proc.wait()
            if code!=0:
                raise subprocess.CalledProcessError(code,cmd)
        except Exception as e:
            hint=f"install bundle failed:{e}"
            log(hint)
            log("suggestion:verify network connectivity, pip index accessibility, and administrator permissions")
            print(hint)
        if attempt>=max_attempts:
            remain=sorted(set(mod for mod,_ in miss))
            msg="自动安装失败，请手动安装缺失依赖: "+", ".join(remain)+f"\n请检查网络或权限设置，并查看日志:{log_path}"
            print(msg)
            _msgbox("依赖安装",msg)
            break
        time.sleep(1)
    miss_final=[]
    for mod,_ in core:
        if importlib.util.find_spec(mod) is None:
            miss_final.append(mod)
    if miss_final:
        msg="依赖未满足:"+",".join(miss_final)
        _msgbox("依赖检查",msg)
        raise RuntimeError(msg)
    d={"numpy":_ver("numpy"),"opencv-python":_ver("opencv-python","opencv-python"),"psutil":_ver("psutil"),"mss":_ver("mss"),"PySide6":_ver("PySide6"),"pywin32":_ver("pywin32","pywin32"),"torch":_ver("torch"),"torchvision":_ver("torchvision"),"scipy":_ver("scipy"),"scikit-learn":_ver("scikit-learn","scikit-learn"),"pynvml":_ver("pynvml"),"comtypes":_ver("comtypes")}
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
if os.name=="nt":
    try:
        import comtypes.client as _ct_client
    except:
        _ct_client=None
else:
    _ct_client=None
try:
    import pynvml as _nv
    _nv.nvmlInit()
except:
    _nv=None
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QComboBox,QLabel,QCheckBox,QMessageBox,QProgressBar,QTableWidget,QDialog,QDialogButtonBox,QTableWidgetItem,QHeaderView,QAbstractItemView,QSizePolicy
from PySide6.QtCore import QTimer,Qt,Signal,QObject,QSize
from PySide6.QtGui import QImage,QPixmap,QColor
def _gpu_util_mem():
    try:
        if _nv is None:
            return None,None
        count=_nv.nvmlDeviceGetCount()
        if count<=0:
            return None,None
        util=0.0
        mem=0.0
        for i in range(count):
            h=_nv.nvmlDeviceGetHandleByIndex(i)
            u=_nv.nvmlDeviceGetUtilizationRates(h)
            info=_nv.nvmlDeviceGetMemoryInfo(h)
            util=max(util,float(u.gpu))
            mem=max(mem,float(info.used)*100.0/max(1.0,float(info.total)))
        return util,mem
    except:
        return None,None
class RewardComposer:
    def __init__(self):
        self.summary={"win_rate":0.5,"objective":0.0,"task":0.0}
        self.lock=threading.Lock()
        self.last_scan=0.0
    def _scan(self):
        if time.time()-self.last_scan<5.0:
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            wins=0
            total=0
            objective_sum=0.0
            objective_count=0
            task_sum=0.0
            task_count=0
            for path in glob.glob(os.path.join(logs_dir,"*.json*")):
                if os.path.isdir(path):
                    continue
                try:
                    if path.endswith(".jsonl"):
                        with open(path,"r",encoding="utf-8") as f:
                            for line in f:
                                obj=json.loads(line)
                                if isinstance(obj,dict):
                                    wins,total,objective_sum,objective_count,task_sum,task_count=self._accumulate(obj,wins,total,objective_sum,objective_count,task_sum,task_count)
                    else:
                        with open(path,"r",encoding="utf-8") as f:
                            obj=json.load(f)
                            if isinstance(obj,dict):
                                wins,total,objective_sum,objective_count,task_sum,task_count=self._accumulate(obj,wins,total,objective_sum,objective_count,task_sum,task_count)
                            elif isinstance(obj,list):
                                for item in obj:
                                    if isinstance(item,dict):
                                        wins,total,objective_sum,objective_count,task_sum,task_count=self._accumulate(item,wins,total,objective_sum,objective_count,task_sum,task_count)
                except:
                    continue
            rate=wins/max(1,total)
            objective=objective_sum/max(1,objective_count)
            task=task_sum/max(1,task_count)
            self.summary={"win_rate":max(0.0,min(1.0,rate)),"objective":max(0.0,objective),"task":max(0.0,task)}
            self.last_scan=time.time()
        finally:
            self.lock.release()
    def _accumulate(self,obj,wins,total,objective_sum,objective_count,task_sum,task_count):
        result=str(obj.get("result",""))
        if result:
            total+=1
            low=result.lower()
            if low in ["win","victory","获胜","success"]:
                wins+=1
        for key in ["objective_score","score","kills","damage","economy","tower_destroyed","stars"]:
            if key in obj:
                try:
                    objective_sum+=float(obj.get(key,0.0))
                    objective_count+=1
                except:
                    continue
        for key in ["tasks","task_progress","missions","quest","quests"]:
            if key in obj:
                val=obj.get(key)
                if isinstance(val,(int,float)):
                    task_sum+=float(val)
                    task_count+=1
                elif isinstance(val,dict):
                    try:
                        total_val=sum(float(v) for v in val.values())
                        task_sum+=total_val
                        task_count+=len(val)
                    except:
                        continue
        return wins,total,objective_sum,objective_count,task_sum,task_count
    def metrics(self):
        self._scan()
        return dict(self.summary)
    def evaluate(self,event):
        metrics=self.metrics()
        base=0.1
        typ=str(event.get("type",""))
        if typ=="left":
            base+=0.18
        elif typ=="right":
            base+=0.12
        elif typ=="middle":
            base+=0.08
        duration=max(0.0,float(event.get("duration",0.0) or 0.0))
        inside=float(event.get("ins_press",0))+float(event.get("ins_release",0))
        inside=max(0.0,min(2.0,inside))
        moves=event.get("moves") or []
        span=0.0
        if moves:
            xs=[m[1] for m in moves if isinstance(m,(list,tuple)) and len(m)>=3]
            ys=[m[2] for m in moves if isinstance(m,(list,tuple)) and len(m)>=3]
            if xs and ys:
                span=(max(xs)-min(xs)+max(ys)-min(ys))/4096.0
        combo=len(event.get("clip_ids") or [])
        synergy=0.6*metrics.get("win_rate",0.5)+0.25*min(1.5,metrics.get("objective",0.0)/10.0)+0.15*min(1.5,metrics.get("task",0.0)/5.0)
        reward=base*(0.7+synergy)
        reward+=min(0.5,span)
        reward+=min(0.45,combo*0.06*(1.0+metrics.get("win_rate",0.5)))
        reward+=min(0.4,duration*0.5)
        reward+=min(0.3,inside*0.1)
        if event.get("source")=="ai":
            reward*=0.95
        penalty=0.0
        if duration>1.5:
            penalty+=0.1
        if span>0.8:
            penalty+=0.08
        reward=max(0.0,reward-penalty)
        return max(0.0,min(2.2,reward))
class ModelManifest:
    def __init__(self,cfg):
        self.repo=cfg.get("model_repo",cfg_defaults["model_repo"])
        self.file=cfg.get("model_file",cfg_defaults["model_file"])
        self.sha_value=cfg.get("model_sha256",cfg_defaults["model_sha256"]) or ""
    def url(self):
        return self.repo+self.file
    def target_path(self):
        return os.path.join(models_dir,self.file)
    def ensure(self,progress_cb=None):
        path=self.target_path()
        if os.path.exists(path):
            if not self.sha_value:
                if progress_cb:
                    progress_cb(100,"模型已就绪")
                return path
            if self._hash(path)==self.sha_value:
                if progress_cb:
                    progress_cb(100,"模型已就绪")
                return path
            try:
                os.remove(path)
            except:
                pass
        if progress_cb:
            progress_cb(5,"下载模型")
        tmp=self._download_with_progress(progress_cb)
        if not tmp:
            try:
                os.remove(path+".tmp")
            except:
                pass
            fall=self._recover_fallback(progress_cb)
            if fall:
                return fall
            raise RuntimeError("模型下载失败")
        calc=self._hash(tmp)
        if self.sha_value:
            if calc!=self.sha_value:
                try:
                    os.remove(tmp)
                except:
                    pass
                raise RuntimeError("模型哈希校验失败")
        else:
            self.sha_value=calc
            self._update_cfg_hash(calc)
        shutil.move(tmp,path)
        if progress_cb:
            progress_cb(100,"模型准备完成")
        return path
    def _download_with_progress(self,cb):
        url=self.url()
        target=self.target_path()
        tmp=target+".tmp"
        attempts=0
        while attempts<3:
            try:
                existing=os.path.getsize(tmp) if os.path.exists(tmp) else 0
                headers={"User-Agent":"Mozilla/5.0"}
                if existing>0:
                    headers["Range"]=f"bytes={existing}-"
                req=urllib.request.Request(url,headers=headers)
                with urllib.request.urlopen(req,timeout=60) as r:
                    total=r.length or 0
                    cr=r.headers.get("Content-Range")
                    if cr:
                        try:
                            total=int(cr.split("/")[-1])
                        except:
                            total=0
                    status=getattr(r,"status",200)
                    if existing>0 and (status!=206 or not cr):
                        existing=0
                        with open(tmp,"wb") as f:
                            f.truncate(0)
                    mode="ab" if existing>0 else "wb"
                    done=existing
                    chunk=262144
                    with open(tmp,mode) as f:
                        if existing>0:
                            f.seek(0,os.SEEK_END)
                        while True:
                            data=r.read(chunk)
                            if not data:
                                break
                            f.write(data)
                            done+=len(data)
                            if cb and total>0:
                                cb(min(90,int(done*80/max(1,total))+10),"下载中")
                        f.flush()
                        os.fsync(f.fileno())
                if total and done<total:
                    raise IOError("incomplete download")
                if cb:
                    cb(95,"下载完成")
                return tmp
            except Exception as e:
                log(f"model download error:{e}")
                attempts+=1
                time.sleep(1)
        return None
    def _hash(self,path):
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1<<20),b""):
                h.update(chunk)
        return h.hexdigest()
    def _update_cfg_hash(self,val):
        try:
            with open(cfg_path,"r",encoding="utf-8") as f:
                cfg=json.load(f)
        except:
            cfg=dict(cfg_defaults)
        cfg["model_sha256"]=val
        cfg["model_sha256_backup"]=val
        try:
            with open(cfg_path,"w",encoding="utf-8") as f:
                json.dump(cfg,f,ensure_ascii=False,indent=2)
                f.flush()
                os.fsync(f.fileno())
        except:
            pass
    def _recover_fallback(self,progress_cb):
        try:
            if progress_cb:
                progress_cb(15,"尝试恢复模型")
            backup=self._find_backup_file()
            if backup and os.path.exists(backup):
                target=self.target_path()
                if os.path.abspath(backup)!=os.path.abspath(target):
                    shutil.copy2(backup,target)
                if self.sha_value:
                    if self._hash(target)!=self.sha_value:
                        os.remove(target)
                        backup=None
                if backup:
                    if progress_cb:
                        progress_cb(95,"恢复模型")
                    return target
        except:
            pass
        try:
            fallback=self._default_model_path()
            if fallback:
                target=self.target_path()
                if os.path.abspath(fallback)!=os.path.abspath(target):
                    shutil.copy2(fallback,target)
                if self.sha_value:
                    if self._hash(target)!=self.sha_value:
                        os.remove(target)
                        return None
                if progress_cb:
                    progress_cb(95,"使用本地模型")
                return target
        except:
            pass
        try:
            data=ModelInitializer.default_bytes()
            target=self.target_path()
            with open(target,"wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            calc=self._hash(target)
            self.sha_value=calc
            self._update_cfg_hash(calc)
            if progress_cb:
                progress_cb(98,"生成默认模型")
            return target
        except:
            return None
    def _find_backup_file(self):
        try:
            cfg=json.load(open(cfg_path,"r",encoding="utf-8"))
        except:
            cfg=dict(cfg_defaults)
        prefer=cfg.get("model_fallback")
        if prefer and os.path.exists(prefer):
            return prefer
        backup_hash=cfg.get("model_sha256_backup")
        if backup_hash:
            cand=glob.glob(os.path.join(models_dir,"*.npz"))
            for fp in cand:
                try:
                    if self._hash(fp)==backup_hash:
                        return fp
                except:
                    continue
        existing=glob.glob(os.path.join(models_dir,"*.npz"))
        if existing:
            return existing[0]
        return None
    def _default_model_path(self):
        fallback=os.path.join(models_dir,"default_policy.npz")
        if os.path.exists(fallback):
            return fallback
        try:
            data=ModelInitializer.default_bytes()
            with open(fallback,"wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            return fallback
        except:
            return None
class JSONLWriter:
    def __init__(self,path):
        self.path=path
        self.f=open(self.path,"a",encoding="utf-8")
        self.lock=threading.Lock()
        self.buffer=[]
        self.flush_limit=48
        self.flush_interval=0.5
        self.last_flush=time.time()
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
                        if c is not None:
                            try:
                                exp=int(c)
                            except:
                                break
                            if exp!=JSONLWriter._crc_of(obj):
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
                self.buffer.append(line)
                now=time.time()
                if len(self.buffer)>=self.flush_limit or now-self.last_flush>=self.flush_interval:
                    self._flush_locked()
            return True
        except:
            return False
    def _flush_locked(self,force=False):
        if not self.buffer and not force:
            return
        self.f.writelines(self.buffer)
        self.buffer.clear()
        self.f.flush()
        os.fsync(self.f.fileno())
        self.last_flush=time.time()
    def close(self):
        try:
            with self.lock:
                self._flush_locked(True)
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
        target=cap*0.9
        files=[]
        for r,_,fs in os.walk(exp_dir):
            for n in fs:
                try:
                    p=os.path.join(r,n)
                    files.append((os.path.getmtime(p),p,os.path.getsize(p)))
                except:
                    pass
        files.sort()
        for _,p,size in files:
            if t<=target:
                break
            try:
                os.remove(p)
                t-=size
            except:
                pass
        for r,ds,_ in os.walk(exp_dir,topdown=False):
            if ds:
                continue
            try:
                if not os.listdir(r):
                    os.rmdir(r)
            except:
                pass
        if t>cap:
            dirs=[]
            for d in glob.glob(os.path.join(exp_dir,"*")):
                if not os.path.isdir(d):
                    continue
                try:
                    dirs.append((os.path.getmtime(d),d))
                except:
                    pass
            dirs.sort()
            for _,d in dirs:
                shutil.rmtree(d,ignore_errors=True)
                t=DiskManager.total_bytes()
                if t<=target:
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
        self.core=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU())
    def forward(self,x):
        return self.core(x)
class PolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=VisionBackbone()
        self.reducer=nn.Conv2d(192,192,1)
        self.temporal=nn.GRU(192,192,batch_first=True)
        self.transformer=PolicyTransformer()
        self.head=PolicyHead(192)
        self.click_proj=nn.Linear(192,2)
        self.path_proj=nn.Linear(192,1)
        self.data_map_proj=nn.Linear(192,1)
        self.data_head=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,192),nn.GELU(),nn.Linear(192,16))
        self.value=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,192),nn.GELU(),nn.Linear(192,1))
        self.traj_encoder=nn.GRU(4,192,batch_first=True)
        self.sequence_attn=nn.MultiheadAttention(192,6,batch_first=True)
    def forward(self,img,context,history=None,trajectory=None,traj_mask=None):
        feat=self.backbone(img)
        feat=self.reducer(feat)
        B,C,H,W=feat.shape
        tokens=feat.flatten(2).permute(0,2,1)
        if history is not None and history.shape[1]>0:
            h,_=self.temporal(history)
            tokens=torch.cat([h,tokens],dim=1)
        if context is not None and context.shape[1]>0:
            tokens=torch.cat([context,tokens],dim=1)
        if trajectory is not None and trajectory.shape[1]>0:
            traj_feat,_=self.traj_encoder(trajectory)
            if traj_mask is not None and traj_mask.shape[:2]==traj_feat.shape[:2]:
                mask=traj_mask.unsqueeze(-1)
                denom=mask.sum(dim=1).clamp(min=1.0)
                pooled=(traj_feat*mask).sum(dim=1)/denom
            else:
                pooled=traj_feat.mean(dim=1)
            tokens=torch.cat([pooled.unsqueeze(1),tokens],dim=1)
        attn,_=self.sequence_attn(tokens,tokens,tokens,need_weights=False)
        enc=self.transformer(attn)
        core=self.head(enc)
        value=self.value(core.mean(dim=1))
        tail=core[:, -H*W:, :]
        logits=self.click_proj(tail).view(B,H,W,2).permute(0,3,1,2)
        path=self.path_proj(tail).view(B,H,W).unsqueeze(1)
        data_map=self.data_map_proj(tail).view(B,H,W).unsqueeze(1)
        data_vec=self.data_head(core.mean(dim=1))
        return logits,value,path,data_map,data_vec
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
        current=model.state_dict()
        with np.load(path,allow_pickle=False) as d:
            for k in current.keys():
                if k in d.files:
                    current[k]=torch.from_numpy(d[k])
        model.load_state_dict(current,strict=False)
class TemporalPlanner(nn.Module):
    def __init__(self,dim=192):
        super().__init__()
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim,nhead=6,dim_feedforward=dim*4,batch_first=True,activation="gelu"),num_layers=3)
        self.policy=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
    def forward(self,seq):
        enc=self.encoder(seq)
        plan=self.policy(enc)
        return plan
class SemanticAggregator(nn.Module):
    def __init__(self,dim=128,out_dim=192):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),nn.Conv2d(64,dim,3,2,1),nn.GELU())
        self.gru=nn.GRU(dim,dim,batch_first=True)
        self.proj=nn.Linear(dim,out_dim)
    def forward(self,x):
        feat=self.conv(x)
        B,C,H,W=feat.shape
        seq=feat.view(B,C,H*W).permute(0,2,1)
        out,_=self.gru(seq)
        pooled=out.mean(dim=1)
        return self.proj(pooled)
class GameSceneInterpreter(nn.Module):
    def __init__(self,hidden=192):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.Conv2d(128,hidden,3,2,1),nn.GELU())
        self.temporal=nn.GRU(hidden,hidden,batch_first=True)
        self.proj=nn.Linear(hidden,hidden)
    def forward(self,x,state=None):
        if state is not None:
            state=state.detach()
        feat=self.spatial(x)
        pooled=F.adaptive_avg_pool2d(feat,(1,1)).view(feat.shape[0],1,feat.shape[1])
        out,state=self.temporal(pooled,state)
        emb=self.proj(out[:, -1, :])
        return emb,state
class MultiTargetDecision(nn.Module):
    def __init__(self,dim=192):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(dim*3),nn.Linear(dim*3,dim),nn.GELU(),nn.Linear(dim,dim//2),nn.GELU(),nn.Linear(dim//2,3))
    def forward(self,scene,plan,ui):
        joined=torch.cat([scene,plan,ui],dim=-1)
        return torch.softmax(self.net(joined),dim=-1)
class AdaptiveHyperParams:
    def __init__(self,cfg):
        self.bounds={"batch_size":(4,16),"gamma":(0.8,0.99),"lam":(0.5,0.95),"confidence_floor":(0.05,0.4),"primary_threshold":(0.3,0.85),"long_press_threshold":(0.45,0.95),"drag_threshold":(0.3,0.85),"multi_gate":(0.1,0.9),"confidence_margin":(0.05,0.35),"drag_steps":(8,64),"parallel_streams":(1,4),"entropy":(0.0,0.1)}
        defaults={"batch_size":6,"gamma":0.92,"lam":0.85,"confidence_floor":0.18,"primary_threshold":0.55,"long_press_threshold":0.78,"drag_threshold":0.58,"multi_gate":0.5,"confidence_margin":0.1,"drag_steps":28,"parallel_streams":2,"entropy":0.02}
        saved=cfg.get("hyperparam_state") if isinstance(cfg.get("hyperparam_state"),dict) else {}
        vals=dict(defaults)
        for k,v in saved.items():
            if k in defaults:
                lo,hi=self.bounds[k]
                try:
                    vals[k]=min(hi,max(lo,float(v)))
                except:
                    vals[k]=defaults[k]
        self.values=vals
        self.pref_map={}
        self.prefers_high=True
        self.save_cb=None
        self.set_preferences(cfg.get("ui_preferences"))
        self.metrics=collections.deque(maxlen=240)
        self.ai_history=collections.deque(maxlen=360)
        self.resource=collections.deque(maxlen=240)
        self.last_update=time.time()
        self._last_snapshot=None
    def snapshot(self):
        data=dict(self.values)
        data["prefers_high"]=self.prefers_high
        data["ui_preferences"]=dict(self.pref_map)
        return data
    def set_save_callback(self,cb):
        self.save_cb=cb
    def set_orientation(self,prefer_high):
        self.prefers_high=bool(prefer_high)
        self._notify()
    def set_preferences(self,prefs):
        clean=self._normalize_prefs(prefs)
        if clean!=getattr(self,"pref_map",{}):
            self.pref_map=clean
        vals=[v for v in clean.values() if v in ["higher","lower"]]
        if vals:
            hi=sum(1 for v in vals if v=="higher")
            self.prefers_high=hi>=((len(vals)+1)//2)
        else:
            self.prefers_high=True
        self._notify()
    def _normalize_prefs(self,prefs):
        data={}
        if isinstance(prefs,dict):
            for k,v in prefs.items():
                if not isinstance(k,str):
                    continue
                nv=str(v).lower()
                if nv.startswith("low"):
                    data[k]="lower"
                elif nv.startswith("high"):
                    data[k]="higher"
                elif nv.startswith("无关") or nv.startswith("ignore"):
                    data[k]="ignore"
        if "__default__" not in data:
            data["__default__"]=cfg_defaults["ui_preferences"]["__default__"]
        return data
    def observe_event(self,event):
        if not isinstance(event,dict):
            return
        if event.get("source")!="ai":
            return
        duration=float(event.get("duration",0))
        inside=float(event.get("ins_release",0))
        coverage=float(event.get("ins_press",0))
        self.ai_history.append((duration,inside,coverage,time.time()))
    def observe_metrics(self,fps,motion,resource,mode):
        self.metrics.append((float(fps),float(motion),float(resource),time.time(),mode))
    def observe_resource(self,cpu,mem,gpu,vram):
        self.resource.append((float(cpu),float(mem),float(gpu),float(vram),time.time()))
    def adaptive_entropy(self):
        vals=[max(item[0],item[1],item[2],item[3]) for item in self.resource] if self.resource else []
        if not vals:
            return self.values["entropy"]
        avg=sum(vals)/len(vals)
        target=0.02 if avg<70 else (0.06 if avg<85 else 0.08)
        self.values["entropy"]=min(self.bounds["entropy"][1],max(self.bounds["entropy"][0],target))
        return self.values["entropy"]
    def adjust(self):
        if time.time()-self.last_update<2.5:
            return
        self.last_update=time.time()
        if self.metrics:
            recent=[m for m in self.metrics if time.time()-m[3]<30]
            if recent:
                avg_motion=sum(m[1] for m in recent)/len(recent)
                avg_fps=sum(m[0] for m in recent)/len(recent)
                pressure=sum(m[2] for m in recent)/len(recent)
                if avg_motion>24 and avg_fps<self.values["drag_steps"]:
                    self._shift("drag_steps",3)
                if avg_motion<10 and avg_fps>self.values["drag_steps"]*1.8:
                    self._shift("drag_steps",-2)
                if pressure>90:
                    self._shift("batch_size",-1)
                    self._shift_float("gamma",-0.01)
                elif pressure<70:
                    self._shift("batch_size",1)
                    self._shift_float("gamma",0.005)
        if self.ai_history:
            recent=[h for h in self.ai_history if time.time()-h[3]<45]
            if recent:
                avg_inside=sum(h[1] for h in recent)/len(recent)
                avg_duration=sum(h[0] for h in recent)/len(recent)
                avg_start=sum(h[2] for h in recent)/len(recent)
                if avg_inside<0.6:
                    self._shift_float("primary_threshold",-0.02)
                    self._shift_float("confidence_floor",0.01)
                else:
                    self._shift_float("primary_threshold",0.015)
                if avg_duration>0.45:
                    self._shift_float("long_press_threshold",0.02)
                else:
                    self._shift_float("long_press_threshold",-0.02)
                if avg_start<0.4:
                    self._shift_float("multi_gate",-0.03)
                else:
                    self._shift_float("multi_gate",0.02)
        self._notify()
    def _shift(self,key,delta):
        lo,hi=self.bounds[key]
        self.values[key]=int(min(hi,max(lo,self.values[key]+delta)))
    def _shift_float(self,key,delta):
        lo,hi=self.bounds[key]
        self.values[key]=float(min(hi,max(lo,self.values[key]+delta)))
    def _notify(self):
        cb=getattr(self,"save_cb",None)
        if cb:
            snap=tuple((k,self.values[k]) for k in sorted(self.values.keys()))
            if snap!=self._last_snapshot:
                self._last_snapshot=snap
                cb(self.values)
class StrategyEngine:
    def __init__(self,manifest):
        self.manifest=manifest
        self.model=PolicyModel()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.context_dim=192
        self.ctx_buffer=collections.deque(maxlen=8)
        self.frame_memory=collections.deque(maxlen=12)
        self.temporal_memory=collections.deque(maxlen=20)
        self.semantic=SemanticAggregator()
        self.semantic.to(self.device)
        self.planner=TemporalPlanner().to(self.device)
        self.scene=GameSceneInterpreter().to(self.device)
        self.decision=MultiTargetDecision(self.context_dim).to(self.device)
        self.scene_state=None
        self.loaded_path=None
        self.last_value=0.0
        self.hparams=None
        self.prefers_high=True
        self.pref_map={"__default__":"higher"}
        self.pref_default="higher"
        self.policy_size=(320,200)
        self.scene_side=256
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
    def attach_hparams(self,hparams):
        self.hparams=hparams
    def set_resolution(self,policy_size,scene_size=None):
        try:
            w=int(policy_size[0])
            h=int(policy_size[1])
        except:
            w,h=320,200
        w=max(224,w)
        h=max(160,h)
        self.policy_size=(w,h)
        if scene_size is not None:
            try:
                if isinstance(scene_size,(list,tuple)):
                    side=max(int(scene_size[0]),int(scene_size[-1]))
                else:
                    side=int(scene_size)
            except:
                side=max(w,h)
            self.scene_side=max(224,min(640,side))
        else:
            self.scene_side=max(224,min(640,int(max(self.policy_size)*0.9)))
    def set_orientation(self,prefer_high):
        self.prefers_high=bool(prefer_high)
    def update_preferences(self,prefs):
        if not isinstance(prefs,dict):
            prefs={}
        clean={}
        for k,v in prefs.items():
            if not isinstance(k,str):
                continue
            clean[k]=self._normalize_pref(v)
        if "__default__" not in clean:
            clean["__default__"]=self.pref_map.get("__default__",self.pref_default)
        self.pref_map=clean
        self.pref_default=clean.get("__default__","higher")
        vals=[v for k,v in clean.items() if k!="__default__" and v in ["higher","lower"]]
        if not vals and self.pref_default in ["higher","lower"]:
            vals=[self.pref_default]
        if vals:
            hi=sum(1 for v in vals if v=="higher")
            self.prefers_high=hi>=((len(vals)+1)//2)
        else:
            self.prefers_high=True
    def _normalize_pref(self,val):
        txt=str(val).lower()
        if txt.startswith("low") or txt.startswith("下"):
            return "lower"
        if txt.startswith("无关") or txt.startswith("ignore"):
            return "ignore"
        if txt.startswith("high") or txt.startswith("上"):
            return "higher"
        return self.pref_default
    def _pref_for_label(self,label):
        if isinstance(label,str) and label in self.pref_map:
            return self.pref_map[label]
        return self.pref_map.get("__default__",self.pref_default)
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
            vec[7]=float(e.get("ins_press",0))
            vec[8]=float(e.get("ins_release",0))
            vec[9]=float(e.get("duration",0))
            ctx.append(vec[:self.context_dim])
        if ctx:
            tens=torch.tensor(ctx,dtype=torch.float32,device=self.device).unsqueeze(0)
            self.ctx_buffer.append(tens)
        if len(self.ctx_buffer)==0:
            return torch.zeros((1,0,self.context_dim),dtype=torch.float32,device=self.device)
        joined=torch.cat(list(self.ctx_buffer),dim=1)
        return joined[:,-8:,:]
    def _history_tensor(self):
        if len(self.temporal_memory)==0:
            return torch.zeros((1,0,self.context_dim),dtype=torch.float32,device=self.device)
        hist=torch.stack(list(self.temporal_memory),dim=1)
        return hist[:,-12:,:]
    def update_temporal(self,embedding):
        if embedding is None:
            return
        self.temporal_memory.append(embedding.detach())
    def embed_frame(self,img):
        with torch.no_grad():
            ten=torch.from_numpy(cv2.resize(img,(self.scene_side,self.scene_side))).to(self.device).float().permute(2,0,1).unsqueeze(0)/255.0
            feat=self.semantic(ten)
            return feat
    def integrate_ui(self,score,ui_map):
        if ui_map is None:
            return score
        if score.shape!=ui_map.shape:
            ui=cv2.resize(ui_map,score.shape[::-1],interpolation=cv2.INTER_CUBIC)
        else:
            ui=ui_map
        return np.clip(score*0.6+ui*0.4,0,1)
    def synthesize_goal_map(self,ui_elements,w,h):
        if not ui_elements:
            return None
        goal=np.zeros((h,w),dtype=np.float32)
        sx=w/1280.0
        sy=h/800.0
        for el in ui_elements:
            bounds=el.get("bounds",[0,0,0,0])
            ox1=float(bounds[0])
            oy1=float(bounds[1])
            ox2=float(bounds[2])
            oy2=float(bounds[3])
            x1=int(max(0,min(w-1,round(ox1*sx))))
            y1=int(max(0,min(h-1,round(oy1*sy))))
            x2=int(max(x1+1,min(w,round(ox2*sx))))
            y2=int(max(y1+1,min(h,round(oy2*sy))))
            label=el.get("type","")
            pref=self._pref_for_label(label)
            conf=max(0.0,min(1.0,float(el.get("confidence",0.5))))
            interaction=max(0.0,min(1.0,float(el.get("interaction",0))))
            if pref=="ignore":
                continue
            if pref=="lower":
                conf=1.0-conf
                interaction=1.0-interaction
            score=min(1.0,max(0.0,conf*0.6+interaction*0.6))
            goal[y1:y2,x1:x2]+=score
        if goal.max()>0:
            goal/=goal.max()
        return goal
    def local_maxima(self,hm,top=80,dist=14):
        if hm is None:
            return []
        arr=np.array(hm,dtype=np.float32,copy=True)
        if arr.ndim==3:
            arr=arr.squeeze()
        if arr.ndim!=2:
            return []
        arr=np.nan_to_num(arr,copy=False)
        h,w=arr.shape
        if h==0 or w==0 or top<=0:
            return []
        arr[arr<0]=0
        result=[]
        work=arr.copy()
        dist=max(1,int(dist))
        for _ in range(int(top)):
            idx=np.argmax(work)
            val=float(work.flat[idx])
            if val<=0:
                break
            y=idx//w
            x=idx%w
            result.append((int(x),int(y),val))
            x1=max(0,x-dist)
            x2=min(w,x+dist+1)
            y1=max(0,y-dist)
            y2=min(h,y+dist+1)
            work[y1:y2,x1:x2]=0
        return result
    def predict(self,img,events=None,ui_elements=None,heat_prior=None,event_prior=None):
        self.ensure_integrity()
        h,w=img.shape[:2]
        inp=cv2.resize(img,self.policy_size)
        ten=torch.from_numpy(inp).to(self.device).float().permute(2,0,1).unsqueeze(0)/255.0
        hp_vals=self.hparams.values if self.hparams else {"confidence_floor":0.18,"primary_threshold":0.55,"long_press_threshold":0.78,"drag_threshold":0.58,"multi_gate":0.5,"confidence_margin":0.1,"drag_steps":28,"parallel_streams":1}
        ctx=self.set_context(events or [])
        hist=self._history_tensor()
        with torch.no_grad():
            logits,value,path_map,data_map,data_vec=self.model(ten,ctx,hist,None,None)
            prob=torch.sigmoid(logits)
            base_value=float(torch.sigmoid(value).mean().item())
            path_heat=torch.sigmoid(path_map)[0,0].detach().cpu().numpy()
            data_heat=torch.sigmoid(data_map)[0,0].detach().cpu().numpy()
            vec=torch.tanh(data_vec).detach().cpu().numpy()
        if vec.size>0:
            data_strength=float(np.mean(vec))
            value_score=0.7*base_value+0.3*(0.5+0.5*data_strength)
        else:
            value_score=base_value
        self.last_value=value_score
        left=prob[0,0].cpu().numpy()
        self.frame_memory.append(left)
        frame_embed=self.embed_frame(img)
        self.update_temporal(frame_embed)
        plan_in=torch.stack(list(self.temporal_memory),dim=1) if len(self.temporal_memory)>0 else torch.zeros((1,0,self.context_dim),dtype=torch.float32,device=self.device)
        scene_img=cv2.resize(img,(self.scene_side,self.scene_side))
        scene_tensor=torch.from_numpy(scene_img).to(self.device).float().permute(2,0,1).unsqueeze(0)/255.0
        with torch.no_grad():
            scene_vec,self.scene_state=self.scene(scene_tensor,self.scene_state)
        if plan_in.shape[1]>0:
            with torch.no_grad():
                plan=self.planner(plan_in)
                plan_vec=plan.mean(dim=1)
                gate_plan=float(torch.sigmoid(plan_vec.mean()).item())
        else:
            plan_vec=torch.zeros((1,self.context_dim),dtype=torch.float32,device=self.device)
            gate_plan=0.5
        ui_summary=self._ui_summary_tensor(ui_elements,w,h)
        with torch.no_grad():
            decision=self.decision(scene_vec,plan_vec,ui_summary)
        multi_factor=float(decision[0,1].item())
        risk_factor=float(decision[0,2].item())
        gate=float(min(1.0,max(0.0,0.32*gate_plan+0.36*multi_factor+0.2*risk_factor+0.12*value_score)))
        score=cv2.resize(left,(w,h),interpolation=cv2.INTER_CUBIC)
        path_overlay=cv2.resize(path_heat,(w,h),interpolation=cv2.INTER_CUBIC)
        data_overlay=cv2.resize(data_heat,(w,h),interpolation=cv2.INTER_CUBIC)
        score=np.clip(score*0.55+path_overlay*0.3+data_overlay*0.15,0,1)
        goal=self.synthesize_goal_map(ui_elements,w,h)
        if goal is not None:
            if not self.prefers_high:
                goal=1.0-np.clip(goal,0,1)
            score=self.integrate_ui(score,goal)
        if heat_prior is not None:
            score=score*(0.5+0.3*gate)+heat_prior*(0.5-0.3*gate)
        if event_prior is not None:
            score=score*(0.6+0.2*gate)+event_prior*(0.4-0.2*gate)
        score=np.clip(score,0,1)
        tactical=self._multi_target_plan(score,ui_elements,multi_factor,risk_factor)
        y,x=np.unravel_index(np.argmax(score),score.shape)
        conf=float(score[y,x])
        grid=(score.shape[1],score.shape[0])
        base={"kind":"idle","heat":score,"confidence":conf,"grid":grid,"tactical":tactical,"value":value_score}
        if conf<hp_vals.get("confidence_floor",0.18):
            return base
        btn="left" if conf>hp_vals.get("primary_threshold",0.55) or risk_factor>0.45 else "right"
        gy,gx=np.gradient(score)
        vx=float(gx[y,x])
        vy=float(gy[y,x])
        norm=math.sqrt(vx*vx+vy*vy)+1e-6
        vx/=norm
        vy/=norm
        def make_tap(ix,iy,button,conf_level):
            return {"kind":"tap","coord":(int(ix),int(iy)),"button":button,"duration":0.06+0.04*gate,"grid":grid,"confidence":float(conf_level),"path":[(int(ix),int(iy))],"release":(int(ix),int(iy))}
        if conf>hp_vals.get("long_press_threshold",0.78):
            act={"kind":"long_press","coord":(int(x),int(y)),"button":btn,"duration":0.35+0.45*gate,"grid":grid,"confidence":conf,"path":[(int(x),int(y))],"release":(int(x),int(y))}
        elif conf>hp_vals.get("drag_threshold",0.58):
            extent=max(score.shape[1],score.shape[0])*0.18*(0.6+conf)
            dx=vx*extent
            dy=vy*extent
            ex=int(max(0,min(score.shape[1]-1,round(x+dx))))
            ey=int(max(0,min(score.shape[0]-1,round(y+dy))))
            if ex==x and ey==y:
                ex=int(max(0,min(score.shape[1]-1,round(x+(random.random()-0.5)*extent))))
                ey=int(max(0,min(score.shape[0]-1,round(y+(random.random()-0.5)*extent))))
            steps=max(8,int(hp_vals.get("drag_steps",28)))
            drag_path=[]
            for i in range(steps):
                u=i/max(steps-1,1)
                ix=int(round(x+(ex-x)*u))
                iy=int(round(y+(ey-y)*u))
                drag_path.append((ix,iy))
            act={"kind":"drag","coord":(int(x),int(y)),"button":btn,"duration":0.18+0.22*gate,"grid":grid,"confidence":conf,"path":drag_path,"release":(int(ex),int(ey))}
        else:
            act=make_tap(x,y,btn,conf)
        peaks=self.local_maxima(score,top=8,dist=36)
        extra=[]
        budget=1+int(multi_factor>hp_vals.get("multi_gate",0.5))+int(multi_factor>0.62)
        for (px,py,pv) in peaks:
            if len(extra)>=budget:
                break
            if px==x and py==y:
                continue
            if pv<hp_vals.get("multi_gate",0.5)+hp_vals.get("confidence_margin",0.1)*multi_factor:
                continue
            choice="left" if pv>hp_vals.get("primary_threshold",0.55) or risk_factor>0.55 else "right"
            extra.append(make_tap(px,py,choice,pv))
        if extra:
            parallel=max(1,int(hp_vals.get("parallel_streams",1)))
            seq=[dict(act)]
            for item in extra:
                seq.append(item)
            for item in seq:
                item["heat_shape"]=grid
            if parallel>1:
                streams=[]
                for idx,item in enumerate(seq):
                    target=idx%parallel
                    if len(streams)<=target:
                        streams.append([])
                    streams[target].append(item)
                combo={"kind":"parallel","streams":streams,"grid":grid,"confidence":conf,"heat":score,"button":btn}
                return combo
            combo={"kind":"combo","sequence":seq,"grid":grid,"confidence":conf,"heat":score,"button":btn}
            return combo
        act["heat"]=score
        act["heat_shape"]=grid
        return act
    def _ui_summary_tensor(self,ui_elements,w,h):
        vec=torch.zeros((1,self.context_dim),dtype=torch.float32,device=self.device)
        if not ui_elements:
            return vec
        total=len(ui_elements)
        conf=sum(float(el.get("confidence",0)) for el in ui_elements)
        interact=sum(float(el.get("interaction",0)) for el in ui_elements)
        areas=[]
        for el in ui_elements:
            b=el.get("bounds",[0,0,0,0])
            width=max(1.0,float(b[2])-float(b[0]))
            height=max(1.0,float(b[3])-float(b[1]))
            areas.append(width*height/(max(1.0,w)*max(1.0,h)))
        avg_area=sum(areas)/len(areas) if areas else 0.0
        vec[0,0]=min(1.0,total/64.0)
        vec[0,1]=min(1.0,conf/max(1,total))
        vec[0,2]=min(1.0,interact/max(1,total))
        vec[0,3]=min(1.0,avg_area)
        vec[0,4]=min(1.0,max(areas) if areas else 0.0)
        vec[0,5]=min(1.0,min(areas) if areas else 0.0)
        conf_vals=[float(el.get("confidence",0)) for el in ui_elements]
        inter_vals=[float(el.get("interaction",0)) for el in ui_elements]
        vec[0,6]=sum(conf_vals)/max(1,total)
        vec[0,7]=float(sum(1 for c in conf_vals if c>0.7))/max(1,total)
        vec[0,8]=float(np.std(conf_vals)) if conf_vals else 0.0
        vec[0,9]=float(sum(1 for v in inter_vals if v>0.5))/max(1,total)
        return vec
    def _multi_target_plan(self,score,ui_elements,multi_factor,risk_factor):
        h,w=score.shape
        flat=score.reshape(-1)
        indices=np.argpartition(-flat,min(len(flat)-1,8))[:8]
        coords=[]
        for idx in indices:
            y=idx//w
            x=idx%w
            coords.append((int(x),int(y),float(score[y,x])))
        coords.sort(key=lambda t:t[2],reverse=True)
        plan=[]
        budget=1+int(multi_factor>0.3)+int(risk_factor>0.45)+int(self.last_value>0.62)
        used=set()
        for x,y,val in coords:
            if len(plan)>=budget:
                break
            key=(x//32,y//32)
            if key in used:
                continue
            used.add(key)
            plan.append({"x":x,"y":y,"confidence":val})
        if ui_elements and plan:
            for item in plan:
                best=None
                best_score=0.0
                for el in ui_elements:
                    b=el.get("bounds",[0,0,0,0])
                    if item["x"]>=b[0] and item["x"]<=b[2] and item["y"]>=b[1] and item["y"]<=b[3]:
                        score=float(el.get("confidence",0))*0.6+float(el.get("interaction",0))*0.4
                        if score>best_score:
                            best_score=score
                            best=el.get("raw_label",el.get("display","unknown"))
                item["ui_type"]=best or "unknown"
        return plan
    def train_incremental(self,batches,progress_cb=None,total=None):
        self.model.train()
        opt=optim.AdamW(list(self.model.parameters())+list(self.semantic.parameters())+list(self.planner.parameters()),lr=1e-4,weight_decay=1e-4)
        counted=total
        if counted is None:
            counted=len(batches) if hasattr(batches,"__len__") else 0
        processed=0
        for batch in batches:
            if batch is None:
                continue
            if len(batch)==4:
                img,ctx,label,hist=batch
                bs=img.shape[0]
                data=torch.zeros((bs,16),dtype=torch.float32)
                traj=torch.zeros((bs,1,4),dtype=torch.float32)
                mask=torch.ones((bs,1),dtype=torch.float32)
                val=torch.zeros((bs,1),dtype=torch.float32)
                adv=torch.zeros((bs,1),dtype=torch.float32)
                weight=torch.ones((bs,1),dtype=torch.float32)
            elif len(batch)==7:
                img,ctx,label,hist,val,adv,weight=batch
                bs=img.shape[0]
                data=torch.zeros((bs,16),dtype=torch.float32)
                traj=torch.zeros((bs,1,4),dtype=torch.float32)
                mask=torch.ones((bs,1),dtype=torch.float32)
            else:
                img,ctx,label,hist,data,traj,mask,val,adv,weight=batch
            processed+=1
            img=img.to(self.device)
            ctx=ctx.to(self.device).float()
            hist=hist.to(self.device).float()
            label=label.to(self.device).float()
            data=data.to(self.device).float()
            traj=traj.to(self.device).float()
            mask=mask.to(self.device).float()
            opt.zero_grad()
            val_t=val.to(self.device).float()
            adv_t=adv.to(self.device).float()
            weight_t=weight.to(self.device).float()
            logits,value,path_pred,data_map_pred,data_vec=self.model(img,ctx,hist,traj,mask)
            click_target=label[:,0:2]
            path_target=label[:,2:3]
            map_target=label[:,3:4]
            if logits.shape[-2:]!=click_target.shape[-2:]:
                click_target=F.interpolate(click_target,size=logits.shape[-2:],mode="bilinear",align_corners=False)
                path_target=F.interpolate(path_target,size=logits.shape[-2:],mode="bilinear",align_corners=False)
                map_target=F.interpolate(map_target,size=logits.shape[-2:],mode="bilinear",align_corners=False)
            base_loss=F.binary_cross_entropy_with_logits(logits,click_target)
            weighted=F.binary_cross_entropy_with_logits(logits,click_target,reduction="none")
            rl_loss=(weighted*(weight_t.view(weight_t.shape[0],1,1,1))).mean()
            value_loss=F.smooth_l1_loss(value,val_t)
            prob=torch.sigmoid(logits)
            entropy=-(prob*torch.log(prob+1e-6)+(1-prob)*torch.log(1-prob+1e-6)).mean()
            entropy_weight=self.hparams.adaptive_entropy() if self.hparams else 0.02
            neg_factor=torch.relu(-adv_t).mean()
            path_loss=F.mse_loss(torch.sigmoid(path_pred),path_target)
            map_loss=F.mse_loss(torch.sigmoid(data_map_pred),map_target)
            data_loss=F.smooth_l1_loss(data_vec,data)
            loss=0.4*base_loss+0.25*rl_loss+0.3*value_loss+0.12*path_loss+0.08*map_loss+0.15*data_loss-entropy_weight*entropy+0.05*neg_factor
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            nn.utils.clip_grad_norm_(self.semantic.parameters(),1.0)
            nn.utils.clip_grad_norm_(self.planner.parameters(),1.0)
            opt.step()
            if progress_cb:
                denom=counted if counted else max(processed*2,1)
                msg_total=counted if counted else "?"
                progress_cb(60+int(35*processed/max(1,denom)),f"训练进度 {processed}/{msg_total}")
        self.model.eval()
        self.semantic.eval()
        self.planner.eval()
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
            try:
                target_pid=win32process.GetWindowThreadProcessId(th)[1]
            except:
                target_pid=None
            blocked=0
            total=0
            for (px,py) in pts:
                total+=1
                try:
                    hw=win32gui.WindowFromPoint((px,py))
                except:
                    hw=0
                if hw==0:
                    continue
                rt=Foreground.top(hw)
                if rt==th:
                    continue
                if Foreground._ignore(rt,target_pid):
                    continue
                if not Foreground._above(rt,th):
                    continue
                blocked+=1
                if blocked>total//2:
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
    def _ignore(hwnd,target_pid=None):
        try:
            if hwnd==0:
                return True
            if target_pid is not None:
                try:
                    pid=win32process.GetWindowThreadProcessId(hwnd)[1]
                    if pid==target_pid:
                        return True
                except:
                    pass
            if not win32gui.IsWindow(hwnd):
                return True
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if win32gui.IsIconic(hwnd):
                return True
            cloaked=ctypes.c_int(0)
            ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(hwnd),Foreground.DWMWA_CLOAKED,ctypes.byref(cloaked),ctypes.sizeof(cloaked))
            if int(cloaked.value)!=0:
                return True
            ex=win32gui.GetWindowLong(hwnd,win32con.GWL_EXSTYLE)
            if ex & win32con.WS_EX_TRANSPARENT:
                return True
            if ex & win32con.WS_EX_LAYERED:
                alpha=ctypes.c_ubyte()
                flags=ctypes.c_uint()
                color=ctypes.c_uint()
                if ctypes.windll.user32.GetLayeredWindowAttributes(ctypes.wintypes.HWND(hwnd),ctypes.byref(color),ctypes.byref(alpha),ctypes.byref(flags)):
                    if alpha.value==0:
                        return True
            return False
        except:
            return False
    @staticmethod
    def _above(candidate,target):
        try:
            if candidate==0 or target==0:
                return False
            cur=win32gui.GetWindow(target,win32con.GW_HWNDPREV)
            steps=0
            while cur and steps<512:
                if cur==candidate:
                    return True
                cur=win32gui.GetWindow(cur,win32con.GW_HWNDPREV)
                steps+=1
            return False
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
    signal_data_ready=Signal(list)
    signal_window_stats=Signal(list)
    MARK=0x22ACE777
    def __init__(self):
        super().__init__()
        with open(cfg_path,"r",encoding="utf-8") as f:
            self.cfg=json.load(f)
        self.manifest=ModelManifest(self.cfg)
        if not self.cfg.get("model_sha256") and self.manifest.sha_value:
            self.cfg["model_sha256"]=self.manifest.sha_value
            with open(cfg_path,"w",encoding="utf-8") as f:
                json.dump(self.cfg,f,ensure_ascii=False,indent=2)
                f.flush()
                os.fsync(f.fileno())
        self.strategy=StrategyEngine(self.manifest)
        self.hyper=AdaptiveHyperParams(self.cfg)
        self.hyper.set_save_callback(self._save_hparams)
        self.strategy.attach_hparams(self.hyper)
        self.validator=AutoValidator(self)
        self.io_q=queue.Queue(maxsize=1024)
        self.io_thread=threading.Thread(target=self._io_worker,daemon=True)
        self.io_thread.start()
        self.prev_lock=threading.Lock()
        self.rect_lock=threading.Lock()
        self.mode="initializing"
        self.last_user_activity=time.time()
        self.selected_hwnd=None
        self.selected_title=""
        self.session_id=str(uuid.uuid4())
        self.running=False
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
        self.ui_preferences=self._normalize_preferences(self.cfg.get("ui_preferences"))
        self.ui_prefers_high=self._aggregate_preferences(self.ui_preferences,self.ui_default_pref)
        self.data_preferences=self._normalize_data_preferences(self.cfg.get("data_preferences"))
        self.data_prefers_high=self._aggregate_preferences(self.data_preferences,self.data_default_pref)
        self.hyper.set_preferences(self.ui_preferences)
        self.strategy.update_preferences(self.ui_preferences)
        self.policy_base=(320,200)
        self.policy_resolution,self.scene_resolution,self.resolution_scale=self._compute_policy_resolution()
        self.strategy.set_resolution(self.policy_resolution,self.scene_resolution)
        self.reward_engine=RewardComposer()
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
        self.data_points=[]
        self.model_ready_event=threading.Event()
        self.model_error=None
        self.metric_history=collections.deque(maxlen=180)
        self.resource_history=collections.deque(maxlen=120)
        self.activity_events=collections.deque(maxlen=360)
        self.window_stats={}
        self.trend_window=collections.deque(maxlen=36)
        self.data_color_map={}
        self._data_id_lock=threading.Lock()
        self._data_id_map={}
        self._data_alias_map={}
        self._color_index=0
        self.visibility_state=None
        self.visibility_reason=""
        self._last_clarity_boost=0.0
        self._pending_window=None
        self._pending_waiter=None
        self._pending_lock=threading.Lock()
        self._init_day_files()
        threading.Thread(target=self._ensure_model_bg,daemon=True).start()
        self.signal_window_stats.emit([])
    def _ensure_model_bg(self):
        try:
            self.signal_modelprog.emit(1,"准备模型")
            path=self.manifest.ensure(lambda p,t:self.signal_modelprog.emit(p,t))
            self.strategy.ensure_loaded(lambda p,t:self.signal_modelprog.emit(p,t))
            self.model_loaded=True
            self.model_ready_event.set()
            ModelMeta.update([path],{"start":None,"end":None},{"note":"初始化校验"})
            self.signal_modelver.emit(f"v{ModelMeta.read().get('version',0)}")
            self._apply_pending_window()
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
    def _sanitize_data_points(self,items,cw,ch):
        sanitized=[]
        idx=0
        for it in items:
            if not isinstance(it,dict):
                continue
            txt_src=it.get("text")
            raw_val=it.get("value")
            val_int=None
            if isinstance(raw_val,(int,float,str)):
                try:
                    val_int=int(str(raw_val).strip())
                except:
                    val_int=None
            if val_int is None or val_int<0:
                txt_src=it.get("text")
                if isinstance(txt_src,str):
                    digits=re.findall(r"\d+",txt_src)
                    if digits:
                        try:
                            cand=int("".join(digits))
                        except:
                            cand=None
                        if cand is not None and cand>=0:
                            val_int=cand
            if val_int is None or val_int<0:
                continue
            nb=it.get("norm_bounds")
            if not isinstance(nb,list) or len(nb)!=4:
                absb=it.get("abs_bounds")
                win=it.get("window_size")
                if isinstance(absb,list) and len(absb)==4 and isinstance(win,list) and len(win)==2:
                    try:
                        ow=max(1,int(win[0]))
                        oh=max(1,int(win[1]))
                        nb=[float(absb[0])/ow,float(absb[1])/oh,float(absb[2])/ow,float(absb[3])/oh]
                    except:
                        nb=None
            if not isinstance(nb,list) or len(nb)!=4:
                continue
            try:
                nx1=float(nb[0])
                ny1=float(nb[1])
                nx2=float(nb[2])
                ny2=float(nb[3])
            except:
                continue
            nx1=max(0.0,min(1.0,nx1))
            ny1=max(0.0,min(1.0,ny1))
            nx2=max(0.0,min(1.0,nx2))
            ny2=max(0.0,min(1.0,ny2))
            if nx2<=nx1 or ny2<=ny1:
                continue
            dw=cw if cw>0 else int((it.get("window_size") or [0,0])[0] or 0)
            dh=ch if ch>0 else int((it.get("window_size") or [0,0])[1] or 0)
            if dw<=0 or dh<=0:
                continue
            ax1=int(max(0,min(dw-1,round(nx1*dw))))
            ay1=int(max(0,min(dh-1,round(ny1*dh))))
            ax2=int(max(ax1+1,min(dw,round(nx2*dw))))
            ay2=int(max(ay1+1,min(dh,round(ny2*dh))))
            name=str(it.get("name") or f"数据{idx+1}")
            raw_key=it.get("key")
            color=it.get("color")
            conf=float(max(0.0,min(1.0,it.get("confidence",0.0))))
            trend=it.get("trend")
            try:
                trend_val=float(trend)
            except:
                trend_val=None
            pref=str(it.get("preference") or self.preference_for_data(name))
            if pref not in ["higher","lower","ignore"]:
                pref=self.preference_for_data(name)
            raw_bounds=it.get("bounds")
            if isinstance(raw_bounds,(list,tuple)) and len(raw_bounds)==4:
                try:
                    bounds_val=[int(raw_bounds[0]),int(raw_bounds[1]),int(raw_bounds[2]),int(raw_bounds[3])]
                except:
                    bounds_val=[ax1,ay1,ax2,ay2]
            else:
                bounds_val=[ax1,ay1,ax2,ay2]
            display_txt=str(txt_src) if isinstance(txt_src,str) and txt_src.strip() else str(val_int)
            stable_key=self.resolve_data_key(raw_key if raw_key is not None else f"dp_{idx}",[nx1,ny1,nx2,ny2],display_txt or name)
            if not isinstance(color,(list,tuple)) or len(color)!=3:
                color=self.ensure_data_color(stable_key)
            else:
                color=tuple(int(c) for c in color)
            entry={"name":name,"bounds":bounds_val,"abs_bounds":[ax1,ay1,ax2,ay2],"window_size":[dw,dh],"value":val_int,"trend":trend_val,"confidence":conf,"preference":pref,"text":display_txt,"color":[int(color[0]),int(color[1]),int(color[2])],"key":stable_key,"norm_bounds":[nx1,ny1,nx2,ny2]}
            sanitized.append(entry)
            idx+=1
        return sanitized
    def set_data_points(self,items):
        try:
            cw=max(0,int(self.client_rect[2]-self.client_rect[0]))
            ch=max(0,int(self.client_rect[3]-self.client_rect[1]))
        except:
            cw=0
            ch=0
        sanitized=self._sanitize_data_points(items if isinstance(items,list) else [],cw,ch)
        with self.rect_lock:
            self.data_points=[dict(p) for p in sanitized]
        return [dict(p) for p in sanitized]
    def remove_data_point(self,key):
        if not key:
            return (self.current_data_snapshot(),False)
        with self.rect_lock:
            if not isinstance(self.data_points,list):
                return ([],False)
            current=[dict(p) for p in self.data_points]
            remain=[p for p in current if p.get("key")!=key]
            if len(remain)==len(current):
                return (current,False)
            self.data_points=[dict(p) for p in remain]
        try:
            with open(data_cache_path,"w",encoding="utf-8") as f:
                json.dump(remain,f,ensure_ascii=False,indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            log(f"data remove save fail:{e}")
        return ([dict(p) for p in remain],True)
    def current_data_snapshot(self):
        with self.rect_lock:
            return [dict(p) for p in self.data_points] if isinstance(self.data_points,list) else []
    def _rescale_data_points(self,cr):
        try:
            cw=max(0,int(cr[2]-cr[0]))
            ch=max(0,int(cr[3]-cr[1]))
        except:
            cw=0
            ch=0
        if cw<=0 or ch<=0:
            return
        with self.rect_lock:
            current=[dict(p) for p in self.data_points] if isinstance(self.data_points,list) else []
        if not current:
            return
        sanitized=self._sanitize_data_points(current,cw,ch)
        with self.rect_lock:
            self.data_points=[dict(p) for p in sanitized]
    def update_rects(self,wr,cr):
        if not isinstance(wr,tuple):
            wr=tuple(wr) if wr else (0,0,0,0)
        if not isinstance(cr,tuple):
            cr=tuple(cr) if cr else (0,0,0,0)
        changed=cr!=self.client_rect
        self.rect=wr
        self.client_rect=cr
        if changed:
            self._rescale_data_points(cr)
    def _save_hparams(self,vals):
        serial={}
        for k,v in vals.items():
            try:
                serial[k]=float(v) if isinstance(v,(int,float)) else v
            except:
                serial[k]=v
        self.save_cfg("hyperparam_state",serial)
    def _norm_pref_value(self,val,default):
        raw=str(val).strip()
        txt=raw.lower()
        lower_tokens=["low","lower","越低","更低","低","越低越好","更低越好"]
        for token in lower_tokens:
            if txt.startswith(token) or token in txt or raw.startswith(token) or token in raw:
                return "lower"
        ignore_tokens=["无关","忽略","ignore","irrelevant","无影响"]
        for token in ignore_tokens:
            if txt.startswith(token) or token in txt or raw.startswith(token) or token in raw:
                return "ignore"
        higher_tokens=["high","higher","越高","更高","高","越高越好","更高越好"]
        for token in higher_tokens:
            if txt.startswith(token) or token in txt or raw.startswith(token) or token in raw:
                return "higher"
        return default
    def _normalize_pref_bundle(self,prefs,default,attr):
        data={"__default__":default}
        if isinstance(prefs,dict):
            for k,v in prefs.items():
                if not isinstance(k,str):
                    continue
                val=self._norm_pref_value(v,default)
                if k=="__default__":
                    data["__default__"]=val
                else:
                    data[k]=val
        setattr(self,attr,data["__default__"])
        return data
    def _normalize_preferences(self,prefs):
        return self._normalize_pref_bundle(prefs,cfg_defaults["ui_preferences"]["__default__"],"ui_default_pref")
    def _normalize_data_preferences(self,prefs):
        return self._normalize_pref_bundle(prefs,cfg_defaults["data_preferences"]["__default__"],"data_default_pref")
    def _aggregate_preferences(self,prefs,default):
        if not isinstance(prefs,dict):
            return True
        vals=[v for k,v in prefs.items() if k!="__default__" and v in ["higher","lower"]]
        base=prefs.get("__default__",default)
        if not vals and base in ["higher","lower"]:
            vals=[base]
        if not vals:
            return True
        hi=sum(1 for v in vals if v=="higher")
        return hi>=((len(vals)+1)//2)
    def preference_for_label(self,label):
        if isinstance(label,str) and label in self.ui_preferences:
            return self.ui_preferences[label]
        return self.ui_preferences.get("__default__",self.ui_default_pref)
    def update_ui_preference(self,label,mode):
        key="__default__" if not label else str(label)
        val=self._norm_pref_value(mode,self.ui_default_pref)
        if key=="__default__":
            self.ui_preferences["__default__"]=val
            self.ui_default_pref=val
        else:
            self.ui_preferences[key]=val
        self.ui_prefers_high=self._aggregate_preferences(self.ui_preferences,self.ui_default_pref)
        self.hyper.set_preferences(self.ui_preferences)
        self.strategy.update_preferences(self.ui_preferences)
        self._save_preferences()
        txt="越高越好" if val=="higher" else "越低越好" if val=="lower" else "无关"
        target="全局" if key=="__default__" else key
        self.signal_tip.emit(f"{target}目标:{txt}")
    def _save_preferences(self):
        self.save_cfg("ui_preferences",self.ui_preferences)
    def preference_for_data(self,name):
        if isinstance(name,str) and name in self.data_preferences:
            return self.data_preferences[name]
        return self.data_preferences.get("__default__",self.data_default_pref)
    def update_data_preference(self,name,mode):
        key="__default__" if not name else str(name)
        val=self._norm_pref_value(mode,self.data_default_pref)
        if key=="__default__":
            self.data_preferences["__default__"]=val
            self.data_default_pref=val
        else:
            self.data_preferences[key]=val
        self.data_prefers_high=self._aggregate_preferences(self.data_preferences,self.data_default_pref)
        self.save_cfg("data_preferences",self.data_preferences)
        txt="越高越好" if val=="higher" else "越低越好" if val=="lower" else "无关"
        target="全局数据" if key=="__default__" else key
        self.signal_tip.emit(f"{target}目标:{txt}")
    def resolve_data_key(self,raw_key,norm,text):
        key=''
        if isinstance(raw_key,str) and raw_key:
            key=raw_key
        elif raw_key is not None:
            key=str(raw_key)
        label=str(text).strip() if isinstance(text,str) else ''
        bounds=[float(norm[0]),float(norm[1]),float(norm[2]),float(norm[3])] if isinstance(norm,(list,tuple)) and len(norm)==4 else None
        with self._data_id_lock:
            if key and key in self._data_id_map:
                info=self._data_id_map[key]
                if bounds:
                    info['norm']=bounds
                if label:
                    info['text']=label
                info['last']=time.time()
                return key
            if key and key in self._data_alias_map:
                stable=self._data_alias_map[key]
                info=self._data_id_map.get(stable)
                if info is None:
                    self._data_id_map[stable]={'norm':bounds,'text':label,'last':time.time()}
                else:
                    if bounds:
                        info['norm']=bounds
                    if label:
                        info['text']=label
                    info['last']=time.time()
                if key!=stable:
                    self._data_alias_map[key]=stable
                return stable
            best_id=None
            best_score=None
            if bounds:
                for sid,info in self._data_id_map.items():
                    dist=self._data_distance(info.get('norm'),bounds)
                    if label and info.get('text'):
                        if label==info.get('text'):
                            dist*=0.5
                        else:
                            dist*=1.1
                    if best_score is None or dist<best_score:
                        best_id=sid
                        best_score=dist
                if best_score is not None and best_score<0.18:
                    info=self._data_id_map[best_id]
                    if bounds:
                        info['norm']=bounds
                    if label:
                        info['text']=label
                    info['last']=time.time()
                    if key:
                        self._data_alias_map[key]=best_id
                    return best_id
            stable=key if key and key.startswith('data-') and key not in self._data_id_map else f"data-{uuid.uuid4().hex[:8]}"
            self._data_id_map[stable]={'norm':bounds,'text':label,'last':time.time()}
            if key and key!=stable:
                self._data_alias_map[key]=stable
            return stable
    def ensure_data_color(self,key):
        ref=str(key) if key else str(uuid.uuid4())
        if ref in self.data_color_map:
            return self.data_color_map[ref]
        color=self._next_color()
        self.data_color_map[ref]=color
        return color
    def _data_distance(self,a,b):
        if not a or not b:
            return 1.0
        try:
            ax1,ay1,ax2,ay2=float(a[0]),float(a[1]),float(a[2]),float(a[3])
            bx1,by1,bx2,by2=float(b[0]),float(b[1]),float(b[2]),float(b[3])
        except Exception:
            return 1.0
        aw=max(1e-6,ax2-ax1)
        ah=max(1e-6,ay2-ay1)
        bw=max(1e-6,bx2-bx1)
        bh=max(1e-6,by2-by1)
        acx=(ax1+ax2)/2.0
        acy=(ay1+ay2)/2.0
        bcx=(bx1+bx2)/2.0
        bcy=(by1+by2)/2.0
        dc=abs(acx-bcx)+abs(acy-bcy)
        ds=abs(aw-bw)+abs(ah-bh)
        return dc*1.1+ds
    def _next_color(self):
        golden=0.618033988749895
        h=(self._color_index*golden)%1.0
        self._color_index+=1
        s=0.72
        v=0.95
        r,g,b=self._hsv_to_rgb(h,s,v)
        return (int(b),int(g),int(r))
    def _hsv_to_rgb(self,h,s,v):
        i=int(h*6.0)
        f=h*6.0-i
        p=v*(1.0-s)
        q=v*(1.0-f*s)
        t=v*(1.0-(1.0-f)*s)
        i=i%6
        if i==0:
            r,g,b=v,t,p
        elif i==1:
            r,g,b=q,v,p
        elif i==2:
            r,g,b=p,v,t
        elif i==3:
            r,g,b=p,q,v
        elif i==4:
            r,g,b=t,p,v
        else:
            r,g,b=v,p,q
        return int(r*255),int(g*255),int(b*255)
    def preview_visual(self,img):
        if img is None or not isinstance(img,np.ndarray):
            return img
        base=np.ascontiguousarray(img.copy())
        with self.rect_lock:
            items=[dict(p) for p in self.data_points] if isinstance(self.data_points,list) else []
        if not items:
            return base
        h,w=base.shape[:2]
        overlay=np.zeros_like(base)
        outlines=[]
        for it in items:
            rect=self._preview_rect(it,w,h)
            if rect is None:
                continue
            x1,y1,x2,y2=rect
            color=it.get("color")
            key=it.get("key")
            if key and (not color or len(color)!=3):
                color=self.ensure_data_color(key)
                it["color"]=[int(color[0]),int(color[1]),int(color[2])]
            if not color or len(color)!=3:
                continue
            col=(int(color[0]),int(color[1]),int(color[2]))
            cv2.rectangle(overlay,(x1,y1),(x2-1,y2-1),col,-1)
            outlines.append((x1,y1,x2,y2,col))
        if not outlines:
            return base
        blended=cv2.addWeighted(base,0.65,overlay,0.35,0)
        for x1,y1,x2,y2,col in outlines:
            cv2.rectangle(blended,(x1,y1),(x2-1,y2-1),col,2)
        return blended
    def _preview_rect(self,item,w,h):
        try:
            cw=max(0,int(self.client_rect[2]-self.client_rect[0]))
            ch=max(0,int(self.client_rect[3]-self.client_rect[1]))
        except:
            cw=0
            ch=0
        if cw>0 and ch>0:
            dw=cw
            dh=ch
        else:
            win=item.get("window_size")
            if isinstance(win,list) and len(win)==2:
                try:
                    dw=max(1,int(win[0]))
                    dh=max(1,int(win[1]))
                except:
                    dw=w
                    dh=h
            else:
                dw=w
                dh=h
        nb=item.get("norm_bounds")
        ax1=ay1=ax2=ay2=None
        if nb and len(nb)==4:
            try:
                nx1=float(nb[0])
                ny1=float(nb[1])
                nx2=float(nb[2])
                ny2=float(nb[3])
            except:
                nx1=ny1=nx2=ny2=None
            if nx1 is not None and nx2 is not None and ny1 is not None and ny2 is not None and nx2>nx1 and ny2>ny1:
                ax1=nx1*dw
                ay1=ny1*dh
                ax2=nx2*dw
                ay2=ny2*dh
        if ax1 is None:
            absb=item.get("abs_bounds")
            if absb and len(absb)==4:
                try:
                    ax1=float(absb[0])
                    ay1=float(absb[1])
                    ax2=float(absb[2])
                    ay2=float(absb[3])
                except:
                    return None
            else:
                bounds=item.get("bounds")
                if not bounds or len(bounds)!=4:
                    return None
                try:
                    ax1=float(bounds[0])
                    ay1=float(bounds[1])
                    ax2=float(bounds[2])
                    ay2=float(bounds[3])
                except:
                    return None
                base_w=1280.0
                base_h=800.0
                ax1=ax1*dw/base_w
                ay1=ay1*dh/base_h
                ax2=ax2*dw/base_w
                ay2=ay2*dh/base_h
        sx=w/float(dw)
        sy=h/float(dh)
        x1=int(max(0,min(w-1,math.floor(ax1*sx))))
        y1=int(max(0,min(h-1,math.floor(ay1*sy))))
        x2=int(max(x1+1,min(w,math.ceil(ax2*sx))))
        y2=int(max(y1+1,min(h,math.ceil(ay2*sy))))
        if x2<=x1 or y2<=y1:
            return None
        return x1,y1,x2,y2
    def _align_dim(self,val,step):
        if step<=0:
            return int(val)
        return int(max(step,(val//step)*step))
    def _compute_policy_resolution(self):
        base=self.policy_base
        scale=1.0
        try:
            cores=psutil.cpu_count(logical=False) or psutil.cpu_count()
        except:
            cores=None
        if cores and cores>=12:
            scale+=0.25
        elif cores and cores>=8:
            scale+=0.15
        try:
            ram=psutil.virtual_memory().total/(1<<30)
        except:
            ram=16.0
        if ram>48:
            scale+=0.25
        elif ram>24:
            scale+=0.15
        gu,gm=_gpu_util_mem()
        if gm and gm>70:
            scale+=0.35
        elif gm and gm>40:
            scale+=0.2
        if gu and gu>85:
            scale-=0.1
        scale=max(0.8,min(1.8,scale))
        w=self._align_dim(int(base[0]*scale),16)
        h=self._align_dim(int(base[1]*scale),8)
        w=max(224,w)
        h=max(160,h)
        scene=max(224,min(640,int(max(w,h)*0.9)))
        return (w,h),(scene,scene),scale
    def _apply_resolution_scale(self):
        base=self.policy_base
        w=self._align_dim(int(base[0]*self.resolution_scale),16)
        h=self._align_dim(int(base[1]*self.resolution_scale),8)
        w=max(224,w)
        h=max(160,h)
        self.policy_resolution=(w,h)
        side=max(224,min(640,int(max(w,h)*0.9)))
        self.scene_resolution=(side,side)
        self.strategy.set_resolution(self.policy_resolution,self.scene_resolution)
    def _auto_adjust_resolution(self,motion,resource):
        target=self.resolution_scale
        if resource>95:
            target-=0.18
        elif resource>90:
            target-=0.1
        elif resource<58 and motion>18:
            target+=0.12
        elif resource<52 and motion>10:
            target+=0.08
        elif resource<68 and motion>16:
            target+=0.08
        elif resource<58 and motion>24:
            target+=0.12
        target=max(0.7,min(1.9,target))
        if abs(target-self.resolution_scale)>0.05:
            self.resolution_scale=target
            self._apply_resolution_scale()
    def _boost_clarity(self,lap,resource):
        try:
            now=time.time()
            if lap<14.0 and resource<88.0 and now-self._last_clarity_boost>1.8:
                self.resolution_scale=min(1.9,self.resolution_scale+0.1)
                self._apply_resolution_scale()
                self._last_clarity_boost=now
        except:
            pass
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
        if not hwnd:
            return
        if self.selected_hwnd==hwnd:
            if self.selected_title!=title:
                self.selected_title=title
                if self.model_ready_event.is_set() and not self.model_error:
                    self.signal_window.emit(title)
            return
        with self._pending_lock:
            if self._pending_window and self._pending_window[0]==hwnd and self.selected_hwnd is None:
                self._pending_window=(hwnd,title)
                self.selected_title=title
                return
            self._pending_window=(hwnd,title)
        self.selected_hwnd=None
        self.selected_title=title
        if self.model_ready_event.is_set() and not self.model_error:
            self._apply_pending_window()
            return
        self.signal_window.emit(f"{title}(待模型)")
        if self._pending_waiter and self._pending_waiter.is_alive():
            return
        self._pending_waiter=threading.Thread(target=self._await_model_then_bind,daemon=True)
        self._pending_waiter.start()
        self.signal_tip.emit("等待模型准备后绑定窗口")
    def _await_model_then_bind(self):
        try:
            self.model_ready_event.wait()
            if self.model_error:
                return
            self._apply_pending_window()
        except:
            pass
    def _apply_pending_window(self):
        with self._pending_lock:
            pending=self._pending_window
            self._pending_window=None
        if not pending:
            return
        hwnd,title=pending
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
    def update_visibility(self,visible,context):
        reason=str(context) if context is not None else ""
        if visible:
            if self.visibility_state is True and reason==self.visibility_reason:
                return
            self.visibility_state=True
            self.visibility_reason=reason
            if reason=="learning":
                msg="窗口可见，继续学习"
            elif reason=="training":
                msg="窗口可见，继续训练"
            elif reason:
                msg=reason
            else:
                msg="窗口可见"
            self.signal_tip.emit(msg)
        else:
            if self.visibility_state is False and reason==self.visibility_reason:
                return
            self.visibility_state=False
            self.visibility_reason=reason
            msg=reason if reason else "窗口不可见，已暂停"
            self.signal_tip.emit(msg)
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
    def _enhance_frame(self,img):
        try:
            if img is None or not isinstance(img,np.ndarray) or img.size==0:
                return img
            blur=cv2.GaussianBlur(img,(0,0),1.0)
            sharp=cv2.addWeighted(img,1.45,blur,-0.45,0)
            lab=cv2.cvtColor(sharp,cv2.COLOR_BGR2LAB)
            l,a,b=cv2.split(lab)
            clahe=cv2.createCLAHE(clipLimit=2.2,tileGridSize=(8,8))
            l=clahe.apply(l)
            merged=cv2.merge((l,a,b))
            enhanced=cv2.cvtColor(merged,cv2.COLOR_LAB2BGR)
            return enhanced
        except:
            return img
    def adapt_fps(self,prev,curr):
        if prev is None or curr is None:
            if curr is not None:
                gray=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
                lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
                self.metric_history.append({"delta":0.0,"mag":0.0,"lap":lap,"time":time.time()})
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
            gray=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
            lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            now=time.time()
            entry={"delta":v,"mag":mag,"lap":lap,"time":now}
            self.metric_history.append(entry)
            while self.metric_history and now-self.metric_history[0]["time"]>6.0:
                self.metric_history.popleft()
            short_vals=[m["delta"]*0.5+m["mag"]*36.0+m["lap"]*0.08 for m in list(self.metric_history)[-12:]]
            long_vals=[m["delta"]*0.5+m["mag"]*36.0+m["lap"]*0.08 for m in self.metric_history]
            short_avg=sum(short_vals)/len(short_vals) if short_vals else 0.0
            long_avg=sum(long_vals)/len(long_vals) if long_vals else 0.0
            trend=short_avg-long_avg*0.85
            cpu=psutil.cpu_percent(interval=None)
            mem=psutil.virtual_memory().percent
            gu,gm=_gpu_util_mem()
            if gu is None:
                gu=0.0
            if gm is None:
                gm=0.0
            res_entry={"cpu":cpu,"mem":mem,"gpu":gu,"vram":gm,"time":now}
            self.resource_history.append(res_entry)
            while self.resource_history and now-self.resource_history[0]["time"]>10.0:
                self.resource_history.popleft()
            avg_cpu=sum(r["cpu"] for r in self.resource_history)/len(self.resource_history)
            avg_mem=sum(r["mem"] for r in self.resource_history)/len(self.resource_history)
            avg_gpu=sum(r["gpu"] for r in self.resource_history)/len(self.resource_history)
            avg_vram=sum(r["vram"] for r in self.resource_history)/len(self.resource_history)
            motion_score=v*0.35+mag*48.0+lap*0.02
            motion_score=motion_score*max(0.6,min(1.6,self._window_weight()))
            resource_pressure=max(avg_cpu,avg_mem,avg_gpu,avg_vram)
            self._auto_adjust_resolution(motion_score,resource_pressure)
            self._boost_clarity(lap,resource_pressure)
            adaptive_target=self.fps
            if motion_score>28 or trend>6:
                adaptive_target=min(self.max_fps,adaptive_target+8.0)
            elif motion_score>18 or trend>3:
                adaptive_target=min(self.max_fps,adaptive_target+4.0)
            elif motion_score<6 and trend<-2:
                adaptive_target=max(self.min_fps,adaptive_target-4.0)
            elif motion_score<12:
                adaptive_target=max(self.min_fps,adaptive_target-1.5)
            if resource_pressure>94:
                adaptive_target=max(self.min_fps,adaptive_target-8.0)
            elif resource_pressure>88:
                adaptive_target=max(self.min_fps,adaptive_target-4.0)
            elif resource_pressure<70 and motion_score>14:
                adaptive_target=min(self.max_fps,adaptive_target+2.0)
            self.trend_window.append(adaptive_target)
            smoothed=sum(self.trend_window)/len(self.trend_window)
            inertia=0.6
            new_fps=self.fps*inertia+smoothed*(1.0-inertia)
            if abs(new_fps-self.fps)<0.2:
                new_fps=adaptive_target
            self.fps=max(self.min_fps,min(self.max_fps,new_fps))
            if hasattr(self,"hyper"):
                self.hyper.observe_metrics(self.fps,motion_score,resource_pressure,self.mode)
                self.hyper.observe_resource(avg_cpu,avg_mem,avg_gpu,avg_vram)
                self.hyper.adjust()
        except:
            pass
    def _window_weight(self):
        title=self.selected_title or ""
        weight=1.0
        stats=self.window_stats.get(title)
        if stats:
            now=time.time()
            recent=[item for item in list(stats.get("recent",[])) if now-item[0]<=90]
            if recent:
                freq=len(recent)/90.0
                dur=sum(r[1] for r in recent)/max(1.0,len(recent))
                ai_ratio=sum(1 for r in recent if len(r)>=3 and r[2]=="ai")/max(1,len(recent))
                weight+=min(0.6,freq*5.0)+min(0.4,dur*1.4)+min(0.3,ai_ratio*0.8)
                if freq<0.01 and dur<0.2:
                    weight=max(0.7,weight*0.75)
        title_lower=title.lower()
        if any(k in title_lower for k in ["game","游戏","3d","渲染"]):
            weight=max(weight,1.25)
        elif any(k in title_lower for k in ["浏览器","browser","chrome","edge","视频","player"]):
            weight=max(weight,1.1)
        elif any(k in title_lower for k in ["excel","表格","editor","文档"]):
            weight=min(weight,0.95)
        return max(0.6,min(1.8,weight))
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
            now_ns=time.time_ns()
            now=now_ns/1_000_000_000.0
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
            fid=str(now_ns)
            base=os.path.join(exp_dir,self.day_tag,"frames")
            path=os.path.join(base,f"{fid}.png")
            while os.path.exists(path):
                fid=str(int(fid)+1)
                path=os.path.join(base,f"{fid}.png")
            enhanced=self._enhance_frame(img)
            resized=cv2.resize(enhanced,(2560,1600),interpolation=cv2.INTER_LANCZOS4)
            ok,buf=cv2.imencode(".png",resized,[int(cv2.IMWRITE_PNG_COMPRESSION),self.png_comp])
            if not ok:
                return None
            self.io_q.put((path,buf))
            src="user" if mode=="learning" else ("ai" if mode=="training" else mode)
            rec={"id":fid,"ts":now,"ts_ns":now_ns,"source":src,"mode":mode,"window_title":self.selected_title,"rect":[0,0,2560,1600],"path":path,"filename":f"{fid}.png","w":2560,"h":1600,"session_id":self.session_id}
            self.frames_writer.append(rec)
            with self.prev_lock:
                self.prev_img=img.copy()
            self._last_f_ts=now
            return fid
        except:
            return None
    def record_op(self,source,typ,press_t,px,py,release_t,rx,ry,moves,frame_id,ins_press,ins_release,clip_ids=None,mode=None):
        try:
            self._init_day_files()
            cx,cy=self.client_rect[0],self.client_rect[1]
            cw=max(1,self.client_rect[2]-self.client_rect[0])
            ch=max(1,self.client_rect[3]-self.client_rect[1])
            plx,ply=self._scale2560(px-cx,py-cy,cw,ch)
            rlx,rly=self._scale2560(rx-cx,ry-cy,cw,ch)
            def norm_ts(ts):
                if not ts:
                    return 0,0.0
                if isinstance(ts,(int,float)) and ts>1e10:
                    ns=int(ts)
                    sec=ns/1_000_000_000.0
                    return ns,sec
                sec=float(ts)
                ns=int(sec*1_000_000_000)
                return ns,sec
            press_ns,press_sec=norm_ts(press_t)
            release_ns,release_sec=norm_ts(release_t)
            mm=[]
            mm_ns=[]
            if moves:
                for (t,x,y,ins) in moves:
                    ns,sec=norm_ts(t)
                    sx,sy=self._scale2560(x-cx,y-cy,cw,ch)
                    mm.append((sec,int(sx),int(sy),int(ins)))
                    mm_ns.append(ns)
            duration=release_sec-press_sec if release_sec and press_sec else 0.0
            snapshot=self.current_data_snapshot()
            traj_seq=[]
            if mm:
                start=mm[0][0]
                end=mm[-1][0]
                span=max(end-start,1e-3)
                for sec,lx,ly,inside in mm:
                    traj_seq.append({"t":max(0.0,min(1.0,(sec-start)/span)),"x":max(0.0,min(1.0,float(lx)/2560.0)),"y":max(0.0,min(1.0,float(ly)/1600.0)),"inside":int(inside)})
            if not traj_seq:
                traj_seq.append({"t":0.0,"x":max(0.0,min(1.0,float(plx)/2560.0)),"y":max(0.0,min(1.0,float(ply)/1600.0)),"inside":int(ins_press)})
            if all(item.get("t",0.0)<0.999 for item in traj_seq):
                traj_seq.append({"t":1.0,"x":max(0.0,min(1.0,float(rlx)/2560.0)),"y":max(0.0,min(1.0,float(rly)/1600.0)),"inside":int(ins_release)})
            path_points=[(max(0.0,min(1.0,float(plx)/2560.0)),max(0.0,min(1.0,float(ply)/1600.0)))]
            path_points.extend((item.get("x",0.0),item.get("y",0.0)) for item in traj_seq)
            path_points.append((max(0.0,min(1.0,float(rlx)/2560.0)),max(0.0,min(1.0,float(rly)/1600.0))))
            data_links=[]
            for item in snapshot:
                nb=item.get("norm_bounds")
                if not isinstance(nb,(list,tuple)) or len(nb)!=4:
                    continue
                try:
                    nx1=float(nb[0])
                    ny1=float(nb[1])
                    nx2=float(nb[2])
                    ny2=float(nb[3])
                except:
                    continue
                nx1=max(0.0,min(1.0,nx1))
                ny1=max(0.0,min(1.0,ny1))
                nx2=max(nx1,min(1.0,nx2))
                ny2=max(ny1,min(1.0,ny2))
                hit=False
                for (nx,ny) in path_points:
                    if nx>=nx1 and nx<=nx2 and ny>=ny1 and ny<=ny2:
                        hit=True
                        break
                if not hit:
                    continue
                data_links.append({"key":item.get("key"),"value":item.get("value"),"preference":item.get("preference"),"norm_bounds":[nx1,ny1,nx2,ny2],"confidence":float(item.get("confidence",0.5))})
            obj={"id":str(uuid.uuid4()),"source":source,"mode":mode or "unknown","type":typ,"press_t":press_sec,"press_t_ns":press_ns,"press_x":px,"press_y":py,"press_lx":plx,"press_ly":ply,"release_t":release_sec,"release_t_ns":release_ns,"release_x":rx,"release_y":ry,"release_lx":rlx,"release_ly":rly,"moves":mm,"moves_ns":mm_ns,"window_title":self.selected_title,"rect":[0,0,2560,1600],"frame_id":frame_id,"ins_press":int(ins_press),"ins_release":int(ins_release),"clip_ids":clip_ids or [],"session_id":self.session_id,"duration":float(duration)}
            obj["trajectory"]=traj_seq
            obj["data_snapshot"]=snapshot
            obj["data_bindings"]=data_links
            self.events_writer.append(obj)
            self.event_count+=1
            self.signal_counts.emit(self.event_count)
            self._update_activity(obj)
            if hasattr(self,"hyper"):
                self.hyper.observe_event(obj)
                self.hyper.adjust()
        except:
            pass
    def record_missed_region(self,img,meta):
        try:
            if img is None or img.size==0:
                return
            self._init_day_files()
            day=os.path.join(exp_dir,self.day_tag)
            miss_dir=os.path.join(day,"missed")
            os.makedirs(miss_dir,exist_ok=True)
            stamp=time.time_ns()
            path=os.path.join(miss_dir,f"{stamp}.png")
            resized=cv2.resize(img,(2560,1600),interpolation=cv2.INTER_LANCZOS4)
            ok,buf=cv2.imencode(".png",resized,[int(cv2.IMWRITE_PNG_COMPRESSION),self.png_comp])
            if not ok:
                return
            with open(path,"wb") as f:
                f.write(buf.tobytes())
                f.flush()
                os.fsync(f.fileno())
            entry=dict(meta)
            entry["id"]=str(uuid.uuid4())
            entry["path"]=path
            entry["timestamp"]=time.time()
            meta_path=os.path.join(miss_dir,"meta.jsonl")
            with open(meta_path,"a",encoding="utf-8") as f:
                f.write(json.dumps(entry,ensure_ascii=False)+"\n")
        except:
            pass
    def _update_activity(self,event):
        try:
            now=time.time()
            duration=max(0.0,float(event.get("duration",0.0) or 0.0))
            window=event.get("window_title") or self.selected_title or ""
            record={"time":now,"duration":duration,"window":window,"source":event.get("source","unknown")}
            self.activity_events.append(record)
            stats=self.window_stats.get(window)
            if stats is None:
                stats={"recent":collections.deque(maxlen=180),"count":0,"duration":0.0}
                self.window_stats[window]=stats
            stats["count"]+=1
            stats["duration"]+=duration
            stats["recent"].append((now,duration,record["source"]))
            while stats["recent"] and now-stats["recent"][0][0]>120:
                stats["recent"].popleft()
            while self.activity_events and now-self.activity_events[0]["time"]>90:
                self.activity_events.popleft()
            self._emit_window_stats()
        except:
            pass
    def _emit_window_stats(self):
        try:
            now=time.time()
            rows=[]
            for title,data in self.window_stats.items():
                recent=list(data.get("recent",[]))
                total=int(data.get("count",0))
                duration=float(data.get("duration",0.0))
                if recent:
                    last_gap=max(0.0,now-recent[-1][0])
                    avg=sum(r[1] for r in recent)/max(1,len(recent))
                    ai=sum(1 for r in recent if len(r)>=3 and r[2]=="ai")/max(1,len(recent))
                else:
                    last_gap=9999.0
                    avg=0.0
                    ai=0.0
                rows.append({"window":title,"count":total,"avg_duration":avg,"ai_ratio":ai,"recent_gap":last_gap,"total_duration":duration})
            rows=sorted(rows,key=lambda x:x.get("recent_gap",9999.0))[:60]
            self.signal_window_stats.emit(rows)
        except:
            pass
    def activity_density(self):
        now=time.time()
        while self.activity_events and now-self.activity_events[0]["time"]>60:
            self.activity_events.popleft()
        if not self.activity_events:
            return 0.0
        total=len(self.activity_events)
        duration=sum(item["duration"] for item in self.activity_events)
        freq=total/60.0
        dur=duration/max(1.0,total)
        return max(0.0,min(2.0,freq*2.5+dur))
    def record_kbd(self,ts):
        try:
            sec=ts/1_000_000_000.0 if isinstance(ts,(int,float)) and ts>1e6 else float(ts)
            if sec-self.last_kbd_log_ts<0.3:
                return
            self.last_kbd_log_ts=sec
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
        act=self.strategy.predict(cv2.resize(img,(2560,1600)),events=self._recent_events(),ui_elements=self.ui_elements,heat_prior=sal,event_prior=prior_events)
        return act
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
    def _recent_events(self,limit=64,source=None,mode=None):
        events=[]
        if self.events_path and os.path.exists(self.events_path):
            try:
                with open(self.events_path,"r",encoding="utf-8") as f:
                    for line in f:
                        obj=json.loads(line)
                        if obj.get("window_title")!=self.selected_title:
                            continue
                        if obj.get("source") not in ["user","ai"]:
                            continue
                        if obj.get("type") not in ["left","right","middle"]:
                            continue
                        if source and obj.get("source")!=source:
                            continue
                        if mode and obj.get("mode")!=mode:
                            continue
                        events.append(obj)
                        if len(events)>max(limit*4,4000):
                            events=events[-max(limit*4,4000):]
            except:
                pass
        return events[-limit:]
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
                    stamp=time.time_ns()
                    for k,v in list(self.mouse_pressed.items()):
                        if k in self.mouse_pressed:
                            inside=1 if self.within(x,y) else 0
                            v["moves"].append((stamp,x,y,inside))
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
                                press_ns=time.time_ns()
                                self.mouse_pressed[btn]={"t":press_ns,"x":x,"y":y,"moves":[(press_ns,x,y,1)],"pre":[]}
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
                                    release_ns=time.time_ns()
                                    self.st.record_op("user",btn,d["t"],d["x"],d["y"],release_ns,x,y,d["moves"],rid,ip,ir,pre,"learning")
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
                    t=time.time_ns()
                    self.st.last_user_activity=time.time()
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
            self.st.update_rects(wr,cr)
            if cr==(0,0,0,0) or not self.st.selected_hwnd or not Foreground.ready(self.st.selected_hwnd,cr):
                self.st.update_visibility(False,"学习暂停：窗口不可见或被遮挡")
                time.sleep(0.12)
                continue
            img=self.st.capture_client()
            if img is None:
                time.sleep(0.06)
                continue
            self.st.update_visibility(True,"learning")
            fid=self.st.record_frame(img,"learning",force=False)
            self.st.adapt_fps(prev,img)
            self.st.prev_img=img
            self.st.prev_imgs.append(img)
            prev=img
            if self.ui.preview_enabled():
                disp=self.st.preview_visual(img)
                if isinstance(disp,np.ndarray):
                    buf=np.ascontiguousarray(disp)
                else:
                    buf=np.ascontiguousarray(img)
                h,w=buf.shape[:2]
                qi=QImage(buf.data,w,h,3*w,QImage.Format_BGR888)
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
            self.st.update_rects(wr,cr)
            if cr==(0,0,0,0) or not self.st.selected_hwnd or not Foreground.ready(self.st.selected_hwnd,cr):
                self.st.update_visibility(False,"训练暂停：窗口不可见或被遮挡")
                time.sleep(0.10)
                continue
            if not Foreground.ensure_front(self.st.selected_hwnd):
                self.st.update_visibility(False,"训练暂停：等待窗口激活")
                time.sleep(0.06)
                continue
            if Foreground.occluded(self.st.selected_hwnd,cr):
                self.st.update_visibility(False,"训练暂停：窗口被遮挡")
                time.sleep(0.10)
                continue
            img=self.st.capture_client()
            if img is None:
                time.sleep(0.06)
                continue
            self.st.update_visibility(True,"training")
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
                disp=self.st.preview_visual(img)
                if isinstance(disp,np.ndarray):
                    buf=np.ascontiguousarray(disp)
                else:
                    buf=np.ascontiguousarray(img)
                h,w=buf.shape[:2]
                qi=QImage(buf.data,w,h,3*w,QImage.Format_BGR888)
                self.st.signal_preview.emit(QPixmap.fromImage(qi))
            kind=act.get("kind") if isinstance(act,dict) else "idle"
            heat=act.get("heat") if isinstance(act,dict) else None
            if kind=="idle":
                time.sleep(max(1.0/self.st.fps,0.01))
                continue
            if time.time()<cool_t:
                time.sleep(max(1.0/self.st.fps,0.01))
                continue
            if self.st.interrupt_ai:
                break
            cooldown=self._execute_action(act,heat,fid)
            if cooldown is not None:
                cool_t=time.time()+max(0.12,cooldown)
            time.sleep(max(1.0/self.st.fps,0.01))
    def _grid(self,act,heat):
        if isinstance(act,dict):
            if act.get("grid"):
                g=act.get("grid")
                return (int(g[0]),int(g[1]))
            if act.get("heat_shape"):
                g=act.get("heat_shape")
                return (int(g[0]),int(g[1]))
        if isinstance(heat,np.ndarray):
            return (heat.shape[1],heat.shape[0])
        cw=max(1,self.st.client_rect[2]-self.st.client_rect[0])
        ch=max(1,self.st.client_rect[3]-self.st.client_rect[1])
        return (cw,ch)
    def _coord_abs(self,pt,grid):
        cw=max(1,self.st.client_rect[2]-self.st.client_rect[0])
        ch=max(1,self.st.client_rect[3]-self.st.client_rect[1])
        sw=max(1,grid[0])
        sh=max(1,grid[1])
        nx=float(pt[0])/max(1,sw-1)
        ny=float(pt[1])/max(1,sh-1)
        bx=self.st.client_rect[0]+int(round(nx*(cw-1)))
        by=self.st.client_rect[1]+int(round(ny*(ch-1)))
        return self.st.clamp_abs(bx,by)
    def _cursor_path(self,start,end,steps):
        cx,cy=start
        bx,by=end
        c1x=cx+(bx-cx)*0.3+random.uniform(-12,12)
        c1y=cy+(by-cy)*0.3+random.uniform(-12,12)
        c2x=cx+(bx-cx)*0.7+random.uniform(-12,12)
        c2y=cy+(by-cy)*0.7+random.uniform(-12,12)
        pts=[]
        steps=max(steps,3)
        for i in range(1,steps+1):
            u=i/float(steps)
            ux=(1-u)**3*cx+3*(1-u)**2*u*c1x+3*(1-u)*u**2*c2x+u**3*bx
            uy=(1-u)**3*cy+3*(1-u)**2*u*c1y+3*(1-u)*u**2*c2y+u**3*by
            pts.append((int(ux),int(uy)))
        return pts
    def _segment_path(self,start,end,steps):
        pts=[]
        steps=max(steps,1)
        for i in range(1,steps+1):
            u=i/float(steps)
            x=int(round(start[0]+(end[0]-start[0])*u))
            y=int(round(start[1]+(end[1]-start[1])*u))
            pts.append((x,y))
        return pts
    def _within(self,x,y):
        r=self.st.client_rect
        return x>=r[0] and x<=r[2] and y>=r[1] and y<=r[3]
    def _execute_action(self,act,heat,fid):
        if not isinstance(act,dict):
            return None
        if hasattr(self.st,"validator"):
            self.st.validator.enqueue_strategy(act,fid)
        if act.get("kind")=="combo":
            total=0.0
            seq=act.get("sequence") or []
            for idx,item in enumerate(seq):
                if self.st.interrupt_ai:
                    break
                cool=self._execute_single(item,heat,fid)
                if cool is not None:
                    total+=cool
                if idx<len(seq)-1:
                    time.sleep(max(0.02,min(0.12,float(item.get("duration",0.05)))))
            return total if total>0 else 0.25
        if act.get("kind")=="parallel":
            streams=act.get("streams") or []
            if not streams:
                return None
            total=0.0
            longest=max(len(s) for s in streams)
            for step in range(longest):
                for seq in streams:
                    if step>=len(seq):
                        continue
                    if self.st.interrupt_ai:
                        return total if total>0 else 0.25
                    cool=self._execute_single(seq[step],heat,fid)
                    if cool is not None and cool>total:
                        total=cool
            return total if total>0 else 0.25
        return self._execute_single(act,heat,fid)
    def _execute_single(self,act,heat,fid):
        if not isinstance(act,dict):
            return None
        grid=self._grid(act,heat)
        coord=act.get("coord") or act.get("start")
        if coord is None:
            return None
        btn=act.get("button","left")
        duration=float(act.get("duration",0.05))
        path=act.get("path") or [tuple(coord)]
        seq=[(int(p[0]),int(p[1])) for p in path]
        abs_seq=[self._coord_abs(p,grid) for p in seq]
        if not abs_seq:
            return None
        try:
            cur=win32api.GetCursorPos()
        except:
            cur=abs_seq[0]
        pre=self._cursor_path(cur,abs_seq[0],max(10,int(self.st.fps*0.6)))
        mv=[]
        if pre:
            if not self.inj.move_path(pre,step_delay=max(0.004,0.2/self.st.fps)):
                return None
            for (px,py) in pre:
                mv.append((time.time_ns(),px,py,1))
        if self.st.interrupt_ai:
            return None
        press_ns=time.time_ns()
        self.inj.down(btn)
        mv.append((press_ns,abs_seq[0][0],abs_seq[0][1],1))
        for i in range(1,len(abs_seq)):
            seg=self._segment_path(abs_seq[i-1],abs_seq[i],max(6,int(self.st.fps*0.4)))
            if not seg:
                continue
            if not self.inj.move_path(seg,step_delay=max(0.004,0.18/self.st.fps)):
                try:
                    self.inj.up(btn)
                except:
                    pass
                return None
            for (px,py) in seg:
                mv.append((time.time_ns(),px,py,1))
        hold=max(0.01,duration)
        time.sleep(hold)
        self.inj.up(btn)
        release_ns=time.time_ns()
        rel=abs_seq[-1]
        ins_press=1 if self._within(abs_seq[0][0],abs_seq[0][1]) else 0
        ins_release=1 if self._within(rel[0],rel[1]) else 0
        mv.append((release_ns,rel[0],rel[1],1))
        self.st.record_op("ai",btn,press_ns,abs_seq[0][0],abs_seq[0][1],release_ns,rel[0],rel[1],mv,fid,ins_press,ins_release,[],"training")
        return hold+0.25
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
        self.frame_cache=collections.OrderedDict()
        self.cache_limit=256
        hp=getattr(self.st,"hyper",None)
        self.gamma=hp.values.get("gamma",0.92) if hp else 0.92
        self.lam=hp.values.get("lam",0.85) if hp else 0.85
        self.batch_size=hp.values.get("batch_size",6) if hp else 6
        self.trace=0.0
        self.baseline=0.0
        self.reward_engine=getattr(self.st,"reward_engine",RewardComposer())
    def run(self):
        self.st.optimizing=True
        try:
            self.trace=0.0
            self.baseline=0.0
            self.st.signal_optprog.emit(10,"收集数据")
            all_days=sorted([p for p in glob.glob(os.path.join(exp_dir,"*")) if os.path.isdir(p)])
            sample_count,tsmin,tsmax=self._count_samples(all_days)
            if self.cancel_flag.is_set():
                self._done(False)
                return
            trained_batches_count=0
            trained_sample_count=0
            if sample_count>0:
                total_batches=max(1,(sample_count+self.batch_size-1)//self.batch_size)
                self.st.signal_optprog.emit(45,"整理样本")
                self.st.signal_optprog.emit(50,"构建批次")
                batch_iter=self._stream_batches(all_days,total_batches)
                self.st.signal_optprog.emit(55,"准备训练")
                self.st.strategy.train_incremental(batch_iter,lambda p,t:self.st.signal_optprog.emit(p,t),total_batches)
                trained_batches_count=total_batches
                trained_sample_count=sample_count
            else:
                frames_pool=self._collect_frames(all_days,limit=256)
                events=self._enrich_events([],frames_pool)
                if len(events)==0:
                    batches=self._create_default_batches()
                else:
                    synth=self._sort_events(events)
                    batches=self._build_batches(synth,frames_pool)
                self.st.signal_optprog.emit(55,"准备训练")
                self.st.strategy.train_incremental(batches,lambda p,t:self.st.signal_optprog.emit(p,t),len(batches))
                trained_batches_count=len(batches)
                trained_sample_count=sum(int(b[0].shape[0]) for b in batches) if batches else 0
            path=self.st.manifest.target_path()
            ModelIO.save(self.st.strategy.model,path)
            stamp=time.strftime("%Y%m%d_%H%M%S")
            suffix=Path(path).suffix or ".npz"
            backup_name=f"{Path(path).stem}_{stamp}_{time.time_ns()%1000000:06d}{suffix}"
            backup_path=os.path.join(models_dir,backup_name)
            try:
                shutil.copy2(path,backup_path)
            except:
                backup_path=None
            calc=self.st.manifest._hash(path)
            self.st.manifest.sha_value=calc
            self.st.manifest._update_cfg_hash(calc)
            self.st.cfg["model_sha256"]=calc
            self.st.cfg["model_sha256_backup"]=calc
            if backup_path:
                self.st.cfg["model_fallback"]=backup_path
            with open(cfg_path,"w",encoding="utf-8") as f:
                json.dump(self.st.cfg,f,ensure_ascii=False,indent=2)
                f.flush()
                os.fsync(f.fileno())
            files=[path]
            if backup_path and os.path.exists(backup_path):
                files.append(backup_path)
            ModelMeta.update(files,{"start":tsmin,"end":tsmax},{"samples":trained_sample_count,"batches":trained_batches_count})
            self.st.strategy.ensure_loaded(lambda p,t:None)
            self.st.model_loaded=True
            ver=f"v{ModelMeta.read().get('version',0)}"
            self.st.model_ver_str=ver
            self.st.signal_modelver.emit(ver)
            self._done(True)
        except Exception as e:
            log(f"opt error:{e}")
            self._done(False)
    def _done(self,ok):
        self.st.optimizing=False
        self.cb(ok)
    def _get_frame_image(self,path):
        if not path or not os.path.exists(path):
            return None
        try:
            if path in self.frame_cache:
                img=self.frame_cache.pop(path)
                self.frame_cache[path]=img
                return img
            buf=np.fromfile(path,dtype=np.uint8)
            img=cv2.imdecode(buf,cv2.IMREAD_COLOR)
            if img is not None:
                self.frame_cache[path]=img
                if len(self.frame_cache)>self.cache_limit:
                    self.frame_cache.popitem(last=False)
            return img
        except:
            return None
    def _count_samples(self,days):
        total=0
        tsmin=None
        tsmax=None
        total_days=max(1,len(days))
        processed=0
        for d in days:
            if self.cancel_flag.is_set():
                break
            frames=self._load_frames_meta(d)
            ep=os.path.join(d,"events.jsonl")
            if os.path.exists(ep):
                JSONLWriter.repair(ep)
                hist=collections.deque(maxlen=64)
                with open(ep,"r",encoding="utf-8") as f:
                    for line in f:
                        try:
                            e=json.loads(line)
                        except:
                            continue
                        hist.append(e)
                        t=self._event_time(e)
                        if t is not None:
                            if tsmin is None or t<tsmin:
                                tsmin=t
                            if tsmax is None or t>tsmax:
                                tsmax=t
                        if self._qualify_event(e) and self._valid_label(e,frames):
                            total+=1
                hist.clear()
            processed+=1
            self.st.signal_optprog.emit(20+int(20*processed/max(1,total_days)),"解析数据")
        return total,tsmin,tsmax
    def _load_frames_meta(self,day,limit=None):
        frames={}
        fp=os.path.join(day,"frames.jsonl")
        if os.path.exists(fp):
            JSONLWriter.repair(fp)
            with open(fp,"r",encoding="utf-8") as f:
                for line in f:
                    if limit is not None and len(frames)>=limit:
                        break
                    try:
                        r=json.loads(line)
                        frames[r["id"]]=r
                    except:
                        pass
        return frames
    def _collect_frames(self,days,limit=None):
        collected={}
        for d in days:
            if self.cancel_flag.is_set():
                break
            remain=None
            if limit is not None:
                remain=max(0,limit-len(collected))
                if remain<=0:
                    break
            frames=self._load_frames_meta(d,remain)
            collected.update(frames)
            if limit is not None and len(collected)>=limit:
                break
        return collected
    def _qualify_event(self,e):
        if e.get("type") not in ["left","right","middle"]:
            return False
        if e.get("mode") not in ["learning","training"]:
            return False
        return True
    def _valid_label(self,e,frames):
        fid=e.get("frame_id")
        if not fid:
            return False
        fr=frames.get(fid)
        if not fr:
            size=e.get("frame_size") if isinstance(e.get("frame_size"),(list,tuple)) else None
            if not size or len(size)<2:
                return False
            try:
                w=int(size[0])
                h=int(size[1])
            except:
                return False
        else:
            try:
                w=int(fr.get("w",0))
                h=int(fr.get("h",0))
            except:
                return False
        if w<=0 or h<=0:
            return False
        try:
            x=int(float(e.get("press_lx",0))/2560.0*w)
            y=int(float(e.get("press_ly",0))/1600.0*h)
        except:
            return False
        if x<0 or x>=w or y<0 or y>=h:
            return False
        return True
    def _stream_batches(self,days,total_batches):
        bs=self.batch_size
        pending=[]
        produced=0
        for d in days:
            if self.cancel_flag.is_set():
                return
            frames=self._load_frames_meta(d)
            ep=os.path.join(d,"events.jsonl")
            if not os.path.exists(ep):
                continue
            JSONLWriter.repair(ep)
            hist=collections.deque(maxlen=64)
            with open(ep,"r",encoding="utf-8") as f:
                for line in f:
                    if self.cancel_flag.is_set():
                        return
                    try:
                        e=json.loads(line)
                    except:
                        continue
                    hist.append(e)
                    if not self._qualify_event(e):
                        continue
                    fid=e.get("frame_id")
                    fr=frames.get(fid)
                    if not fr and e.get("frame_path"):
                        fr={"path":e.get("frame_path"),"w":(e.get("frame_size") or [2560,1600])[0],"h":(e.get("frame_size") or [2560,1600])[1]}
                    if not fr:
                        continue
                    img=self._get_frame_image(fr.get("path"))
                    if img is None and e.get("frame_path"):
                        img=self._get_frame_image(e.get("frame_path"))
                    if img is None:
                        continue
                    label,data_vec,traj=self._make_label(e,img.shape[1],img.shape[0])
                    if label is None:
                        continue
                    ctx=self._context_tensor(list(hist),len(hist)-1)
                    hist_t=self._history_tensor(list(hist),len(hist)-1)
                    size=self.st.policy_resolution
                    tensor_img=torch.from_numpy(cv2.resize(img,size)).permute(2,0,1).float()/255.0
                    reward=self._reward_of_event(e)
                    value,adv,weight=self._rl_signal(reward)
                    pending.append((tensor_img,ctx,label,data_vec,traj,hist_t,value,adv,weight))
                    if len(pending)>=bs:
                        produced+=1
                        yield self._stack_batch(pending[:bs])
                        del pending[:bs]
                        if total_batches>0:
                            prog=45+int(5*min(produced,total_batches)/total_batches)
                            self.st.signal_optprog.emit(min(prog,55),"构建批次")
            hist.clear()
        if pending:
            yield self._stack_batch(pending)
    def _stack_batch(self,samples):
        imgs=[]
        ctxs=[]
        grids=[]
        datas=[]
        trajs=[]
        lengths=[]
        hists=[]
        vals=[]
        advs=[]
        weights=[]
        for (img,ctx,label,data_vec,traj,hist,val,adv,weight) in samples:
            imgs.append(img.float())
            ctxs.append(ctx.to(torch.float32) if isinstance(ctx,torch.Tensor) else torch.tensor(ctx,dtype=torch.float32))
            grids.append(label.to(torch.float32) if isinstance(label,torch.Tensor) else torch.tensor(label,dtype=torch.float32))
            datas.append(data_vec.to(torch.float32) if isinstance(data_vec,torch.Tensor) else torch.tensor(data_vec,dtype=torch.float32))
            traj_tensor=traj if isinstance(traj,torch.Tensor) else torch.tensor(traj,dtype=torch.float32)
            if traj_tensor.ndim==1:
                traj_tensor=traj_tensor.unsqueeze(0)
            if traj_tensor.numel()==0 or traj_tensor.shape[0]==0:
                traj_tensor=torch.zeros((1,4),dtype=torch.float32)
            trajs.append(traj_tensor.to(torch.float32))
            lengths.append(int(traj_tensor.shape[0]))
            hists.append(hist.to(torch.float32) if isinstance(hist,torch.Tensor) else torch.tensor(hist,dtype=torch.float32))
            vals.append(float(val))
            advs.append(float(adv))
            weights.append(float(weight))
        max_len=max(lengths) if lengths else 1
        padded=[]
        mask_list=[]
        for traj_tensor,length in zip(trajs,lengths):
            steps=traj_tensor.shape[0]
            if steps<max_len:
                pad=torch.zeros((max_len-steps,traj_tensor.shape[1]),dtype=torch.float32)
                padded.append(torch.cat([traj_tensor,pad],dim=0))
            else:
                padded.append(traj_tensor[:max_len])
            mask=torch.zeros((max_len,),dtype=torch.float32)
            mask[:min(length,max_len)]=1.0
            mask_list.append(mask)
        return (torch.stack(imgs,dim=0),torch.stack(ctxs,dim=0),torch.stack(grids,dim=0),torch.stack(hists,dim=0),torch.stack(datas,dim=0),torch.stack(padded,dim=0),torch.stack(mask_list,dim=0),torch.tensor(vals,dtype=torch.float32).unsqueeze(1),torch.tensor(advs,dtype=torch.float32).unsqueeze(1),torch.tensor(weights,dtype=torch.float32).unsqueeze(1))
    def _reward_of_event(self,e):
        if not isinstance(e,dict):
            return 0.0
        try:
            reward=float(self.reward_engine.evaluate(e))
        except:
            reward=0.0
        if reward<=0.0:
            dur=max(0.0,float(e.get("duration",0.0) or 0.0))
            base=0.08+min(0.36,dur*0.4)
            if e.get("type")=="left":
                base+=0.12
            elif e.get("type")=="right":
                base+=0.08
            reward=base
        return max(0.0,min(2.2,reward))
    def _rl_signal(self,reward):
        r=float(reward)
        self.trace=self.trace*self.gamma*self.lam+r
        self.baseline=self.baseline*0.97+r*0.03
        adv=self.trace-self.baseline
        adv=max(-2.5,min(2.5,adv))
        value=max(-3.0,min(3.0,self.trace))
        weight=max(0.1,min(2.5,0.5+abs(adv)*0.7))
        if adv<0:
            weight*=0.6
        return value,adv,weight
    def _sort_events(self,events):
        return sorted(events,key=lambda e:self._event_time(e) or 0)
    def _event_time(self,e):
        if "press_t_ns" in e and e.get("press_t_ns"):
            try:
                return float(e.get("press_t_ns",0))/1_000_000_000.0
            except:
                return None
        if "press_t" in e:
            try:
                return float(e.get("press_t",0))
            except:
                return None
        if "ts_ns" in e and e.get("ts_ns"):
            try:
                return float(e.get("ts_ns",0))/1_000_000_000.0
            except:
                return None
        if "ts" in e:
            try:
                return float(e.get("ts",0))
            except:
                return None
        return None
    def _build_batches(self,events,frames):
        seqs=[]
        for idx,e in enumerate(events):
            if e.get("type") not in ["left","right","middle"]:
                continue
            if e.get("mode") not in ["learning","training"]:
                continue
            fid=e.get("frame_id")
            fr=frames.get(fid)
            if not fr and e.get("frame_path"):
                fr={"path":e.get("frame_path"),"w":(e.get("frame_size") or [2560,1600])[0],"h":(e.get("frame_size") or [2560,1600])[1]}
            if not fr:
                continue
            img=self._get_frame_image(fr.get("path"))
            if img is None and e.get("frame_path"):
                img=self._get_frame_image(e.get("frame_path"))
            if img is None:
                continue
            size=self.st.policy_resolution
            img=cv2.resize(img,size)
            label,data_vec,traj=self._make_label(e,img.shape[1],img.shape[0])
            if label is None:
                continue
            ctx=self._context_tensor(events,idx)
            hist=self._history_tensor(events,idx)
            reward=self._reward_of_event(e)
            seqs.append((img,ctx,label,data_vec,traj,hist,reward))
        random.shuffle(seqs)
        batches=[]
        bs=self.batch_size
        for i in range(0,len(seqs),bs):
            chunk=seqs[i:i+bs]
            samples=[]
            for (img,ctx,label,data_vec,traj,hist,reward) in chunk:
                value,adv,weight=self._rl_signal(reward)
                samples.append((torch.from_numpy(img).permute(2,0,1).float()/255.0,ctx,label,data_vec,traj,hist,value,adv,weight))
            if not samples:
                continue
            batches.append(self._stack_batch(samples))
        return batches
    def _trajectory_points(self,e,w,h):
        pts=[]
        traj=e.get("trajectory")
        if isinstance(traj,list) and traj:
            for item in traj:
                if not isinstance(item,dict):
                    continue
                try:
                    t=float(item.get("t",0.0))
                    x=float(item.get("x",0.0))
                    y=float(item.get("y",0.0))
                    inside=float(item.get("inside",0.0))
                except:
                    continue
                if x>1.0 or y>1.0:
                    x=x/2560.0
                    y=y/1600.0
                pts.append((t,x,y,inside))
        if not pts:
            moves=e.get("moves") if isinstance(e.get("moves"),list) else []
            for mv in moves:
                if not isinstance(mv,(list,tuple)) or len(mv)<4:
                    continue
                try:
                    t=float(mv[0])
                    x=float(mv[1])/2560.0
                    y=float(mv[2])/1600.0
                    inside=float(mv[3])
                except:
                    continue
                pts.append((t,x,y,inside))
        if not pts:
            px=float(e.get("press_lx",0))/2560.0
            py=float(e.get("press_ly",0))/1600.0
            rx=float(e.get("release_lx",px*2560.0))/2560.0
            ry=float(e.get("release_ly",py*1600.0))/1600.0
            pts=[(0.0,px,py,1.0),(max(1.0,float(e.get("duration",0.06))),rx,ry,float(e.get("ins_release",0)))]
        pts=[(p[0],max(0.0,min(1.0,p[1])),max(0.0,min(1.0,p[2])),max(0.0,min(1.0,p[3]))) for p in pts]
        pts.sort(key=lambda item:item[0])
        start=pts[0][0]
        end=pts[-1][0]
        span=max(end-start,1e-3)
        norm=[((p[0]-start)/span, p[1], p[2], p[3]) for p in pts]
        return norm
    def _data_targets(self,e,grid_h,grid_w):
        data_map=np.zeros((grid_h,grid_w),dtype=np.float32)
        vec=np.zeros((8,2),dtype=np.float32)
        bindings=e.get("data_bindings") if isinstance(e.get("data_bindings"),list) else []
        idx=0
        for b in bindings:
            if idx>=8:
                break
            if not isinstance(b,dict):
                continue
            nb=b.get("norm_bounds")
            if not isinstance(nb,(list,tuple)) or len(nb)!=4:
                continue
            try:
                nx1=float(nb[0])
                ny1=float(nb[1])
                nx2=float(nb[2])
                ny2=float(nb[3])
            except:
                continue
            nx1=max(0.0,min(1.0,nx1))
            ny1=max(0.0,min(1.0,ny1))
            nx2=max(nx1+1e-4,min(1.0,nx2))
            ny2=max(ny1+1e-4,min(1.0,ny2))
            gx1=int(max(0,min(grid_w-1,math.floor(nx1*grid_w))))
            gy1=int(max(0,min(grid_h-1,math.floor(ny1*grid_h))))
            gx2=int(max(gx1+1,min(grid_w,math.ceil(nx2*grid_w))))
            gy2=int(max(gy1+1,min(grid_h,math.ceil(ny2*grid_h))))
            weight=float(max(0.0,min(1.0,b.get("confidence",0.5))))
            val=float(b.get("value",0.0) or 0.0)
            score=(math.tanh(val/500.0)+1.0)/2.0
            pref=str(b.get("preference",""))
            pref_norm=1.0 if pref.lower().startswith("high") or pref.startswith("越高") else (-1.0 if pref.lower().startswith("low") or pref.startswith("越低") else 0.0)
            vec[idx,0]=score
            vec[idx,1]=pref_norm
            data_map[gy1:gy2,gx1:gx2]+=weight*score
            idx+=1
        if data_map.max()>0:
            data_map=data_map/data_map.max()
        return data_map,vec.reshape(-1)
    def _trajectory_tensor(self,points):
        if not points:
            return torch.zeros((1,4),dtype=torch.float32)
        arr=np.zeros((len(points),4),dtype=np.float32)
        for i,(t,x,y,inside) in enumerate(points):
            arr[i,0]=float(max(0.0,min(1.0,t)))
            arr[i,1]=float(max(0.0,min(1.0,x)))
            arr[i,2]=float(max(0.0,min(1.0,y)))
            arr[i,3]=float(max(0.0,min(1.0,inside)))
        return torch.from_numpy(arr)
    def _make_label(self,e,w,h):
        x=int(float(e.get("press_lx",0))/2560.0*w)
        y=int(float(e.get("press_ly",0))/1600.0*h)
        if x<0 or x>=w or y<0 or y>=h:
            return None,None,None
        grid_h=13
        grid_w=20
        gx=int(max(0,min(grid_w-1,round(x*grid_w/max(1,w)))))
        gy=int(max(0,min(grid_h-1,round(y*grid_h/max(1,h)))))
        grid=torch.zeros((4,grid_h,grid_w),dtype=torch.float32)
        if e.get("type")=="left":
            grid[0,gy,gx]=1.0
        if e.get("type")=="right":
            grid[1,gy,gx]=1.0
        points=self._trajectory_points(e,w,h)
        for _,nx,ny,inside in points:
            tx=int(max(0,min(grid_w-1,round(nx*(grid_w-1)))))
            ty=int(max(0,min(grid_h-1,round(ny*(grid_h-1)))))
            grid[2,ty,tx]+=1.0+inside*0.2
        total=float(grid[2].sum())
        if total>0:
            grid[2]=grid[2]/total
        data_map,data_vec=self._data_targets(e,grid_h,grid_w)
        grid[3]=torch.from_numpy(data_map.astype(np.float32))
        data_tensor=torch.from_numpy(data_vec.astype(np.float32))
        traj_tensor=self._trajectory_tensor(points)
        return grid,data_tensor,traj_tensor
    def _context_tensor(self,events,idx):
        ctx=[]
        dim=self.st.strategy.context_dim
        start=max(0,idx-8)
        subset=events[start:idx]
        for e in subset[-8:]:
            if e.get("type") not in ["left","right","middle"]:
                continue
            vec=[0.0]*dim
            vec[0]=1.0 if e.get("type")=="left" else 0.0
            vec[1]=1.0 if e.get("type")=="right" else 0.0
            vec[2]=1.0 if e.get("source")=="ai" else 0.0
            vec[3]=float(e.get("press_lx",0))/2560.0
            vec[4]=float(e.get("press_ly",0))/1600.0
            vec[5]=float(e.get("release_lx",0))/2560.0
            vec[6]=float(e.get("release_ly",0))/1600.0
            vec[7]=float(e.get("ins_press",0))
            vec[8]=float(e.get("ins_release",0))
            vec[9]=float(e.get("duration",0))
            ctx.append(vec[:dim])
        while len(ctx)<8:
            ctx.insert(0,[0.0]*dim)
        return torch.tensor(ctx,dtype=torch.float32)
    def _history_tensor(self,events,idx):
        dim=self.st.strategy.context_dim
        start=max(0,idx-12)
        subset=events[start:idx]
        seq=[]
        for e in subset[-12:]:
            if e.get("type") not in ["left","right","middle"]:
                continue
            vec=[0.0]*dim
            vec[0]=1.0 if e.get("type")=="left" else 0.0
            vec[1]=1.0 if e.get("type")=="right" else 0.0
            vec[2]=1.0 if e.get("source")=="ai" else 0.0
            vec[3]=float(e.get("press_lx",0))/2560.0
            vec[4]=float(e.get("press_ly",0))/1600.0
            vec[5]=float(e.get("release_lx",0))/2560.0
            vec[6]=float(e.get("release_ly",0))/1600.0
            vec[7]=float(e.get("ins_press",0))
            vec[8]=float(e.get("ins_release",0))
            vec[9]=float(e.get("duration",0))
            seq.append(vec[:dim])
        while len(seq)<12:
            seq.insert(0,[0.0]*dim)
        return torch.tensor(seq,dtype=torch.float32)
    def _enrich_events(self,events,frames):
        enriched=list(events)
        if len(enriched)==0:
            enriched.extend(self._synthesize_from_frames(frames,limit=64))
        if len(enriched)<30:
            aug=[]
            for e in enriched:
                aug.append(self._augment_event(e))
                if len(enriched)+len(aug)>=30:
                    break
            enriched.extend(aug[:max(0,30-len(enriched))])
        missed=self._load_missed_events(limit=24)
        if missed:
            enriched.extend(missed)
        return enriched
    def _load_missed_events(self,limit=24):
        entries=[]
        for d in sorted(glob.glob(os.path.join(exp_dir,"*")),reverse=True):
            meta=os.path.join(d,"missed","meta.jsonl")
            if not os.path.exists(meta):
                continue
            try:
                with open(meta,"r",encoding="utf-8") as f:
                    for line in f:
                        if len(entries)>=limit:
                            break
                        entry=json.loads(line)
                        if isinstance(entry,dict):
                            entries.append(entry)
            except:
                continue
            if len(entries)>=limit:
                break
        picked=sorted(entries,key=lambda e:e.get("timestamp",0),reverse=True)[:limit]
        events=[]
        for entry in picked:
            path=entry.get("path")
            if not path or not os.path.exists(path):
                continue
            win=entry.get("window_size") or [2560,1600]
            try:
                w=int(win[0]) if isinstance(win,(list,tuple)) else 2560
                h=int(win[1]) if isinstance(win,(list,tuple)) else 1600
            except:
                w=2560
                h=1600
            nb=entry.get("norm_bounds") or [0.0,0.0,1.0,1.0]
            try:
                cx=float(nb[0]+nb[2])/2.0
                cy=float(nb[1]+nb[3])/2.0
            except:
                cx=0.5
                cy=0.5
            press_lx=int(max(0,min(2560,cx*2560.0)))
            press_ly=int(max(0,min(1600,cy*1600.0)))
            key=entry.get("key") or entry.get("id") or f"miss_{uuid.uuid4()}"
            binding={"key":key,"value":entry.get("value"),"preference":entry.get("preference","ignore"),"norm_bounds":nb,"confidence":float(entry.get("confidence",0.3))}
            eid=entry.get("id") or str(uuid.uuid4())
            event={"id":eid,"source":"user","mode":"learning","type":"left","press_lx":press_lx,"press_ly":press_ly,"release_lx":press_lx,"release_ly":press_ly,"duration":0.05,"ins_press":1,"ins_release":1,"frame_id":eid,"frame_path":path,"frame_size":[w,h],"trajectory":[{"t":0.0,"x":cx,"y":cy,"inside":1.0},{"t":1.0,"x":cx,"y":cy,"inside":1.0}],"data_bindings":[binding]}
            events.append(event)
        return events
    def _augment_event(self,e):
        base=dict(e)
        base["id"]=str(uuid.uuid4())
        px=float(e.get("press_lx",0))
        py=float(e.get("press_ly",0))
        jx=int(max(0,min(2560,round(px+random.randint(-32,32)))))
        jy=int(max(0,min(1600,round(py+random.randint(-32,32)))))
        base["press_lx"]=jx
        base["press_ly"]=jy
        base["release_lx"]=jx
        base["release_ly"]=jy
        base["press_x"]=float(e.get("press_x",0))+random.uniform(-16,16)
        base["press_y"]=float(e.get("press_y",0))+random.uniform(-16,16)
        base["release_x"]=base["press_x"]
        base["release_y"]=base["press_y"]
        press_sec=float(e.get("press_t",time.time()))+random.uniform(-0.2,0.2)
        press_ns=int(press_sec*1_000_000_000)
        dur=max(0.01,float(e.get("duration",0.05))+random.uniform(-0.02,0.02))
        release_sec=press_sec+dur
        release_ns=int(release_sec*1_000_000_000)
        base["press_t"]=press_sec
        base["press_t_ns"]=press_ns
        base["release_t"]=release_sec
        base["release_t_ns"]=release_ns
        base["duration"]=float(dur)
        base["moves"]=e.get("moves",[])
        base["moves_ns"]=e.get("moves_ns",[])
        base["source"]=e.get("source","user")
        base["mode"]=e.get("mode","learning")
        base["type"]=e.get("type","left")
        base["window_title"]=e.get("window_title",self.st.selected_title)
        base["session_id"]=self.st.session_id
        return base
    def _synthesize_from_frames(self,frames,limit=60):
        items=list(frames.values())
        random.shuffle(items)
        synth=[]
        self.st.strategy.ensure_loaded(lambda p,t:None)
        for rec in items[:limit]:
            path=rec.get("path")
            if not path or not os.path.exists(path):
                continue
            img=self._get_frame_image(path)
            if img is None:
                continue
            try:
                act=self.st.strategy.predict(cv2.resize(img,(2560,1600)),events=[],ui_elements=self.st.ui_elements,heat_prior=None,event_prior=None)
            except Exception as ex:
                log(f"synth predict fail:{ex}")
                continue
            step=None
            if isinstance(act,dict):
                kind=act.get("kind")
                if kind in ["tap","long_press","drag"]:
                    step=act
                elif kind=="combo":
                    seq=act.get("sequence") or []
                    if seq:
                        step=seq[0]
            if not isinstance(step,dict):
                continue
            coord=step.get("coord") or step.get("start")
            if coord is None:
                continue
            btn=step.get("button","left")
            press_lx=int(max(0,min(2560,round(coord[0]))))
            press_ly=int(max(0,min(1600,round(coord[1]))))
            now_ns=time.time_ns()
            now_sec=now_ns/1_000_000_000.0
            dur=max(0.02,float(step.get("duration",0.05)))
            release=step.get("release") or coord
            relx=int(max(0,min(2560,round(release[0]))))
            rely=int(max(0,min(1600,round(release[1]))))
            release_ns=int(now_ns+dur*1_000_000_000)
            release_sec=release_ns/1_000_000_000.0
            synth.append({"id":str(uuid.uuid4()),"source":"ai","mode":"training","type":btn,"press_t":now_sec,"press_t_ns":now_ns,"press_x":0.0,"press_y":0.0,"press_lx":press_lx,"press_ly":press_ly,"release_t":release_sec,"release_t_ns":release_ns,"release_x":0.0,"release_y":0.0,"release_lx":relx,"release_ly":rely,"moves":[],"moves_ns":[],"window_title":rec.get("window_title",self.st.selected_title),"rect":[0,0,2560,1600],"frame_id":rec.get("id"),"ins_press":1,"ins_release":1,"clip_ids":[],"session_id":self.st.session_id,"duration":dur})
        return synth
    def _create_default_batches(self):
        hp=getattr(self.st,"hyper",None)
        bs=int(max(2,hp.values.get("batch_size",6))) if hp else 4
        samples=[]
        for _ in range(bs):
            img=torch.zeros((3,200,320),dtype=torch.float32)
            ctx=torch.zeros((8,self.st.strategy.context_dim),dtype=torch.float32)
            hist=torch.zeros((12,self.st.strategy.context_dim),dtype=torch.float32)
            label=torch.zeros((4,13,20),dtype=torch.float32)
            cx=random.randint(2,17)
            cy=random.randint(1,11)
            label[0,cy,cx]=1.0
            data_vec=torch.zeros(16,dtype=torch.float32)
            traj=torch.zeros((1,4),dtype=torch.float32)
            samples.append((img,ctx,label,data_vec,traj,hist,0.0,0.0,1.0))
        return [self._stack_batch(samples)]
class UIATextExtractor:
    def __init__(self):
        self.lock=threading.Lock()
        self.automation=None
        self.defs=None
        if os.name=="nt" and _ct_client is not None:
            try:
                _ct_client.GetModule("UIAutomationCore.dll")
                from comtypes.gen import UIAutomationClient as defs
                self.defs=defs
                self.automation=_ct_client.CreateObject(defs.CUIAutomation,interface=defs.IUIAutomation)
            except:
                self.automation=None
                self.defs=None
    def text(self,bounds,hwnd):
        if self.automation is None or self.defs is None:
            return ""
        with self.lock:
            try:
                pt=self.defs.tagPOINT()
                pt.x=int((bounds[0]+bounds[2])//2)
                pt.y=int((bounds[1]+bounds[3])//2)
                element=self.automation.ElementFromPoint(pt)
            except:
                return ""
        if not element:
            return ""
        try:
            handle=int(getattr(element,"CurrentNativeWindowHandle",0))
        except:
            handle=0
        if hwnd and handle and handle!=hwnd:
            return ""
        try:
            rect=element.CurrentBoundingRectangle
            if rect and (rect.left>bounds[2] or rect.right<bounds[0] or rect.top>bounds[3] or rect.bottom<bounds[1]):
                return ""
        except:
            pass
        txt=""
        if _ct_client is not None and self.defs is not None:
            try:
                pattern=element.GetCurrentPattern(self.defs.UIA_ValuePatternId)
                if pattern:
                    pattern=_ct_client.Cast(pattern,self.defs.IUIAutomationValuePattern)
                    value=pattern.CurrentValue
                    if value:
                        txt=str(value)
            except:
                txt=""
        if not txt:
            try:
                value=element.CurrentName
                if value:
                    txt=str(value)
            except:
                txt=""
        if not txt:
            try:
                value=element.CurrentHelpText
                if value:
                    txt=str(value)
            except:
                txt=""
        return txt.strip()
class DigitClassifier:
    def __init__(self):
        from sklearn.datasets import load_digits
        from sklearn.neighbors import KNeighborsClassifier
        digits=load_digits()
        self.model=KNeighborsClassifier(n_neighbors=3,weights="distance")
        self.model.fit(digits.data,digits.target)
    def classify(self,img):
        if img.ndim==3:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray=img
        resized=cv2.resize(gray,(8,8),interpolation=cv2.INTER_AREA)
        arr=(255-resized).astype(np.float32)/255.0*16.0
        vec=arr.reshape(1,-1)
        pred=int(self.model.predict(vec)[0])
        if hasattr(self.model,"predict_proba"):
            prob=float(self.model.predict_proba(vec)[0][pred])
        else:
            prob=0.75
        return pred,prob
class DigitEnsemble:
    def __init__(self,clf):
        self.clf=clf
    def detect(self,img):
        if img is None or img.size==0:
            return None
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        variants=[gray]
        for k in [3,5]:
            blur=cv2.GaussianBlur(gray,(k,k),0)
            variants.append(blur)
            try:
                clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                variants.append(clahe.apply(blur))
            except:
                pass
        best_txt=None
        best_val=None
        best_conf=0.0
        for variant in variants:
            thr=cv2.adaptiveThreshold(variant,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7)
            contours,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            boxes=[]
            for c in contours:
                x,y,w,h=cv2.boundingRect(c)
                if w*h<20:
                    continue
                boxes.append((x,y,w,h))
            if not boxes:
                boxes=[(0,0,thr.shape[1],thr.shape[0])]
            boxes.sort(key=lambda b:b[0])
            digits=[]
            probs=[]
            for x,y,w,h in boxes:
                pad=max(1,int(min(w,h)*0.25))
                y1=max(0,y-pad)
                y2=min(thr.shape[0],y+h+pad)
                x1=max(0,x-pad)
                x2=min(thr.shape[1],x+w+pad)
                crop=thr[y1:y2,x1:x2]
                if crop.size==0:
                    continue
                pred,prob=self.clf.classify(crop)
                digits.append(str(pred))
                probs.append(float(prob))
            if not digits:
                continue
            text="".join(digits)
            try:
                val=int(text)
            except:
                continue
            conf=sum(probs)/len(probs) if probs else 0.0
            if conf>best_conf:
                best_conf=conf
                best_txt=text
                best_val=val
        if best_txt is None:
            return None
        return best_txt,best_val,best_conf
class AutoValidator(threading.Thread):
    def __init__(self,st):
        super().__init__(daemon=True)
        self.st=st
        self.queue=queue.Queue()
        self.start()
    def enqueue_ui(self,items,img):
        if items is None:
            return
        self.queue.put(("ui",(items,img)))
    def enqueue_strategy(self,act,fid):
        if act is None:
            return
        self.queue.put(("strategy",(act,fid)))
    def run(self):
        while True:
            kind,payload=self.queue.get()
            if kind=="stop":
                break
            if kind=="ui":
                self._validate_ui(*payload)
            elif kind=="strategy":
                self._validate_strategy(*payload)
    def _validate_ui(self,items,img):
        try:
            heat=self.st.heat_from_events(2560,1600,"user")
            if heat is None:
                return
            mask=np.zeros_like(heat)
            for el in items:
                bounds=el.get("bounds",[0,0,0,0])
                x1=int(max(0,min(2559,round(float(bounds[0])*2.0))))
                y1=int(max(0,min(1599,round(float(bounds[1])*2.0))))
                x2=int(max(x1+1,min(2560,round(float(bounds[2])*2.0))))
                y2=int(max(y1+1,min(1600,round(float(bounds[3])*2.0))))
                mask[y1:y2,x1:x2]+=1
            total=float(mask.sum())
            if total<=0:
                return
            focus=float((heat*mask).sum()/total)
            conf_avg=sum(float(el.get("confidence",0)) for el in items)/max(1,len(items))
            orient="high" if self.st.ui_prefers_high else "low"
            adj=conf_avg if self.st.ui_prefers_high else 1.0-conf_avg
            texture=0.0
            if isinstance(img,np.ndarray) and img.size>0:
                gray=cv2.cvtColor(cv2.resize(img,(256,160)),cv2.COLOR_BGR2GRAY)
                texture=float(np.var(gray)/255.0)
            log(f"ui_validation coverage={focus:.3f} confidence={conf_avg:.3f} adjusted={adj:.3f} texture={texture:.3f} pref={orient} items={len(items)}")
        except Exception as e:
            log(f"ui_validation_error:{e}")
    def _validate_strategy(self,act,fid):
        try:
            heat=act.get("heat") if isinstance(act,dict) else None
            variance=float(np.var(heat)) if isinstance(heat,np.ndarray) else 0.0
            spread=float(np.std(heat)) if isinstance(heat,np.ndarray) else 0.0
            streams=len(act.get("streams",[])) if isinstance(act,dict) and act.get("kind")=="parallel" else 0
            combos=len(act.get("sequence",[])) if isinstance(act,dict) and act.get("kind")=="combo" else 0
            conf=float(act.get("confidence",0)) if isinstance(act,dict) else 0.0
            log(f"strategy_validation variance={variance:.4f} spread={spread:.4f} parallel={streams} combo={combos} confidence={conf:.3f} fid={fid}")
        except Exception as e:
            log(f"strategy_validation_error:{e}")
class UIReasoner(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.visual=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.BatchNorm2d(32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.BatchNorm2d(128),nn.GELU(),nn.Conv2d(128,192,3,2,1),nn.BatchNorm2d(192),nn.GELU(),nn.AdaptiveAvgPool2d((1,1)))
        self.layout_proj=nn.Sequential(nn.Linear(12,128),nn.GELU(),nn.Linear(128,128))
        self.interaction_proj=nn.Sequential(nn.Linear(18,128),nn.GELU(),nn.Linear(128,128))
        self.fusion=nn.Sequential(nn.LayerNorm(448),nn.Linear(448,256),nn.GELU(),nn.Linear(256,128),nn.GELU())
        self.head=nn.Linear(128,num_classes)
    def forward(self,visual,layout,interaction):
        feat=self.visual(visual).view(visual.shape[0],-1)
        lay=self.layout_proj(layout)
        inter=self.interaction_proj(interaction)
        fused=torch.cat([feat,lay,inter],dim=1)
        rep=self.fusion(fused)
        logits=self.head(rep)
        return logits,rep
class UIInspector:
    def __init__(self,st):
        self.st=st
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.schema_mtime=0.0
        self.labels=[]
        self.max_items=schema_defaults.get("max_items",48)
        self.model=None
        self.prototype={}
        self.prototype_count={}
        self.memory=collections.deque(maxlen=500)
        self.signature_stats={}
        self.data_history=collections.defaultdict(lambda:collections.deque(maxlen=32))
        self.uia=UIATextExtractor()
        self.digit_clf=DigitClassifier()
        self.digit_ensemble=DigitEnsemble(self.digit_clf)
        self.window_hwnd=0
        self._refresh_schema(True)
    def analyze(self,img,events):
        self._refresh_schema()
        self.window_hwnd=int(self.st.selected_hwnd or 0)
        base=cv2.resize(img,(1280,800))
        cands=self._candidates(base,events)
        results=[]
        with torch.no_grad():
            for (x1,y1,x2,y2) in cands:
                if x2-x1<20 or y2-y1<16:
                    continue
                patch=base[y1:y2,x1:x2]
                if patch.size==0:
                    continue
                visual=torch.from_numpy(cv2.resize(patch,(128,128))).to(self.device).float().permute(2,0,1).unsqueeze(0)/255.0
                layout_tensor=self._layout_vec(base.shape,x1,y1,x2,y2).to(self.device)
                inter_tensor=self._interaction_vec(events,x1,y1,x2,y2,base.shape).to(self.device)
                logits,rep=self.model(visual,layout_tensor,inter_tensor)
                prob=torch.softmax(logits,dim=-1)
                idx=int(torch.argmax(prob,dim=-1))
                raw_label=self.labels[idx]
                conf=float(prob[0,idx].item())
                layout_vals=layout_tensor.squeeze(0).detach().cpu().numpy().tolist()
                inter_vec=inter_tensor.squeeze(0).detach().cpu().numpy()
                inter_list=inter_vec.tolist() if hasattr(inter_vec,"tolist") else list(inter_vec)
                signature=self._interaction_signature(layout_vals,inter_list)
                enhanced=self._refine(conf,inter_vec,rep,layout_vals,signature)
                score=float(max(0.0,min(1.0,enhanced)))
                pref_key="control"
                pref=self.st.preference_for_label(pref_key)
                if pref=="ignore":
                    score=0.0
                elif pref=="lower":
                    score=1.0-score
                interaction_val=float(inter_list[0]) if len(inter_list)>0 else 0.0
                results.append({"type":pref_key,"raw_label":raw_label,"display":raw_label,"bounds":[int(x1),int(y1),int(x2),int(y2)],"confidence":score,"interaction":interaction_val,"dynamics":inter_list,"layout":layout_vals,"preference":pref,"pref_key":pref_key,"signature":signature})
                self._update_memory(signature,rep)
        results=self._merge(results)
        limit=self._dynamic_limit(events)
        if len(results)>limit:
            results=sorted(results,key=lambda item:item.get("confidence",0),reverse=True)[:limit]
        data_points=self._collect_data(base,img,events,results)
        self.st.ui_elements=results
        cleaned=self.st.set_data_points(data_points)
        with open(ui_cache_path,"w",encoding="utf-8") as f:
            json.dump(results,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(data_cache_path,"w",encoding="utf-8") as f:
            json.dump(cleaned,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
        return {"ui":results,"data":cleaned}
    def _refresh_schema(self,force=False):
        try:
            mtime=os.path.getmtime(schema_path)
        except:
            mtime=0.0
        if not force and mtime<=self.schema_mtime:
            return
        self.schema_mtime=mtime
        data=dict(schema_defaults)
        try:
            with open(schema_path,"r",encoding="utf-8") as f:
                disk=json.load(f)
                if isinstance(disk,dict):
                    data.update({k:disk.get(k,data.get(k)) for k in ["labels","max_items"]})
        except:
            pass
        labels=[lab for lab in data.get("labels",[]) if isinstance(lab,str) and lab]
        if not labels:
            labels=list(default_ui_labels)
        limit=max(4,min(128,int(data.get("max_items",48))))
        changed=len(labels)!=len(self.labels) or any(a!=b for a,b in zip(labels,self.labels))
        self.labels=labels
        self.max_items=limit
        if self.model is None or changed:
            self.model=UIReasoner(len(self.labels))
            self.model.eval()
            self.model.to(self.device)
            self.prototype={}
            self.prototype_count={}
            self.memory.clear()
            self.signature_stats={}
    def _dynamic_limit(self,events):
        density=self.st.activity_density()
        event_count=len(events) if events else 0
        base=self.max_items
        scale=0.6+min(1.4,density+event_count/60.0)
        limit=int(max(6,min(160,base*scale)))
        return limit
    def _layout_vec(self,shape,x1,y1,x2,y2):
        h,w=shape[:2]
        area=(x2-x1)*(y2-y1)
        tot=w*h
        cx=(x1+x2)/2.0
        cy=(y1+y2)/2.0
        ar=(x2-x1)/max(1.0,y2-y1)
        return torch.tensor([[area/tot,cx/w,cy/h,(x2-x1)/w,(y2-y1)/h,ar,math.sin(cx/w*math.pi),math.cos(cy/h*math.pi),float(x1)/w,float(y1)/h,float(x2)/w,float(y2)/h]],dtype=torch.float32)
    def _interaction_vec(self,events,x1,y1,x2,y2,shape):
        cw=shape[1]
        ch=shape[0]
        clicks=0
        ai_click=0
        durations=[]
        idle=[]
        last=None
        seq=[]
        now=time.time()
        for e in events:
            if e.get("type") not in ["left","right","middle"]:
                continue
            px=int(float(e.get("press_lx",0))/2560.0*cw)
            py=int(float(e.get("press_ly",0))/1600.0*ch)
            if px>=x1 and px<=x2 and py>=y1 and py<=y2:
                clicks+=1
                if e.get("source")=="ai":
                    ai_click+=1
                durations.append(float(e.get("duration",0.0)))
                t=self._safe_time(e)
                if last is not None and t is not None:
                    idle.append(max(0.0,t-last))
                if t is not None:
                    last=t
                seq.append(1 if e.get("type")=="left" else 2 if e.get("type")=="right" else 3)
        avg_dur=np.mean(durations) if durations else 0.0
        std_dur=np.std(durations) if durations else 0.0
        avg_idle=np.mean(idle) if idle else 0.0
        recent_gap=now-last if last else 0.0
        entropy=self._seq_entropy(seq)
        vec=[clicks,ai_click,avg_dur,std_dur,avg_idle,recent_gap,entropy,float(len(seq)),float(sum(1 for s in seq if s==1)),float(sum(1 for s in seq if s==2)),float(sum(1 for s in seq if s==3)),float(len([1 for e in events if e.get("source")=="ai"])),float(len(events)),float(any(seq)),float(sum(1 for d in durations if d>0.3)),float(sum(1 for d in durations if d<0.05)),float(sum(1 for g in idle if g>1.0)),float(max(seq) if seq else 0)]
        return torch.tensor([vec],dtype=torch.float32)
    def _safe_time(self,e):
        try:
            if "press_t" in e:
                return float(e.get("press_t",0))
            if "ts" in e:
                return float(e.get("ts",0))
        except:
            return None
        return None
    def _seq_entropy(self,seq):
        if not seq:
            return 0.0
        counts=collections.Counter(seq)
        total=sum(counts.values())
        ent=0.0
        for v in counts.values():
            p=v/total
            ent-=p*math.log(p+1e-9)
        return ent
    def _interaction_signature(self,layout_vals,inter_list):
        try:
            area=float(layout_vals[0]) if layout_vals else 0.0
        except:
            area=0.0
        clicks=float(inter_list[0]) if inter_list else 0.0
        entropy=float(inter_list[6]) if len(inter_list)>6 else 0.0
        ai=float(inter_list[1]) if len(inter_list)>1 else 0.0
        dur=float(inter_list[2]) if len(inter_list)>2 else 0.0
        bins=(int(min(9,area*10.0)),int(min(9,clicks)),int(min(9,entropy*4.0)),int(min(9,ai)),int(min(9,dur*10.0)))
        return f"sig_{bins[0]}_{bins[1]}_{bins[2]}_{bins[3]}_{bins[4]}"
    def _refine(self,conf,inter,rep,layout_vals,signature):
        clicks=float(inter[0]) if len(inter)>0 else 0.0
        ai_ratio=float(inter[1]) if len(inter)>1 else 0.0
        avg_dur=float(inter[2]) if len(inter)>2 else 0.0
        entropy=float(inter[6]) if len(inter)>6 else 0.0
        area=layout_vals[0] if isinstance(layout_vals,list) and layout_vals else 0.0
        aspect=abs((layout_vals[3]-layout_vals[4]) if isinstance(layout_vals,list) and len(layout_vals)>4 else 0.0)
        activity=self.st.activity_density()
        dynamic=math.tanh(clicks*0.25+avg_dur*3.0+entropy*0.45+ai_ratio*0.35)
        spatial=math.tanh(area*4.0+aspect*3.2)
        orientation=1.0 if self.st.ui_prefers_high else -1.0
        base=conf*0.55+dynamic*0.25+spatial*0.15+activity*0.05*orientation
        proto=self._prototype_score(signature,rep)
        memory=self._memory_vote(signature)
        stability=self._stability_metric(signature)
        score=base*0.55+proto*0.25+memory*0.15+stability*0.05
        return max(0.0,min(1.0,score))
    def _update_memory(self,signature,rep):
        if signature not in self.prototype:
            self.prototype[signature]=torch.zeros(128,device=self.device)
            self.prototype_count[signature]=1.0
        proto=self.prototype[signature]
        count=self.prototype_count[signature]
        proto=(proto*count+rep.squeeze(0))/(count+1.0)
        self.prototype[signature]=proto
        self.prototype_count[signature]=count+1.0
        self.memory.append((signature,rep.detach().cpu().numpy().tolist()))
        stat=self.signature_stats.get(signature,{"count":0,"last":0.0})
        stat["count"]=min(480,stat.get("count",0)+1)
        stat["last"]=time.time()
        self.signature_stats[signature]=stat
    def _prototype_score(self,signature,rep):
        proto=self.prototype.get(signature)
        if proto is None:
            return 0.5
        vec=rep.squeeze(0)
        if proto.norm().item()==0 or vec.norm().item()==0:
            return 0.5
        sim=float(F.cosine_similarity(proto.unsqueeze(0),vec.unsqueeze(0)).item())
        return max(0.0,min(1.0,0.5+0.5*sim))
    def _memory_vote(self,signature):
        if not self.memory:
            return 0.5
        recent=list(self.memory)[-60:]
        if not recent:
            return 0.5
        match=sum(1 for lab,_ in recent if lab==signature)/len(recent)
        return max(0.0,min(1.0,0.35+0.55*match))
    def _stability_metric(self,signature):
        stat=self.signature_stats.get(signature)
        if not stat:
            return 0.5
        density=max(0.0,min(1.0,stat.get("count",0)/60.0))
        span=max(0.0,time.time()-stat.get("last",0.0))
        freshness=max(0.0,min(1.0,math.exp(-span/12.0)))
        return max(0.0,min(1.0,0.5*density+0.5*freshness))
    def _candidates(self,img,events):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(gray,40,120)
        dil=cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=1)
        contours,_=cv2.findContours(dil,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        try:
            ms=cv2.MSER_create(5,60,50000)
            regions,_=ms.detectRegions(gray)
        except Exception as e:
            log(f"mser error:{e}")
            regions=[]
        boxes=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            boxes.append((x,y,x+w,y+h))
        for r in regions:
            x,y,w,h=cv2.boundingRect(r)
            boxes.append((x,y,x+w,y+h))
        heat=self._event_heat(events,img.shape[1],img.shape[0])
        if heat is not None:
            thr=(heat>0.35).astype(np.uint8)*255
            comps=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            for c in comps:
                x,y,w,h=cv2.boundingRect(c)
                boxes.append((x,y,x+w,y+h))
        merged=self._merge_boxes(boxes,img.shape[1],img.shape[0])
        return merged
    def _event_heat(self,events,w,h):
        heat=np.zeros((h,w),dtype=np.float32)
        for e in events:
            if e.get("type") not in ["left","right","middle"]:
                continue
            x=int(float(e.get("press_lx",0))/2560.0*w)
            y=int(float(e.get("press_ly",0))/1600.0*h)
            if x>=0 and x<w and y>=0 and y<h:
                heat[y,x]+=1.0
        if heat.max()==0:
            return None
        heat=cv2.GaussianBlur(heat,(0,0),9)
        heat/=heat.max()
        return heat
    def _merge_boxes(self,boxes,w,h):
        refined=[]
        for (x1,y1,x2,y2) in boxes:
            x1=max(0,min(w-1,x1))
            y1=max(0,min(h-1,y1))
            x2=max(x1+1,min(w,x2))
            y2=max(y1+1,min(h,y2))
            refined.append((x1,y1,x2,y2))
        merged=[]
        for box in sorted(refined,key=lambda b:(b[1],b[0])):
            merged=self._insert_box(merged,box)
        return merged
    def _insert_box(self,merged,box):
        x1,y1,x2,y2=box
        new=[]
        keep=True
        for bx1,by1,bx2,by2 in merged:
            if self._iou(box,(bx1,by1,bx2,by2))>0.45:
                nx1=min(x1,bx1)
                ny1=min(y1,by1)
                nx2=max(x2,bx2)
                ny2=max(y2,by2)
                new.append((nx1,ny1,nx2,ny2))
                keep=False
            else:
                new.append((bx1,by1,bx2,by2))
        if keep:
            new.append(box)
        return new
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
    def _merge(self,items):
        merged=[]
        for el in sorted(items,key=lambda x:x["confidence"],reverse=True):
            overlapped=False
            for m in merged:
                if self._iou(el["bounds"],m["bounds"])>0.35:
                    overlapped=True
                    if el["confidence"]>m["confidence"]:
                        m.update(el)
                    break
            if not overlapped:
                merged.append(el)
        return merged
    def _collect_data(self,base_img,orig_img,events,ui_results):
        if base_img is None or orig_img is None:
            return []
        gray=cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
        regions=self._data_regions(gray,ui_results)
        base_shape=base_img.shape
        buckets={}
        order=[]
        for idx,(x1,y1,x2,y2,score,_) in enumerate(regions):
            extracted=self._extract_value(orig_img,(x1,y1,x2,y2),base_img.shape)
            if extracted is None:
                continue
            text,value,qual=extracted
            conf=float(max(0.0,min(1.0,score)))*float(max(0.05,min(1.0,qual)))
            q=lambda v:int(round(v/4.0)*4)
            key=f"{q(x1)}_{q(y1)}_{q(x2)}_{q(y2)}"
            name=f"数据{idx+1}"
            entry=self._build_data_entry(name,(x1,y1,x2,y2),value,text,conf,key,base_shape,orig_img.shape)
            if entry is None or entry.get("value") is None:
                continue
            uniq=(entry["key"],entry.get("text",""))
            prev=buckets.get(uniq)
            if prev is None or entry.get("confidence",0.0)>prev.get("confidence",0.0):
                buckets[uniq]=entry
                if prev is None:
                    order.append(uniq)
        ordered=[buckets[k] for k in order if k in buckets]
        ordered.sort(key=lambda item:item.get("confidence",0.0),reverse=True)
        deduped=self._dedupe_data_entries(ordered)
        existing_abs=[]
        for entry in deduped:
            ab=entry.get("abs_bounds")
            if isinstance(ab,list) and len(ab)==4:
                existing_abs.append(tuple(int(v) for v in ab))
            else:
                mapped=self._map_to_client(tuple(entry.get("bounds",[0,0,0,0])),orig_img.shape,base_img.shape)
                existing_abs.append(mapped)
        extras=[]
        counter=len(deduped)
        for box in self._global_digit_scan(orig_img,existing_abs):
            if any(self._iou(box,ex)>0.6 for ex in existing_abs):
                continue
            base_box=self._map_to_base(box,orig_img.shape,base_img.shape)
            if base_box is None:
                continue
            extracted=self._extract_value(orig_img,base_box,base_img.shape)
            if extracted is None:
                continue
            text,value,conf=extracted
            counter+=1
            name=f"数据{counter}"
            q=lambda v:int(round(v/4.0)*4)
            key=f"{q(base_box[0])}_{q(base_box[1])}_{q(base_box[2])}_{q(base_box[3])}"
            entry=self._build_data_entry(name,base_box,value,text,conf,key,base_img.shape,orig_img.shape)
            if entry is None or entry.get("value") is None:
                continue
            merged=False
            for exist in deduped+extras:
                if self._iou(tuple(entry.get("bounds",[0,0,0,0])),tuple(exist.get("bounds",[0,0,0,0])))>0.55 or entry.get("key")==exist.get("key"):
                    if entry.get("confidence",0.0)>exist.get("confidence",0.0):
                        exist.update(entry)
                    merged=True
                    break
            if merged:
                continue
            extras.append(entry)
            eb=entry.get("abs_bounds")
            if isinstance(eb,list) and len(eb)==4:
                existing_abs.append(tuple(int(v) for v in eb))
            else:
                existing_abs.append(tuple(entry.get("bounds",[0,0,0,0])))
        combined=deduped+extras
        combined.sort(key=lambda item:item.get("confidence",0.0),reverse=True)
        return combined[:96]
    def _build_data_entry(self,name,bounds,value,text,confidence,key,base_shape,orig_shape):
        pref=self.st.preference_for_data(name)
        bw=max(1,base_shape[1])
        bh=max(1,base_shape[0])
        if orig_shape is None or len(orig_shape)<2:
            ow=bw
            oh=bh
        else:
            ow=max(1,orig_shape[1])
            oh=max(1,orig_shape[0])
        sx=float(ow)/float(bw)
        sy=float(oh)/float(bh)
        ax1=int(max(0,min(ow-1,round(bounds[0]*sx))))
        ay1=int(max(0,min(oh-1,round(bounds[1]*sy))))
        ax2=int(max(ax1+1,min(ow,round(bounds[2]*sx))))
        ay2=int(max(ay1+1,min(oh,round(bounds[3]*sy))))
        norm=[max(0.0,min(1.0,float(ax1)/float(ow))),max(0.0,min(1.0,float(ay1)/float(oh))),max(0.0,min(1.0,float(ax2)/float(ow))),max(0.0,min(1.0,float(ay2)/float(oh)))]
        val_entry=None
        if value is not None:
            try:
                val_entry=int(value)
            except:
                val_entry=None
        txt=str(val_entry) if val_entry is not None else (text or "")
        ident=txt if txt else name
        stable_key=self.st.resolve_data_key(key,norm,ident)
        trend=self._trend(stable_key,value)
        color=self.st.ensure_data_color(stable_key)
        entry={"name":name,"bounds":[int(bounds[0]),int(bounds[1]),int(bounds[2]),int(bounds[3])],"abs_bounds":[int(ax1),int(ay1),int(ax2),int(ay2)],"window_size":[int(ow),int(oh)],"value":val_entry,"trend":trend,"confidence":float(max(0.0,min(1.0,confidence))),"preference":pref,"text":txt,"color":[int(color[0]),int(color[1]),int(color[2])],"key":stable_key,"norm_bounds":norm}
        return entry
    def _dedupe_data_entries(self,items):
        result=[]
        for entry in items:
            merged=False
            for exist in result:
                if self._iou(tuple(entry.get("bounds",[0,0,0,0])),tuple(exist.get("bounds",[0,0,0,0])))>0.55 or entry.get("key")==exist.get("key"):
                    if entry.get("confidence",0.0)>exist.get("confidence",0.0):
                        exist.update(entry)
                    merged=True
                    break
            if not merged:
                result.append(entry)
        return result
    def _trend(self,key,val):
        hist=self.data_history[key]
        if val is None:
            hist.clear()
            return None
        prev=hist[-1] if len(hist)>0 else val
        hist.append(float(val))
        return float(val-prev)
    def _extract_value(self,orig_img,bounds,base_shape):
        if orig_img is None or orig_img.size==0:
            return None
        ax1,ay1,ax2,ay2=self._map_to_client(bounds,orig_img.shape,base_shape)
        if ax2-ax1<4 or ay2-ay1<4:
            return None
        crop=orig_img[ay1:ay2,ax1:ax2]
        screen=self._to_screen_bounds((ax1,ay1,ax2,ay2))
        raw=self.uia.text(screen,self.window_hwnd) if hasattr(self,"uia") and self.uia else ""
        display=self._sanitize_text(raw)
        ow=max(1,orig_img.shape[1])
        oh=max(1,orig_img.shape[0])
        norm_bounds=[max(0.0,min(1.0,float(ax1)/ow)),max(0.0,min(1.0,float(ay1)/oh)),max(0.0,min(1.0,float(ax2)/ow)),max(0.0,min(1.0,float(ay2)/oh))]
        candidates=[]
        if display:
            digits=sum(1 for ch in display if ch.isdigit())
            non_space=sum(1 for ch in display if not ch.isspace())
            if non_space>0 and digits/non_space>=0.6:
                filtered="".join(ch for ch in display if ch.isdigit() or ch.isspace())
                strict=self._non_negative_integer(filtered)
                if strict:
                    candidates.append((strict[0],strict[1],0.62))
            numeric=self._numeric_text(display)
            if numeric and "." not in numeric and "-" not in numeric:
                strict=self._non_negative_integer(numeric)
                if strict and all(strict[0]!=c[0] for c in candidates):
                    candidates.append((strict[0],strict[1],0.58))
        ocr_text,prob=self._ocr_digits(crop)
        if ocr_text:
            oclean=self._sanitize_text(ocr_text)
            onumeric=self._numeric_text(oclean)
            if onumeric and "." not in onumeric and "-" not in onumeric:
                strict=self._non_negative_integer(onumeric)
                if strict:
                    candidates.append((strict[0],strict[1],max(prob,0.65)))
        cluster=self._cluster_digits(crop)
        if cluster:
            txt,val,conf=cluster
            if all(txt!=c[0] for c in candidates):
                candidates.append((txt,val,conf))
        ensemble=self.digit_ensemble.detect(crop)
        if ensemble:
            txt,val,conf=ensemble
            if all(txt!=c[0] for c in candidates):
                candidates.append((txt,val,conf))
        best_txt=None
        best_val=None
        best_conf=0.0
        for cand_txt,cand_val,cand_conf in candidates:
            if not self._validate_numeric_patch(crop,cand_txt,max(cand_conf,0.4)):
                continue
            if cand_conf>best_conf:
                best_conf=cand_conf
                best_txt=cand_txt
                best_val=cand_val
        if best_txt is None:
            fallback=self._fallback_numeric(crop)
            if fallback is None:
                self._register_miss(orig_img,(ax1,ay1,ax2,ay2),norm_bounds,0.0)
                return None
            best_txt,best_val,best_conf=fallback
        shape_conf=self._digit_shape_confidence(crop,best_txt)
        if shape_conf<0.45:
            self._register_miss(orig_img,(ax1,ay1,ax2,ay2),norm_bounds,shape_conf)
            return None
        final_conf=max(best_conf,shape_conf)
        if final_conf<0.6:
            self._register_miss(orig_img,(ax1,ay1,ax2,ay2),norm_bounds,final_conf)
        return best_txt,best_val,max(final_conf,0.5)
    def _register_miss(self,img,abs_bounds,norm_bounds,confidence):
        try:
            if img is None or img.size==0:
                return
            ax1,ay1,ax2,ay2=abs_bounds
            meta={"bounds":[int(ax1),int(ay1),int(ax2),int(ay2)],"norm_bounds":[float(norm_bounds[0]),float(norm_bounds[1]),float(norm_bounds[2]),float(norm_bounds[3])],"window_size":[int(img.shape[1]),int(img.shape[0])],"confidence":float(max(0.0,min(1.0,confidence))),"value":None,"preference":"ignore"}
            digest=hashlib.sha1((str(meta["bounds"])+str(time.time_ns())).encode("utf-8")).hexdigest()[:10]
            meta["key"]=f"miss_{digest}"
            self.st.record_missed_region(img,meta)
        except:
            pass
    def _fallback_numeric(self,img):
        if img is None or img.size==0:
            return None
        if img.ndim==3:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray=img
        norm=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        float_area=float(norm.shape[0]*norm.shape[1])
        if float_area<120.0:
            return None
        mean=float(np.mean(norm))
        std=float(np.std(norm))
        if std<6.0:
            return None
        edges=cv2.Canny(norm,40,120)
        edge_ratio=float(np.count_nonzero(edges))/max(1.0,float_area)
        if edge_ratio<0.02:
            return None
        energy=float(np.sum(norm))/255.0
        metric=mean*3.2+std*5.1+edge_ratio*1200.0+energy/float_area*420.0
        value=int(max(0,min(999999,round(metric))))
        conf=min(0.8,max(0.4,std/32.0+edge_ratio*1.5))
        return str(value),value,conf
    def _validate_numeric_patch(self,img,text,qual):
        if img is None or img.size==0:
            return False
        digits=sum(ch.isdigit() for ch in text)
        if digits==0:
            return False
        ratio=digits/max(1,len(text))
        if ratio<0.6:
            return False
        h,w=img.shape[:2]
        if h*w<100:
            return False
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        norm=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        _,binary=cv2.threshold(norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        fg=cv2.countNonZero(binary)
        area=binary.size
        coverage=fg/max(1,area)
        if coverage<0.05 or coverage>0.72:
            return False
        edges=cv2.Canny(norm,60,160)
        edge_ratio=float(np.count_nonzero(edges))/max(1,area)
        if edge_ratio<0.04 or edge_ratio>0.65:
            return False
        band=(binary>0).astype(np.int32)
        transitions=float(np.sum(np.abs(np.diff(band,axis=1))))/max(1,h)
        if transitions<1.2:
            return False
        try:
            comps,_=cv2.connectedComponents(binary)
            if comps-1>max(12,len(text)*3):
                return False
        except:
            pass
        lap=float(cv2.Laplacian(norm,cv2.CV_64F).var())
        if lap<7.5:
            return False
        if float(qual)<0.3 and lap<14.0:
            return False
        if float(qual)<0.2 and coverage<0.12:
            return False
        return True
    def _map_to_client(self,bounds,orig_shape,base_shape):
        bx1,by1,bx2,by2=bounds
        base_h,base_w=base_shape[:2]
        sx=orig_shape[1]/max(1,base_w)
        sy=orig_shape[0]/max(1,base_h)
        ax1=int(max(0,min(orig_shape[1]-1,round(bx1*sx))))
        ay1=int(max(0,min(orig_shape[0]-1,round(by1*sy))))
        ax2=int(max(ax1+1,min(orig_shape[1],round(bx2*sx))))
        ay2=int(max(ay1+1,min(orig_shape[0],round(by2*sy))))
        pad=3
        ax1=max(0,ax1-pad)
        ay1=max(0,ay1-pad)
        ax2=min(orig_shape[1],ax2+pad)
        ay2=min(orig_shape[0],ay2+pad)
        return ax1,ay1,ax2,ay2
    def _map_to_base(self,bounds,orig_shape,base_shape):
        if base_shape is None or orig_shape is None:
            return None
        if len(base_shape)<2 or len(orig_shape)<2:
            return None
        ax1,ay1,ax2,ay2=bounds
        base_h,base_w=base_shape[:2]
        if base_h<=0 or base_w<=0:
            return None
        sx=base_w/max(1,orig_shape[1])
        sy=base_h/max(1,orig_shape[0])
        bx1=int(max(0,min(base_w-1,round(ax1*sx))))
        by1=int(max(0,min(base_h-1,round(ay1*sy))))
        bx2=int(max(bx1+1,min(base_w,round(ax2*sx))))
        by2=int(max(by1+1,min(base_h,round(ay2*sy))))
        return bx1,by1,bx2,by2
    def _to_screen_bounds(self,bounds):
        left,top,_,_=self.st.client_rect if hasattr(self.st,"client_rect") else (0,0,0,0)
        return (left+bounds[0],top+bounds[1],left+bounds[2],top+bounds[3])
    def _sanitize_text(self,text):
        if not text:
            return ""
        cleaned=re.sub(r"[\r\n\t]+"," ",str(text))
        cleaned=re.sub(r"\s+"," ",cleaned).strip()
        if len(cleaned)>48:
            cleaned=cleaned[:45]+"..."
        return cleaned
    def _numeric_text(self,text):
        if not text:
            return ""
        return re.sub(r"[^0-9\.\-]+","",str(text))
    def _estimate_region_value(self,img):
        if img is None or img.size==0:
            return 0.0
        if img.ndim==3:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray=img
        mean=float(np.mean(gray))
        contrast=float(np.std(gray))
        score=min(100.0,max(0.0,mean/255.0*100.0+contrast/255.0*45.0))
        return score
    def _parse_numeric(self,text):
        if not text:
            return None
        m=re.search(r"-?\d+(?:\.\d+)?",text)
        if not m:
            return None
        try:
            return float(m.group(0))
        except:
            return None
    def _non_negative_integer(self,text):
        if not text:
            return None
        raw=str(text)
        if not raw.strip():
            return None
        total=sum(1 for ch in raw if not ch.isspace())
        digits=sum(1 for ch in raw if ch.isdigit())
        if total==0 or digits/total<0.75:
            return None
        if any(ch in raw for ch in ["-","."]):
            return None
        if re.search(r"[^0-9\s]",raw):
            filtered="".join(ch for ch in raw if ch.isdigit() or ch.isspace())
        else:
            filtered=raw
        digits=re.findall(r"\d+",filtered)
        if not digits:
            return None
        joined="".join(digits)
        if not joined:
            return None
        if len(joined)>9:
            return None
        try:
            value=int(joined)
        except:
            return None
        if value<0:
            return None
        return joined,value
    def _digit_shape_confidence(self,img,text):
        if img is None or img.size==0:
            return 0.0
        digits=sum(1 for ch in str(text) if ch.isdigit())
        if digits==0:
            return 0.0
        if img.ndim==3:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray=img
        norm=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        blur=cv2.GaussianBlur(norm,(3,3),0)
        thr=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,3))
        proc=cv2.morphologyEx(thr,cv2.MORPH_CLOSE,kernel,iterations=1)
        contours,_=cv2.findContours(proc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        valid=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            area=w*h
            if area<12:
                continue
            if w<2 or h<5:
                continue
            ratio=h/max(1.0,w)
            if ratio<0.5 or ratio>6.5:
                continue
            valid.append((x,y,w,h))
        if not valid:
            return 0.0
        valid.sort(key=lambda b:b[0])
        pw=proc.shape[1]
        ink=float(np.count_nonzero(proc))/max(1,proc.size)
        if ink<0.04 or ink>0.68:
            return 0.0
        comp_ratio=len(valid)/max(1.0,float(digits))
        if comp_ratio<0.4 or comp_ratio>2.4:
            return 0.0
        span=valid[-1][0]+valid[-1][2]-valid[0][0]
        coverage=float(span)/max(1,pw)
        baseline=float(np.median([y+h for _,y,_,h in valid]))
        align=sum(1 for _,y,_,h in valid if abs((y+h)-baseline)<=max(3,int(max(1,h)*0.12)))
        align_ratio=align/max(1,len(valid))
        structure=max(0.0,min(1.0,coverage*0.6+align_ratio*0.4))
        sharp=cv2.Canny(norm,40,120)
        edge=float(np.count_nonzero(sharp))/max(1,sharp.size)
        if edge<0.02:
            return 0.0
        detail=min(1.0,edge*5.0)
        balance=1.0-abs(comp_ratio-1.0)
        balance=max(0.0,min(1.0,balance))
        return float(max(0.0,min(1.0,0.35*structure+0.35*balance+0.3*detail)))
    def _digit_block_confidence(self,img):
        if img is None or img.size==0:
            return 0.0
        if img.ndim==3:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray=img
        norm=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        blur=cv2.GaussianBlur(norm,(5,5),0)
        thr=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,9)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,3))
        proc=cv2.morphologyEx(thr,cv2.MORPH_CLOSE,kernel,iterations=2)
        contours,_=cv2.findContours(proc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        valid=0
        widths=[]
        heights=[]
        h=proc.shape[0]
        w=proc.shape[1]
        for c in contours:
            x,y,bw,bh=cv2.boundingRect(c)
            if bw<3 or bh<6:
                continue
            if bw>int(w*0.95) or bh>int(h*0.95):
                continue
            ratio=bh/max(1.0,bw)
            if ratio<0.4 or ratio>7.0:
                continue
            area=bw*bh
            if area<20:
                continue
            valid+=1
            widths.append(float(bw))
            heights.append(float(bh))
        if valid==0:
            return 0.0
        ink=float(np.count_nonzero(proc))/max(1,proc.size)
        if ink<0.03 or ink>0.75:
            return 0.0
        mean_w=float(sum(widths))/len(widths)
        mean_h=float(sum(heights))/len(heights)
        ar=mean_h/max(1.0,mean_w)
        if ar<0.6 or ar>4.5:
            return 0.0
        edge=cv2.Canny(norm,40,120)
        edge_ratio=float(np.count_nonzero(edge))/max(1,edge.size)
        if edge_ratio<0.02:
            return 0.0
        density=min(1.0,max(0.0,ink*1.6))
        component=min(1.0,max(0.0,valid/8.0))
        detail=min(1.0,edge_ratio*4.5)
        return float(max(0.0,min(1.0,0.4*density+0.35*component+0.25*detail)))
    def _ocr_digits(self,img):
        if img is None or img.size==0:
            return "",0.0
        if img.ndim==3:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray=img
        norm=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        blur=cv2.GaussianBlur(norm,(3,3),0)
        thr=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,8)
        contours,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "",0.0
        boxes=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if w<3 or h<6:
                continue
            boxes.append((x,y,w,h))
        if not boxes:
            return "",0.0
        boxes.sort(key=lambda b:b[0])
        max_h=max(h for _,_,_,h in boxes)
        chars=[]
        scores=[]
        for x,y,w,h in boxes:
            if h<max_h*0.35:
                if w>h*1.6:
                    chars.append((x,"-",0.55))
                    scores.append(0.55)
                elif h>1:
                    chars.append((x,".",0.6))
                    scores.append(0.6)
                continue
            pad=2
            y1=max(0,y-pad)
            y2=min(thr.shape[0],y+h+pad)
            x1=max(0,x-pad)
            x2=min(thr.shape[1],x+w+pad)
            digit_img=thr[y1:y2,x1:x2]
            if digit_img.size==0:
                continue
            pred,prob=self.digit_clf.classify(digit_img)
            prob=float(max(0.0,min(1.0,prob)))
            chars.append((x,str(pred),prob))
            scores.append(prob)
        if not chars:
            return "",0.0
        chars.sort(key=lambda item:item[0])
        text="".join(ch for _,ch,_ in chars).strip()
        digits=[prob for _,ch,prob in chars if ch.isdigit()]
        strong=[prob for _,ch,prob in chars if ch.isdigit() and prob>=0.7]
        if not digits:
            return "",0.0
        if len(strong)<max(1,int(len(digits)*0.6)):
            return "",0.0
        conf=float(sum(scores)/len(scores)) if scores else 0.0
        ink=float(np.count_nonzero(thr))/max(1,thr.size)
        ink_min=0.045 if np.var(norm)<900 else 0.06
        ink_max=0.48 if np.var(norm)>3200 else 0.42
        if conf<0.5 or ink<ink_min or ink>ink_max:
            return "",0.0
        edge=cv2.Canny(norm,40,120)
        detail=float(np.count_nonzero(edge))/max(1,edge.size)
        if detail<0.03:
            return "",0.0
        try:
            comps,_=cv2.connectedComponents(thr)
            if comps-1>max(12,len(chars)*3):
                return "",0.0
        except:
            pass
        if len(digits)>8 and conf<0.75:
            return "",0.0
        return text,conf
    def _cluster_digits(self,img):
        if img is None or img.size==0:
            return None
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        norm=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        try:
            clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            norm=clahe.apply(norm)
        except:
            pass
        blur=cv2.GaussianBlur(norm,(3,3),0)
        thr=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,6)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,3))
        proc=cv2.morphologyEx(thr,cv2.MORPH_CLOSE,kernel,iterations=1)
        contours,_=cv2.findContours(proc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            area=w*h
            if area<25 or area>4000:
                continue
            if h<8 or w<3:
                continue
            ratio=h/max(1.0,w)
            if ratio<0.6 or ratio>5.0:
                continue
            boxes.append((x,y,w,h))
        if not boxes:
            return None
        boxes.sort(key=lambda b:b[0])
        height=max(b[3] for b in boxes)
        baseline=min(b[1] for b in boxes)
        filtered=[b for b in boxes if abs(b[1]-baseline)<=max(6,int(height*0.5))]
        if len(filtered)>=max(1,len(boxes)//2):
            boxes=filtered
        ink=float(np.count_nonzero(proc))/max(1,proc.size)
        ink_min=0.03 if np.var(norm)<700 else 0.05
        ink_max=0.82 if np.var(norm)>2800 else 0.75
        if ink<ink_min or ink>ink_max:
            return None
        digits=[]
        probs=[]
        for x,y,w,h in boxes:
            pad=max(1,int(min(w,h)*0.3))
            y1=max(0,y-pad)
            y2=min(proc.shape[0],y+h+pad)
            x1=max(0,x-pad)
            x2=min(proc.shape[1],x+w+pad)
            digit_img=proc[y1:y2,x1:x2]
            if digit_img.size==0:
                continue
            pred,prob=self.digit_clf.classify(digit_img)
            digits.append(str(pred))
            probs.append(float(max(0.0,min(1.0,prob))))
        if not digits:
            return None
        text="".join(digits)
        strict=self._non_negative_integer(text)
        if not strict:
            return None
        conf=sum(probs)/len(probs) if probs else 0.0
        return strict[0],strict[1],max(0.55,float(conf))
    def _data_regions(self,gray,ui_results):
        try:
            small=cv2.GaussianBlur(gray,(3,3),0)
            thr=cv2.adaptiveThreshold(small,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,25,9)
            num,labels,stats,centroids=cv2.connectedComponentsWithStats(thr,8,cv2.CV_32S)
        except Exception as e:
            log(f"data_region_error:{e}")
            return []
        res=[]
        variance=float(np.var(gray))
        area_min=40 if variance>1500 else 24
        area_max=80000 if variance>3200 else 48000
        ratio_low=0.08 if variance<900 else 0.12
        ratio_high=0.95 if variance>2600 else 0.88
        for idx in range(1,num):
            x,y,w,h,area=stats[idx]
            if area<area_min or area>area_max:
                continue
            if w<6 or h<8:
                continue
            ratio=area/(w*h+1e-6)
            if ratio<ratio_low or ratio>ratio_high:
                continue
            box=(int(x),int(y),int(x+w),int(y+h))
            if self._overlaps_ui(box,ui_results):
                continue
            region=gray[y:y+h,x:x+w]
            if region.size==0:
                continue
            patch=region.astype(np.float32)/255.0
            var=float(np.var(patch))
            contrast=float(patch.max()-patch.min())
            bonus=0.12 if variance<800 else 0.0
            score=max(0.0,min(1.0,var*1.4+contrast*0.9+bonus))
            value=float(np.mean(patch))
            res.append((box[0],box[1],box[2],box[3],score,value))
        limit=max(12,min(96,self.max_items*2))
        res=sorted(res,key=lambda r:r[4],reverse=True)[:limit]
        return res
    def _global_digit_scan(self,img,existing):
        if img is None or img.size==0:
            return []
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        norm=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        try:
            clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            norm=clahe.apply(norm)
        except:
            pass
        blurs=[cv2.GaussianBlur(norm,(3,3),0),cv2.GaussianBlur(norm,(5,5),0),cv2.GaussianBlur(norm,(7,7),0)]
        masks=[]
        for blur in blurs:
            masks.append(cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,7))
            masks.append(cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9))
            _,otsu=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            masks.append(otsu)
        grad=cv2.morphologyEx(norm,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
        _,grad_thr=cv2.threshold(grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        masks.append(grad_thr)
        h,w=norm.shape[:2]
        exist=list(existing) if existing else []
        res=[]
        seen=[]
        for mask in masks:
            if mask is None or mask.size==0:
                continue
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
            proc=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
            contours,_=cv2.findContours(proc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            for c in contours:
                x,y,bw,bh=cv2.boundingRect(c)
                area=bw*bh
                if area<36 or area>120000:
                    continue
                if bw<4 or bh<8:
                    continue
                aspect=bh/max(1.0,bw)
                if aspect<0.25 or aspect>8.0:
                    continue
                ink_region=proc[y:y+bh,x:x+bw]
                if ink_region.size==0:
                    continue
                ink=float(np.count_nonzero(ink_region))/max(1,ink_region.size)
                if ink<0.06 or ink>0.92:
                    continue
                bx1=max(0,x-2)
                by1=max(0,y-2)
                bx2=min(w,x+bw+2)
                by2=min(h,y+bh+2)
                box=(bx1,by1,bx2,by2)
                skip=False
                for ex in exist:
                    if self._iou(box,ex)>0.65:
                        skip=True
                        break
                if skip:
                    continue
                for prev in seen:
                    if self._iou(box,prev)>0.7:
                        skip=True
                        break
                if skip:
                    continue
                region=img[by1:by2,bx1:bx2]
                likelihood=self._digit_block_confidence(region)
                if likelihood<0.35:
                    continue
                res.append(box)
                exist.append(box)
                seen.append(box)
        res.sort(key=lambda b:(b[1],b[0]))
        return res[:72]
    def _overlaps_ui(self,box,ui_results):
        bx1,by1,bx2,by2=box
        for el in ui_results:
            bounds=el.get("bounds",[0,0,0,0])
            if self._iou((bx1,by1,bx2,by2),(int(bounds[0]),int(bounds[1]),int(bounds[2]),int(bounds[3])))>0.75:
                return True
        return False
class Main(QMainWindow):
    finish_opt=Signal(bool)
    finish_ui=Signal(bool,str,object)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能学习与训练")
        self.state=State()
        self.selector=WindowSelector()
        self.inspector=UIInspector(self.state)
        self.hook=LowLevelHook(self.state)
        self.hook.start()
        self._selected_hwnd_ui=None
        top=QWidget()
        mid=QWidget()
        bot=QWidget()
        self.cmb=QComboBox()
        self.btn_opt=QPushButton("优化")
        self.btn_ui=QPushButton("识别")
        self.chk_preview=QCheckBox("预览开关")
        self.chk_preview.setChecked(self.state.preview_on)
        self.lbl_mode=QLabel("模式:初始化")
        self.lbl_fps=QLabel("帧率:0")
        self.lbl_rec=QLabel("0B")
        self.lbl_ver=QLabel("v0")
        self.lbl_tip=QLabel("初始化")
        self.progress=QProgressBar()
        self.progress.setRange(0,100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("0%")
        self.model_progress=QProgressBar()
        self.model_progress.setRange(0,100)
        self.model_progress.setValue(0)
        self.model_progress.setTextVisible(True)
        self.model_progress.setFormat("0%")
        self.preview_label=QLabel()
        self.preview_label.setFixedSize(QSize(640,400))
        self.preview_label.setVisible(self.state.preview_on)
        self.ui_table_title=QLabel("UI识别结果列表")
        self.data_table_title=QLabel("窗口内数据列表")
        self.ui_table=QTableWidget()
        self.ui_table.setColumnCount(5)
        self.ui_table.setHorizontalHeaderLabels(["类型","区域","置信度","交互强度","目标"])
        self.ui_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui_table.verticalHeader().setVisible(False)
        self.ui_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ui_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ui_table.setFocusPolicy(Qt.NoFocus)
        self.ui_table.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.pref_editors=[]
        self.data_table=QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels(["名称","数值","趋势","置信度","目标"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.data_table.setFocusPolicy(Qt.NoFocus)
        self.data_table.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.data_pref_editors=[]
        lt=QHBoxLayout(top)
        lt.addWidget(self.cmb,1)
        lt.addWidget(self.btn_opt)
        lt.addWidget(self.btn_ui)
        lt.addWidget(self.chk_preview)
        lm=QHBoxLayout(mid)
        left_col=QVBoxLayout()
        left_col.addWidget(self.preview_label,1)
        lm.addLayout(left_col,1)
        right_panel=QVBoxLayout()
        right_panel.addWidget(self.ui_table_title)
        right_panel.addWidget(self.ui_table,1)
        right_panel.addWidget(self.data_table_title)
        right_panel.addWidget(self.data_table,1)
        lm.addLayout(right_panel,1)
        lb=QHBoxLayout(bot)
        lb.addWidget(self.lbl_mode)
        lb.addWidget(self.lbl_fps)
        lb.addStretch(1)
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
        self.state.signal_data_ready.connect(self.on_data_ready)
        self.on_data_ready([])
        self.ui_table_title.setObjectName("uiTableTitle")
        self.data_table_title.setObjectName("dataTableTitle")
        self.finish_opt.connect(self._handle_opt_finished)
        self.finish_ui.connect(self._handle_ui_finished)
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
        self._ui_thread=None
        self._ui_busy=False
        self._ui_paused=False
        self._ui_last_frame=None
        self._ui_restore_state=Qt.WindowNoState
        self._ui_restore_required=False
        self._data_items=[]
        self.start_wait_timer=QTimer(self)
        self.start_wait_timer.timeout.connect(self.on_model_ready_check)
        self.start_wait_timer.start(200)
        self.refresh_windows()
    def preview_enabled(self):
        return self.chk_preview.isChecked()
    def on_preview_changed(self,_):
        enabled=self.chk_preview.isChecked()
        self.state.save_cfg("preview_on",enabled)
        self.preview_label.setVisible(enabled)
        if not enabled:
            self.preview_label.clear()
    def on_pref_changed_item(self,label,val):
        self.state.update_ui_preference(label,val)
    def on_data_pref_changed(self,combo,val):
        if not isinstance(combo,QComboBox):
            return
        name=str(combo.property("data_name") or "")
        key=combo.property("data_key")
        prev=combo.property("current_pref") or "higher"
        if val=="删除":
            self._handle_data_delete(combo,name,key)
            return
        mapping={"越高越好":"higher","越低越好":"lower","无关":"ignore"}
        pref=mapping.get(val,prev if prev in ["higher","lower","ignore"] else "higher")
        combo.setProperty("current_pref",pref)
        self.state.update_data_preference(name,val)
        if isinstance(key,str) and key:
            for it in self._data_items:
                if it.get("key")==key:
                    it["preference"]=pref
                    break
    def on_preview(self,px):
        if not self.preview_enabled():
            return
        try:
            self.preview_label.setPixmap(px.scaled(self.preview_label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        except:
            pass
    def on_mode(self,s):
        self.lbl_mode.setText(f"模式:{s}")
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
        info=f"{int(val)}%"
        if txt:
            info=f"{info} {txt}"
        self.progress.setFormat(info)
    def on_modelprog(self,val,txt):
        self.model_progress.setValue(int(val))
        info=f"{int(val)}%"
        if txt:
            info=f"{info} {txt}"
        self.model_progress.setFormat(info)
    def on_ui_ready(self,items):
        self.ui_table.setRowCount(0)
        limited=items[:60]
        self.ui_table.setRowCount(len(limited))
        self.pref_editors=[]
        mapping={"higher":0,"lower":1,"ignore":2}
        for idx,it in enumerate(limited):
            display=str(it.get("display",it.get("raw_label",it.get("type",""))))
            pref_key=str(it.get("pref_key",it.get("type","")))
            b=str(it.get("bounds",""))
            c=f"{float(it.get('confidence',0)):.2f}"
            inter=f"{float(it.get('interaction',0)):.2f}"
            self.ui_table.setItem(idx,0,QTableWidgetItem(display))
            self.ui_table.setItem(idx,1,QTableWidgetItem(b))
            ci=QTableWidgetItem(c)
            ci.setTextAlignment(Qt.AlignCenter)
            self.ui_table.setItem(idx,2,ci)
            ii=QTableWidgetItem(inter)
            ii.setTextAlignment(Qt.AlignCenter)
            self.ui_table.setItem(idx,3,ii)
            combo=QComboBox()
            combo.addItems(["越高越好","越低越好","无关"])
            pref=it.get("preference") or self.state.preference_for_label(pref_key)
            idx_pref=mapping.get(pref,0)
            combo.blockSignals(True)
            combo.setCurrentIndex(idx_pref)
            combo.blockSignals(False)
            combo.currentTextChanged.connect(lambda val,lab=pref_key:self.on_pref_changed_item(lab,val))
            self.ui_table.setCellWidget(idx,4,combo)
            self.pref_editors.append(combo)
    def on_data_ready(self,items):
        self.data_table.setRowCount(0)
        limited=items[:60]
        self._data_items=[dict(it) for it in limited]
        self.data_table.setRowCount(len(self._data_items))
        self.data_pref_editors=[]
        for idx,it in enumerate(self._data_items):
            name=str(it.get("name",""))
            num=it.get("value",None)
            val="--"
            num_int=None
            if isinstance(num,int):
                num_int=num
            else:
                try:
                    num_int=int(str(num).strip())
                except:
                    num_int=None
            if isinstance(num_int,int) and num_int>=0:
                val=str(num_int)
            else:
                txt=it.get("text","")
                if isinstance(txt,str) and txt.strip():
                    val=txt.strip()
            trend_val=it.get("trend",None)
            trend="--" if trend_val is None else f"{float(trend_val):+.2f}"
            conf_val=it.get("confidence",None)
            conf="--" if conf_val is None else f"{float(conf_val)*100.0:.0f}%"
            self.data_table.setItem(idx,0,QTableWidgetItem(name))
            self.data_table.setItem(idx,1,QTableWidgetItem(val))
            tr=QTableWidgetItem(trend)
            tr.setTextAlignment(Qt.AlignCenter)
            self.data_table.setItem(idx,2,tr)
            cf=QTableWidgetItem(conf)
            cf.setTextAlignment(Qt.AlignCenter)
            self.data_table.setItem(idx,3,cf)
            combo=QComboBox()
            combo.addItems(["越高越好","越低越好","无关","删除"])
            pref=it.get("preference") or self.state.preference_for_data(name)
            if pref not in ["higher","lower","ignore"]:
                pref=self.state.preference_for_data(name)
            it["preference"]=pref
            combo.blockSignals(True)
            combo.setCurrentIndex(self._data_pref_index(pref))
            combo.blockSignals(False)
            key_ref=it.get("key") or name
            combo.setProperty("data_name",name)
            combo.setProperty("data_key",key_ref)
            combo.setProperty("current_pref",pref)
            combo.currentTextChanged.connect(lambda val,widget=combo:self.on_data_pref_changed(widget,val))
            self.data_table.setCellWidget(idx,4,combo)
            self.data_pref_editors.append(combo)
            stable_key=key_ref
            color=it.get("color")
            if not isinstance(color,(list,tuple)) or len(color)!=3:
                color=self.state.ensure_data_color(stable_key)
                it["color"]=[int(color[0]),int(color[1]),int(color[2])]
            if color and len(color)==3:
                b=int(color[0])
                g=int(color[1])
                r=int(color[2])
                bg=QColor((r+255)//2,(g+255)//2,(b+255)//2)
                fg=QColor(0,0,0) if 0.299*r+0.587*g+0.114*b>186 else QColor(255,255,255)
                for col_idx in range(4):
                    item=self.data_table.item(idx,col_idx)
                    if item:
                        item.setBackground(bg)
                        item.setForeground(fg)
                combo.setStyleSheet(f"background-color: rgba({r},{g},{b},120); color: rgb({fg.red()},{fg.green()},{fg.blue()});")
            else:
                combo.setStyleSheet("")
    def _data_pref_index(self,pref):
        if pref=="lower":
            return 1
        if pref=="ignore":
            return 2
        return 0
    def _reset_data_combo(self,combo,pref):
        combo.blockSignals(True)
        combo.setCurrentIndex(self._data_pref_index(pref))
        combo.blockSignals(False)
        combo.setProperty("current_pref",pref)
    def _handle_data_delete(self,combo,name,key):
        prev=combo.property("current_pref") or "higher"
        box=QMessageBox(self)
        box.setWindowTitle("删除确认")
        box.setText(f"确定删除“{name}”数据吗？")
        box.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        yes=box.button(QMessageBox.Yes)
        no=box.button(QMessageBox.No)
        if yes:
            yes.setText("确定")
        if no:
            no.setText("取消")
        result=box.exec()
        if result!=QMessageBox.Yes:
            self._reset_data_combo(combo,prev)
            return
        remain,removed=self.state.remove_data_point(key or name)
        if not removed:
            self._reset_data_combo(combo,prev)
            return
        self.state.signal_tip.emit(f"已删除数据:{name}")
        self.state.signal_data_ready.emit(remain)
        self._refresh_preview_overlay()
    def _refresh_preview_overlay(self):
        if not self.preview_enabled():
            return
        frame=None
        if isinstance(self.state.prev_img,np.ndarray):
            frame=self.state.prev_img
        elif isinstance(self._ui_last_frame,np.ndarray):
            frame=self._ui_last_frame
        if frame is None:
            return
        disp=self.state.preview_visual(frame)
        if isinstance(disp,np.ndarray):
            buf=np.ascontiguousarray(disp)
        else:
            buf=np.ascontiguousarray(frame)
        h,w=buf.shape[:2]
        qi=QImage(buf.data,w,h,3*w,QImage.Format_BGR888)
        self.state.signal_preview.emit(QPixmap.fromImage(qi))
    def refresh_windows(self):
        mapping=self.selector.refresh()
        keys=sorted(list(mapping.keys()))
        self.cmb.blockSignals(True)
        self.cmb.clear()
        self.cmb.addItems(keys)
        self.cmb.blockSignals(False)
        target_hwnd=self.state.selected_hwnd
        pending=getattr(self.state,"_pending_window",None)
        if target_hwnd is None and pending:
            target_hwnd=pending[0]
        if target_hwnd is None and self._selected_hwnd_ui:
            target_hwnd=self._selected_hwnd_ui
        selected_key=None
        if target_hwnd:
            for k in keys:
                if mapping[k]==target_hwnd:
                    selected_key=k
                    break
        if selected_key is not None:
            idx=keys.index(selected_key)
            self.cmb.blockSignals(True)
            self.cmb.setCurrentIndex(idx)
            self.cmb.blockSignals(False)
            self._selected_hwnd_ui=mapping[selected_key]
            if self.state.selected_title!=selected_key or self.state.selected_hwnd!=self._selected_hwnd_ui:
                self.state.set_window(self._selected_hwnd_ui,selected_key)
            return
        if keys:
            self.cmb.blockSignals(True)
            self.cmb.setCurrentIndex(0)
            self.cmb.blockSignals(False)
            new_key=keys[0]
            new_hwnd=mapping[new_key]
            if self._selected_hwnd_ui!=new_hwnd or self.state.selected_hwnd!=new_hwnd:
                self._selected_hwnd_ui=new_hwnd
                self.state.set_window(new_hwnd,new_key)
            else:
                self._selected_hwnd_ui=new_hwnd
        else:
            self._selected_hwnd_ui=None
    def on_sel_changed(self,_):
        key=self.cmb.currentText()
        hwnd=self.selector.map.get(key)
        if not hwnd:
            return
        if hwnd==self._selected_hwnd_ui and self.state.selected_hwnd==hwnd and self.state.selected_title==key:
            return
        self._selected_hwnd_ui=hwnd
        self.state.set_window(hwnd,key)
    def set_mode(self,mode,force=False):
        if not self.state.model_ready_event.is_set():
            return
        if self.state.optimizing:
            return
        if self._ui_busy:
            return
        if not self.state.running:
            return
        if mode==self.state.mode and not force:
            thread=self.learning_thread if mode=="learning" else self.training_thread
            if thread and thread.is_alive():
                return
        self.state.stop_event.set()
        time.sleep(0.05)
        self.state.stop_event.clear()
        self.state.mode=mode
        if mode=="learning":
            self.lbl_mode.setText("模式:学习")
            self.state.signal_mode.emit("学习")
            self.learning_thread=LearningThread(self.state,self)
            self.learning_thread.start()
        elif mode=="training":
            if not self.state.model_loaded:
                QMessageBox.information(self,"提示","模型不可用，保持学习模式")
                self.state.mode="learning"
                self.lbl_mode.setText("模式:学习")
                self.learning_thread=LearningThread(self.state,self)
                self.learning_thread.start()
                return
            self.lbl_mode.setText("模式:训练")
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
        self.lbl_fps.setText(f"帧率:{int(self.state.fps)}")
    def on_usage(self):
        cpu=int(psutil.cpu_percent())
        mem=int(psutil.virtual_memory().percent)
        gu,gm=_gpu_util_mem()
        info=f"CPU:{cpu}% MEM:{mem}%"
        if gu is not None:
            info+=f" GPU:{int(gu)}% VRAM:{int(gm)}%"
        self.statusBar().showMessage(info)
    def on_opt(self):
        if self.state.optimizing:
            return
        if self._ui_busy:
            return
        if not self.state.model_ready_event.is_set():
            QMessageBox.information(self,"提示","模型尚未准备完成，请稍候")
            return
        try:
            self.state.wait_model()
        except ModelIntegrityError as e:
            QMessageBox.critical(self,"错误",str(e))
            return
        self.state.stop_event.set()
        self.state.running=False
        self.btn_opt.setEnabled(False)
        self.btn_ui.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setFormat("0%")
        self.lbl_mode.setText("模式:优化中")
        self.state.signal_mode.emit("优化中")
        self._optim_flag.clear()
        self.lbl_tip.setText("正在离线优化")
        th=OptimizerThread(self.state,self,self._on_opt_done,self._optim_flag)
        th.start()
    def _on_opt_done(self,ok):
        self.finish_opt.emit(ok)
    def _handle_opt_finished(self,ok):
        msg="优化完成" if ok else "优化失败"
        QMessageBox.information(self,"提示",msg)
        self.btn_opt.setEnabled(True)
        self.btn_ui.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("0%")
        self.state.stop_event.clear()
        self.state.running=True
        self.lbl_tip.setText("优化完成，已恢复学习" if ok else "优化失败，继续使用现有模型")
        if ok:
            try:
                ver=f"v{ModelMeta.read().get('version',0)}"
            except Exception:
                ver=self.lbl_ver.text()
            self.state.signal_modelver.emit(ver)
        self.set_mode("learning",force=True)
    def _prepare_ui_minimize(self):
        self._ui_restore_state=self.windowState()
        self._ui_restore_required=False
        if not (self._ui_restore_state&Qt.WindowMinimized):
            self._ui_restore_required=True
            self.showMinimized()
            QApplication.processEvents()
            time.sleep(0.15)
        else:
            QApplication.processEvents()
            time.sleep(0.05)
    def _restore_ui_window(self):
        if not self._ui_restore_required:
            return
        state=self._ui_restore_state
        try:
            if state&Qt.WindowFullScreen:
                self.showFullScreen()
            elif state&Qt.WindowMaximized:
                self.showMaximized()
            else:
                self.showNormal()
        except:
            self.showNormal()
        QApplication.processEvents()
        try:
            self.activateWindow()
            self.raise_()
        except:
            pass
        self._ui_restore_required=False
        self._ui_restore_state=Qt.WindowNoState
    def on_ui(self):
        if self.state.optimizing or self._ui_busy:
            return
        if not self.state.selected_hwnd:
            QMessageBox.information(self,"提示","未选择窗口")
            return
        paused=False
        if self.state.mode=="learning" and self.learning_thread and self.learning_thread.is_alive():
            paused=True
            self.state.stop_event.set()
            try:
                self.learning_thread.join(timeout=1.0)
            except:
                pass
            self.state.stop_event.clear()
        self._prepare_ui_minimize()
        img=None
        for _ in range(6):
            frame=self.state.capture_client()
            if isinstance(frame,np.ndarray):
                img=frame
                break
            time.sleep(0.05)
        if img is None:
            self._restore_ui_window()
            QMessageBox.information(self,"提示","无法获取窗口画面")
            if paused:
                self._ui_paused=False
                self.set_mode("learning",force=True)
            return
        events=self.state._recent_events(source="user",mode="learning")
        self._ui_busy=True
        self._ui_paused=paused
        self.btn_opt.setEnabled(False)
        self.btn_ui.setEnabled(False)
        self.lbl_tip.setText("正在执行UI识别")
        self._ui_last_frame=img.copy() if isinstance(img,np.ndarray) else None
        def task():
            ok=False
            err=""
            res=[]
            try:
                res=self.inspector.analyze(img,events)
                ok=True
            except Exception as e:
                err=str(e)
                log(f"ui detect error:{e}")
            self.finish_ui.emit(ok,err,res)
        self._ui_thread=threading.Thread(target=task,daemon=True)
        self._ui_thread.start()
    def _handle_ui_finished(self,ok,err,res):
        self._restore_ui_window()
        if ok:
            items=res.get("ui",[]) if isinstance(res,dict) else (res or [])
            data_items=res.get("data",[]) if isinstance(res,dict) else []
            self.state.signal_ui_ready.emit(items)
            self.state.signal_data_ready.emit(data_items)
            QMessageBox.information(self,"提示","UI识别完成，结果已列出")
            self.lbl_tip.setText("UI识别完成，已恢复学习")
            if hasattr(self.state,"validator"):
                self.state.validator.enqueue_ui(items,self._ui_last_frame)
        else:
            QMessageBox.warning(self,"提示","UI识别失败"+(f":{err}" if err else ""))
            self.lbl_tip.setText("UI识别失败，保持学习")
        self.btn_opt.setEnabled(True)
        self.btn_ui.setEnabled(True)
        self._ui_busy=False
        if self._ui_paused:
            self.state.stop_event.clear()
        self._ui_paused=False
        self._ui_thread=None
        self._ui_last_frame=None
        self.set_mode("learning",force=True)
    def on_model_ready_check(self):
        if not self.state.model_ready_event.is_set():
            return
        self.start_wait_timer.stop()
        if self.state.model_error:
            self.lbl_tip.setText(f"模型错误:{self.state.model_error}")
            QMessageBox.critical(self,"错误",self.state.model_error)
            return
        self.state.running=True
        self.state.stop_event.clear()
        self.lbl_tip.setText("资源已就绪，进入学习模式")
        self.set_mode("learning",force=True)
    def closeEvent(self,event):
        self.state.stop_event.set()
        self.state.running=False
        self.hook.stop()
        self.state.io_q.put(None)
        if hasattr(self.state,"validator"):
            self.state.validator.queue.put(("stop",None))
        event.accept()
def main():
    app=QApplication(sys.argv)
    w=Main()
    w.resize(960,720)
    w.show()
    sys.exit(app.exec())
if __name__=="__main__":
    main()
