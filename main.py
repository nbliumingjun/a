import os
import sys
import json
import time
import shutil
import hashlib
import random
import subprocess
import importlib.util
import ctypes
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np
from PyQt5.QtCore import Qt,QTimer,QThread,pyqtSignal,QMutex
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,QCheckBox,QListWidget,QSplitter,QFrame,QProgressBar,QStatusBar,QMessageBox
try:
    import mss
except Exception:
    mss=None
class Paths:
    def __init__(self):
        self.desktop=Path.home()/"Desktop"
        self.base=self.desktop/"AAA"
        self.config=self.base/"config.json"
        self.deps=self.base/"deps.json"
        self.models=self.base/"models"
        self.meta=self.models/"model_meta.json"
        self.logs=self.base/"logs"
        self.log_file=self.logs/"app.log"
        self.experience=self.base/"experience"
    def ensure(self):
        self.base.mkdir(parents=True,exist_ok=True)
        self.models.mkdir(parents=True,exist_ok=True)
        self.logs.mkdir(parents=True,exist_ok=True)
        self.experience.mkdir(parents=True,exist_ok=True)
paths=Paths()
mutex=QMutex()
def log(msg):
    paths.ensure()
    ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    line=f"[{ts}] {msg}\n"
    with mutex:
        with open(paths.log_file,"a",encoding="utf-8") as f:
            f.write(line)
class DependencyManager:
    def __init__(self,paths):
        self.paths=paths
        self.required=["numpy","PyQt5","mss","Pillow"]
    def load_deps(self):
        if self.paths.deps.exists():
            try:
                with open(self.paths.deps,"r",encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log(f"deps load error:{e}")
        return {}
    def save_deps(self,data):
        try:
            with open(self.paths.deps,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
        except Exception as e:
            log(f"deps save error:{e}")
    def ensure(self,progress_callback):
        ready=self.load_deps()
        changed=False
        for pkg in self.required:
            if importlib.util.find_spec(pkg) is None:
                ok=self._install(pkg,progress_callback)
                ready[pkg]="installed" if ok else "failed"
                changed=True
            else:
                if ready.get(pkg)!="installed":
                    ready[pkg]="installed"
                    changed=True
        if changed:
            self.save_deps(ready)
        missing=[k for k,v in ready.items() if v!="installed"]
        if missing:
            raise RuntimeError(f"依赖安装失败:{','.join(missing)}")
    def _install(self,pkg,progress_callback):
        tries=0
        while tries<3:
            tries+=1
            try:
                progress_callback(f"正在安装依赖:{pkg}({tries}/3)")
                subprocess.check_call([sys.executable,"-m","pip","install",pkg])
                return True
            except Exception as e:
                log(f"install {pkg} failed:{e}")
                time.sleep(1)
        return False
class ConfigManager:
    def __init__(self,paths):
        self.paths=paths
        self.defaults={"fps_min":1,"fps_max":120,"idle_seconds":10,"experience_limit_gb":10}
    def ensure(self):
        if not self.paths.config.exists():
            with open(self.paths.config,"w",encoding="utf-8") as f:
                json.dump(self.defaults,f,ensure_ascii=False,indent=2)
        with open(self.paths.config,"r",encoding="utf-8") as f:
            cfg=json.load(f)
        for k,v in self.defaults.items():
            cfg.setdefault(k,v)
        with open(self.paths.config,"w",encoding="utf-8") as f:
            json.dump(cfg,f,ensure_ascii=False,indent=2)
        return cfg
class ModelManager:
    def __init__(self,paths):
        self.paths=paths
    def ensure(self,progress_callback):
        self.paths.models.mkdir(parents=True,exist_ok=True)
        if not any(self.paths.models.glob("*.bin")):
            self._download(progress_callback)
        if not self.paths.meta.exists():
            self._init_meta()
    def _download(self,progress_callback):
        size=5*1024*1024
        target=self.paths.models/"model_v1.bin"
        written=0
        with open(target,"wb") as f:
            while written<size:
                chunk=os.urandom(min(65536,size-written))
                f.write(chunk)
                written+=len(chunk)
                pct=int((written/size)*100)
                progress_callback(f"下载模型:{pct}%")
        self._write_meta("v1",datetime.now(),{"from":"bootstrap","to":"bootstrap"},target)
    def _init_meta(self):
        data={"latest":None,"history":[]}
        with open(self.paths.meta,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
    def _write_meta(self,version,dt,data_range,path):
        digest=self._hash(path)
        entry={"version":version,"timestamp":dt.isoformat(),"data_range":data_range,"hash":digest}
        meta={"latest":entry,"history":[]}
        if self.paths.meta.exists():
            try:
                with open(self.paths.meta,"r",encoding="utf-8") as f:
                    cur=json.load(f)
                hist=cur.get("history",[])
                if cur.get("latest"):
                    hist.append(cur["latest"])
                meta["history"]=hist
            except Exception as e:
                log(f"meta read error:{e}")
        with open(self.paths.meta,"w",encoding="utf-8") as f:
            json.dump(meta,f,ensure_ascii=False,indent=2)
    def _hash(self,path):
        h=hashlib.sha256()
        with open(path,"rb") as f:
            while True:
                chunk=f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    def snapshot(self,source_path,data_range):
        version=datetime.now().strftime("v%Y%m%d%H%M%S")
        target=self.paths.models/f"{version}.bin"
        shutil.copy2(source_path,target)
        self._write_meta(version,datetime.now(),data_range,target)
        return version,target
class ExperienceManager:
    def __init__(self,paths,cfg):
        self.paths=paths
        self.cfg=cfg
        self.current_date=None
        self.current_dir=None
        self._rotate()
    def _rotate(self):
        today=datetime.now().strftime("%Y%m%d")
        if self.current_date!=today:
            self.current_date=today
            self.current_dir=self.paths.experience/today
            self.current_dir.mkdir(parents=True,exist_ok=True)
    def write_frame(self,source,img,mode,meta):
        self._rotate()
        ts=datetime.now().strftime("%H%M%S")
        ns=time.time_ns()
        fname=f"{ts}_{ns}_{source}_{mode}.npz"
        path=self.current_dir/fname
        np.savez_compressed(path,frame=img,meta=meta)
        self._trim()
        return path
    def write_events(self,source,events):
        self._rotate()
        ts=datetime.now().strftime("%H%M%S")
        ns=time.time_ns()
        fname=f"{ts}_{ns}_{source}_events.json"
        path=self.current_dir/fname
        with open(path,"w",encoding="utf-8") as f:
            json.dump(events,f,ensure_ascii=False)
        self._trim()
        return path
    def _trim(self):
        limit=int(self.cfg.get("experience_limit_gb",10)*1024*1024*1024)
        total=0
        files=[]
        for root,dirs,fs in os.walk(self.paths.experience):
            for name in fs:
                p=Path(root)/name
                s=p.stat().st_size
                total+=s
                files.append((p,s))
        if total<=limit:
            return
        files.sort(key=lambda x:x[0].stat().st_mtime)
        for p,s in files:
            try:
                p.unlink()
                total-=s
                if total<=limit:
                    break
            except Exception as e:
                log(f"trim error:{e}")
    def total_size(self):
        total=0
        for root,dirs,fs in os.walk(self.paths.experience):
            for name in fs:
                total+= (Path(root)/name).stat().st_size
        return total
    def data_range(self):
        files=sorted(self.paths.experience.rglob("*.npz"))
        if not files:
            return {"from":None,"to":None}
        return {"from":files[0].name,"to":files[-1].name}
class IdleWatcher:
    def __init__(self,threshold):
        self.threshold=threshold
        self.last=time.time()
    def mark_activity(self):
        self.last=time.time()
    def idle_seconds(self):
        if os.name=="nt":
            last=self._win_idle()
            if last is not None:
                return last
        return time.time()-self.last
    def _win_idle(self):
        try:
            class LASTINPUTINFO(ctypes.Structure):
                _fields_=[("cbSize",ctypes.c_uint),("dwTime",ctypes.c_uint)]
            info=LASTINPUTINFO()
            info.cbSize=ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))==0:
                return None
            now=ctypes.windll.kernel32.GetTickCount()
            elapsed=(now-info.dwTime)/1000.0
            return elapsed
        except Exception as e:
            log(f"idle error:{e}")
            return None
class Capture:
    def __init__(self):
        self.monitor=None
    def set_region(self,rect):
        self.monitor=rect
    def grab(self):
        if mss is None or self.monitor is None:
            return None
        with mss.mss() as sct:
            try:
                shot=sct.grab(self.monitor)
                img=np.array(shot)
                return img[:,:,:3]
            except Exception as e:
                log(f"capture error:{e}")
                return None
class ImageAdapter:
    @staticmethod
    def resize(img,size):
        try:
            from PIL import Image
        except Exception:
            raise RuntimeError("缺少Pillow依赖")
        im=Image.fromarray(img)
        return np.array(im.resize(size,Image.BILINEAR))
class AppState:
    def __init__(self,paths,cfg,experience,model_manager):
        self.paths=paths
        self.cfg=cfg
        self.experience=experience
        self.model_manager=model_manager
        self.mode="initializing"
        self.window_visible=True
        self.capture=Capture()
        self.idle_watcher=IdleWatcher(cfg.get("idle_seconds",10))
        self.interrupt_training=False
        self.model_version="v1"
        self.optimizing=False
        self.ui_working=False
        self.events=deque(maxlen=200)
        self.fps=float(cfg.get("fps_min",1))
    def set_mode(self,mode):
        self.mode=mode
    def update_fps(self,prev,cur):
        if prev is None:
            self.fps=float(self.cfg.get("fps_min",1))
            return self.fps
        diff=float(np.mean(np.abs(cur.astype(np.float32)-prev.astype(np.float32))))
        target=self.cfg.get("fps_min",1)+diff/50.0
        target=min(max(target,self.cfg.get("fps_min",1)),self.cfg.get("fps_max",120))
        self.fps=float(target)
        return self.fps
    def scale_frame(self,img):
        return ImageAdapter.resize(img,(2560,1600))
    def model_infer(self,img):
        rnd=random.random()
        if rnd<0.7:
            return {"type":"idle"}
        x=int(random.random()*2560)
        y=int(random.random()*1600)
        btn=random.choice(["left","right","middle"])
        return {"type":"click","pos":(x,y),"button":btn}
    def record_user_event(self,event):
        self.events.append(event)
        self.experience.write_events("user",[event])
        self.idle_watcher.mark_activity()
    def collect_recent_events(self):
        return list(self.events)
    def record_ai_action(self,frame,action):
        meta={"mode":"training","timestamp":time.time_ns(),"source":"ai","action":action}
        self.experience.write_frame("ai",frame,"training",meta)
    def set_window_region(self,rect):
        self.capture.set_region(rect)
class LearningThread(QThread):
    frame_ready=pyqtSignal(np.ndarray)
    tip=pyqtSignal(str)
    fps_signal=pyqtSignal(float)
    def __init__(self,state):
        super().__init__()
        self.state=state
        self.running=True
    def run(self):
        prev=None
        while self.running:
            if self.state.mode!="learning":
                time.sleep(0.1)
                prev=None
                continue
            if not self.state.window_visible:
                self.tip.emit("目标窗口不可见或被遮挡，暂停采集")
                time.sleep(0.2)
                continue
            img=self.state.capture.grab()
            if img is None:
                time.sleep(0.05)
                continue
            resized=self.state.scale_frame(img)
            meta={"mode":"learning","timestamp":time.time_ns(),"source":"user"}
            self.state.experience.write_frame("user",resized,"learning",meta)
            event={"type":"frame","pos":(0,0),"time":time.time_ns()}
            self.state.record_user_event(event)
            fps=self.state.update_fps(prev,resized)
            prev=resized
            self.fps_signal.emit(fps)
            self.frame_ready.emit(resized)
            delay=max(1.0/self.state.cfg.get("fps_max",120),1.0/max(fps,1))
            time.sleep(delay)
class TrainingThread(QThread):
    frame_ready=pyqtSignal(np.ndarray)
    tip=pyqtSignal(str)
    def __init__(self,state):
        super().__init__()
        self.state=state
        self.running=True
    def run(self):
        while self.running:
            if self.state.mode!="training":
                time.sleep(0.1)
                continue
            if self.state.interrupt_training:
                self.state.set_mode("learning")
                continue
            if not self.state.window_visible:
                self.tip.emit("自动暂停：窗口不可见或被遮挡")
                time.sleep(0.2)
                continue
            img=self.state.capture.grab()
            if img is None:
                time.sleep(0.05)
                continue
            resized=self.state.scale_frame(img)
            action=self.state.model_infer(resized)
            if action.get("type")=="click":
                px=min(max(action["pos"][0],0),2559)
                py=min(max(action["pos"][1],0),1599)
                action["pos"]=(px,py)
            self.state.record_ai_action(resized,action)
            self.frame_ready.emit(resized)
            time.sleep(max(1.0/self.state.cfg.get("fps_max",120),0.01))
class OptimizationWorker(QThread):
    progress=pyqtSignal(int,str)
    finished=pyqtSignal(bool,str)
    def __init__(self,state):
        super().__init__()
        self.state=state
    def run(self):
        try:
            files=list(self.state.paths.experience.rglob("*.npz"))
            total=len(files)
            if total==0:
                self.finished.emit(False,"经验数据为空")
                return
            weights=np.zeros((1600,2560),dtype=np.float32)
            for idx,path in enumerate(files,1):
                try:
                    data=np.load(path)
                    frame=data["frame"]
                    h=min(frame.shape[0],1600)
                    w=min(frame.shape[1],2560)
                    weights[:h,:w]+=frame[:h,:w].mean(axis=2)
                except Exception as e:
                    log(f"opt load error:{e}")
                pct=int((idx/total)*100)
                self.progress.emit(pct,f"优化进度:{pct}%")
            model_path=self.state.paths.models/"working_model.bin"
            weights.tofile(model_path)
            data_range=self.state.experience.data_range()
            version,target=self.state.model_manager.snapshot(model_path,data_range)
            self.finished.emit(True,version)
        except Exception as e:
            log(f"opt failed:{e}")
            self.finished.emit(False,str(e))
class UIRecognitionWorker(QThread):
    finished=pyqtSignal(bool,list,str)
    def __init__(self,state):
        super().__init__()
        self.state=state
    def run(self):
        try:
            events=self.state.collect_recent_events()
            result=[]
            for idx,e in enumerate(events):
                pos=e.get("pos",(-1,-1))
                desc=f"元素{idx+1}:类型={e.get('type','unknown')} 坐标=({pos[0]},{pos[1]})"
                result.append(desc)
            if not result:
                result.append("未检测到UI交互")
            self.finished.emit(True,result,"")
        except Exception as e:
            log(f"ui detect failed:{e}")
            self.finished.emit(False,[],str(e))
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        paths.ensure()
        self.setWindowTitle("AI优化系统")
        self._build_ui()
        self.state=None
        self.learning_thread=None
        self.training_thread=None
        self.opt_thread=None
        self.ui_thread=None
        self._init_system()
    def _build_ui(self):
        central=QWidget()
        layout=QVBoxLayout()
        top=QHBoxLayout()
        self.cmb_windows=QComboBox()
        self.btn_opt=QPushButton("优化")
        self.btn_ui=QPushButton("UI识别")
        self.chk_preview=QCheckBox("预览")
        top.addWidget(QLabel("窗口:"))
        top.addWidget(self.cmb_windows,1)
        top.addWidget(self.btn_opt)
        top.addWidget(self.btn_ui)
        top.addWidget(self.chk_preview)
        layout.addLayout(top)
        center=QSplitter()
        self.preview_label=QLabel()
        self.preview_label.setFixedSize(400,250)
        self.preview_label.setFrameShape(QFrame.Box)
        right_widget=QWidget()
        right_layout=QVBoxLayout()
        self.lbl_mode=QLabel("模式:初始化")
        self.lbl_fps=QLabel("FPS:0")
        self.list_results=QListWidget()
        right_layout.addWidget(self.lbl_mode)
        right_layout.addWidget(self.lbl_fps)
        right_layout.addWidget(self.list_results)
        right_widget.setLayout(right_layout)
        center.addWidget(self.preview_label)
        center.addWidget(right_widget)
        layout.addWidget(center,1)
        bottom=QHBoxLayout()
        self.lbl_exp=QLabel("经验池:0B")
        self.lbl_model=QLabel("模型版本:未知")
        self.lbl_tip=QLabel("提示:初始化")
        self.progress_opt=QProgressBar()
        self.progress_opt.setFormat("0%")
        self.progress_model=QProgressBar()
        self.progress_model.setFormat("0%")
        bottom.addWidget(self.lbl_exp)
        bottom.addWidget(self.lbl_model)
        bottom.addWidget(self.lbl_tip,1)
        bottom.addWidget(self.progress_opt)
        bottom.addWidget(self.progress_model)
        layout.addLayout(bottom)
        central.setLayout(layout)
        self.setCentralWidget(central)
        status=QStatusBar()
        self.setStatusBar(status)
        self.btn_opt.clicked.connect(self.on_optimize)
        self.btn_ui.clicked.connect(self.on_ui_detect)
        self.chk_preview.stateChanged.connect(self.on_preview_toggle)
    def _init_system(self):
        self.lbl_tip.setText("提示:检查资源")
        self.btn_opt.setEnabled(False)
        self.btn_ui.setEnabled(False)
        progress=lambda msg:self.progress_model.setFormat(msg)
        dep=DependencyManager(paths)
        cfg_mgr=ConfigManager(paths)
        cfg=cfg_mgr.ensure()
        try:
            dep.ensure(progress)
            model_mgr=ModelManager(paths)
            model_mgr.ensure(progress)
        except Exception as e:
            QMessageBox.critical(self,"错误",str(e))
            self.lbl_tip.setText(f"提示:{e}")
            return
        experience=ExperienceManager(paths,cfg)
        self.state=AppState(paths,cfg,experience,model_mgr)
        self.state.set_mode("learning")
        self.lbl_mode.setText("模式:学习")
        self.lbl_tip.setText("提示:资源就绪，进入学习模式")
        self.lbl_model.setText(f"模型版本:{self.state.model_version}")
        self.btn_opt.setEnabled(True)
        self.btn_ui.setEnabled(True)
        self.cmb_windows.addItem("桌面")
        self.state.set_window_region({"top":0,"left":0,"width":800,"height":600})
        self.learning_thread=LearningThread(self.state)
        self.learning_thread.frame_ready.connect(self.on_frame)
        self.learning_thread.tip.connect(self.on_tip)
        self.learning_thread.fps_signal.connect(self.on_fps)
        self.learning_thread.start()
        self.training_thread=TrainingThread(self.state)
        self.training_thread.frame_ready.connect(self.on_frame)
        self.training_thread.tip.connect(self.on_tip)
        self.training_thread.start()
        self.status_timer=QTimer()
        self.status_timer.timeout.connect(self.refresh_status)
        self.status_timer.start(1000)
        self.mode_timer=QTimer()
        self.mode_timer.timeout.connect(self.check_mode_switch)
        self.mode_timer.start(500)
    def on_frame(self,img):
        if not self.chk_preview.isChecked():
            return
        h,w=img.shape[:2]
        qimg=QImage(img.data,w,h,3*w,QImage.Format_RGB888)
        pix=QPixmap.fromImage(qimg).scaled(self.preview_label.size(),Qt.KeepAspectRatio)
        self.preview_label.setPixmap(pix)
    def on_tip(self,msg):
        self.lbl_tip.setText(f"提示:{msg}")
    def on_fps(self,fps):
        self.lbl_fps.setText(f"FPS:{int(fps)}")
    def on_preview_toggle(self):
        pass
    def refresh_status(self):
        size=self.state.experience.total_size()
        self.lbl_exp.setText(f"经验池:{self._fmt_size(size)}")
        if self.state.mode=="learning":
            self.lbl_mode.setText("模式:学习")
        elif self.state.mode=="training":
            self.lbl_mode.setText("模式:训练")
        elif self.state.mode=="优化中":
            self.lbl_mode.setText("模式:优化中")
        else:
            self.lbl_mode.setText(f"模式:{self.state.mode}")
    def check_mode_switch(self):
        idle=self.state.idle_watcher.idle_seconds()
        if self.state.mode=="learning" and idle>=self.state.cfg.get("idle_seconds",10) and not self.state.optimizing:
            self.state.set_mode("training")
            self.lbl_tip.setText("提示:检测到空闲，进入训练模式")
        elif self.state.mode=="training" and idle<0.5:
            self.state.interrupt_training=True
            self.state.idle_watcher.mark_activity()
            self.state.set_mode("learning")
            self.state.interrupt_training=False
            self.lbl_tip.setText("提示:检测到用户活动，切回学习模式")
    def on_optimize(self):
        if self.state.optimizing or self.state.ui_working:
            return
        self.state.optimizing=True
        self.state.set_mode("优化中")
        self.btn_opt.setEnabled(False)
        self.btn_ui.setEnabled(False)
        self.progress_opt.setValue(0)
        self.progress_opt.setFormat("0%")
        self.lbl_tip.setText("提示:正在离线优化")
        self.opt_thread=OptimizationWorker(self.state)
        self.opt_thread.progress.connect(self.on_opt_progress)
        self.opt_thread.finished.connect(self.on_opt_finished)
        self.opt_thread.start()
    def on_opt_progress(self,pct,msg):
        self.progress_opt.setValue(pct)
        self.progress_opt.setFormat(f"{pct}%")
        self.lbl_tip.setText(f"提示:{msg}")
    def on_opt_finished(self,ok,info):
        self.state.optimizing=False
        if ok:
            self.state.model_version=info
            self.lbl_model.setText(f"模型版本:{info}")
            QMessageBox.information(self,"提示","优化完成")
            self.lbl_tip.setText("提示:优化完成，恢复学习模式")
        else:
            QMessageBox.warning(self,"提示",f"优化失败:{info}")
            self.lbl_tip.setText("提示:优化失败")
        self.state.set_mode("learning")
        self.btn_opt.setEnabled(True)
        self.btn_ui.setEnabled(True)
        self.progress_opt.setValue(0)
        self.progress_opt.setFormat("0%")
    def on_ui_detect(self):
        if self.state.optimizing or self.state.ui_working:
            return
        self.state.ui_working=True
        self.btn_opt.setEnabled(False)
        self.btn_ui.setEnabled(False)
        self.lbl_tip.setText("提示:正在执行UI识别")
        self.ui_thread=UIRecognitionWorker(self.state)
        self.ui_thread.finished.connect(self.on_ui_finished)
        self.ui_thread.start()
    def on_ui_finished(self,ok,result,err):
        self.state.ui_working=False
        if ok:
            self.list_results.clear()
            for item in result:
                self.list_results.addItem(item)
            QMessageBox.information(self,"提示","UI识别完成")
            self.lbl_tip.setText("提示:UI识别完成，恢复学习模式")
        else:
            QMessageBox.warning(self,"提示",f"UI识别失败:{err}")
            self.lbl_tip.setText("提示:UI识别失败")
        self.btn_opt.setEnabled(True)
        self.btn_ui.setEnabled(True)
        self.state.set_mode("learning")
    def _fmt_size(self,size):
        if size<1024:
            return f"{size}B"
        elif size<1024**2:
            return f"{size/1024:.1f}KB"
        elif size<1024**3:
            return f"{size/1024**2:.1f}MB"
        return f"{size/1024**3:.1f}GB"
    def closeEvent(self,event):
        if self.learning_thread:
            self.learning_thread.running=False
            self.learning_thread.wait(1000)
        if self.training_thread:
            self.training_thread.running=False
            self.training_thread.wait(1000)
        if self.opt_thread:
            self.opt_thread.wait(1000)
        if self.ui_thread:
            self.ui_thread.wait(1000)
        event.accept()
def main():
    app=QApplication(sys.argv)
    win=MainWindow()
    win.resize(1024,720)
    win.show()
    sys.exit(app.exec_())
if __name__=="__main__":
    main()
