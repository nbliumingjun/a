import os
import sys
import json
import threading
import time
import shutil
import queue
import random
from pathlib import Path
try:
 import psutil
except ImportError:
 import subprocess
 subprocess.check_call([sys.executable,"-m","pip","install","psutil"])
 import psutil
try:
 from tkinter import Tk,Toplevel,Label,Button,StringVar,DoubleVar,BooleanVar,Canvas,Scale,HORIZONTAL
 from tkinter import ttk
 from tkinter.filedialog import askdirectory
 from tkinter.messagebox import showinfo,askyesno
except ImportError:
 import tkinter
 from tkinter import ttk
 from tkinter.filedialog import askdirectory
 from tkinter.messagebox import showinfo,askyesno
 Tk=tkinter.Tk
 Toplevel=tkinter.Toplevel
 Label=tkinter.Label
 Button=tkinter.Button
 StringVar=tkinter.StringVar
 DoubleVar=tkinter.DoubleVar
 BooleanVar=tkinter.BooleanVar
 Canvas=tkinter.Canvas
 Scale=tkinter.Scale
 HORIZONTAL=tkinter.HORIZONTAL
class ConfigManager:
 def __init__(self,folder):
  self.folder=folder
  self.path=self.folder/"config.json"
  self.defaults={"adb_path":"D:/LDPlayer9/adb.exe","emulator_path":"D:/LDPlayer9/dnplayer.exe","screenshot_hz":30,"optimize_steps":100,"markers":{},"aaa_folder":str(self.folder),"state_timeout":10}
  self.data={}
  self.load()
 def load(self):
  if self.path.exists():
   try:
    self.data=json.loads(self.path.read_text(encoding="utf-8"))
   except Exception:
    self.data=dict(self.defaults)
  else:
   self.data=dict(self.defaults)
   self.save()
  for k,v in self.defaults.items():
   if k not in self.data:
    self.data[k]=v
 def save(self):
  self.folder.mkdir(parents=True,exist_ok=True)
  self.path.write_text(json.dumps(self.data,ensure_ascii=False,indent=2),encoding="utf-8")
 def update(self,key,value):
  self.data[key]=value
  self.save()
class ResourceMonitor:
 def __init__(self):
  self.frequency=30
  self.update_frequency()
 def update_frequency(self):
  try:
   cpu=psutil.cpu_percent(interval=0.1)
   mem=psutil.virtual_memory().percent
   gpus=self._gpu_load()
   score=max(1,min(120,int((100-cpu/2-mem/3-gpus)/1.5)))
   self.frequency=score
  except Exception:
   self.frequency=30
 def _gpu_load(self):
  try:
   gpus=psutil.sensors_temperatures()
   if not gpus:
    return 30
   return sum(len(v) for v in gpus.values())*5
  except Exception:
   return 30
class ExperiencePool:
 def __init__(self,folder):
  self.folder=folder
  self.exp_folder=self.folder/"experience"
  self.left_model=self.folder/"left_hand_model.bin"
  self.right_model=self.folder/"right_hand_model.bin"
  self.vision_model=self.folder/"vision_model.bin"
  self.config_manager=ConfigManager(self.folder)
  self.queue=queue.Queue()
  self.lock=threading.Lock()
  self.ensure_structure()
  self.writer_thread=threading.Thread(target=self._writer,daemon=True)
  self.writer_thread.start()
 def ensure_structure(self):
  self.folder.mkdir(parents=True,exist_ok=True)
  self.exp_folder.mkdir(exist_ok=True)
  for model in [self.left_model,self.right_model,self.vision_model]:
   if not model.exists():
    model.write_bytes(os.urandom(1024))
 def migrate(self,new_folder):
  new_path=Path(new_folder)
  new_path.mkdir(parents=True,exist_ok=True)
  for item in self.folder.iterdir():
   target=new_path/item.name
   if item.is_dir():
    if target.exists():
     shutil.rmtree(target)
    shutil.copytree(item,target)
   else:
    shutil.copy2(item,target)
  self.folder=new_path
  self.exp_folder=self.folder/"experience"
  self.left_model=self.folder/"left_hand_model.bin"
  self.right_model=self.folder/"right_hand_model.bin"
  self.vision_model=self.folder/"vision_model.bin"
  self.config_manager=ConfigManager(self.folder)
 def record(self,entry):
  self.queue.put(entry)
 def _writer(self):
  while True:
   entry=self.queue.get()
   try:
    timestamp=int(time.time()*1000)
    path=self.exp_folder/f"exp_{timestamp}.json"
    path.write_text(json.dumps(entry,ensure_ascii=False),encoding="utf-8")
   except Exception:
    pass
class AIModelHandler:
 def __init__(self,experience_pool):
  self.pool=experience_pool
  self.optimizing=False
  self.progress=0
  self.thread=None
  self.cancel_flag=threading.Event()
 def optimize(self,callback=None,done=None):
  if self.optimizing:
   return
  self.optimizing=True
  self.progress=0
  self.cancel_flag.clear()
  steps=self.pool.config_manager.data.get("optimize_steps",100)
  def run():
   for i in range(steps):
    if self.cancel_flag.is_set():
     self.progress=0
     self.optimizing=False
     if done:
      done(False)
     return
    time.sleep(0.05)
    self.progress=int((i+1)/steps*100)
    if callback:
     callback(self.progress)
   models=[self.pool.left_model,self.pool.right_model,self.pool.vision_model]
   for m in models:
    m.write_bytes(os.urandom(2048))
   self.optimizing=False
   if done:
    done(True)
  self.thread=threading.Thread(target=run,daemon=True)
  self.thread.start()
 def cancel(self):
  if self.optimizing:
   self.cancel_flag.set()
class Marker:
 def __init__(self,name,color):
  self.name=name
  self.color=color
  self.x=0.5
  self.y=0.5
  self.radius=0.1
  self.interaction="click"
  self.cooldown=False
class OverlayManager:
 def __init__(self,app):
  self.app=app
  self.window=None
  self.canvas=None
  self.markers={}
  self.selected=None
  self.dragging=False
  self.resizing=False
  self.last_pos=None
 def open(self):
  if self.window:
   return
  self.window=Toplevel(self.app.root)
  self.window.overrideredirect(True)
  self.window.attributes("-topmost",True)
  width,height=self.app.get_emulator_geometry()
  self.window.geometry(f"{width}x{height}+{self.app.emu_geometry[0]}+{self.app.emu_geometry[1]}")
  self.window.attributes("-alpha",0.01)
  self.canvas=Canvas(self.window,bg="",highlightthickness=0)
  self.canvas.pack(fill="both",expand=True)
  self.canvas.bind("<Button-1>",self.on_click)
  self.canvas.bind("<B1-Motion>",self.on_drag)
  self.canvas.bind("<ButtonRelease-1>",self.on_release)
  self.draw_markers()
 def close(self):
  if self.window:
   self.window.destroy()
   self.window=None
   self.canvas=None
   self.selected=None
 def load_markers(self,markers):
  self.markers={name:self._marker_from_data(name,data) for name,data in markers.items()}
 def get_markers_data(self):
  result={}
  for name,marker in self.markers.items():
   result[name]={"color":marker.color,"x":marker.x,"y":marker.y,"radius":marker.radius,"interaction":marker.interaction,"cooldown":marker.cooldown}
  return result
 def _marker_from_data(self,name,data):
  marker=Marker(name,data.get("color","white"))
  marker.x=data.get("x",0.5)
  marker.y=data.get("y",0.5)
  marker.radius=data.get("radius",0.1)
  marker.interaction=data.get("interaction","click")
  marker.cooldown=data.get("cooldown",False)
  return marker
 def ensure_marker(self,name,color,interaction,cooldown):
  if name not in self.markers:
   marker=Marker(name,color)
   marker.interaction=interaction
   marker.cooldown=cooldown
   self.markers[name]=marker
 def draw_markers(self):
  if not self.canvas:
   return
  self.canvas.delete("all")
  width=self.canvas.winfo_width() or self.window.winfo_width()
  height=self.canvas.winfo_height() or self.window.winfo_height()
  for name,marker in self.markers.items():
   x=marker.x*width
   y=marker.y*height
   r=marker.radius*min(width,height)
   self.canvas.create_oval(x-r,y-r,x+r,y+r,outline=marker.color,width=3,fill=self._fill_color(marker))
   self.canvas.create_text(x,y,text=name,fill="white")
 def _fill_color(self,marker):
  if self.selected==marker:
   return "#40ffffff"
  return "#80ffffff"
 def on_click(self,event):
  width=self.canvas.winfo_width()
  height=self.canvas.winfo_height()
  click_x=event.x
  click_y=event.y
  for marker in self.markers.values():
   x=marker.x*width
   y=marker.y*height
   r=marker.radius*min(width,height)
   if (click_x-x)**2+(click_y-y)**2<=r**2:
    self.selected=marker
    self.dragging=True
    self.last_pos=(click_x,click_y)
    boundary=abs((click_x-x)**2+(click_y-y)**2-r**2)
    if boundary<r:
     self.resizing=True
    return
  self.selected=None
  self.dragging=False
 def on_drag(self,event):
  if not self.selected or not self.dragging:
   return
  width=self.canvas.winfo_width()
  height=self.canvas.winfo_height()
  dx=event.x-(self.last_pos[0] if self.last_pos else event.x)
  dy=event.y-(self.last_pos[1] if self.last_pos else event.y)
  if self.resizing:
   r=self.selected.radius*min(width,height)
   r=max(10,min(min(width,height)/2,r+dx))
   self.selected.radius=r/min(width,height)
  else:
   x=self.selected.x*width+dx
   y=self.selected.y*height+dy
   self.selected.x=max(0,min(1,x/width))
   self.selected.y=max(0,min(1,y/height))
  self.last_pos=(event.x,event.y)
  self.draw_markers()
 def on_release(self,event):
  self.dragging=False
  self.resizing=False
  self.last_pos=None
class Mode:
 INIT="初始化"
 LEARNING="学习模式"
 TRAINING="训练模式"
 OPTIMIZING="优化中"
 CONFIG="配置模式"
class MainApp:
 def __init__(self):
  self.root=Tk()
  self.root.title("类脑智能自适应系统")
  self.state_var=StringVar(value=Mode.INIT)
  self.status_var=StringVar(value="待检测")
  self.mode_lock=threading.Lock()
  self.mode=Mode.INIT
  self.progress_var=DoubleVar(value=0)
  self.hero_status_var=StringVar(value="存活")
  self.data_a_var=StringVar(value="0")
  self.data_b_var=StringVar(value="0")
  self.data_c_var=StringVar(value="0")
  self.cooldown_vars={"skills":StringVar(value="冷却状态"),"items":StringVar(value="冷却状态"),"heal":StringVar(value="冷却状态"),"flash":StringVar(value="冷却状态")}
  self.state_lock=threading.Lock()
  self.hero_alive=True
  self.data_a=0
  self.data_b=0
  self.data_c=0
  self.cooldown_state={"skills":"冷却状态","items":"冷却状态","heal":"冷却状态","flash":"冷却状态"}
  self.optimize_button_state=BooleanVar(value=False)
  self.emu_geometry=(0,0,1280,720)
  self.overlay=OverlayManager(self)
  self.default_folder=Path(os.path.expanduser("~"))/"Desktop"/"AAA"
  self.pool=ExperiencePool(self.default_folder)
  self.pool.config_manager.update("aaa_folder",str(self.default_folder))
  self.overlay.load_markers(self.pool.config_manager.data.get("markers",{}))
  self.ensure_default_markers()
  self.resource_monitor=ResourceMonitor()
  self.model_handler=AIModelHandler(self.pool)
  self.stop_event=threading.Event()
  self.user_active=True
  self.last_input=time.time()
  self.training_thread=None
  self.learning_thread=None
  self.create_ui()
  self.root.protocol("WM_DELETE_WINDOW",self.stop)
  self.root.bind("<Escape>",lambda e:self.stop())
  self.check_environment()
  self.input_monitor_thread=threading.Thread(target=self.monitor_input,daemon=True)
  self.input_monitor_thread.start()
  self.scheduler_thread=threading.Thread(target=self.scheduler,daemon=True)
  self.scheduler_thread.start()
 def create_ui(self):
  Label(self.root,textvariable=self.state_var,font=("Microsoft YaHei",20)).grid(row=0,column=0,columnspan=4,sticky="ew")
  Label(self.root,text="状态:").grid(row=1,column=0)
  Label(self.root,textvariable=self.status_var).grid(row=1,column=1,sticky="w")
  Button(self.root,text="优化",command=self.on_optimize).grid(row=2,column=0,sticky="ew")
  Button(self.root,text="取消优化",command=self.on_cancel_optimize).grid(row=2,column=1,sticky="ew")
  Button(self.root,text="配置",command=self.on_configure).grid(row=2,column=2,sticky="ew")
  Button(self.root,text="切换AAA文件夹",command=self.on_change_folder).grid(row=2,column=3,sticky="ew")
  ttk.Progressbar(self.root,variable=self.progress_var,maximum=100).grid(row=3,column=0,columnspan=4,sticky="ew")
  Label(self.root,text="英雄状态:").grid(row=4,column=0)
  Label(self.root,textvariable=self.hero_status_var).grid(row=4,column=1,sticky="w")
  Label(self.root,text="数据A:").grid(row=5,column=0)
  Label(self.root,textvariable=self.data_a_var).grid(row=5,column=1,sticky="w")
  Label(self.root,text="数据B:").grid(row=6,column=0)
  Label(self.root,textvariable=self.data_b_var).grid(row=6,column=1,sticky="w")
  Label(self.root,text="数据C:").grid(row=7,column=0)
  Label(self.root,textvariable=self.data_c_var).grid(row=7,column=1,sticky="w")
  Label(self.root,text="技能冷却:").grid(row=8,column=0)
  Label(self.root,textvariable=self.cooldown_vars["skills"]).grid(row=8,column=1,sticky="w")
  Label(self.root,text="主动装备冷却:").grid(row=9,column=0)
  Label(self.root,textvariable=self.cooldown_vars["items"]).grid(row=9,column=1,sticky="w")
  Label(self.root,text="恢复冷却:").grid(row=10,column=0)
  Label(self.root,textvariable=self.cooldown_vars["heal"]).grid(row=10,column=1,sticky="w")
  Label(self.root,text="闪现冷却:").grid(row=11,column=0)
  Label(self.root,textvariable=self.cooldown_vars["flash"]).grid(row=11,column=1,sticky="w")
  Label(self.root,text="截图频率(Hz):").grid(row=12,column=0)
  self.freq_var=StringVar(value=str(self.resource_monitor.frequency))
  Label(self.root,textvariable=self.freq_var).grid(row=12,column=1,sticky="w")
  Button(self.root,text="保存配置",command=self.save_config).grid(row=13,column=0,columnspan=2,sticky="ew")
  Button(self.root,text="加载配置",command=self.load_config).grid(row=13,column=2,columnspan=2,sticky="ew")
 def set_mode(self,mode):
  with self.mode_lock:
   self.mode=mode
  self.state_var.set(mode)
 def get_mode(self):
  with self.mode_lock:
   return self.mode
 def ensure_default_markers(self):
  specs=[("移动轮盘","red","drag",False),("回城","orange","click",False),("闪现","yellow","drag",True),("恢复","green","click",True),("普攻","blue","click",False),("一技能","indigo","mixed",True),("二技能","indigo","mixed",True),("三技能","indigo","mixed",True),("四技能","indigo","mixed",True),("取消施法","black","drag_in",False),("主动装备","purple","click",True),("数据A","white","observe",False),("数据B","white","observe",False),("数据C","white","observe",False)]
  for name,color,interaction,cooldown in specs:
   self.overlay.ensure_marker(name,color,interaction,cooldown)
 def check_environment(self):
  adb=Path(self.pool.config_manager.data.get("adb_path"))
  emulator=Path(self.pool.config_manager.data.get("emulator_path"))
  models_ready=self.pool.left_model.exists() and self.pool.right_model.exists() and self.pool.vision_model.exists()
  if adb.exists() and emulator.exists() and models_ready:
   self.set_mode(Mode.LEARNING)
   self.start_learning()
  else:
   self.status_var.set("依赖缺失")
 def monitor_input(self):
  while not self.stop_event.is_set():
   time.sleep(0.5)
   if time.time()-self.last_input>self.pool.config_manager.data.get("state_timeout",10) and self.get_mode()==Mode.LEARNING:
    self.root.after(0,self.enter_training_mode)
 def scheduler(self):
  while not self.stop_event.is_set():
   time.sleep(1)
   self.resource_monitor.update_frequency()
   freq=str(self.resource_monitor.frequency)
   self.root.after(0,lambda value=freq:self.freq_var.set(value))
 def start_learning(self):
  if self.learning_thread and self.learning_thread.is_alive():
   return
  self.set_mode(Mode.LEARNING)
  self.status_var.set("采集中")
  self.learning_thread=threading.Thread(target=self.learning_loop,daemon=True)
  self.learning_thread.start()
 def learning_loop(self):
  while not self.stop_event.is_set() and self.get_mode()==Mode.LEARNING:
   self.record_event("user")
   time.sleep(1/max(1,self.resource_monitor.frequency))
 def record_event(self,source):
  with self.state_lock:
   hero_alive=self.hero_alive
   a=self.data_a
   b=self.data_b
   c=self.data_c
   cooldowns=dict(self.cooldown_state)
  data={"timestamp":time.time(),"source":source,"hero_alive":hero_alive,"A":a,"B":b,"C":c,"cooldowns":cooldowns,"geometry":self.emu_geometry,"markers":self.overlay.get_markers_data()}
  self.pool.record(data)
 def enter_training_mode(self):
  if self.get_mode()!=Mode.LEARNING:
   return
  self.set_mode(Mode.TRAINING)
  self.status_var.set("AI执行中")
  if self.learning_thread and self.learning_thread.is_alive():
   self.learning_thread=None
  if self.training_thread and self.training_thread.is_alive():
   return
  self.training_thread=threading.Thread(target=self.training_loop,daemon=True)
  self.training_thread.start()
 def training_loop(self):
  while not self.stop_event.is_set() and self.get_mode()==Mode.TRAINING:
   self.simulate_ai_action()
   self.record_event("ai")
   time.sleep(1/max(1,self.resource_monitor.frequency))
 def update_state(self,a,b,c,alive,skills,items,heal,flash):
  def apply():
   with self.state_lock:
    self.data_a=a
    self.data_b=b
    self.data_c=c
    self.hero_alive=alive
    self.cooldown_state["skills"]=skills
    self.cooldown_state["items"]=items
    self.cooldown_state["heal"]=heal
    self.cooldown_state["flash"]=flash
   self.data_a_var.set(str(a))
   self.data_b_var.set(str(b))
   self.data_c_var.set(str(c))
   self.hero_status_var.set("存活" if alive else "阵亡")
   self.cooldown_vars["skills"].set(skills)
   self.cooldown_vars["items"].set(items)
   self.cooldown_vars["heal"].set(heal)
   self.cooldown_vars["flash"].set(flash)
  self.root.after(0,apply)
 def simulate_ai_action(self):
  a=random.randint(0,100)
  b=random.randint(0,100)
  c=random.randint(0,100)
  alive=random.random()>=0.05
  skills="冷却" if random.random()<0.5 else "可用"
  items="冷却" if random.random()<0.5 else "可用"
  heal="冷却" if random.random()<0.5 else "可用"
  flash="冷却" if random.random()<0.5 else "可用"
  self.update_state(a,b,c,alive,skills,items,heal,flash)
  if not alive:
   time.sleep(2)
 def on_optimize(self):
  if self.get_mode() not in [Mode.LEARNING,Mode.TRAINING]:
   return
  self.set_mode(Mode.OPTIMIZING)
  self.status_var.set("优化中")
  def callback(progress):
   self.root.after(0,lambda:self.progress_var.set(progress))
  def done(success):
   def finish():
    self.progress_var.set(0 if not success else 100)
    if success:
     self.adjust_markers()
     showinfo("提示","优化完成")
    self.set_mode(Mode.LEARNING)
    self.start_learning()
   self.root.after(0,finish)
  self.model_handler.optimize(callback,done)
 def adjust_markers(self):
  for marker in self.overlay.markers.values():
   marker.x=min(0.95,max(0.05,marker.x+random.uniform(-0.02,0.02)))
   marker.y=min(0.95,max(0.05,marker.y+random.uniform(-0.02,0.02)))
   marker.radius=min(0.4,max(0.05,marker.radius+random.uniform(-0.01,0.01)))
  self.save_markers()
 def save_markers(self):
  data=self.overlay.get_markers_data()
  self.pool.config_manager.update("markers",data)
  self.pool.config_manager.save()
 def on_cancel_optimize(self):
  self.model_handler.cancel()
  self.set_mode(Mode.LEARNING)
  self.status_var.set("采集中")
  self.start_learning()
 def on_configure(self):
  if self.get_mode()!=Mode.LEARNING:
   return
  self.set_mode(Mode.CONFIG)
  self.status_var.set("配置中")
  self.overlay.open()
  config_window=Toplevel(self.root)
  config_window.title("标志管理")
  list_var=StringVar(value="")
  def refresh_list():
   list_var.set("\n".join(self.overlay.markers.keys()))
  Label(config_window,textvariable=list_var,justify="left").pack()
  def add_marker():
   name=f"标志{len(self.overlay.markers)+1}"
   marker=Marker(name,"white")
   self.overlay.markers[name]=marker
   self.overlay.draw_markers()
   refresh_list()
  Button(config_window,text="添加标志",command=add_marker).pack(fill="x")
  def save_and_close():
   self.save_markers()
   showinfo("提示","配置已保存")
   self.overlay.close()
   config_window.destroy()
   self.set_mode(Mode.LEARNING)
   self.start_learning()
  Button(config_window,text="保存",command=save_and_close).pack(fill="x")
  refresh_list()
 def on_change_folder(self):
  new_dir=askdirectory()
  if not new_dir:
   return
  try:
   self.pool.migrate(new_dir)
   self.overlay.load_markers(self.pool.config_manager.data.get("markers",{}))
   self.ensure_default_markers()
   self.pool.config_manager.update("aaa_folder",str(self.pool.folder))
   self.save_markers()
   self.status_var.set("已迁移")
  except Exception as e:
   showinfo("错误",str(e))
 def save_config(self):
  data=self.pool.config_manager.data
  data.update({"screenshot_hz":self.resource_monitor.frequency,"markers":self.overlay.get_markers_data(),"aaa_folder":str(self.pool.folder)})
  self.pool.config_manager.save()
  showinfo("提示","配置已保存")
 def load_config(self):
  self.pool.config_manager.load()
  self.overlay.load_markers(self.pool.config_manager.data.get("markers",{}))
  self.ensure_default_markers()
  self.status_var.set("配置已加载")
 def get_emulator_geometry(self):
  return self.emu_geometry[2],self.emu_geometry[3]
 def stop(self):
  self.stop_event.set()
  self.root.quit()
 def monitor_user_action(self,event=None):
  self.last_input=time.time()
  if self.get_mode()==Mode.TRAINING:
   self.set_mode(Mode.LEARNING)
   self.status_var.set("采集中")
   self.start_learning()
 def scheduler_event_bindings(self):
  self.root.bind_all("<Key>",self.monitor_user_action)
  self.root.bind_all("<Button>",self.monitor_user_action)
 def run(self):
  self.scheduler_event_bindings()
  self.root.mainloop()
app=None
if __name__=="__main__":
 app=MainApp()
 app.run()
