import os, subprocess, io, threading, time, random, math, platform
from collections import deque, namedtuple
try:
    import psutil
except:
    psutil=None
try:
    import GPUtil
except:
    GPUtil=None
import numpy as np
from PIL import Image
import pytesseract
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk

class SharedState:
    def __init__(self):
        self.lock=threading.Lock()
        self.kills=0
        self.deaths=0
        self.assists=0
        self.alive=0
        self.cd1=0
        self.cd2=0
        self.cd3=0
        self.cd_item=0
        self.running=False
        self.epsilon=1.0
        self.device="cpu"
        self.hw_profile={}

def hardware_profile():
    profile={}
    profile["platform"]=platform.platform()
    if 'psutil' in globals() and psutil:
        profile["cpu_count"]=psutil.cpu_count(logical=True)
        profile["memory_gb"]=round(psutil.virtual_memory().total/(1024**3),2)
    else:
        profile["cpu_count"]=os.cpu_count()
        profile["memory_gb"]=None
    gpus=[]
    if 'GPUtil' in globals() and GPUtil:
        try:
            g_list=GPUtil.getGPUs()
            for g in g_list:
                gpus.append({"name":g.name,"memory_total_mb":g.memoryTotal})
        except:
            gpus=[]
    profile["gpus"]=gpus
    if torch.cuda.is_available():
        profile["cuda_gpu"]=torch.cuda.get_device_name(0)
        profile["cuda_total_mem_mb"]=round(torch.cuda.get_device_properties(0).total_memory/(1024**2),2)
    else:
        profile["cuda_gpu"]=None
        profile["cuda_total_mem_mb"]=None
    return profile

class QNetwork(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim):
        super().__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class DeepRLAgent:
    def __init__(self,device,hidden_dim=128,buffer_size=10000,batch_size=64,gamma=0.99,lr=1e-4,target_sync=1000):
        self.device=device
        self.obs_dim=8
        self.action_dim=11
        self.policy_net=QNetwork(self.obs_dim,self.action_dim,hidden_dim).to(self.device)
        self.target_net=QNetwork(self.obs_dim,self.action_dim,hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=lr)
        self.replay_buffer=deque(maxlen=buffer_size)
        self.batch_size=batch_size
        self.gamma=gamma
        self.step_count=0
        self.target_sync=target_sync
        self.Transition=namedtuple("Transition",("state","action","reward","next_state","done"))
        self.epsilon_start=1.0
        self.epsilon_end=0.1
        self.epsilon_decay=50000
    def epsilon(self):
        return self.epsilon_end+(self.epsilon_start-self.epsilon_end)*math.exp(-1.0*self.step_count/self.epsilon_decay)
    def select_action(self,obs_vec):
        eps=self.epsilon()
        if random.random()<eps:
            return random.randrange(self.action_dim),eps
        with torch.no_grad():
            t=torch.tensor(obs_vec,dtype=torch.float32,device=self.device).unsqueeze(0)
            q=self.policy_net(t)
            a=int(torch.argmax(q,dim=1).item())
            return a,eps
    def store(self,state,action,reward,next_state,done):
        self.replay_buffer.append(self.Transition(state,action,reward,next_state,done))
    def train_step(self):
        if len(self.replay_buffer)<self.batch_size:
            return
        batch=random.sample(self.replay_buffer,self.batch_size)
        state_batch=torch.tensor([b.state for b in batch],dtype=torch.float32,device=self.device)
        action_batch=torch.tensor([b.action for b in batch],dtype=torch.int64,device=self.device).unsqueeze(1)
        reward_batch=torch.tensor([b.reward for b in batch],dtype=torch.float32,device=self.device).unsqueeze(1)
        next_state_batch=torch.tensor([b.next_state for b in batch],dtype=torch.float32,device=self.device)
        done_batch=torch.tensor([b.done for b in batch],dtype=torch.float32,device=self.device).unsqueeze(1)
        q_values=self.policy_net(state_batch).gather(1,action_batch)
        with torch.no_grad():
            next_actions=torch.argmax(self.policy_net(next_state_batch),dim=1,keepdim=True)
            next_q=self.target_net(next_state_batch).gather(1,next_actions)
            target_q=reward_batch+self.gamma*(1.0-done_batch)*next_q
        loss=nn.functional.smooth_l1_loss(q_values,target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),10.0)
        self.optimizer.step()
        self.step_count+=1
        if self.step_count%self.target_sync==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class VisionModule:
    def __init__(self):
        pass
    def crop_cv(self,pil_img,x,y,w,h,scale_x,scale_y):
        sx=int(x*scale_x)
        sy=int(y*scale_y)
        sw=int(w*scale_x)
        sh=int(h*scale_y)
        sx=max(0,sx)
        sy=max(0,sy)
        ex=min(pil_img.width,sx+sw)
        ey=min(pil_img.height,sy+sh)
        crop_pil=pil_img.crop((sx,sy,ex,ey))
        return cv2.cvtColor(np.array(crop_pil),cv2.COLOR_RGB2BGR)
    def region_brightness(self,cv_img):
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    def ocr_digits(self,cv_img):
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        txt=pytesseract.image_to_string(th,config="--psm 7 -c tessedit_char_whitelist=0123456789")
        digits="".join([c for c in txt if c.isdigit()])
        if digits=="":
            return 0
        try:
            return int(digits)
        except:
            return 0

class GameInterface:
    def __init__(self,adb_path,base_w=2560,base_h=1600):
        self.adb_path=adb_path
        self.base_w=base_w
        self.base_h=base_h
        self.screen_w=base_w
        self.screen_h=base_h
        self.vision=VisionModule()
        self.last_screenshot=None
        self.coords={"joystick":(166,915,536),"recall":(1083,1263,162),"heal":(1271,1263,162),"flash":(1467,1263,162),"skill1":(1672,1220,195),"skill2":(1825,956,195),"skill3":(2088,803,195),"cancel":(2165,252,250),"attack_min":(1915,1296,123),"attack_tower":(2241,1014,123),"active_item":(2092,544,161),"kills_box":(1904,122,56,56),"deaths_box":(1996,122,56,56),"assists_box":(2087,122,56,56),"minimap_box":(0,72,453,453)}
        self.prev_metrics={"kills":0,"deaths":0,"assists":0,"alive":0,"cd1":0,"cd2":0,"cd3":0,"cd_item":0}
    def adb_tap(self,x,y):
        try:
            subprocess.call([self.adb_path,"shell","input","tap",str(int(x)),str(int(y))])
        except:
            pass
    def adb_swipe(self,x1,y1,x2,y2,duration_ms):
        try:
            subprocess.call([self.adb_path,"shell","input","swipe",str(int(x1)),str(int(y1)),str(int(x2)),str(int(y2)),str(int(duration_ms))])
        except:
            pass
    def get_screenshot(self):
        try:
            png_bytes=subprocess.check_output([self.adb_path,"exec-out","screencap","-p"],timeout=5)
            img=Image.open(io.BytesIO(png_bytes)).convert("RGB")
            self.last_screenshot=img
            self.screen_w,self.screen_h=img.size
            return img
        except:
            return self.last_screenshot
    def scaled_xy(self,x,y):
        sx=int(x*self.screen_w/self.base_w)
        sy=int(y*self.screen_h/self.base_h)
        return sx,sy
    def circle_center(self,tlx,tly,d):
        return tlx+d/2.0,tly+d/2.0
    def tap_circle(self,tlx,tly,d):
        cx,cy=self.circle_center(tlx,tly,d)
        sx,sy=self.scaled_xy(cx,cy)
        self.adb_tap(sx,sy)
    def random_move(self):
        tlx,tly,d=self.coords["joystick"]
        cx,cy=self.circle_center(tlx,tly,d)
        r=d/2.0*0.8
        ang=random.random()*2.0*math.pi
        ex=cx+r*math.cos(ang)
        ey=cy+r*math.sin(ang)
        sx,sy=self.scaled_xy(cx,cy)
        exs,eys=self.scaled_xy(ex,ey)
        self.adb_swipe(sx,sy,exs,eys,200)
    def attack_minion(self):
        tlx,tly,d=self.coords["attack_min"]
        self.tap_circle(tlx,tly,d)
    def attack_tower(self):
        tlx,tly,d=self.coords["attack_tower"]
        self.tap_circle(tlx,tly,d)
    def cast_skill(self,key):
        tlx,tly,d=self.coords[key]
        self.tap_circle(tlx,tly,d)
    def use_item(self):
        tlx,tly,d=self.coords["active_item"]
        self.tap_circle(tlx,tly,d)
    def heal(self):
        tlx,tly,d=self.coords["heal"]
        self.tap_circle(tlx,tly,d)
    def recall(self):
        tlx,tly,d=self.coords["recall"]
        self.tap_circle(tlx,tly,d)
    def flash_forward(self):
        tlx,tly,d=self.coords["flash"]
        cx,cy=self.circle_center(tlx,tly,d)
        ex=cx+150.0
        ey=cy
        sx,sy=self.scaled_xy(cx,cy)
        exs,eys=self.scaled_xy(ex,ey)
        self.adb_swipe(sx,sy,exs,eys,50)
    def get_metrics(self):
        img=self.get_screenshot()
        if img is None:
            return self.prev_metrics
        w,h=img.size
        scale_x=w/self.base_w
        scale_y=h/self.base_h
        v=self.vision
        def crop_circle_box(coord_key):
            tlx,tly,d=self.coords[coord_key]
            return v.crop_cv(img,tlx,tly,d,d,scale_x,scale_y)
        def crop_rect_box(box_key):
            tlx,tly,w_,h_=self.coords[box_key]
            return v.crop_cv(img,tlx,tly,w_,h_,scale_x,scale_y)
        kills_img=crop_rect_box("kills_box")
        deaths_img=crop_rect_box("deaths_box")
        assists_img=crop_rect_box("assists_box")
        skill1_img=crop_circle_box("skill1")
        skill2_img=crop_circle_box("skill2")
        skill3_img=crop_circle_box("skill3")
        item_img=crop_circle_box("active_item")
        attack_img=crop_circle_box("attack_min")
        kills_val=v.ocr_digits(kills_img)
        deaths_val=v.ocr_digits(deaths_img)
        assists_val=v.ocr_digits(assists_img)
        b1=v.region_brightness(skill1_img)
        b2=v.region_brightness(skill2_img)
        b3=v.region_brightness(skill3_img)
        bitem=v.region_brightness(item_img)
        balive=v.region_brightness(attack_img)
        th_skill=80.0
        th_alive=50.0
        cd1=1 if b1>th_skill else 0
        cd2=1 if b2>th_skill else 0
        cd3=1 if b3>th_skill else 0
        cd_item=1 if bitem>th_skill else 0
        alive=1 if balive>th_alive else 0
        metrics={"kills":kills_val,"deaths":deaths_val,"assists":assists_val,"alive":alive,"cd1":cd1,"cd2":cd2,"cd3":cd3,"cd_item":cd_item}
        self.prev_metrics=metrics
        return metrics
    def metrics_to_obs(self,m):
        kills=min(m["kills"],20)/20.0
        deaths=min(m["deaths"],20)/20.0
        assists=min(m["assists"],20)/20.0
        alive=float(m["alive"])
        cd1=float(m["cd1"])
        cd2=float(m["cd2"])
        cd3=float(m["cd3"])
        cd_item=float(m["cd_item"])
        obs=[kills,deaths,assists,alive,cd1,cd2,cd3,cd_item]
        return obs
    def compute_reward(self,prev_metrics,curr_metrics):
        dk=curr_metrics["kills"]-prev_metrics["kills"]
        da=curr_metrics["assists"]-prev_metrics["assists"]
        dd=curr_metrics["deaths"]-prev_metrics["deaths"]
        r=2.0*dk+1.0*da-3.0*dd
        return float(r)
    def execute_action(self,action_idx,metrics_prev):
        if action_idx==0:
            return
        if action_idx==1:
            if metrics_prev["alive"]==1:
                self.random_move()
            return
        if action_idx==2:
            if metrics_prev["alive"]==1:
                self.attack_minion()
            return
        if action_idx==3:
            if metrics_prev["alive"]==1:
                self.attack_tower()
            return
        if action_idx==4:
            if metrics_prev["alive"]==1 and metrics_prev["cd1"]==1:
                self.cast_skill("skill1")
            return
        if action_idx==5:
            if metrics_prev["alive"]==1 and metrics_prev["cd2"]==1:
                self.cast_skill("skill2")
            return
        if action_idx==6:
            if metrics_prev["alive"]==1 and metrics_prev["cd3"]==1:
                self.cast_skill("skill3")
            return
        if action_idx==7:
            if metrics_prev["alive"]==1 and metrics_prev["cd_item"]==1:
                self.use_item()
            return
        if action_idx==8:
            if metrics_prev["alive"]==1:
                self.heal()
            return
        if action_idx==9:
            if metrics_prev["alive"]==1:
                self.recall()
            return
        if action_idx==10:
            if metrics_prev["alive"]==1:
                self.flash_forward()
            return

class BotController:
    def __init__(self,shared,agent,adb_path="D:\\LDPlayer9\\adb.exe",dnplayer_path="D:\\LDPlayer9\\dnplayer.exe"):
        self.shared=shared
        self.agent=agent
        self.adb_path=adb_path
        self.dnplayer_path=dnplayer_path
        self.game=GameInterface(self.adb_path)
        self.running_event=threading.Event()
        self.thread=None
    def start(self):
        if self.running_event.is_set():
            return
        self.running_event.set()
        self.thread=threading.Thread(target=self.loop,daemon=True)
        self.thread.start()
    def stop(self):
        self.running_event.clear()
        with self.shared.lock:
            self.shared.running=False
    def loop(self):
        metrics_prev=self.game.get_metrics()
        obs_prev=self.game.metrics_to_obs(metrics_prev)
        while self.running_event.is_set():
            action,eps=self.agent.select_action(obs_prev)
            self.game.execute_action(action,metrics_prev)
            time.sleep(0.2)
            metrics_curr=self.game.get_metrics()
            obs_curr=self.game.metrics_to_obs(metrics_curr)
            reward=self.game.compute_reward(metrics_prev,metrics_curr)
            self.agent.store(obs_prev,action,reward,obs_curr,False)
            self.agent.train_step()
            with self.shared.lock:
                self.shared.kills=metrics_curr["kills"]
                self.shared.deaths=metrics_curr["deaths"]
                self.shared.assists=metrics_curr["assists"]
                self.shared.alive=metrics_curr["alive"]
                self.shared.cd1=metrics_curr["cd1"]
                self.shared.cd2=metrics_curr["cd2"]
                self.shared.cd3=metrics_curr["cd3"]
                self.shared.cd_item=metrics_curr["cd_item"]
                self.shared.epsilon=eps
                self.shared.running=True
            metrics_prev=metrics_curr
            obs_prev=obs_curr

class BotUI:
    def __init__(self,shared,controller):
        self.shared=shared
        self.controller=controller
        self.root=tk.Tk()
        self.root.title("Honor of Kings Deep RL Bot")
        self.vars={}
        self.vars["kills"]=tk.StringVar()
        self.vars["deaths"]=tk.StringVar()
        self.vars["assists"]=tk.StringVar()
        self.vars["alive"]=tk.StringVar()
        self.vars["cd1"]=tk.StringVar()
        self.vars["cd2"]=tk.StringVar()
        self.vars["cd3"]=tk.StringVar()
        self.vars["cd_item"]=tk.StringVar()
        self.vars["epsilon"]=tk.StringVar()
        self.vars["status"]=tk.StringVar()
        self.vars["device"]=tk.StringVar()
        self.vars["hw"]=tk.StringVar()
        frame_state=tk.Frame(self.root)
        frame_state.pack(side="top",fill="x")
        tk.Label(frame_state,text="Kills").grid(row=0,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["kills"]).grid(row=0,column=1,sticky="w")
        tk.Label(frame_state,text="Deaths").grid(row=1,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["deaths"]).grid(row=1,column=1,sticky="w")
        tk.Label(frame_state,text="Assists").grid(row=2,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["assists"]).grid(row=2,column=1,sticky="w")
        tk.Label(frame_state,text="Alive").grid(row=3,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["alive"]).grid(row=3,column=1,sticky="w")
        tk.Label(frame_state,text="Skill1 Ready").grid(row=4,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd1"]).grid(row=4,column=1,sticky="w")
        tk.Label(frame_state,text="Skill2 Ready").grid(row=5,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd2"]).grid(row=5,column=1,sticky="w")
        tk.Label(frame_state,text="Skill3 Ready").grid(row=6,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd3"]).grid(row=6,column=1,sticky="w")
        tk.Label(frame_state,text="Item Ready").grid(row=7,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd_item"]).grid(row=7,column=1,sticky="w")
        tk.Label(frame_state,text="ε").grid(row=8,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["epsilon"]).grid(row=8,column=1,sticky="w")
        tk.Label(frame_state,text="Status").grid(row=9,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["status"]).grid(row=9,column=1,sticky="w")
        control_frame=tk.Frame(self.root)
        control_frame.pack(side="top",fill="x")
        tk.Button(control_frame,text="Start AI",command=self.start_bot).grid(row=0,column=0,sticky="we")
        tk.Button(control_frame,text="Stop AI",command=self.stop_bot).grid(row=0,column=1,sticky="we")
        hw_frame=tk.Frame(self.root)
        hw_frame.pack(side="top",fill="x")
        tk.Label(hw_frame,text="Device").grid(row=0,column=0,sticky="w")
        tk.Label(hw_frame,textvariable=self.vars["device"]).grid(row=0,column=1,sticky="w")
        tk.Label(hw_frame,text="Hardware").grid(row=1,column=0,sticky="w")
        tk.Label(hw_frame,textvariable=self.vars["hw"]).grid(row=1,column=1,sticky="w")
        self.update_loop()
    def start_bot(self):
        self.controller.start()
    def stop_bot(self):
        self.controller.stop()
    def update_loop(self):
        with self.shared.lock:
            self.vars["kills"].set(str(self.shared.kills))
            self.vars["deaths"].set(str(self.shared.deaths))
            self.vars["assists"].set(str(self.shared.assists))
            self.vars["alive"].set("Yes" if self.shared.alive==1 else "No")
            self.vars["cd1"].set("Ready" if self.shared.cd1==1 else "CD")
            self.vars["cd2"].set("Ready" if self.shared.cd2==1 else "CD")
            self.vars["cd3"].set("Ready" if self.shared.cd3==1 else "CD")
            self.vars["cd_item"].set("Ready" if self.shared.cd_item==1 else "CD")
            self.vars["epsilon"].set("{:.3f}".format(self.shared.epsilon))
            self.vars["status"].set("Running" if self.shared.running else "Stopped")
            self.vars["device"].set(self.shared.device)
            hw_text=""
            if "cpu_count" in self.shared.hw_profile:
                hw_text+="CPU:"+str(self.shared.hw_profile["cpu_count"])+" "
            if "memory_gb" in self.shared.hw_profile and self.shared.hw_profile["memory_gb"] is not None:
                hw_text+="RAM:"+str(self.shared.hw_profile["memory_gb"])+"GB "
            if "cuda_gpu" in self.shared.hw_profile and self.shared.hw_profile["cuda_gpu"]:
                hw_text+="GPU:"+self.shared.hw_profile["cuda_gpu"]
            self.vars["hw"].set(hw_text)
        self.root.after(500,self.update_loop)
    def run(self):
        self.root.mainloop()

def main():
    shared=SharedState()
    hw=hardware_profile()
    device="cuda" if torch.cuda.is_available() else "cpu"
    shared.device=device
    shared.hw_profile=hw
    hidden_dim=128 if device=="cuda" else 64
    agent=DeepRLAgent(device,hidden_dim=hidden_dim)
    controller=BotController(shared,agent)
    ui=BotUI(shared,controller)
    ui.run()

if __name__=="__main__":
    main()
