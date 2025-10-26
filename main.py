import os,subprocess,io,threading,time,random,math,platform,json
from collections import deque,namedtuple
from dataclasses import dataclass
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
        self.A=0
        self.B=0
        self.C=0
        self.alive=0
        self.cd1=0
        self.cd2=0
        self.cd3=0
        self.cd_item=0
        self.running=False
        self.epsilon=1.0
        self.device="cpu"
        self.hw_profile={}
        self.adb_path="D:\\LDPlayer9\\adb.exe"
        self.dnplayer_path="D:\\LDPlayer9\\dnplayer.exe"
        self.aaa_dir=os.path.join(os.path.expanduser("~"),"Desktop","AAA")
        self.reward_weights={"kills":3.0,"assists":2.0,"deaths":-4.0}
        self.base_dir=""
class AIModelHub:
    def __init__(self,aaa_dir,device,dtype):
        self.aaa_dir=aaa_dir
        self.device=device
        self.dtype=dtype
        self.skill1_model=None
        self.skill2_model=None
        self.skill3_model=None
        self.move_model=None
        self.attack_model=None
        self.recall_model=None
        self.vision_model=None
        self.models_raw={}
        self.load_models()
    def safe_load(self,path):
        try:
            obj=torch.load(path,map_location=self.device)
            if isinstance(obj,nn.Module):
                try:
                    obj.to(self.device)
                    try:
                        obj.to(self.dtype)
                    except:
                        pass
                    obj.eval()
                except:
                    pass
            return obj
        except:
            return None
    def load_models(self):
        if not os.path.isdir(self.aaa_dir):
            return
        m1=os.path.join(self.aaa_dir,"skill1_model.pt")
        m2=os.path.join(self.aaa_dir,"skill2_model.pt")
        m3=os.path.join(self.aaa_dir,"skill3_model.pt")
        mm=os.path.join(self.aaa_dir,"move_model.pt")
        ma=os.path.join(self.aaa_dir,"attack_model.pt")
        mr=os.path.join(self.aaa_dir,"recall_model.pt")
        mv=os.path.join(self.aaa_dir,"vision_model.pt")
        if os.path.isfile(m1):
            self.skill1_model=self.safe_load(m1)
        if os.path.isfile(m2):
            self.skill2_model=self.safe_load(m2)
        if os.path.isfile(m3):
            self.skill3_model=self.safe_load(m3)
        if os.path.isfile(mm):
            self.move_model=self.safe_load(mm)
        if os.path.isfile(ma):
            self.attack_model=self.safe_load(ma)
        if os.path.isfile(mr):
            self.recall_model=self.safe_load(mr)
        if os.path.isfile(mv):
            self.vision_model=self.safe_load(mv)
        for f in glob.glob(os.path.join(self.aaa_dir,"*.pt"))+glob.glob(os.path.join(self.aaa_dir,"*.pth")):
            k=os.path.basename(f)
            if k not in self.models_raw:
                self.models_raw[k]=self.safe_load(f)

class QNetwork(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,dtype):
        super().__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.bn1=nn.LayerNorm(hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.bn2=nn.LayerNorm(hidden_dim)
        self.fc3=nn.Linear(hidden_dim,output_dim)
        self.dtype_used=dtype
        self.to(dtype)
    def forward(self,x):
        x=x.to(self.dtype_used)
        x=torch.relu(self.bn1(self.fc1(x)))
        x=torch.relu(self.bn2(self.fc2(x)))
        x=self.fc3(x)
        return x

class DeepRLAgent:
    def __init__(self,device,dtype,hidden_dim=128,buffer_size=10000,batch_size=64,gamma=0.99,lr=1e-4,target_sync=1000,epsilon_decay=50000,experience_dir=None):
        self.device=device
        self.dtype=dtype
        self.obs_dim=8
        self.action_dim=11
        self.policy_net=QNetwork(self.obs_dim,self.action_dim,hidden_dim,self.dtype).to(self.device)
        self.target_net=QNetwork(self.obs_dim,self.action_dim,hidden_dim,self.dtype).to(self.device)
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
        self.epsilon_decay=epsilon_decay
        self.experience_dir=experience_dir
        self.last_save_step=0
        self.load_experience()
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
            self.step_count+=1
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
        loss=nn.functional.smooth_l1_loss(q_values.to(target_q.dtype),target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),10.0)
        self.optimizer.step()
        self.step_count+=1
        if self.step_count%self.target_sync==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.experience_dir is not None and self.step_count-self.last_save_step>=500:
            self.save_experience()
            self.last_save_step=self.step_count
    def save_experience(self):
        try:
            if not os.path.isdir(self.experience_dir):
                os.makedirs(self.experience_dir,exist_ok=True)
            buf_states=[]
            buf_actions=[]
            buf_rewards=[]
            buf_next=[]
            buf_done=[]
            for t in self.replay_buffer:
                buf_states.append(t.state)
                buf_actions.append(t.action)
                buf_rewards.append(t.reward)
                buf_next.append(t.next_state)
                buf_done.append(t.done)
            data={"state":np.array(buf_states,dtype=np.float32),"action":np.array(buf_actions,dtype=np.int64),"reward":np.array(buf_rewards,dtype=np.float32),"next_state":np.array(buf_next,dtype=np.float32),"done":np.array(buf_done,dtype=np.int8),"step":self.step_count}
            path=os.path.join(self.experience_dir,"buffer.npz")
            np.savez_compressed(path,**data)
            weights_path=os.path.join(self.experience_dir,"policy.pt")
            torch.save(self.policy_net.state_dict(),weights_path)
        except:
            pass
    def load_experience(self):
        try:
            if self.experience_dir is None:
                return
            path=os.path.join(self.experience_dir,"buffer.npz")
            if os.path.isfile(path):
                loaded=np.load(path,allow_pickle=True)
                st=loaded["state"]
                ac=loaded["action"]
                rw=loaded["reward"]
                ns=loaded["next_state"]
                dn=loaded["done"]
                for i in range(len(st)):
                    self.replay_buffer.append(self.Transition(st[i].tolist(),int(ac[i]),float(rw[i]),ns[i].tolist(),bool(dn[i])))
            weights_path=os.path.join(self.experience_dir,"policy.pt")
            if os.path.isfile(weights_path):
                sd=torch.load(weights_path,map_location=self.device)
                self.policy_net.load_state_dict(sd,strict=False)
                self.target_net.load_state_dict(self.policy_net.state_dict())
        except:
            pass

@dataclass
class CircleRegion:
    x:int
    y:int
    d:int

@dataclass
class RectRegion:
    x:int
    y:int
    w:int
    h:int

class VisionModule:
    def __init__(self,aihub):
        self.prev_ready={}
        self.aihub=aihub
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
    def region_stats(self,cv_img):
        hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2HSV)
        mean_v=float(np.mean(hsv[:,:,2]))
        mean_s=float(np.mean(hsv[:,:,1]))
        std_v=float(np.std(hsv[:,:,2]))
        return mean_v,mean_s,std_v
    def cooldown_ready(self,key,cv_img):
        mean_v,mean_s,std_v=self.region_stats(cv_img)
        bright=mean_v>110
        saturated=mean_s>60
        stable=std_v<55
        ready=1 if (bright and saturated and stable) else 0
        if key in self.prev_ready:
            ready=max(ready,self.prev_ready[key]*0.5)
        self.prev_ready[key]=ready
        return ready
    def alive_state(self,cv_img):
        mean_v,mean_s,std_v=self.region_stats(cv_img)
        return 1 if (mean_v>90 and mean_s>50) else 0
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
    def __init__(self,adb_path,base_w=2560,base_h=1600,aihub=None):
        self.adb_path=adb_path
        self.base_w=base_w
        self.base_h=base_h
        self.screen_w=base_w
        self.screen_h=base_h
        self.vision=VisionModule(aihub)
        self.last_screenshot=None
        self.coords={
            "joystick":CircleRegion(166,915,536),
            "recall":CircleRegion(1083,1263,162),
            "heal":CircleRegion(1271,1263,162),
            "flash":CircleRegion(1467,1263,162),
            "skill1":CircleRegion(1672,1220,195),
            "skill2":CircleRegion(1825,956,195),
            "skill3":CircleRegion(2088,803,195),
            "cancel":CircleRegion(2165,252,250),
            "attack_min":CircleRegion(1915,1296,123),
            "attack_tower":CircleRegion(2241,1014,123),
            "active_item":CircleRegion(2092,544,161),
            "A_box":RectRegion(1904,122,56,56),
            "B_box":RectRegion(1996,122,56,56),
            "C_box":RectRegion(2087,122,56,56),
            "minimap_box":RectRegion(0,72,453,453)
        }
        self.prev_metrics={"A":0,"B":0,"C":0,"alive":0,"cd1":0,"cd2":0,"cd3":0,"cd_item":0}
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
    def joystick_move_random(self):
        c=self.coords["joystick"]
        cx,cy=self.circle_center(c.x,c.y,c.d)
        r=c.d/2.0*0.8
        ang=random.random()*2.0*math.pi
        ex=cx+r*math.cos(ang)
        ey=cy+r*math.sin(ang)
        sx,sy=self.scaled_xy(cx,cy)
        exs,eys=self.scaled_xy(ex,ey)
        self.adb_swipe(sx,sy,exs,eys,200)
    def basic_attack_minion(self):
        c=self.coords["attack_min"]
        self.tap_circle(c.x,c.y,c.d)
    def basic_attack_tower(self):
        c=self.coords["attack_tower"]
        self.tap_circle(c.x,c.y,c.d)
    def cast_skill(self,key):
        c=self.coords[key]
        self.tap_circle(c.x,c.y,c.d)
    def use_item(self):
        c=self.coords["active_item"]
        self.tap_circle(c.x,c.y,c.d)
    def heal(self):
        c=self.coords["heal"]
        self.tap_circle(c.x,c.y,c.d)
    def recall(self):
        c=self.coords["recall"]
        self.tap_circle(c.x,c.y,c.d)
        time.sleep(8.0)
    def flash_forward(self):
        c=self.coords["flash"]
        cx,cy=self.circle_center(c.x,c.y,c.d)
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
            c=self.coords[coord_key]
            return v.crop_cv(img,c.x,c.y,c.d,c.d,scale_x,scale_y)
        def crop_rect_box(box_key):
            r=self.coords[box_key]
            return v.crop_cv(img,r.x,r.y,r.w,r.h,scale_x,scale_y)
        A_img=crop_rect_box("A_box")
        B_img=crop_rect_box("B_box")
        C_img=crop_rect_box("C_box")
        skill1_img=crop_circle_box("skill1")
        skill2_img=crop_circle_box("skill2")
        skill3_img=crop_circle_box("skill3")
        item_img=crop_circle_box("active_item")
        attack_img=crop_circle_box("attack_min")
        A_val=v.ocr_digits(A_img)
        B_val=v.ocr_digits(B_img)
        C_val=v.ocr_digits(C_img)
        cd1=v.cooldown_ready("skill1",skill1_img)
        cd2=v.cooldown_ready("skill2",skill2_img)
        cd3=v.cooldown_ready("skill3",skill3_img)
        cd_item=v.cooldown_ready("item",item_img)
        alive=v.alive_state(attack_img)
        metrics={"A":A_val,"B":B_val,"C":C_val,"alive":alive,"cd1":cd1,"cd2":cd2,"cd3":cd3,"cd_item":cd_item}
        self.prev_metrics=metrics
        return metrics
    def metrics_to_obs(self,m):
        A_norm=min(m["A"],20)/20.0
        B_norm=min(m["B"],20)/20.0
        C_norm=min(m["C"],20)/20.0
        alive=float(m["alive"])
        cd1=float(m["cd1"])
        cd2=float(m["cd2"])
        cd3=float(m["cd3"])
        cd_item=float(m["cd_item"])
        obs=[A_norm,B_norm,C_norm,alive,cd1,cd2,cd3,cd_item]
        return obs
    def compute_reward(self,prev_metrics,curr_metrics):
        dA=curr_metrics["A"]-prev_metrics["A"]
        dC=curr_metrics["C"]-prev_metrics["C"]
        dB=curr_metrics["B"]-prev_metrics["B"]
        r=3.0*dA+2.0*dC-4.0*dB
        return float(r)
    def execute_action(self,action_idx,metrics_prev):
        if action_idx==0:
            return
        if action_idx==1:
            if metrics_prev["alive"]==1:
                self.joystick_move_random()
            return
        if action_idx==2:
            if metrics_prev["alive"]==1:
                self.basic_attack_minion()
            return
        if action_idx==3:
            if metrics_prev["alive"]==1:
                self.basic_attack_tower()
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
    def __init__(self,shared,agent,aihub):
        self.shared=shared
        self.agent=agent
        self.aihub=aihub
        self.adb_path=shared.adb_path
        self.dnplayer_path=shared.dnplayer_path
        self.game=GameInterface(self.adb_path,aihub=self.aihub)
        self.running_event=threading.Event()
        self.thread=None
    def update_paths(self,adb_path,dnplayer_path):
        self.adb_path=adb_path
        self.dnplayer_path=dnplayer_path
        self.game=GameInterface(self.adb_path,aihub=self.aihub)
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
                self.shared.A=metrics_curr["A"]
                self.shared.B=metrics_curr["B"]
                self.shared.C=metrics_curr["C"]
                self.shared.alive=metrics_curr["alive"]
                self.shared.cd1=metrics_curr["cd1"]
                self.shared.cd2=metrics_curr["cd2"]
                self.shared.cd3=metrics_curr["cd3"]
                self.shared.cd_item=metrics_curr["cd_item"]
                self.shared.epsilon=eps
                self.shared.running=True
                self.shared.adb_path=self.adb_path
                self.shared.dnplayer_path=self.dnplayer_path
            metrics_prev=metrics_curr
            obs_prev=obs_curr

class BotUI:
    def __init__(self,shared,controller):
        self.shared=shared
        self.controller=controller
        self.root=tk.Tk()
        self.root.title("AI 强化学习 深度学习 控制面板")
        self.vars={}
        self.vars["A"]=tk.StringVar()
        self.vars["B"]=tk.StringVar()
        self.vars["C"]=tk.StringVar()
        self.vars["alive"]=tk.StringVar()
        self.vars["cd1"]=tk.StringVar()
        self.vars["cd2"]=tk.StringVar()
        self.vars["cd3"]=tk.StringVar()
        self.vars["cd_item"]=tk.StringVar()
        self.vars["epsilon"]=tk.StringVar()
        self.vars["status"]=tk.StringVar()
        self.vars["device"]=tk.StringVar()
        self.vars["hw"]=tk.StringVar()
        self.vars["adb"]=tk.StringVar(value=self.shared.adb_path)
        self.vars["dnplayer"]=tk.StringVar(value=self.shared.dnplayer_path)
        self.vars["aaa"]=tk.StringVar(value=self.shared.aaa_dir)
        frame_state=tk.Frame(self.root)
        frame_state.pack(side="top",fill="x")
        tk.Label(frame_state,text="A").grid(row=0,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["A"]).grid(row=0,column=1,sticky="w")
        tk.Label(frame_state,text="B").grid(row=1,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["B"]).grid(row=1,column=1,sticky="w")
        tk.Label(frame_state,text="C").grid(row=2,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["C"]).grid(row=2,column=1,sticky="w")
        tk.Label(frame_state,text="Alive").grid(row=3,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["alive"]).grid(row=3,column=1,sticky="w")
        tk.Label(frame_state,text="Skill1").grid(row=4,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd1"]).grid(row=4,column=1,sticky="w")
        tk.Label(frame_state,text="Skill2").grid(row=5,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd2"]).grid(row=5,column=1,sticky="w")
        tk.Label(frame_state,text="Skill3").grid(row=6,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd3"]).grid(row=6,column=1,sticky="w")
        tk.Label(frame_state,text="Item").grid(row=7,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["cd_item"]).grid(row=7,column=1,sticky="w")
        tk.Label(frame_state,text="ε").grid(row=8,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["epsilon"]).grid(row=8,column=1,sticky="w")
        tk.Label(frame_state,text="Status").grid(row=9,column=0,sticky="w")
        tk.Label(frame_state,textvariable=self.vars["status"]).grid(row=9,column=1,sticky="w")
        control_frame=tk.Frame(self.root)
        control_frame.pack(side="top",fill="x")
        tk.Button(control_frame,text="Start AI",command=self.start_bot).grid(row=0,column=0,sticky="we")
        tk.Button(control_frame,text="Stop AI",command=self.stop_bot).grid(row=0,column=1,sticky="we")
        tk.Label(control_frame,text="ADB").grid(row=1,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["adb"],width=48).grid(row=1,column=1,sticky="we")
        tk.Label(control_frame,text="dnplayer").grid(row=2,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["dnplayer"],width=48).grid(row=2,column=1,sticky="we")
        tk.Label(control_frame,text="AAA").grid(row=3,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["aaa"],width=48).grid(row=3,column=1,sticky="we")
        tk.Button(control_frame,text="Apply Paths",command=self.apply_paths).grid(row=4,column=0,columnspan=2,sticky="we")
        hw_frame=tk.Frame(self.root)
        hw_frame.pack(side="top",fill="x")
        tk.Label(hw_frame,text="Device").grid(row=0,column=0,sticky="w")
        tk.Label(hw_frame,textvariable=self.vars["device"]).grid(row=0,column=1,sticky="w")
        tk.Label(hw_frame,text="Hardware").grid(row=1,column=0,sticky="w")
        tk.Label(hw_frame,textvariable=self.vars["hw"]).grid(row=1,column=1,sticky="w")
        self.update_loop()
    def start_bot(self):
        self.apply_paths()
        self.controller.start()
    def stop_bot(self):
        self.controller.stop()
    def apply_paths(self):
        adb_path=self.vars["adb"].get()
        dnplayer_path=self.vars["dnplayer"].get()
        aaa_path=self.vars["aaa"].get()
        if os.path.exists(adb_path):
            self.controller.update_paths(adb_path,dnplayer_path)
            with self.shared.lock:
                self.shared.adb_path=adb_path
                self.shared.dnplayer_path=dnplayer_path
        if os.path.isdir(aaa_path):
            with self.shared.lock:
                self.shared.aaa_dir=aaa_path
    def update_loop(self):
        with self.shared.lock:
            self.vars["A"].set(str(self.shared.A))
            self.vars["B"].set(str(self.shared.B))
            self.vars["C"].set(str(self.shared.C))
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
                hw_text+=f"CPU:{self.shared.hw_profile['cpu_count']} "
            if "memory_gb" in self.shared.hw_profile and self.shared.hw_profile["memory_gb"] is not None:
                hw_text+=f"RAM:{self.shared.hw_profile['memory_gb']}GB "
            if "cuda_gpu" in self.shared.hw_profile and self.shared.hw_profile["cuda_gpu"]:
                hw_text+=f"GPU:{self.shared.hw_profile['cuda_gpu']}"
            self.vars["hw"].set(hw_text)
        self.root.after(500,self.update_loop)
    def run(self):
        self.root.mainloop()

def adaptive_hyperparams(hw,device):
    cpu=hw.get("cpu_count",4) if isinstance(hw,dict) else 4
    mem=hw.get("memory_gb",4) if isinstance(hw,dict) else 4
    gpu_mem=hw.get("cuda_total_mem_mb",0) if isinstance(hw,dict) else 0
    hidden=256 if gpu_mem and gpu_mem>6144 else (192 if cpu>=8 else 96)
    batch=128 if gpu_mem and gpu_mem>8192 else (96 if cpu>=8 else 48)
    buffer=30000 if mem and mem>12 else (20000 if mem and mem>8 else 12000)
    lr=1e-4 if gpu_mem and gpu_mem>0 else 3e-4
    target_sync=800 if device=="cuda" else 1200
    epsilon_decay=80000 if gpu_mem and gpu_mem>0 else 60000
    return {"hidden_dim":hidden,"batch_size":batch,"buffer_size":buffer,"lr":lr,"target_sync":target_sync,"epsilon_decay":epsilon_decay}

def main():
    shared=SharedState()
    hw=hardware_profile()
    device="cuda" if torch.cuda.is_available() else "cpu"
    gpu_mem=None
    if hw.get("cuda_total_mem_mb",None) is not None:
        gpu_mem=hw["cuda_total_mem_mb"]
    dtype=torch.float16 if device=="cuda" and gpu_mem and gpu_mem>8192 else torch.float32
    shared.device=device
    shared.hw_profile=hw
    params=adaptive_hyperparams(hw,device)
    aaa_dir=shared.aaa_dir
    experience_dir=os.path.join(aaa_dir,"experience")
    agent=DeepRLAgent(device,dtype,hidden_dim=params["hidden_dim"],buffer_size=params["buffer_size"],batch_size=params["batch_size"],lr=params["lr"],target_sync=params["target_sync"],epsilon_decay=params["epsilon_decay"],experience_dir=experience_dir)
    aihub=AIModelHub(aaa_dir,device,dtype)
    controller=BotController(shared,agent,aihub)
    ui=BotUI(shared,controller)
    ui.run()

if __name__=="__main__":
    main()
