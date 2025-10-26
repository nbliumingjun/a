import os,subprocess,threading,time,random,math,platform,json,io,glob
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
@dataclass
class CircleRegion:
    x:float
    y:float
    d:float
@dataclass
class RectRegion:
    x:float
    y:float
    w:float
    h:float
def default_config(base_dir):
    return {"屏幕":{"基准宽度":2560,"基准高度":1600},"路径":{"ADB":"D:\\LDPlayer9\\adb.exe","模拟器":"D:\\LDPlayer9\\dnplayer.exe","AAA":base_dir},"经验目录":"experience","模型文件":{"一技能":"skill1_model.pt","二技能":"skill2_model.pt","三技能":"skill3_model.pt","移动轮盘":"move_model.pt","普攻":"attack_model.pt","回城":"recall_model.pt","视觉":"vision_model.pt"},"识别":{"A":{"左上角":[1904,122],"尺寸":[56,56]},"B":{"左上角":[1996,122],"尺寸":[56,56]},"C":{"左上角":[2087,122],"尺寸":[56,56]},"小地图":{"左上角":[0,72],"尺寸":[453,453]}},"圆形区域":{"移动轮盘":{"左上角":[166,915],"直径":536},"回城":{"左上角":[1083,1263],"直径":162},"恢复":{"左上角":[1271,1263],"直径":162},"闪现":{"左上角":[1467,1263],"直径":162},"一技能":{"左上角":[1672,1220],"直径":195},"二技能":{"左上角":[1825,956],"直径":195},"三技能":{"左上角":[2088,803],"直径":195},"取消施法":{"左上角":[2165,252],"直径":250},"普攻补刀":{"左上角":[1915,1296],"直径":123},"普攻点塔":{"左上角":[2241,1014],"直径":123},"主动装备":{"左上角":[2092,544],"直径":161}},"奖励":{"A":3.0,"C":2.0,"B":-4.0},"动作":{"循环间隔":0.2,"拖动耗时":0.2},"动作冷却":{"回城等待":8.0,"恢复时长":5.0,"闪现位移":[150.0,0.0]},"OCR":{"亮度阈值":110,"饱和阈值":60,"波动阈值":55},"学习":{"折扣":0.99,"学习率":0.0003,"缓冲大小":20000,"批次":64,"隐藏单元":128,"同步步数":1000,"ε衰减":60000}}
class ConfigManager:
    def __init__(self):
        self.home=os.path.expanduser("~")
        self.default_aaa=os.path.join(self.home,"Desktop","AAA")
        self.config_path=None
        self.config=None
        self.load()
    def ensure_dir(self,path):
        os.makedirs(path,exist_ok=True)
    def load(self):
        self.ensure_dir(self.default_aaa)
        self.config_path=os.path.join(self.default_aaa,"配置.json")
        if os.path.isfile(self.config_path):
            with open(self.config_path,"r",encoding="utf-8") as f:
                data=json.load(f)
        else:
            data=default_config(self.default_aaa)
            self.save_config(data)
        if "路径" not in data or "AAA" not in data["路径"]:
            data["路径"]=default_config(self.default_aaa)["路径"]
        aaa_dir=data["路径"].get("AAA",self.default_aaa)
        self.ensure_dir(aaa_dir)
        self.config_path=os.path.join(aaa_dir,"配置.json")
        if not os.path.isfile(self.config_path):
            self.save_config(data)
        else:
            with open(self.config_path,"r",encoding="utf-8") as f:
                try:
                    data=json.load(f)
                except:
                    data=default_config(aaa_dir)
                    self.save_config(data)
        self.config=data
        self.ensure_dir(os.path.join(self.config["路径"]["AAA"],self.config["经验目录"]))
    def save_config(self,data=None):
        target=data if data is not None else self.config
        if target is None:
            target=default_config(self.default_aaa)
        path=self.config_path if self.config_path else os.path.join(self.default_aaa,"配置.json")
        with open(path,"w",encoding="utf-8") as f:
            json.dump(target,f,ensure_ascii=False,indent=2)
    def update_paths(self,adb_path,dn_path,aaa_dir):
        self.config["路径"]["ADB"]=adb_path
        self.config["路径"]["模拟器"]=dn_path
        if aaa_dir and aaa_dir!=self.config["路径"]["AAA"]:
            self.ensure_dir(aaa_dir)
            self.config["路径"]["AAA"]=aaa_dir
        self.config_path=os.path.join(self.config["路径"]["AAA"],"配置.json")
        self.ensure_dir(os.path.join(self.config["路径"]["AAA"],self.config["经验目录"]))
        self.save_config()
class SharedState:
    def __init__(self,config):
        self.lock=threading.Lock()
        self.A=0
        self.B=0
        self.C=0
        self.alive=0
        self.cd1=0
        self.cd2=0
        self.cd3=0
        self.cd_item=0
        self.cd_heal=0
        self.running=False
        self.epsilon=1.0
        self.device="cpu"
        self.hw_profile={}
        self.config=config
        self.config_manager=None
class ModelManager:
    def __init__(self,config,device,dtype):
        self.config=config
        self.device=device
        self.dtype=dtype
        self.models={}
        self.ensure_models()
        self.load_models()
    def ensure_models(self):
        model_dir=self.config["路径"]["AAA"]
        for name,filename in self.config["模型文件"].items():
            path=os.path.join(model_dir,filename)
            if not os.path.isfile(path):
                net=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32))
                torch.save(net,path)
    def load_models(self):
        model_dir=self.config["路径"]["AAA"]
        for name,filename in self.config["模型文件"].items():
            path=os.path.join(model_dir,filename)
            try:
                obj=torch.load(path,map_location=self.device)
                if isinstance(obj,nn.Module):
                    obj.to(self.device)
                    try:
                        obj.to(self.dtype)
                    except:
                        pass
                    obj.eval()
                    self.models[name]=obj
                else:
                    net=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32)).to(self.device)
                    net.load_state_dict(obj,strict=False)
                    net.eval()
                    self.models[name]=net
            except:
                self.models[name]=None
        for f in glob.glob(os.path.join(model_dir,"*.pt"))+glob.glob(os.path.join(model_dir,"*.pth")):
            if f not in [os.path.join(model_dir,x) for x in self.config["模型文件"].values()]:
                try:
                    self.models[os.path.basename(f)]=torch.load(f,map_location=self.device)
                except:
                    self.models[os.path.basename(f)]=None
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
    def __init__(self,device,dtype,config,hidden_dim,buffer_size,batch_size,gamma,lr,target_sync,epsilon_decay,experience_dir):
        self.device=device
        self.dtype=dtype
        self.obs_dim=9
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
        self.config=config
        self.load_experience()
    def epsilon(self):
        return self.epsilon_end+(self.epsilon_start-self.epsilon_end)*math.exp(-1.0*self.step_count/max(1.0,self.epsilon_decay))
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
        if self.experience_dir and self.step_count-self.last_save_step>=500:
            self.save_experience()
            self.last_save_step=self.step_count
    def save_experience(self):
        try:
            os.makedirs(self.experience_dir,exist_ok=True)
            data={"state":np.array([t.state for t in self.replay_buffer],dtype=np.float32),"action":np.array([t.action for t in self.replay_buffer],dtype=np.int64),"reward":np.array([t.reward for t in self.replay_buffer],dtype=np.float32),"next_state":np.array([t.next_state for t in self.replay_buffer],dtype=np.float32),"done":np.array([t.done for t in self.replay_buffer],dtype=np.int8),"step":self.step_count}
            np.savez_compressed(os.path.join(self.experience_dir,"buffer.npz"),**data)
            torch.save(self.policy_net.state_dict(),os.path.join(self.experience_dir,"policy.pt"))
        except:
            pass
    def load_experience(self):
        try:
            if not self.experience_dir:
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
class VisionModule:
    def __init__(self,config):
        self.config=config
        self.prev_ready={}
    def crop_cv(self,pil_img,x,y,w,h,scale_x,scale_y):
        sx=int(x*scale_x)
        sy=int(y*scale_y)
        sw=max(1,int(w*scale_x))
        sh=max(1,int(h*scale_y))
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
        bright=mean_v>self.config["OCR"]["亮度阈值"]
        saturated=mean_s>self.config["OCR"]["饱和阈值"]
        stable=std_v<self.config["OCR"]["波动阈值"]
        ready=1 if bright and saturated and stable else 0
        prev=self.prev_ready.get(key,0.0)
        ready=max(ready,prev*0.5)
        self.prev_ready[key]=ready
        return ready
    def alive_state(self,cv_img):
        mean_v,mean_s,_=self.region_stats(cv_img)
        return 1 if mean_v>self.config["OCR"]["亮度阈值"]*0.8 and mean_s>self.config["OCR"]["饱和阈值"]*0.6 else 0
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
    def __init__(self,config,adb_path):
        self.config=config
        self.adb_path=adb_path
        self.base_w=self.config["屏幕"]["基准宽度"]
        self.base_h=self.config["屏幕"]["基准高度"]
        self.screen_w=self.base_w
        self.screen_h=self.base_h
        self.vision=VisionModule(config)
        self.last_screenshot=None
        circle_conf=self.config["圆形区域"]
        rect_conf=self.config["识别"]
        self.circle_regions={
            "joystick":self._circle(circle_conf.get("移动轮盘")),
            "recall":self._circle(circle_conf.get("回城")),
            "heal":self._circle(circle_conf.get("恢复")),
            "flash":self._circle(circle_conf.get("闪现")),
            "skill1":self._circle(circle_conf.get("一技能")),
            "skill2":self._circle(circle_conf.get("二技能")),
            "skill3":self._circle(circle_conf.get("三技能")),
            "cancel":self._circle(circle_conf.get("取消施法")),
            "attack_min":self._circle(circle_conf.get("普攻补刀")),
            "attack_tower":self._circle(circle_conf.get("普攻点塔")),
            "active_item":self._circle(circle_conf.get("主动装备"))
        }
        self.rect_regions={
            "A":self._rect(rect_conf.get("A")),
            "B":self._rect(rect_conf.get("B")),
            "C":self._rect(rect_conf.get("C")),
            "minimap":self._rect(rect_conf.get("小地图"))
        }
        self.prev_metrics={"A":0,"B":0,"C":0,"alive":0,"cd1":0,"cd2":0,"cd3":0,"cd_item":0,"cd_heal":0}
    def _circle(self,data):
        if not data:
            return CircleRegion(0.0,0.0,1.0)
        tl=data.get("左上角",[0.0,0.0])
        return CircleRegion(float(tl[0]),float(tl[1]),float(data.get("直径",1.0)))
    def _rect(self,data):
        if not data:
            return RectRegion(0.0,0.0,1.0,1.0)
        tl=data.get("左上角",[0.0,0.0])
        sz=data.get("尺寸",[1.0,1.0])
        return RectRegion(float(tl[0]),float(tl[1]),float(sz[0]),float(sz[1]))
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
    def circle_center(self,region):
        return region.x+region.d/2.0,region.y+region.d/2.0
    def tap_circle(self,region):
        cx,cy=self.circle_center(region)
        sx,sy=self.scaled_xy(cx,cy)
        self.adb_tap(sx,sy)
    def joystick_move_random(self):
        region=self.circle_regions["joystick"]
        cx,cy=self.circle_center(region)
        radius=region.d/2.0*0.8
        ang=random.random()*2.0*math.pi
        ex=cx+radius*math.cos(ang)
        ey=cy+radius*math.sin(ang)
        sx,sy=self.scaled_xy(cx,cy)
        exs,eys=self.scaled_xy(ex,ey)
        self.adb_swipe(sx,sy,exs,eys,int(self.config["动作"]["拖动耗时"]*1000))
    def basic_attack_minion(self):
        self.tap_circle(self.circle_regions["attack_min"])
    def basic_attack_tower(self):
        self.tap_circle(self.circle_regions["attack_tower"])
    def cast_skill(self,key):
        self.tap_circle(self.circle_regions[key])
    def use_item(self):
        self.tap_circle(self.circle_regions["active_item"])
    def heal(self):
        self.tap_circle(self.circle_regions["heal"])
    def recall(self):
        self.tap_circle(self.circle_regions["recall"])
        time.sleep(self.config["动作冷却"]["回城等待"])
    def flash_forward(self):
        region=self.circle_regions["flash"]
        cx,cy=self.circle_center(region)
        offset=self.config["动作冷却"]["闪现位移"]
        ex=cx+offset[0]
        ey=cy+offset[1]
        sx,sy=self.scaled_xy(cx,cy)
        exs,eys=self.scaled_xy(ex,ey)
        self.adb_swipe(sx,sy,exs,eys,int(self.config["动作"]["拖动耗时"]*1000))
    def get_metrics(self):
        img=self.get_screenshot()
        if img is None:
            return self.prev_metrics
        w,h=img.size
        scale_x=w/self.base_w
        scale_y=h/self.base_h
        v=self.vision
        def crop_circle(region):
            return v.crop_cv(img,region.x,region.y,region.d,region.d,scale_x,scale_y)
        def crop_rect(region):
            return v.crop_cv(img,region.x,region.y,region.w,region.h,scale_x,scale_y)
        A_img=crop_rect(self.rect_regions["A"])
        B_img=crop_rect(self.rect_regions["B"])
        C_img=crop_rect(self.rect_regions["C"])
        attack_img=crop_circle(self.circle_regions["attack_min"])
        skill1_img=crop_circle(self.circle_regions["skill1"])
        skill2_img=crop_circle(self.circle_regions["skill2"])
        skill3_img=crop_circle(self.circle_regions["skill3"])
        item_img=crop_circle(self.circle_regions["active_item"])
        heal_img=crop_circle(self.circle_regions["heal"])
        metrics={"A":v.ocr_digits(A_img),"B":v.ocr_digits(B_img),"C":v.ocr_digits(C_img),"alive":v.alive_state(attack_img),"cd1":v.cooldown_ready("skill1",skill1_img),"cd2":v.cooldown_ready("skill2",skill2_img),"cd3":v.cooldown_ready("skill3",skill3_img),"cd_item":v.cooldown_ready("item",item_img),"cd_heal":v.cooldown_ready("heal",heal_img)}
        self.prev_metrics=metrics
        return metrics
    def metrics_to_obs(self,m):
        return [min(m["A"],99)/99.0,min(m["B"],99)/99.0,min(m["C"],99)/99.0,float(m["alive"]),float(m["cd1"]),float(m["cd2"]),float(m["cd3"]),float(m["cd_item"]),float(m["cd_heal"])]
    def compute_reward(self,prev_metrics,curr_metrics):
        dA=curr_metrics["A"]-prev_metrics["A"]
        dC=curr_metrics["C"]-prev_metrics["C"]
        dB=curr_metrics["B"]-prev_metrics["B"]
        weights=self.config["奖励"]
        return float(weights["A"]*dA+weights["C"]*dC+weights["B"]*dB)
    def execute_action(self,action_idx,metrics_prev):
        if action_idx==0:
            return
        if action_idx==1 and metrics_prev["alive"]==1:
            self.joystick_move_random()
            return
        if action_idx==2 and metrics_prev["alive"]==1:
            self.basic_attack_minion()
            return
        if action_idx==3 and metrics_prev["alive"]==1:
            self.basic_attack_tower()
            return
        if action_idx==4 and metrics_prev["alive"]==1 and metrics_prev["cd1"]==1:
            self.cast_skill("skill1")
            return
        if action_idx==5 and metrics_prev["alive"]==1 and metrics_prev["cd2"]==1:
            self.cast_skill("skill2")
            return
        if action_idx==6 and metrics_prev["alive"]==1 and metrics_prev["cd3"]==1:
            self.cast_skill("skill3")
            return
        if action_idx==7 and metrics_prev["alive"]==1 and metrics_prev["cd_item"]==1:
            self.use_item()
            return
        if action_idx==8 and metrics_prev["alive"]==1 and metrics_prev["cd_heal"]==1:
            self.heal()
            time.sleep(self.config["动作冷却"]["恢复时长"])
            return
        if action_idx==9 and metrics_prev["alive"]==1:
            self.recall()
            return
        if action_idx==10 and metrics_prev["alive"]==1:
            self.flash_forward()
            return
class BotController:
    def __init__(self,shared,agent,model_manager):
        self.shared=shared
        self.agent=agent
        self.model_manager=model_manager
        self.adb_path=shared.config["路径"]["ADB"]
        self.dnplayer_path=shared.config["路径"]["模拟器"]
        self.game=GameInterface(shared.config,self.adb_path)
        self.running_event=threading.Event()
        self.thread=None
    def update_paths(self,adb_path,dnplayer_path,aaa_path):
        if adb_path:
            self.adb_path=adb_path
        if dnplayer_path:
            self.dnplayer_path=dnplayer_path
        cm=self.shared.config_manager
        target_aaa=aaa_path if aaa_path else cm.config["路径"]["AAA"]
        cm.update_paths(self.adb_path,self.dnplayer_path,target_aaa)
        self.shared.config=cm.config
        self.model_manager=ModelManager(cm.config,self.agent.device,self.agent.dtype)
        self.game=GameInterface(cm.config,self.adb_path)
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
        interval=self.shared.config["动作"]["循环间隔"]
        while self.running_event.is_set():
            action,eps=self.agent.select_action(obs_prev)
            self.game.execute_action(action,metrics_prev)
            time.sleep(interval)
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
                self.shared.cd_heal=metrics_curr["cd_heal"]
                self.shared.epsilon=eps
                self.shared.running=True
        metrics_prev=metrics_curr
        obs_prev=obs_curr
class BotUI:
    def __init__(self,shared,controller,config_manager):
        self.shared=shared
        self.controller=controller
        self.config_manager=config_manager
        self.root=tk.Tk()
        self.root.title("AI强化学习深度学习控制面板")
        self.vars={k:tk.StringVar() for k in ["A","B","C","alive","cd1","cd2","cd3","cd_item","cd_heal","epsilon","status","device","hw","adb","dn","aaa"]}
        self.vars["adb"].set(self.shared.config["路径"]["ADB"])
        self.vars["dn"].set(self.shared.config["路径"]["模拟器"])
        self.vars["aaa"].set(self.shared.config["路径"]["AAA"])
        self.build_ui()
        self.update_loop()
    def build_ui(self):
        frame_state=tk.Frame(self.root)
        frame_state.pack(side="top",fill="x")
        labels=[("A","A"),("B","B"),("C","C"),("存活","alive"),("一技能","cd1"),("二技能","cd2"),("三技能","cd3"),("主动装备","cd_item"),("恢复","cd_heal"),("ε","epsilon"),("状态","status"),("设备","device"),("硬件","hw")]
        for idx,(text,key) in enumerate(labels):
            tk.Label(frame_state,text=text).grid(row=idx,column=0,sticky="w")
            tk.Label(frame_state,textvariable=self.vars[key]).grid(row=idx,column=1,sticky="w")
        control_frame=tk.Frame(self.root)
        control_frame.pack(side="top",fill="x")
        tk.Button(control_frame,text="启动AI",command=self.start_bot).grid(row=0,column=0,sticky="we")
        tk.Button(control_frame,text="停止AI",command=self.stop_bot).grid(row=0,column=1,sticky="we")
        tk.Label(control_frame,text="ADB").grid(row=1,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["adb"],width=48).grid(row=1,column=1,sticky="we")
        tk.Label(control_frame,text="模拟器").grid(row=2,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["dn"],width=48).grid(row=2,column=1,sticky="we")
        tk.Label(control_frame,text="AAA").grid(row=3,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["aaa"],width=48).grid(row=3,column=1,sticky="we")
        tk.Button(control_frame,text="应用路径",command=self.apply_paths).grid(row=4,column=0,columnspan=2,sticky="we")
    def start_bot(self):
        self.apply_paths()
        self.controller.start()
    def stop_bot(self):
        self.controller.stop()
    def apply_paths(self):
        adb_path=self.vars["adb"].get()
        dn_path=self.vars["dn"].get()
        aaa_path=self.vars["aaa"].get()
        if adb_path:
            self.shared.config["路径"]["ADB"]=adb_path
        if dn_path:
            self.shared.config["路径"]["模拟器"]=dn_path
        if aaa_path:
            self.config_manager.update_paths(adb_path,dn_path,aaa_path)
            self.shared.config=self.config_manager.config
        self.controller.update_paths(adb_path,dn_path,aaa_path)
    def update_loop(self):
        with self.shared.lock:
            self.vars["A"].set(str(self.shared.A))
            self.vars["B"].set(str(self.shared.B))
            self.vars["C"].set(str(self.shared.C))
            self.vars["alive"].set("是" if self.shared.alive==1 else "否")
            self.vars["cd1"].set("可用" if self.shared.cd1==1 else "冷却")
            self.vars["cd2"].set("可用" if self.shared.cd2==1 else "冷却")
            self.vars["cd3"].set("可用" if self.shared.cd3==1 else "冷却")
            self.vars["cd_item"].set("可用" if self.shared.cd_item==1 else "冷却")
            self.vars["cd_heal"].set("可用" if self.shared.cd_heal==1 else "冷却")
            self.vars["epsilon"].set("{:.3f}".format(self.shared.epsilon))
            self.vars["status"].set("运行" if self.shared.running else "停止")
            self.vars["device"].set(self.shared.device)
            hw_desc=""
            if "cpu_count" in self.shared.hw_profile:
                hw_desc+=f"CPU:{self.shared.hw_profile['cpu_count']} "
            if "memory_gb" in self.shared.hw_profile and self.shared.hw_profile["memory_gb"] is not None:
                hw_desc+=f"RAM:{self.shared.hw_profile['memory_gb']:.1f}GB "
            if "cuda_gpu" in self.shared.hw_profile and self.shared.hw_profile["cuda_gpu"]:
                hw_desc+=f"GPU:{self.shared.hw_profile['cuda_gpu']}"
            self.vars["hw"].set(hw_desc)
        self.root.after(500,self.update_loop)
    def run(self):
        self.root.mainloop()
def hardware_profile():
    profile={"platform":platform.platform()}
    cpu=os.cpu_count() or 4
    profile["cpu_count"]=cpu
    if psutil:
        try:
            profile["memory_gb"]=psutil.virtual_memory().total/1024**3
        except:
            profile["memory_gb"]=None
    else:
        profile["memory_gb"]=None
    if torch.cuda.is_available():
        idx=torch.cuda.current_device()
        profile["cuda_gpu"]=torch.cuda.get_device_name(idx)
        profile["cuda_total_mem_mb"]=torch.cuda.get_device_properties(idx).total_memory/1024**2
    elif GPUtil:
        try:
            gpus=GPUtil.getGPUs()
            if gpus:
                g=gpus[0]
                profile["cuda_gpu"]=g.name
                profile["cuda_total_mem_mb"]=g.memoryTotal*1024
        except:
            profile["cuda_gpu"]=None
            profile["cuda_total_mem_mb"]=0
    else:
        profile["cuda_gpu"]=None
        profile["cuda_total_mem_mb"]=0
    return profile
def adaptive_hyperparams(hw,config,device):
    base=config["学习"]
    cpu=hw.get("cpu_count",4)
    mem=hw.get("memory_gb",8) or 8
    gpu_mem=hw.get("cuda_total_mem_mb",0) or 0
    hidden=base["隐藏单元"]
    batch=base["批次"]
    buffer=base["缓冲大小"]
    lr=base["学习率"]
    target_sync=base["同步步数"]
    epsilon_decay=base["ε衰减"]
    if gpu_mem>8192:
        hidden=int(hidden*1.5)
        batch=int(batch*1.5)
        buffer=int(buffer*1.5)
        lr=lr*0.7
        epsilon_decay=int(epsilon_decay*1.3)
        target_sync=max(500,int(target_sync*0.8))
    elif cpu>=8 and mem>12:
        hidden=int(hidden*1.3)
        batch=int(batch*1.2)
        buffer=int(buffer*1.2)
        epsilon_decay=int(epsilon_decay*1.1)
    elif mem<6:
        batch=max(16,int(batch*0.6))
        buffer=max(8000,int(buffer*0.5))
        target_sync=int(target_sync*1.2)
    if device=="cpu":
        lr=max(lr,1e-4)
    return {"hidden_dim":hidden,"batch_size":batch,"buffer_size":buffer,"lr":lr,"target_sync":target_sync,"epsilon_decay":epsilon_decay,"gamma":base["折扣"]}
def main():
    config_manager=ConfigManager()
    config=config_manager.config
    shared=SharedState(config)
    shared.config_manager=config_manager
    hw=hardware_profile()
    device="cuda" if torch.cuda.is_available() else "cpu"
    gpu_mem=hw.get("cuda_total_mem_mb",0)
    dtype=torch.float16 if device=="cuda" and gpu_mem>8192 else torch.float32
    shared.device=device
    shared.hw_profile=hw
    params=adaptive_hyperparams(hw,config,device)
    experience_dir=os.path.join(config["路径"]["AAA"],config["经验目录"])
    agent=DeepRLAgent(device,dtype,config,hidden_dim=params["hidden_dim"],buffer_size=params["buffer_size"],batch_size=params["batch_size"],gamma=params["gamma"],lr=params["lr"],target_sync=params["target_sync"],epsilon_decay=params["epsilon_decay"],experience_dir=experience_dir)
    model_manager=ModelManager(config,device,dtype)
    controller=BotController(shared,agent,model_manager)
    ui=BotUI(shared,controller,config_manager)
    ui.run()
if __name__=="__main__":
    main()
