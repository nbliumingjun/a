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
import tkinter.ttk as ttk
from tkinter import messagebox
try:
    from pynput import mouse,keyboard
except:
    mouse=None
    keyboard=None
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
class RegionScaler:
    def __init__(self,config):
        screen=config.get("屏幕",{})
        self.base_w=float(screen.get("基准宽度",2560))
        self.base_h=float(screen.get("基准高度",1600))
        self.circle_conf=config.get("圆形区域",{})
        self.rect_conf=config.get("识别",{})
        self.screen_w=self.base_w
        self.screen_h=self.base_h
    def update(self,width,height):
        self.screen_w=max(1.0,float(width))
        self.screen_h=max(1.0,float(height))
    def _ratio(self,data,key,fallback):
        value=data.get(key)
        if isinstance(value,list) and len(value)>=2:
            return [float(value[0]),float(value[1])]
        return fallback
    def circle(self,name):
        data=self.circle_conf.get(name)
        if not isinstance(data,dict):
            return CircleRegion(0.0,0.0,1.0)
        ratio=self._ratio(data,"左上角比例",self._ratio(data,"左上角",[0.0,0.0]))
        dia_ratio=data.get("直径比例")
        if dia_ratio is None:
            dia=float(data.get("直径",1.0))/self.base_w if self.base_w else 0.0
        else:
            dia=float(dia_ratio)
        return CircleRegion(ratio[0]*self.screen_w,ratio[1]*self.screen_h,dia*self.screen_w)
    def rect(self,name):
        data=self.rect_conf.get(name)
        if not isinstance(data,dict):
            return RectRegion(0.0,0.0,1.0,1.0)
        ratio=self._ratio(data,"左上角比例",self._ratio(data,"左上角",[0.0,0.0]))
        size_ratio=data.get("尺寸比例")
        if isinstance(size_ratio,list) and len(size_ratio)>=2:
            sw=float(size_ratio[0])
            sh=float(size_ratio[1])
        else:
            size=data.get("尺寸",[1.0,1.0])
            sw=float(size[0])/self.base_w if self.base_w else 0.0
            sh=float(size[1])/self.base_h if self.base_h else 0.0
        return RectRegion(ratio[0]*self.screen_w,ratio[1]*self.screen_h,sw*self.screen_w,sh*self.screen_h)
def default_config(base_dir):
    base_w=2560
    base_h=1600
    model_pairs=[("一技能","skill1_model.pt"),("二技能","skill2_model.pt"),("三技能","skill3_model.pt"),("移动轮盘","move_model.pt"),("普攻补刀","attack_last_hit_model.pt"),("普攻点塔","attack_tower_model.pt"),("回城","recall_model.pt"),("视觉","vision_model.pt"),("恢复","heal_model.pt"),("闪现","flash_model.pt"),("小地图","minimap_model.pt"),("主动装备","active_item_model.pt")]
    model_names={k:v for k,v in model_pairs}
    circle_items=[("移动轮盘",166,915,536),("回城",1083,1263,162),("恢复",1271,1263,162),("闪现",1467,1263,162),("一技能",1672,1220,195),("二技能",1825,956,195),("三技能",2088,803,195),("取消施法",2165,252,250),("普攻补刀",1915,1296,123),("普攻点塔",2241,1014,123),("主动装备",2092,544,161)]
    rect_items=[("A",1904,122,56,56),("B",1996,122,56,56),("C",2087,122,56,56),("小地图",0,72,453,453)]
    circle_raw={name:(x,y,d) for name,x,y,d in circle_items}
    rect_raw={name:(x,y,w,h) for name,x,y,w,h in rect_items}
    circle_conf={k:{"左上角":[float(x),float(y)],"左上角比例":[float(x)/base_w,float(y)/base_h],"直径":float(d),"直径比例":float(d)/base_w} for k,(x,y,d) in circle_raw.items()}
    rect_conf={k:{"左上角":[float(x),float(y)],"左上角比例":[float(x)/base_w,float(y)/base_h],"尺寸":[float(w),float(h)],"尺寸比例":[float(w)/base_w,float(h)/base_h]} for k,(x,y,w,h) in rect_raw.items()}
    flash_offset=[150.0,0.0]
    flash_ratio=[flash_offset[0]/base_w,flash_offset[1]/base_h]
    obs_keys=["A","B","C","alive","cd1","cd2","cd3","cd_item","cd_heal","cd_flash","recalling"]
    action_names=["idle","move","attack_minion","attack_tower","skill1","skill2","skill3","item","heal","recall","flash"]
    lr=0.0003
    reward_window=max(12,len(action_names)*6)
    reward_gain=0.3+len(action_names)/100.0
    value_reg=0.25/max(8,len(obs_keys))
    base_interval=0.2
    idle_switch=max(10.0,reward_window*base_interval*0.5)
    learn_sample=max(base_interval,base_interval*2.0)
    train_sample=max(base_interval,base_interval*1.5)
    return {"屏幕":{"基准宽度":base_w,"基准高度":base_h},"路径":{"ADB":"D:\\LDPlayer9\\adb.exe","模拟器":"D:\\LDPlayer9\\dnplayer.exe","AAA":base_dir},"经验目录":"experience","模型文件":model_names,"识别":rect_conf,"圆形区域":circle_conf,"奖励":{"A":3.0,"C":2.0,"B":-4.0},"动作":{"循环间隔":base_interval,"拖动耗时":base_interval,"摇杆幅度":0.8},"动作冷却":{"回城等待":8.0,"恢复时长":5.0,"闪现位移":flash_offset,"闪现位移比例":flash_ratio},"OCR":{"亮度阈值":110,"饱和阈值":60,"波动阈值":55,"存活亮度比":0.8,"存活饱和比":0.6},"学习":{"折扣":0.99,"学习率":lr,"缓冲大小":20000,"批次":64,"隐藏单元":128,"同步步数":1000,"ε衰减":60000,"权重衰减":lr*0.1,"奖励平滑窗口":reward_window,"奖励调节":reward_gain,"价值正则":value_reg},"模式":{"学习静默阈值":idle_switch,"学习采样间隔":learn_sample,"训练采样间隔":train_sample},"观测键":obs_keys,"动作名称":action_names,"数值上限":{"A":99.0,"B":99.0,"C":99.0}}
class ConfigManager:
    def __init__(self):
        self.home=os.path.expanduser("~")
        self.default_aaa=os.path.join(self.home,"Desktop","AAA")
        self.config_path=None
        self.config=None
        self.load()
    def ensure_dir(self,path):
        os.makedirs(path,exist_ok=True)
    def merge_dict(self,base,override):
        if not isinstance(base,dict):
            return override if override is not None else base
        result=dict(base)
        if not isinstance(override,dict):
            return result
        for k,v in override.items():
            if k in result and isinstance(result[k],dict) and isinstance(v,dict):
                result[k]=self.merge_dict(result[k],v)
            elif v is not None:
                result[k]=v
        return result
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
        aaa_dir=os.path.normpath(os.path.abspath(data["路径"].get("AAA",self.default_aaa)))
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
        defaults=default_config(aaa_dir)
        data=self.merge_dict(defaults,data)
        self.config=data
        self.ensure_region_ratios()
        self.ensure_dir(os.path.join(self.config["路径"]["AAA"],self.config["经验目录"]))
    def save_config(self,data=None):
        target=data if data is not None else self.config
        if target is None:
            target=default_config(self.default_aaa)
        path=self.config_path if self.config_path else os.path.join(self.default_aaa,"配置.json")
        with open(path,"w",encoding="utf-8") as f:
            json.dump(target,f,ensure_ascii=False,indent=2)
    def update_paths(self,adb_path,dn_path,aaa_dir):
        if adb_path:
            self.config["路径"]["ADB"]=adb_path
        if dn_path:
            self.config["路径"]["模拟器"]=dn_path
        if aaa_dir:
            normalized=os.path.normpath(os.path.abspath(aaa_dir))
            if normalized!=self.config["路径"]["AAA"]:
                self.ensure_dir(normalized)
                self.config["路径"]["AAA"]=normalized
        self.config_path=os.path.join(self.config["路径"]["AAA"],"配置.json")
        self.ensure_dir(os.path.join(self.config["路径"]["AAA"],self.config["经验目录"]))
        self.save_config()
    def ensure_region_ratios(self):
        if not self.config:
            return
        screen=self.config.get("屏幕",{})
        base_w=float(screen.get("基准宽度",2560))
        base_h=float(screen.get("基准高度",1600))
        circle_conf=self.config.get("圆形区域",{})
        rect_conf=self.config.get("识别",{})
        for name,data in circle_conf.items():
            if not isinstance(data,dict):
                continue
            tl=data.get("左上角") or [0.0,0.0]
            ratio=data.get("左上角比例") or [tl[0]/base_w if base_w else 0.0,tl[1]/base_h if base_h else 0.0]
            data["左上角比例"]=ratio
            data["左上角"]= [ratio[0]*base_w,ratio[1]*base_h]
            dia=data.get("直径") if data.get("直径") is not None else 0.0
            ratio_d=data.get("直径比例") if data.get("直径比例") is not None else (dia/base_w if base_w else 0.0)
            data["直径比例"]=ratio_d
            data["直径"]=ratio_d*base_w
        for name,data in rect_conf.items():
            if not isinstance(data,dict):
                continue
            tl=data.get("左上角") or [0.0,0.0]
            ratio=data.get("左上角比例") or [tl[0]/base_w if base_w else 0.0,tl[1]/base_h if base_h else 0.0]
            data["左上角比例"]=ratio
            data["左上角"]= [ratio[0]*base_w,ratio[1]*base_h]
            size=data.get("尺寸") or [1.0,1.0]
            ratio_s=data.get("尺寸比例") or [size[0]/base_w if base_w else 0.0,size[1]/base_h if base_h else 0.0]
            data["尺寸比例"]=ratio_s
            data["尺寸"]= [ratio_s[0]*base_w,ratio_s[1]*base_h]
        cooldown=self.config.get("动作冷却",{})
        flash=cooldown.get("闪现位移")
        flash_ratio=cooldown.get("闪现位移比例")
        if flash_ratio and not flash:
            cooldown["闪现位移"]= [flash_ratio[0]*base_w,flash_ratio[1]*base_h]
        elif flash and not flash_ratio:
            cooldown["闪现位移比例"]= [flash[0]/base_w if base_w else 0.0,flash[1]/base_h if base_h else 0.0]
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
        self.cd_flash=0
        self.recalling=0
        self.running=False
        self.epsilon=1.0
        self.device="cpu"
        self.hw_profile={}
        self.config=config
        self.config_manager=None
        self.mode="学习"
        self.terminate=False
        self.optimizing=False
        self.optimize_progress=0.0
        self.optimize_iterations=0
        self.optimize_text="待命"
class ModelManager:
    def __init__(self,config,device,dtype):
        self.config=config
        self.device=device
        self.dtype=dtype
        self.models={}
        self.ensure_models()
        self.load_models()
    def model_dims(self):
        learn_conf=self.config.get("学习",{})
        obs_dim=len(self.config.get("观测键",[])) or 32
        act_dim=len(self.config.get("动作名称",[])) or max(16,obs_dim//2)
        hidden=max(16,int(learn_conf.get("隐藏单元",128)))
        obs_dim=max(8,int(obs_dim))
        act_dim=max(8,int(act_dim))
        mid=max(8,int(hidden//2))
        return obs_dim,hidden,act_dim,mid
    def build_template(self):
        obs_dim,hidden,act_dim,mid=self.model_dims()
        return nn.Sequential(nn.Linear(obs_dim,hidden),nn.ReLU(),nn.Linear(hidden,mid),nn.ReLU(),nn.Linear(mid,act_dim))
    def load_single(self,path):
        try:
            loaded=torch.load(path,map_location="cpu")
        except:
            return None
        if isinstance(loaded,nn.Module):
            model=loaded.to(self.device)
            model=model.to(self.dtype)
            model.eval()
            return model
        net=self.build_template()
        if isinstance(loaded,dict):
            try:
                net.load_state_dict(loaded,strict=False)
            except:
                pass
        net=net.to(self.device)
        net=net.to(self.dtype)
        net.eval()
        return net
    def ensure_models(self):
        model_dir=self.config["路径"]["AAA"]
        template=self.build_template()
        state_dict=template.state_dict()
        for name,filename in self.config["模型文件"].items():
            path=os.path.join(model_dir,filename)
            if not os.path.isfile(path):
                torch.save(state_dict,path)
    def load_models(self):
        model_dir=self.config["路径"]["AAA"]
        for name,filename in self.config["模型文件"].items():
            path=os.path.join(model_dir,filename)
            self.models[name]=self.load_single(path)
        for f in glob.glob(os.path.join(model_dir,"*.pt"))+glob.glob(os.path.join(model_dir,"*.pth")):
            if f not in [os.path.join(model_dir,x) for x in self.config["模型文件"].values()]:
                self.models[os.path.basename(f)]=self.load_single(f)
    def infer(self,name,input_vec):
        model=self.models.get(name)
        if model is None:
            return None
        try:
            with torch.no_grad():
                t=torch.tensor(input_vec,dtype=torch.float32,device=self.device)
                if t.ndim==1:
                    t=t.unsqueeze(0)
                out=model(t)
                return out.squeeze(0).detach().cpu().numpy()
        except:
            return None
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
    def __init__(self,device,dtype,config,obs_dim,action_dim,hidden_dim,buffer_size,batch_size,gamma,lr,target_sync,epsilon_decay,experience_dir,weight_decay,value_reg):
        self.device=device
        self.dtype=dtype
        self.hidden_dim=hidden_dim
        self.buffer_size=buffer_size
        self.lr=lr
        self.obs_dim=obs_dim
        self.action_dim=action_dim
        self.build_networks()
        self.weight_decay=weight_decay
        self.value_reg=value_reg
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=lr,weight_decay=self.weight_decay)
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
    def build_networks(self):
        self.policy_net=QNetwork(self.obs_dim,self.action_dim,self.hidden_dim,self.dtype).to(self.device)
        self.target_net=QNetwork(self.obs_dim,self.action_dim,self.hidden_dim,self.dtype).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    def rebuild(self,obs_dim,action_dim,lr=None):
        if obs_dim==self.obs_dim and action_dim==self.action_dim:
            return
        self.obs_dim=obs_dim
        self.action_dim=action_dim
        self.build_networks()
        if lr is not None:
            self.lr=lr
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        self.replay_buffer=deque(maxlen=self.buffer_size)
        self.step_count=0
        self.last_save_step=0
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
        policy_out=self.policy_net(state_batch)
        q_values=policy_out.gather(1,action_batch)
        with torch.no_grad():
            next_policy=self.policy_net(next_state_batch)
            next_actions=torch.argmax(next_policy,dim=1,keepdim=True)
            next_q=self.target_net(next_state_batch).gather(1,next_actions)
            reward_batch=reward_batch.to(next_q.dtype)
            done_batch=done_batch.to(next_q.dtype)
            target_q=reward_batch+self.gamma*(1.0-done_batch)*next_q
        base_loss=nn.functional.smooth_l1_loss(q_values.to(target_q.dtype),target_q)
        reg_loss=self.value_reg*torch.mean(policy_out.to(target_q.dtype)**2)
        loss=base_loss+reg_loss
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
    def offline_optimize(self,progress_callback=None):
        total=len(self.replay_buffer)
        if total<self.batch_size:
            if progress_callback:
                progress_callback(1.0)
            return 0
        cycles=max(1,total//self.batch_size)
        steps=max(10,cycles*5)
        for i in range(steps):
            self.train_step()
            if progress_callback:
                progress_callback((i+1)/steps)
        if progress_callback:
            progress_callback(1.0)
        return steps
class VisionModule:
    def __init__(self,config):
        self.config=config
        self.prev_ready={}
        self.recall_ref=None
        self.recall_trace=0.0
    def crop_cv(self,pil_img,x,y,w,h):
        sx=max(0,int(x))
        sy=max(0,int(y))
        ex=min(pil_img.width,max(1,int(x+w)))
        ey=min(pil_img.height,max(1,int(y+h)))
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
        bright_ratio=self.config["OCR"].get("存活亮度比",0.8)
        sat_ratio=self.config["OCR"].get("存活饱和比",0.6)
        return 1 if mean_v>self.config["OCR"]["亮度阈值"]*bright_ratio and mean_s>self.config["OCR"]["饱和阈值"]*sat_ratio else 0
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
    def recall_signature(self,cv_img):
        if cv_img is None or cv_img.size==0:
            return 0.0,0.0,0.0,0.0
        mean_v,mean_s,std_v=self.region_stats(cv_img)
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        lap=cv2.Laplacian(gray,cv2.CV_32F)
        energy=float(np.mean(np.abs(lap)))
        return mean_v,mean_s,std_v,energy
    def set_recall_baseline(self,cv_img):
        self.recall_ref=self.recall_signature(cv_img)
        return self.recall_ref
    def recall_activity(self,cv_img):
        sig=self.recall_signature(cv_img)
        if self.recall_ref is None:
            self.recall_ref=sig
        base_v,base_s,base_std,base_eng=self.recall_ref
        dv=max(0.0,base_v-sig[0])
        ds=max(0.0,sig[1]-base_s)
        dst=max(0.0,sig[2]-base_std)
        de=max(0.0,sig[3]-base_eng)
        scale=max(base_v+base_s+base_std+base_eng,1.0)
        energy=dv*0.45+ds*0.2+dst*0.2+de*0.15
        progress=float(np.clip(energy/(scale+energy),0.0,1.0))
        threshold=max(0.05,0.08*(base_std+1.0)/scale)
        active=1 if energy>scale*threshold else 0
        if active==1:
            self.recall_trace=self.recall_trace*0.6+progress*0.4
        else:
            decay=0.85
            self.recall_ref=(base_v*decay+sig[0]*(1.0-decay),base_s*decay+sig[1]*(1.0-decay),base_std*decay+sig[2]*(1.0-decay),base_eng*decay+sig[3]*(1.0-decay))
            self.recall_trace=self.recall_trace*0.4
        completed=1 if active==0 and self.recall_trace>0.9 else 0
        return active,self.recall_trace if active==1 else progress,completed
    def hero_home(self,minimap_img):
        if minimap_img is None or getattr(minimap_img,"size",0)==0:
            return 0
        h=minimap_img.shape[0]
        w=minimap_img.shape[1]
        size=max(6,int(min(h,w)*0.18))
        y0=max(0,h-size)
        x0=max(0,w-size)
        region=minimap_img[y0:h,x0:w]
        if region.size==0:
            return 0
        hsv=cv2.cvtColor(region,cv2.COLOR_BGR2HSV)
        lower=np.array([90,40,40],dtype=np.uint8)
        upper=np.array([140,255,255],dtype=np.uint8)
        mask=cv2.inRange(hsv,lower,upper)
        ratio=float(np.sum(mask>0))/mask.size
        energy=float(np.mean(hsv[:,:,2]))/255.0
        return 1 if ratio>0.12 and energy>0.35 else 0
class RecallMonitor:
    def __init__(self,config,vision):
        self.config=config
        self.vision=vision
        self.active=False
        self.start_time=0.0
        self.status="idle"
        self.progress=0.0
        self.timeout=float(self.config["动作冷却"].get("回城等待",8.0))
    def begin(self,button_img,minimap_img):
        self.timeout=float(self.config["动作冷却"].get("回城等待",8.0))
        self.active=True
        self.start_time=time.time()
        self.status="active"
        self.progress=0.0
        if button_img is not None:
            self.vision.set_recall_baseline(button_img)
        if minimap_img is not None:
            self.last_home=self.vision.hero_home(minimap_img)
        else:
            self.last_home=0
    def update(self,button_img,minimap_img):
        if not self.active:
            return False
        active,progress,completed=self.vision.recall_activity(button_img)
        home=self.vision.hero_home(minimap_img) if minimap_img is not None else 0
        now=time.time()
        if home==1 or completed==1:
            self.active=False
            self.status="completed"
            self.progress=1.0
            return False
        if now-self.start_time>=self.timeout:
            self.active=False
            self.status="timeout"
            self.progress=max(self.progress,progress)
            return False
        if active==0 and progress<0.25 and now-self.start_time>0.8:
            self.active=False
            self.status="interrupted"
            self.progress=max(self.progress,progress)
            return False
        self.progress=max(self.progress,progress)
        self.status="active" if active==1 else "stabilizing"
        return True
    def cancel(self):
        self.active=False
        self.status="cancelled"
        self.progress=0.0
class ExperienceRecorder:
    def __init__(self,config):
        self.base_dir=os.path.join(config["路径"]["AAA"],config["经验目录"])
        self.frame_dir=os.path.join(self.base_dir,"frames")
        self.input_path=os.path.join(self.base_dir,"inputs.log")
        self.metric_path=os.path.join(self.base_dir,"metrics.log")
        os.makedirs(self.frame_dir,exist_ok=True)
        self.lock=threading.Lock()
        self.frame_paths=deque()
        learn_conf=config.get("学习",{})
        self.max_frames=max(200,int(learn_conf.get("缓冲大小",20000)//10))
        self.paused=False
    def set_paused(self,value):
        self.paused=bool(value)
    def record_input(self,mode,event_type,data):
        if self.paused:
            return
        payload={"t":time.time(),"mode":mode,"type":event_type,"data":data}
        try:
            with self.lock:
                with open(self.input_path,"a",encoding="utf-8") as f:
                    f.write(json.dumps(payload,ensure_ascii=False)+"\n")
        except:
            pass
    def record_metrics(self,mode,metrics):
        if self.paused:
            return
        payload={"t":time.time(),"mode":mode,"metrics":metrics}
        try:
            with self.lock:
                with open(self.metric_path,"a",encoding="utf-8") as f:
                    f.write(json.dumps(payload,ensure_ascii=False)+"\n")
        except:
            pass
    def record_frame(self,mode,image):
        if image is None:
            return
        if self.paused:
            return
        ts="{:.3f}".format(time.time()).replace(".","_")
        name=f"{mode}_{ts}_{len(self.frame_paths)}.png"
        path=os.path.join(self.frame_dir,name)
        try:
            with self.lock:
                image.save(path)
                self.frame_paths.append(path)
                while len(self.frame_paths)>self.max_frames:
                    old=self.frame_paths.popleft()
                    try:
                        os.remove(old)
                    except:
                        pass
        except:
            pass
class InputMonitor:
    def __init__(self,mode_manager):
        self.mode_manager=mode_manager
        self.keyboard_listener=None
        self.mouse_listener=None
        self.available=mouse is not None and keyboard is not None
        interval=self.mode_manager.config["动作"].get("循环间隔",0.2)
        self.move_interval=max(0.05,float(interval))
        self.last_move=0.0
    def start(self):
        if not self.available:
            return
        try:
            self.keyboard_listener=keyboard.Listener(on_press=self.on_key_press,on_release=self.on_key_release)
            self.keyboard_listener.start()
            self.mouse_listener=mouse.Listener(on_click=self.on_click,on_scroll=self.on_scroll,on_move=self.on_move)
            self.mouse_listener.start()
        except:
            self.available=False
    def stop(self):
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except:
                pass
        if self.mouse_listener:
            try:
                self.mouse_listener.stop()
            except:
                pass
    def on_key_press(self,key):
        self.mode_manager.handle_user_event("键盘",{"key":str(key),"pressed":True})
    def on_key_release(self,key):
        self.mode_manager.handle_user_event("键盘",{"key":str(key),"pressed":False})
    def on_click(self,x,y,button,pressed):
        self.mode_manager.handle_user_event("鼠标",{"button":str(button),"pressed":pressed,"x":x,"y":y})
    def on_scroll(self,x,y,dx,dy):
        self.mode_manager.handle_user_event("鼠标滚轮",{"dx":dx,"dy":dy,"x":x,"y":y})
    def on_move(self,x,y):
        now=time.time()
        if now-self.last_move>=self.move_interval:
            self.last_move=now
            self.mode_manager.handle_user_event("鼠标移动",{"x":x,"y":y})
class ModeManager:
    def __init__(self,shared,controller,recorder,config):
        self.shared=shared
        self.controller=controller
        self.recorder=recorder
        self.config=config
        self.current_mode="学习"
        mode_conf=config.get("模式",{})
        self.idle_threshold=float(mode_conf.get("学习静默阈值",10.0))
        self.learn_interval=float(mode_conf.get("学习采样间隔",0.4))
        self.train_interval=float(mode_conf.get("训练采样间隔",0.2))
        self.last_input=time.time()
        self.last_frame=0.0
        self.running=False
        self.thread=None
        self.controller.set_recorder(recorder)
        self.optimizing=False
        self.optimize_wait=False
    def start(self):
        if self.running:
            return
        with self.shared.lock:
            self.shared.mode=self.current_mode
        self.running=True
        self.thread=threading.Thread(target=self.loop,daemon=True)
        self.thread.start()
    def stop(self):
        self.running=False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
    def handle_user_event(self,event_type,data):
        now=time.time()
        self.last_input=now
        payload={k:str(v) for k,v in data.items()}
        self.recorder.record_input(self.current_mode,event_type,payload)
        if event_type=="键盘" and payload.get("key") in ("Key.esc","esc","<esc>","\u001b"):
            self.shared.terminate=True
            self.controller.stop()
            self.recorder.set_paused(False)
            with self.shared.lock:
                self.shared.mode="学习"
                self.shared.running=False
                self.shared.optimizing=False
            self.current_mode="学习"
            self.last_frame=0.0
            self.optimize_wait=False
            return
        if self.current_mode=="训练":
            self.controller.stop()
            with self.shared.lock:
                self.shared.mode="学习"
                self.shared.running=False
            self.current_mode="学习"
            self.last_frame=0.0
    def switch_to_train(self):
        if self.current_mode=="训练" or self.shared.terminate or self.optimizing or self.optimize_wait:
            return
        self.controller.start()
        with self.shared.lock:
            self.shared.mode="训练"
        self.current_mode="训练"
        self.last_frame=0.0
    def force_learning(self):
        self.controller.stop()
        self.recorder.set_paused(False)
        with self.shared.lock:
            self.shared.mode="学习"
            self.shared.running=False
            self.shared.optimizing=False
        self.current_mode="学习"
        self.last_frame=0.0
        self.last_input=time.time()
        self.optimize_wait=False
    def enter_optimization(self):
        if self.optimizing:
            return
        self.controller.stop()
        self.recorder.set_paused(True)
        with self.shared.lock:
            self.shared.mode="优化"
            self.shared.running=False
            self.shared.optimizing=True
        self.current_mode="优化"
        self.optimizing=True
        self.optimize_wait=False
    def exit_optimization(self):
        self.optimizing=False
        self.optimize_wait=True
        with self.shared.lock:
            self.shared.mode="优化完成"
            self.shared.running=False
            self.shared.optimizing=False
        self.last_frame=0.0
    def finalize_optimization(self):
        self.recorder.set_paused(False)
        with self.shared.lock:
            self.shared.mode="学习"
            self.shared.running=False
            self.shared.optimizing=False
        self.current_mode="学习"
        self.last_input=time.time()
        self.last_frame=0.0
        self.optimize_wait=False
    def loop(self):
        while self.running and not self.shared.terminate:
            now=time.time()
            if self.optimizing or self.optimize_wait:
                time.sleep(0.2)
                continue
            if self.current_mode=="学习" and now-self.last_input>=self.idle_threshold:
                self.switch_to_train()
            interval=self.learn_interval if self.current_mode=="学习" else self.train_interval
            if now-self.last_frame>=interval:
                frame=self.controller.game.get_screenshot() if self.current_mode=="学习" else self.controller.game.last_screenshot
                if self.current_mode=="训练" and frame is None:
                    frame=self.controller.game.get_screenshot()
                if frame is not None:
                    self.recorder.record_frame(self.current_mode,frame)
                    metrics=self.controller.game.get_metrics(frame,update_prev=False)
                    self.recorder.record_metrics(self.current_mode,metrics)
                self.last_frame=now
            time.sleep(max(0.1,interval*0.5))
class GameInterface:
    def __init__(self,config,adb_path,model_manager,experience_dir):
        self.config=config
        self.adb_path=adb_path
        self.model_manager=model_manager
        self.scaler=RegionScaler(config)
        self.screen_w=self.scaler.screen_w
        self.screen_h=self.scaler.screen_h
        self.vision=VisionModule(config)
        self.last_screenshot=None
        self.obs_keys=list(self.config.get("观测键",["A","B","C","alive","cd1","cd2","cd3","cd_item","cd_heal","cd_flash","recalling"]))
        self.metric_caps={k:float(v) for k,v in self.config.get("数值上限",{}).items()}
        self.binary_keys={k for k in self.obs_keys if k=="alive" or k.startswith("cd") or k=="recalling"}
        self.action_order=list(self.config.get("动作名称",["idle","move","attack_minion","attack_tower","skill1","skill2","skill3","item","heal","recall","flash"]))
        self.obs_dim=len(self.obs_keys)
        self.action_dim=len(self.action_order)
        self.joystick_ratio=float(self.config["动作"].get("摇杆幅度",0.8))
        self.circle_map={"joystick":"移动轮盘","recall":"回城","heal":"恢复","flash":"闪现","skill1":"一技能","skill2":"二技能","skill3":"三技能","cancel":"取消施法","attack_min":"普攻补刀","attack_tower":"普攻点塔","active_item":"主动装备"}
        self.rect_map={"A":"A","B":"B","C":"C","minimap":"小地图"}
        self.circle_regions={}
        self.rect_regions={}
        self.refresh_regions()
        self.action_conditions={"idle":lambda m:True,"move":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0,"attack_minion":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0,"attack_tower":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0,"skill1":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0 and m.get("cd1",0)==1,"skill2":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0 and m.get("cd2",0)==1,"skill3":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0 and m.get("cd3",0)==1,"item":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0 and m.get("cd_item",0)==1,"heal":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0 and m.get("cd_heal",0)==1,"recall":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0,"flash":lambda m:m.get("alive",0)==1 and m.get("recalling",0)==0 and m.get("cd_flash",0)==1}
        self.action_funcs={"idle":self.act_idle,"move":self.act_move,"attack_minion":self.act_attack_minion,"attack_tower":self.act_attack_tower,"skill1":self.skill_executor("skill1"),"skill2":self.skill_executor("skill2"),"skill3":self.skill_executor("skill3"),"item":self.act_item,"heal":self.act_heal,"recall":self.act_recall,"flash":self.act_flash}
        self.prev_metrics={"A":0,"B":0,"C":0,"alive":0,"cd1":0,"cd2":0,"cd3":0,"cd_item":0,"cd_heal":0,"cd_flash":0,"recalling":0}
        learn_conf=self.config.get("学习",{})
        window=int(max(8,learn_conf.get("奖励平滑窗口",len(self.action_order)*6)))
        self.reward_memory=deque(maxlen=window)
        self.metric_history={k:deque(maxlen=window) for k in ("A","B","C")}
        self.reward_gain=float(learn_conf.get("奖励调节",0.4))
        self.experience_dir=experience_dir
        os.makedirs(self.experience_dir,exist_ok=True)
        self.skill_path_file=os.path.join(self.experience_dir,"skill_paths.json")
        self.skill_paths=self.load_skill_paths()
        self.recall_monitor=RecallMonitor(self.config,self.vision)
    def refresh_regions(self):
        for alias,name in self.circle_map.items():
            self.circle_regions[alias]=self.scaler.circle(name)
        for alias,name in self.rect_map.items():
            self.rect_regions[alias]=self.scaler.rect(name)
        self.screen_w=self.scaler.screen_w
        self.screen_h=self.scaler.screen_h
    def load_skill_paths(self):
        if not os.path.isfile(self.skill_path_file):
            return {}
        try:
            with open(self.skill_path_file,"r",encoding="utf-8") as f:
                data=json.load(f)
            if isinstance(data,dict):
                return data
        except:
            pass
        return {}
    def save_skill_paths(self):
        try:
            with open(self.skill_path_file,"w",encoding="utf-8") as f:
                json.dump(self.skill_paths,f,ensure_ascii=False)
        except:
            pass
    def experience_path(self,key,center,radius,seed):
        entry=self.skill_paths.get(key)
        if not isinstance(entry,list) or len(entry)==0:
            return None,False
        idx=int(abs(seed)%len(entry))
        data=entry[idx]
        pts=data.get("points")
        cancel=bool(data.get("cancel",False))
        if not isinstance(pts,list):
            return None,cancel
        path=[center]
        for item in pts:
            if not isinstance(item,list) or len(item)<2:
                continue
            px=center[0]+float(item[0])*radius
            py=center[1]+float(item[1])*radius
            path.append((px,py))
        return path,cancel
    def record_skill_path(self,key,path,cancel,center,radius):
        if len(path)<=1:
            return
        norm=[]
        inv=max(radius,1.0)
        for pt in path[1:]:
            norm.append([(pt[0]-center[0])/inv,(pt[1]-center[1])/inv])
        data={"points":norm,"cancel":1 if cancel else 0,"t":time.time()}
        bucket=self.skill_paths.get(key,[])
        bucket=[d for d in bucket if isinstance(d,dict)]
        bucket.append(data)
        limit=max(8,int(self.config["学习"].get("批次",64)//2))
        if len(bucket)>limit:
            bucket=bucket[-limit:]
        self.skill_paths[key]=bucket
        self.save_skill_paths()
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
            self.scaler.update(*img.size)
            self.refresh_regions()
            return img
        except:
            return self.last_screenshot
    def scaled_xy(self,x,y):
        return int(x),int(y)
    def circle_center(self,region):
        return region.x+region.d/2.0,region.y+region.d/2.0
    def tap_circle(self,region):
        cx,cy=self.circle_center(region)
        sx,sy=self.scaled_xy(cx,cy)
        self.adb_tap(sx,sy)
    def skill_executor(self,key):
        def runner(metrics):
            self.cast_skill(key,metrics)
        return runner
    def act_idle(self,metrics):
        return
    def act_move(self,metrics):
        self.joystick_move_random(metrics)
    def act_attack_minion(self,metrics):
        self.basic_attack_minion()
    def act_attack_tower(self,metrics):
        self.basic_attack_tower()
    def act_item(self,metrics):
        self.use_item()
    def act_heal(self,metrics):
        self.heal()
        time.sleep(self.config["动作冷却"]["恢复时长"])
    def act_recall(self,metrics):
        self.recall()
    def act_flash(self,metrics):
        self.flash_forward()
    def joystick_move_random(self,metrics):
        region=self.circle_regions["joystick"]
        cx,cy=self.circle_center(region)
        radius=max(region.d/2.0*self.joystick_ratio,1.0)
        obs=self.metrics_to_obs(metrics)
        seed=time.time()+random.random()
        recorded,_=self.experience_path("joystick",(cx,cy),radius,seed)
        base_speed=float(self.config["动作"].get("拖动耗时",0.2))*1000.0
        base_speed=max(base_speed,1.0)
        if recorded:
            path=list(recorded)
            speed_scale=1.0
        else:
            vec=None
            if self.model_manager:
                vec=self.model_manager.infer("移动轮盘",obs)
                if vec is not None:
                    try:
                        vec=np.array(vec,dtype=np.float32).flatten()
                    except:
                        vec=None
            if vec is not None and vec.size>=2:
                dir_x=float(vec[0])
                dir_y=float(vec[1])
            else:
                ang=(seed*math.tau) if hasattr(math,"tau") else seed*2.0*math.pi
                dir_x=math.cos(ang)
                dir_y=math.sin(ang)
            norm=math.hypot(dir_x,dir_y)
            if norm<=np.finfo(np.float32).eps:
                dir_x=0.0
                dir_y=1.0
                norm=1.0
            dir_x/=norm
            dir_y/=norm
            rotation_bias=float(vec[2]) if vec is not None and vec.size>=3 else random.uniform(-1.0,1.0)
            spin=1 if rotation_bias>=0 else -1
            velocity=float(vec[3]) if vec is not None and vec.size>=4 else random.uniform(0.35,0.9)
            velocity=max(0.25,min(0.95,abs(velocity)))
            arc_hint=float(vec[4]) if vec is not None and vec.size>=5 else random.uniform(math.pi/3.0,math.pi*1.2)
            arc=abs(arc_hint)
            tau=math.tau if hasattr(math,"tau") else math.pi*2.0
            arc=max(math.pi/6.0,min(tau/2.0,arc))
            start_angle=math.atan2(dir_y,dir_x)
            edge=(cx+dir_x*radius,cy+dir_y*radius)
            steps=max(6,int(arc/(math.pi/18.0)))
            path=[(cx,cy),edge]
            for i in range(1,steps+1):
                angle=start_angle+spin*arc*(i/steps)
                px=cx+math.cos(angle)*radius
                py=cy+math.sin(angle)*radius
                path.append((px,py))
            speed_scale=1.0-velocity*0.5
        hold=int(max(1,base_speed*0.3))
        sx,sy=self.scaled_xy(cx,cy)
        self.adb_swipe(sx,sy,sx,sy,hold)
        total=0.0
        segments=[]
        for i in range(len(path)-1):
            x1,y1=path[i]
            x2,y2=path[i+1]
            dist=math.hypot(x2-x1,y2-y1)
            if dist<=0.0:
                continue
            segments.append((x1,y1,x2,y2,dist))
            total+=dist
        if not segments:
            return
        for x1,y1,x2,y2,dist in segments:
            duration=int(max(1,base_speed*(dist/total)*speed_scale))
            sx1,sy1=self.scaled_xy(x1,y1)
            sx2,sy2=self.scaled_xy(x2,y2)
            self.adb_swipe(sx1,sy1,sx2,sy2,duration)
        lx,ly=self.scaled_xy(path[-1][0],path[-1][1])
        release=int(max(1,base_speed*0.25))
        self.adb_swipe(lx,ly,lx,ly,release)
        self.record_skill_path("joystick",path,False,(cx,cy),radius)
    def basic_attack_minion(self):
        self.tap_circle(self.circle_regions["attack_min"])
    def basic_attack_tower(self):
        self.tap_circle(self.circle_regions["attack_tower"])
    def model_direction(self,key,obs):
        name_map={"skill1":"一技能","skill2":"二技能","skill3":"三技能"}
        name=name_map.get(key)
        if not name or not self.model_manager:
            return None
        out=self.model_manager.infer(name,obs)
        if out is None:
            return None
        try:
            arr=np.array(out,dtype=np.float32).flatten()
        except:
            return None
        if arr.size==0:
            return None
        return arr
    def generate_drag_path(self,key,metrics):
        region=self.circle_regions[key]
        center=self.circle_center(region)
        radius=max(region.d/2.0,1.0)
        obs=self.metrics_to_obs(metrics)
        seed=sum(obs)+metrics.get("A",0)+metrics.get("C",0)+time.time()%1.0
        path,cancel=self.experience_path(key,center,radius,seed)
        if not path:
            vec=self.model_direction(key,obs)
            direction=None
            cancel_score=0.0
            curvature=seed
            if vec is not None:
                length=float(np.linalg.norm(vec[:2])) if vec.size>=2 else 0.0
                denom=max(length,np.finfo(np.float32).eps)
                scale=length/(1.0+length) if length>0 else 0.0
                base_mag=radius*self.joystick_ratio*scale if scale>0 else radius*self.joystick_ratio*0.5
                direction=(vec[0]/denom if vec.size>=1 else 1.0,vec[1]/denom if vec.size>=2 else 0.0)
                cancel_score=float(vec[2]) if vec.size>=3 else 0.0
                curvature=float(vec[3]) if vec.size>=4 else seed
                target=(center[0]+direction[0]*base_mag,center[1]+direction[1]*base_mag)
            else:
                ang=seed*math.tau if hasattr(math,"tau") else seed*math.pi*2.0
                direction=(math.cos(ang),math.sin(ang))
                target=(center[0]+direction[0]*radius*self.joystick_ratio,center[1]+direction[1]*radius*self.joystick_ratio)
                curvature=seed
            if direction is None:
                direction=(0.0,1.0)
                target=(center[0],center[1]+radius*self.joystick_ratio)
            path=[center]
            steps=max(3,int(len(self.obs_keys)/2))
            unit=math.sqrt(direction[0]**2+direction[1]**2)
            if unit<=np.finfo(np.float32).eps:
                direction=(0.0,1.0)
                unit=1.0
            perp=(-direction[1]/unit,direction[0]/unit)
            readiness=max(0.0,metrics.get("cd1",0)+metrics.get("cd2",0)+metrics.get("cd3",0)+metrics.get("cd_flash",0)+metrics.get("cd_item",0))
            caution=max(0.0,metrics.get("B",0)+1.0)/(metrics.get("A",0)+metrics.get("C",0)+1.0)
            curve_gain=self.reward_gain/(len(self.obs_keys)+1)
            for i in range(1,steps+1):
                ratio=i/steps
                base_x=center[0]+(target[0]-center[0])*ratio
                base_y=center[1]+(target[1]-center[1])*ratio
                wave=math.sin(curvature*ratio+self.reward_gain*readiness)
                offset=radius*self.joystick_ratio*wave*curve_gain*caution
                path.append((base_x+perp[0]*offset,base_y+perp[1]*offset))
            cancel_prob=1.0/(1.0+math.exp(-cancel_score)) if cancel_score!=0.0 else 0.0
            threshold=0.5+self.reward_gain/(2.0+self.reward_gain)
            cancel=cancel or metrics.get("alive",1)==0 or cancel_prob>threshold
        return path,cancel
    def drag_path(self,path):
        if not path or len(path)<2:
            return
        total=0.0
        segments=[]
        for i in range(len(path)-1):
            x1,y1=path[i]
            x2,y2=path[i+1]
            dist=math.hypot(x2-x1,y2-y1)
            segments.append((x1,y1,x2,y2,dist))
            total+=dist
        base=self.config["动作"].get("拖动耗时",0.2)*1000.0
        base=max(base,1.0)
        for x1,y1,x2,y2,dist in segments:
            duration=int(base*(dist/total if total>0 else 1.0))
            duration=max(duration,1)
            sx,sy=self.scaled_xy(x1,y1)
            ex,ey=self.scaled_xy(x2,y2)
            self.adb_swipe(sx,sy,ex,ey,duration)
    def cast_skill(self,key,metrics):
        path,cancel=self.generate_drag_path(key,metrics)
        if cancel:
            cancel_region=self.circle_regions["cancel"]
            cancel_center=self.circle_center(cancel_region)
            path=list(path)
            path.append(cancel_center)
        self.drag_path(path)
        region=self.circle_regions[key]
        center=self.circle_center(region)
        self.record_skill_path(key,path,cancel,center,max(region.d/2.0,1.0))
    def use_item(self):
        self.tap_circle(self.circle_regions["active_item"])
    def heal(self):
        self.tap_circle(self.circle_regions["heal"])
    def recall(self):
        region=self.circle_regions["recall"]
        minimap_region=self.rect_regions["minimap"]
        self.tap_circle(region)
        base_img=self.get_screenshot()
        if base_img is not None:
            recall_img=self.vision.crop_cv(base_img,region.x,region.y,region.d,region.d)
            minimap_img=self.vision.crop_cv(base_img,minimap_region.x,minimap_region.y,minimap_region.w,minimap_region.h)
        else:
            recall_img=None
            minimap_img=None
        self.recall_monitor.begin(recall_img,minimap_img)
        self.prev_metrics["recalling"]=1
        timeout=float(self.config["动作冷却"].get("回城等待",8.0))
        interval=max(0.1,float(self.config["动作"].get("循环间隔",0.2))*0.5)
        start=time.time()
        while self.recall_monitor.active:
            time.sleep(interval)
            img=self.get_screenshot()
            if img is None:
                if time.time()-start>timeout:
                    self.recall_monitor.cancel()
                    break
                continue
            recall_img=self.vision.crop_cv(img,region.x,region.y,region.d,region.d)
            minimap_img=self.vision.crop_cv(img,minimap_region.x,minimap_region.y,minimap_region.w,minimap_region.h)
            if not self.recall_monitor.update(recall_img,minimap_img):
                break
            if time.time()-start>timeout:
                self.recall_monitor.cancel()
                break
        self.prev_metrics["recalling"]=0
    def flash_forward(self):
        region=self.circle_regions["flash"]
        cx,cy=self.circle_center(region)
        cooldown=self.config["动作冷却"]
        ratio=cooldown.get("闪现位移比例")
        if ratio:
            ex=cx+ratio[0]*self.screen_w
            ey=cy+ratio[1]*self.screen_h
        else:
            offset=cooldown.get("闪现位移",[0.0,0.0])
            ex=cx+offset[0]
            ey=cy+offset[1]
        sx,sy=self.scaled_xy(cx,cy)
        exs,eys=self.scaled_xy(ex,ey)
        self.adb_swipe(sx,sy,exs,eys,int(self.config["动作"]["拖动耗时"]*1000))
    def get_metrics(self,img=None,update_prev=True):
        capture_needed=img is None
        if capture_needed:
            img=self.get_screenshot()
        if img is None:
            return self.prev_metrics
        v=self.vision
        def crop_circle(region):
            return v.crop_cv(img,region.x,region.y,region.d,region.d)
        def crop_rect(region):
            return v.crop_cv(img,region.x,region.y,region.w,region.h)
        A_img=crop_rect(self.rect_regions["A"])
        B_img=crop_rect(self.rect_regions["B"])
        C_img=crop_rect(self.rect_regions["C"])
        minimap_img=crop_rect(self.rect_regions["minimap"])
        attack_img=crop_circle(self.circle_regions["attack_min"])
        skill1_img=crop_circle(self.circle_regions["skill1"])
        skill2_img=crop_circle(self.circle_regions["skill2"])
        skill3_img=crop_circle(self.circle_regions["skill3"])
        item_img=crop_circle(self.circle_regions["active_item"])
        heal_img=crop_circle(self.circle_regions["heal"])
        flash_img=crop_circle(self.circle_regions["flash"])
        recall_img=crop_circle(self.circle_regions["recall"])
        recall_active,recall_trace,recall_completed=v.recall_activity(recall_img)
        home_flag=v.hero_home(minimap_img)
        if self.recall_monitor.active:
            self.recall_monitor.progress=max(self.recall_monitor.progress,recall_trace)
        metrics={"A":v.ocr_digits(A_img),"B":v.ocr_digits(B_img),"C":v.ocr_digits(C_img),"alive":v.alive_state(attack_img),"cd1":v.cooldown_ready("skill1",skill1_img),"cd2":v.cooldown_ready("skill2",skill2_img),"cd3":v.cooldown_ready("skill3",skill3_img),"cd_item":v.cooldown_ready("item",item_img),"cd_heal":v.cooldown_ready("heal",heal_img),"cd_flash":v.cooldown_ready("flash",flash_img),"recalling":1 if (recall_active==1 or self.recall_monitor.active) and home_flag==0 else 0}
        if recall_completed==1 and not self.recall_monitor.active:
            metrics["recalling"]=0
        if home_flag==1 and self.recall_monitor.active:
            self.recall_monitor.active=False
            self.recall_monitor.status="completed"
            self.recall_monitor.progress=1.0
        if home_flag==1 and metrics["recalling"]==1:
            metrics["recalling"]=0
        if update_prev or capture_needed:
            self.prev_metrics=metrics
        return metrics
    def metrics_to_obs(self,m):
        obs=[]
        for key in self.obs_keys:
            value=float(m.get(key,0))
            if key in self.metric_caps:
                cap=max(1.0,self.metric_caps[key])
                obs.append(min(value,cap)/cap)
            elif key in self.binary_keys:
                obs.append(1.0 if value>=1.0 else 0.0)
            else:
                obs.append(value)
        return obs
    def compute_reward(self,prev_metrics,curr_metrics):
        dA=curr_metrics["A"]-prev_metrics["A"]
        dC=curr_metrics["C"]-prev_metrics["C"]
        dB=curr_metrics["B"]-prev_metrics["B"]
        weights=self.config["奖励"]
        for key,delta in (("A",dA),("B",dB),("C",dC)):
            self.metric_history[key].append(delta)
        trend_A=float(np.mean(self.metric_history["A"])) if self.metric_history["A"] else 0.0
        trend_C=float(np.mean(self.metric_history["C"])) if self.metric_history["C"] else 0.0
        trend_B=float(np.mean(self.metric_history["B"])) if self.metric_history["B"] else 0.0
        gain=max(0.0,self.reward_gain)
        boost_A=max(dA,0.0)*weights["A"]*(1.0+gain*np.clip(trend_A,-1.0,1.0))
        boost_C=max(dC,0.0)*weights["C"]*(1.0+gain*np.clip(trend_C,-1.0,1.0))
        penalty=max(dB,0.0)*abs(weights["B"])*(1.0+gain*np.clip(max(0.0,trend_B),0.0,1.5))
        reward=boost_A+boost_C-penalty
        if curr_metrics.get("alive",0)==0 and prev_metrics.get("alive",1)==1:
            reward-=abs(weights["B"])*(1.0+gain)
        flash_delta=curr_metrics.get("cd_flash",0)-prev_metrics.get("cd_flash",0)
        readiness_scale=(abs(weights["A"])+abs(weights["C"]))/(abs(weights["B"])+1.0)
        recall_block=1.0 if curr_metrics.get("recalling",0)==1 else 0.0
        reward+=readiness_scale*(flash_delta-recall_block/(len(self.obs_keys)+1))
        self.reward_memory.append(reward)
        if self.reward_memory:
            smooth=float(np.clip(np.mean(self.reward_memory),-gain,gain))
            reward*=1.0+smooth
        return reward
    def execute_action(self,action_idx,metrics_prev):
        if action_idx<0 or action_idx>=self.action_dim:
            return
        action_name=self.action_order[action_idx]
        condition=self.action_conditions.get(action_name)
        if condition and not condition(metrics_prev):
            return
        if self.recall_monitor.active and action_name!="recall":
            return
        executor=self.action_funcs.get(action_name)
        if executor:
            executor(metrics_prev)
class BotController:
    def __init__(self,shared,agent,model_manager,game):
        self.shared=shared
        self.agent=agent
        self.model_manager=model_manager
        self.game=game
        self.adb_path=self.game.adb_path
        self.dnplayer_path=shared.config["路径"]["模拟器"]
        self.running_event=threading.Event()
        self.thread=None
        self.recorder=None
    def update_paths(self,adb_path,dnplayer_path,aaa_path):
        if adb_path:
            self.adb_path=adb_path
        if dnplayer_path:
            self.dnplayer_path=dnplayer_path
        cm=self.shared.config_manager
        target_aaa=aaa_path if aaa_path else cm.config["路径"]["AAA"]
        cm.update_paths(self.adb_path,self.dnplayer_path,target_aaa)
        self.shared.config=cm.config
        experience_dir=os.path.join(cm.config["路径"]["AAA"],cm.config["经验目录"])
        new_model_manager=ModelManager(cm.config,self.agent.device,self.agent.dtype)
        self.model_manager=new_model_manager
        new_game=GameInterface(cm.config,self.adb_path,self.model_manager,experience_dir)
        if new_game.obs_dim!=self.game.obs_dim or new_game.action_dim!=self.game.action_dim:
            self.agent.rebuild(new_game.obs_dim,new_game.action_dim)
        self.game=new_game
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
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
    def is_running(self):
        return self.running_event.is_set()
    def set_recorder(self,recorder):
        self.recorder=recorder
    def loop(self):
        metrics_prev=self.game.get_metrics()
        obs_prev=self.game.metrics_to_obs(metrics_prev)
        while self.running_event.is_set():
            action,eps=self.agent.select_action(obs_prev)
            self.game.execute_action(action,metrics_prev)
            base_interval=float(self.shared.config["动作"].get("循环间隔",0.2))
            interval=max(0.05,base_interval*(0.5+0.5*eps))
            time.sleep(interval)
            metrics_curr=self.game.get_metrics()
            obs_curr=self.game.metrics_to_obs(metrics_curr)
            reward=self.game.compute_reward(metrics_prev,metrics_curr)
            done=metrics_curr.get("alive",0)==0
            self.agent.store(obs_prev,action,reward,obs_curr,done)
            self.agent.train_step()
            if self.recorder and self.shared.mode=="训练" and self.game.last_screenshot is not None:
                self.recorder.record_frame("训练",self.game.last_screenshot)
                self.recorder.record_metrics("训练",metrics_curr)
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
                self.shared.cd_flash=metrics_curr["cd_flash"]
                self.shared.recalling=metrics_curr["recalling"]
                self.shared.epsilon=eps
                self.shared.running=True
        metrics_prev=metrics_curr
        obs_prev=obs_curr
class OptimizationManager:
    def __init__(self,shared,agent,recorder,mode_manager):
        self.shared=shared
        self.agent=agent
        self.recorder=recorder
        self.mode_manager=mode_manager
        self.lock=threading.Lock()
        self.active=False
        self.thread=None
    def progress_update(self,value):
        val=max(0.0,min(1.0,float(value)))
        with self.shared.lock:
            self.shared.optimize_progress=val
            self.shared.optimize_text="优化中{:.1f}%".format(val*100.0)
    def start(self,callback=None):
        if self.mode_manager.optimize_wait:
            return False
        with self.lock:
            if self.active:
                return False
            self.active=True
        self.mode_manager.enter_optimization()
        with self.shared.lock:
            self.shared.optimize_progress=0.0
            self.shared.optimize_iterations=0
            self.shared.optimize_text="优化准备中"
        def runner():
            steps=0
            try:
                steps=self.agent.offline_optimize(self.progress_update)
            finally:
                self.mode_manager.exit_optimization()
                with self.shared.lock:
                    self.shared.optimize_iterations=steps
                    self.shared.optimize_progress=1.0 if self.shared.optimize_progress<1.0 else self.shared.optimize_progress
                    self.shared.optimize_text="优化完成"
                with self.lock:
                    self.active=False
                if callback:
                    callback(steps)
        self.thread=threading.Thread(target=runner,daemon=True)
        self.thread.start()
        return True
class BotUI:
    def __init__(self,shared,controller,config_manager,mode_manager,optimizer):
        self.shared=shared
        self.controller=controller
        self.config_manager=config_manager
        self.mode_manager=mode_manager
        self.optimizer=optimizer
        self.root=tk.Tk()
        self.root.title("AI强化学习深度学习自适应类脑智能正则化控制面板")
        self.vars={k:tk.StringVar() for k in ["A","B","C","alive","cd1","cd2","cd3","cd_item","cd_heal","cd_flash","recalling","epsilon","status","device","hw","adb","dn","aaa","opt"]}
        self.vars["adb"].set(self.shared.config["路径"]["ADB"])
        self.vars["dn"].set(self.shared.config["路径"]["模拟器"])
        self.vars["aaa"].set(self.shared.config["路径"]["AAA"])
        self.vars["opt"].set(self.shared.optimize_text)
        self.progress_var=tk.DoubleVar(value=0.0)
        self.build_ui()
        self.update_loop()
    def build_ui(self):
        frame_state=tk.Frame(self.root)
        frame_state.pack(side="top",fill="x")
        labels=[("A","A"),("B","B"),("C","C"),("存活","alive"),("一技能","cd1"),("二技能","cd2"),("三技能","cd3"),("主动装备","cd_item"),("恢复","cd_heal"),("闪现","cd_flash"),("回城状态","recalling"),("ε","epsilon"),("状态","status"),("设备","device"),("硬件","hw"),("优化","opt")]
        for idx,(text,key) in enumerate(labels):
            tk.Label(frame_state,text=text).grid(row=idx,column=0,sticky="w")
            tk.Label(frame_state,textvariable=self.vars[key]).grid(row=idx,column=1,sticky="w")
        control_frame=tk.Frame(self.root)
        control_frame.pack(side="top",fill="x")
        tk.Button(control_frame,text="启动AI",command=self.start_bot).grid(row=0,column=0,sticky="we")
        tk.Button(control_frame,text="停止AI",command=self.stop_bot).grid(row=0,column=1,sticky="we")
        tk.Button(control_frame,text="识别优化",command=self.start_optimize).grid(row=0,column=2,sticky="we")
        tk.Label(control_frame,text="ADB").grid(row=1,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["adb"],width=48).grid(row=1,column=1,columnspan=2,sticky="we")
        tk.Label(control_frame,text="模拟器").grid(row=2,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["dn"],width=48).grid(row=2,column=1,columnspan=2,sticky="we")
        tk.Label(control_frame,text="AAA").grid(row=3,column=0,sticky="w")
        tk.Entry(control_frame,textvariable=self.vars["aaa"],width=48).grid(row=3,column=1,columnspan=2,sticky="we")
        ttk.Progressbar(control_frame,variable=self.progress_var,maximum=100.0,length=240).grid(row=4,column=0,columnspan=3,sticky="we")
        tk.Button(control_frame,text="应用路径",command=self.apply_paths).grid(row=5,column=0,columnspan=3,sticky="we")
    def start_bot(self):
        self.apply_paths()
        self.mode_manager.switch_to_train()
    def stop_bot(self):
        self.mode_manager.force_learning()
    def start_optimize(self):
        if not self.optimizer:
            return
        started=self.optimizer.start(lambda steps:self.root.after(0,lambda:self.on_optimize_complete(steps)))
        if started:
            with self.shared.lock:
                self.shared.optimize_text="优化进行中"
            self.vars["opt"].set(self.shared.optimize_text)
    def on_optimize_complete(self,steps):
        messagebox.showinfo("优化完成",f"基于经验池完成{steps}次迭代")
        self.mode_manager.finalize_optimization()
        with self.shared.lock:
            self.shared.optimize_text="待命"
            self.shared.optimize_progress=0.0
        self.vars["opt"].set(self.shared.optimize_text)
        self.progress_var.set(0.0)
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
        if self.shared.terminate:
            self.controller.stop()
            self.root.quit()
            return
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
            self.vars["cd_flash"].set("可用" if self.shared.cd_flash==1 else "冷却")
            self.vars["recalling"].set("回城中" if self.shared.recalling==1 else "空闲")
            self.vars["epsilon"].set("{:.3f}".format(self.shared.epsilon))
            status_prefix="运行" if self.shared.running else "停止"
            self.vars["status"].set(f"{status_prefix}-{self.shared.mode}")
            self.vars["device"].set(self.shared.device)
            hw_desc=""
            if "cpu_count" in self.shared.hw_profile:
                hw_desc+=f"CPU:{self.shared.hw_profile['cpu_count']} "
            if "memory_gb" in self.shared.hw_profile and self.shared.hw_profile["memory_gb"] is not None:
                hw_desc+=f"RAM:{self.shared.hw_profile['memory_gb']:.1f}GB "
            if "cuda_gpu" in self.shared.hw_profile and self.shared.hw_profile["cuda_gpu"]:
                hw_desc+=f"GPU:{self.shared.hw_profile['cuda_gpu']}"
            self.vars["hw"].set(hw_desc)
            self.vars["opt"].set(self.shared.optimize_text)
            self.progress_var.set(self.shared.optimize_progress*100.0)
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
    weight_decay=base.get("权重衰减",lr*0.1)
    value_reg=base.get("价值正则",0.01)
    if gpu_mem>8192:
        hidden=int(hidden*1.5)
        batch=int(batch*1.5)
        buffer=int(buffer*1.5)
        lr=lr*0.7
        epsilon_decay=int(epsilon_decay*1.3)
        target_sync=max(500,int(target_sync*0.8))
        weight_decay=float(weight_decay)*0.8
        value_reg=float(value_reg)*0.7
    elif cpu>=8 and mem>12:
        hidden=int(hidden*1.3)
        batch=int(batch*1.2)
        buffer=int(buffer*1.2)
        epsilon_decay=int(epsilon_decay*1.1)
        weight_decay=float(weight_decay)*0.9
    elif mem<6:
        batch=max(16,int(batch*0.6))
        buffer=max(8000,int(buffer*0.5))
        target_sync=int(target_sync*1.2)
        weight_decay=float(weight_decay)*1.2
        value_reg=float(value_reg)*1.3
    if device=="cpu":
        lr=max(lr,1e-4)
        weight_decay=float(weight_decay)*1.1
    return {"hidden_dim":hidden,"batch_size":batch,"buffer_size":buffer,"lr":lr,"target_sync":target_sync,"epsilon_decay":epsilon_decay,"gamma":base["折扣"],"weight_decay":float(weight_decay),"value_reg":float(value_reg)}
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
    model_manager=ModelManager(config,device,dtype)
    game=GameInterface(config,config["路径"]["ADB"],model_manager,experience_dir)
    agent=DeepRLAgent(device,dtype,config,game.obs_dim,game.action_dim,hidden_dim=params["hidden_dim"],buffer_size=params["buffer_size"],batch_size=params["batch_size"],gamma=params["gamma"],lr=params["lr"],target_sync=params["target_sync"],epsilon_decay=params["epsilon_decay"],experience_dir=experience_dir,weight_decay=params["weight_decay"],value_reg=params["value_reg"])
    controller=BotController(shared,agent,model_manager,game)
    recorder=ExperienceRecorder(config)
    mode_manager=ModeManager(shared,controller,recorder,config)
    optimizer=OptimizationManager(shared,agent,recorder,mode_manager)
    input_monitor=InputMonitor(mode_manager)
    mode_manager.start()
    input_monitor.start()
    ui=BotUI(shared,controller,config_manager,mode_manager,optimizer)
    try:
        ui.run()
    finally:
        shared.terminate=True
        controller.stop()
        mode_manager.stop()
        input_monitor.stop()
if __name__=="__main__":
    main()
