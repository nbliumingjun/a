import os,time,threading,random
from collections import deque
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import torch.amp as amp
    AMP_MODE="new"
except Exception:
    from torch.cuda import amp
    AMP_MODE="old"
import mss
from pynput import keyboard,mouse
try:
    import psutil
except Exception:
    psutil=None
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
KEY_LIST=["w","a","s","d","space","shift","ctrl","1","2","3","4","q","e","r","f","g"]
BASE_TICK_INTERVAL=0.02
IDLE_THRESHOLD=3.0
STACK_N=4
LEARNING_RATE=1e-4
SAVE_INTERVAL_STEPS=200
MODEL_INPUT_SIZE=(84,84)
LOG_PRINT_EVERY=50
EMA_DECAY=0.999
CONF_BASE=0.5
VAL_LOSS_SCALE=0.1
GAMMA=0.97
def get_base_dir():
    desktop=os.path.join(os.path.expanduser("~"),"Desktop")
    base=os.path.join(desktop,"GameAI")
    return base
BASE_DIR=get_base_dir()
DATA_DIR=os.path.join(BASE_DIR,"data")
MODEL_DIR=os.path.join(BASE_DIR,"models")
LOG_DIR=os.path.join(BASE_DIR,"logs")
for d in [BASE_DIR,DATA_DIR,MODEL_DIR,LOG_DIR]:
    os.makedirs(d,exist_ok=True)
SESSION_TIMESTAMP=datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH=os.path.join(LOG_DIR,f"session_{SESSION_TIMESTAMP}.txt")
def log(msg:str):
    stamp=datetime.now().strftime("%H:%M:%S")
    line=f"[{stamp}] {msg}"
    print(line)
    try:
        with open(LOG_PATH,"a",encoding="utf-8") as f:
            f.write(line+"\n")
    except Exception:
        pass
def get_hardware_profile():
    gpu=torch.cuda.is_available()
    vram_bytes=0
    if gpu:
        try:
            prop=torch.cuda.get_device_properties(0)
            vram_bytes=prop.total_memory
        except Exception:
            vram_bytes=0
    ram_bytes=8*(1024**3)
    if psutil is not None:
        try:
            ram_bytes=psutil.virtual_memory().total
        except Exception:
            pass
    cpu_count=os.cpu_count() or 4
    try:
        torch.set_num_threads(min(cpu_count,8))
    except Exception:
        pass
    return {"gpu":gpu,"vram_bytes":vram_bytes,"ram_bytes":ram_bytes,"cpu_count":cpu_count}
def calc_max_replay(p):
    approx_per=32000
    target=int((p["ram_bytes"]*0.3)/approx_per)
    if p["gpu"]:
        target=max(24000,target)
    else:
        target=max(12000,target)
    if target<4000:
        target=4000
    if target>60000:
        target=60000
    return target
def calc_explore(p):
    if p["gpu"]:
        return 0.025
    else:
        return 0.06
def calc_base_ch(p):
    if p["gpu"] and p["vram_bytes"]>16*(1024**3) and p["ram_bytes"]>32*(1024**3):
        return 96
    elif p["gpu"] and p["vram_bytes"]>7*(1024**3) and p["ram_bytes"]>16*(1024**3):
        return 64
    else:
        return 32
class ScreenCapture:
    def __init__(self):
        self.sct=mss.mss()
        best=None
        best_area=0
        for m in self.sct.monitors[1:]:
            area=m["width"]*m["height"]
            if area>best_area:
                best_area=area
                best=m
        if best is None:
            best=self.sct.monitors[0]
        self.monitor=best
        try:
            shorter=min(self.monitor["width"],self.monitor["height"])
            self.dynamic_max_delta=max(10.0,float(shorter)*0.03)
        except Exception:
            self.dynamic_max_delta=30.0
    def grab_frame_bgr(self):
        shot=self.sct.grab(self.monitor)
        frame=np.array(shot,dtype=np.uint8)[...,:3]
        return frame
    def grab_state84(self):
        frame=self.grab_frame_bgr()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        resized=cv2.resize(gray,(MODEL_INPUT_SIZE[1],MODEL_INPUT_SIZE[0]),interpolation=cv2.INTER_AREA)
        state84=resized.astype(np.uint8)
        return state84
class FrameStack:
    def __init__(self,n):
        self.n=n
        self.buf=deque(maxlen=n)
    def push(self,f):
        self.buf.append(f)
    def get_stack(self):
        if len(self.buf)==0:
            return None
        while len(self.buf)<self.n:
            self.buf.appendleft(self.buf[0].copy())
        stack=np.stack(list(self.buf),axis=0).astype(np.uint8)
        return stack
class InputMonitor:
    def __init__(self,key_list,exit_flag_ref,mode_ref,suppress_ref):
        self.key_list=key_list
        self.exit_flag_ref=exit_flag_ref
        self.mode_ref=mode_ref
        self.suppress_ref=suppress_ref
        self.pressed_keys=set()
        self.left_pressed=False
        self.right_pressed=False
        self.dx_accum=0.0
        self.dy_accum=0.0
        self.last_mouse_pos=None
        self.last_input_time=time.time()
        self.lock=threading.Lock()
        self.k_listener=keyboard.Listener(on_press=self._on_key_press,on_release=self._on_key_release)
        self.m_listener=mouse.Listener(on_move=self._on_mouse_move,on_click=self._on_mouse_click)
        self.k_listener.start()
        self.m_listener.start()
    def stop(self):
        try:
            self.k_listener.stop()
        except Exception:
            pass
        try:
            self.m_listener.stop()
        except Exception:
            pass
    def _on_key_press(self,key):
        try:
            if key==keyboard.Key.esc:
                self.exit_flag_ref["exit"]=True
                return
            keystr=None
            if isinstance(key,keyboard.KeyCode):
                if key.char:
                    keystr=key.char.lower()
                else:
                    keystr=None
            else:
                if key==keyboard.Key.space:
                    keystr="space"
                elif key in (keyboard.Key.shift,keyboard.Key.shift_l,keyboard.Key.shift_r):
                    keystr="shift"
                elif key in (keyboard.Key.ctrl,keyboard.Key.ctrl_l,keyboard.Key.ctrl_r):
                    keystr="ctrl"
            if keystr in self.key_list:
                with self.lock:
                    self.pressed_keys.add(keystr)
                    if not self.suppress_ref["suppress"]:
                        self.last_input_time=time.time()
        except Exception:
            pass
    def _on_key_release(self,key):
        try:
            keystr=None
            if isinstance(key,keyboard.KeyCode):
                if key.char:
                    keystr=key.char.lower()
                else:
                    keystr=None
            else:
                if key==keyboard.Key.space:
                    keystr="space"
                elif key in (keyboard.Key.shift,keyboard.Key.shift_l,keyboard.Key.shift_r):
                    keystr="shift"
                elif key in (keyboard.Key.ctrl,keyboard.Key.ctrl_l,keyboard.Key.ctrl_r):
                    keystr="ctrl"
            if keystr in self.key_list:
                with self.lock:
                    if keystr in self.pressed_keys:
                        self.pressed_keys.remove(keystr)
                    if not self.suppress_ref["suppress"]:
                        self.last_input_time=time.time()
        except Exception:
            pass
    def _on_mouse_move(self,x,y):
        try:
            with self.lock:
                if self.last_mouse_pos is not None:
                    dx=x-self.last_mouse_pos[0]
                    dy=y-self.last_mouse_pos[1]
                    self.dx_accum+=dx
                    self.dy_accum+=dy
                self.last_mouse_pos=(x,y)
                if not self.suppress_ref["suppress"]:
                    self.last_input_time=time.time()
        except Exception:
            pass
    def _on_mouse_click(self,x,y,button,pressed):
        try:
            with self.lock:
                if button==mouse.Button.left:
                    self.left_pressed=pressed
                elif button==mouse.Button.right:
                    self.right_pressed=pressed
                if not self.suppress_ref["suppress"]:
                    self.last_input_time=time.time()
        except Exception:
            pass
    def get_and_reset_action(self):
        with self.lock:
            dx=self.dx_accum
            dy=self.dy_accum
            self.dx_accum=0.0
            self.dy_accum=0.0
            left=1.0 if self.left_pressed else 0.0
            right=1.0 if self.right_pressed else 0.0
            keys_vec=np.zeros(len(self.key_list),dtype=np.float32)
            for i,k in enumerate(self.key_list):
                if k in self.pressed_keys:
                    keys_vec[i]=1.0
            snap=dict(dx=dx,dy=dy,mouse_left=left,mouse_right=right,keys_vec=keys_vec,any_active=(abs(dx)>0.0 or abs(dy)>0.0 or left>0.0 or right>0.0 or float(np.sum(keys_vec))>0.0))
        return snap
    def get_idle_seconds(self):
        with self.lock:
            idle=time.time()-self.last_input_time
        return idle
    def human_hint(self):
        with self.lock:
            self.last_input_time=time.time()
class ResidualBlock(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.c1=nn.Conv2d(ch,ch,3,1,1)
        self.c2=nn.Conv2d(ch,ch,3,1,1)
        self.n1=nn.GroupNorm(4,ch)
        self.n2=nn.GroupNorm(4,ch)
    def forward(self,x):
        h=F.relu(self.n1(self.c1(x)),inplace=True)
        h=self.n2(self.c2(h))
        return F.relu(x+h,inplace=True)
class SEBlock(nn.Module):
    def __init__(self,ch,rd=16):
        super().__init__()
        self.fc1=nn.Linear(ch,ch//rd)
        self.fc2=nn.Linear(ch//rd,ch)
    def forward(self,x):
        b,c,h,w=x.shape
        y=x.mean(dim=[2,3])
        y=F.relu(self.fc1(y),inplace=True)
        y=torch.sigmoid(self.fc2(y)).view(b,c,1,1)
        return x*y
class GameBrain(nn.Module):
    def __init__(self,num_keys,stack_n,base_ch):
        super().__init__()
        self.conv1=nn.Conv2d(stack_n,base_ch,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(base_ch,base_ch*2,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(base_ch*2,base_ch*2,kernel_size=3,stride=1,padding=1)
        self.gn1=nn.GroupNorm(4,base_ch)
        self.gn2=nn.GroupNorm(4,base_ch*2)
        self.gn3=nn.GroupNorm(4,base_ch*2)
        self.res1=ResidualBlock(base_ch*2)
        self.res2=ResidualBlock(base_ch*2)
        self.se=SEBlock(base_ch*2)
        with torch.no_grad():
            dummy=torch.zeros(1,stack_n,MODEL_INPUT_SIZE[0],MODEL_INPUT_SIZE[1])
            h=self.forward_conv(dummy)
            flat_dim=h.view(1,-1).shape[1]
        self.fc_embed=nn.Linear(flat_dim,512)
        self.gru_cell=nn.GRUCell(512,512)
        self.head_mouse=nn.Linear(512,2)
        self.head_mousebtn=nn.Linear(512,2)
        self.head_keys=nn.Linear(512,num_keys)
        self.head_value=nn.Linear(512,1)
        self.register_buffer("gru_h",torch.zeros(1,512))
    def forward_conv(self,x):
        h=F.relu(self.gn1(self.conv1(x)),inplace=True)
        h=F.relu(self.gn2(self.conv2(h)),inplace=True)
        h=F.relu(self.gn3(self.conv3(h)),inplace=True)
        h=self.res1(h)
        h=self.res2(h)
        h=self.se(h)
        return h
    def encode(self,x):
        b=x.shape[0]
        h=self.forward_conv(x)
        h=h.view(b,-1)
        h=F.relu(self.fc_embed(h),inplace=True)
        return h
    def forward_train(self,x):
        feat=self.encode(x)
        h0=torch.zeros(feat.shape[0],self.gru_h.shape[1],device=feat.device,dtype=feat.dtype)
        h1=self.gru_cell(feat,h0)
        h1=F.relu(h1,inplace=True)
        mouse_raw=self.head_mouse(h1)
        mouse_tanh=torch.tanh(mouse_raw)
        mouse_btn_logits=self.head_mousebtn(h1)
        key_logits=self.head_keys(h1)
        value=self.head_value(h1)
        return mouse_tanh,mouse_btn_logits,key_logits,value
    def forward_step(self,x):
        feat=self.encode(x)
        if self.gru_h.shape[0]!=feat.shape[0]:
            self.gru_h=torch.zeros(feat.shape[0],self.gru_h.shape[1],device=self.gru_h.device,dtype=self.gru_h.dtype)
        self.gru_h=self.gru_cell(feat,self.gru_h)
        h1=F.relu(self.gru_h,inplace=True)
        mouse_raw=self.head_mouse(h1)
        mouse_tanh=torch.tanh(mouse_raw)
        mouse_btn_logits=self.head_mousebtn(h1)
        key_logits=self.head_keys(h1)
        value=self.head_value(h1)
        return mouse_tanh,mouse_btn_logits,key_logits,value
    def reset_memory(self):
        self.gru_h.zero_()
def maybe_compile(m,is_train):
    return m
class TrainerThread(threading.Thread):
    def __init__(self,device,replay_buffer,replay_lock,train_model,ema_model,policy_model,policy_lock,exit_flag_ref,ready_ref):
        super().__init__(daemon=True)
        self.device=device
        self.replay_buffer=replay_buffer
        self.replay_lock=replay_lock
        self.exit_flag_ref=exit_flag_ref
        self.ready_ref=ready_ref
        self.train_model=train_model
        self.ema_model=ema_model
        self.policy_model=policy_model
        self.policy_lock=policy_lock
        self.optimizer=optim.Adam(self.train_model.parameters(),lr=LEARNING_RATE)
        self.train_steps=0
        self.use_amp=(self.device.type=="cuda")
        if self.use_amp:
            if AMP_MODE=="new":
                self.scaler=amp.GradScaler("cuda")
            else:
                self.scaler=amp.GradScaler()
        else:
            self.scaler=None
        self.ema_decay=EMA_DECAY
    def run(self):
        log("Trainer thread started.")
        while not self.exit_flag_ref["exit"]:
            with self.replay_lock:
                buf_len=len(self.replay_buffer)
            if buf_len<8:
                time.sleep(0.01)
                continue
            batch_size=self.get_dynamic_batch(buf_len)
            batch=self.sample_batch(batch_size)
            if batch is None:
                time.sleep(0.005)
                continue
            states_t,target_mouse_t,target_mousebtn_t,target_keys_t,returns_t,prio_t=batch
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                try:
                    if AMP_MODE=="new":
                        ctx=amp.autocast(device_type="cuda")
                    else:
                        ctx=amp.autocast()
                    with ctx:
                        pred_mouse_tanh,pred_mousebtn_logits,pred_keys_logits,pred_value=self.train_model.forward_train(states_t)
                        mouse_l2=((pred_mouse_tanh-target_mouse_t)**2).mean(dim=1)
                        btn_bce=F.binary_cross_entropy_with_logits(pred_mousebtn_logits,target_mousebtn_t,reduction="none").mean(dim=1)
                        keys_bce=F.binary_cross_entropy_with_logits(pred_keys_logits,target_keys_t,reduction="none").mean(dim=1)
                        value_target=returns_t.unsqueeze(1)
                        val_mse=((pred_value-value_target)**2).mean(dim=1)
                        sample_loss=(10.0*mouse_l2)+btn_bce+keys_bce+(VAL_LOSS_SCALE*val_mse)
                        total_loss=(sample_loss*prio_t).mean()
                        loss_mouse=(mouse_l2*prio_t).mean()
                        loss_btn=(btn_bce*prio_t).mean()
                        loss_keys=(keys_bce*prio_t).mean()
                        loss_val=(val_mse*prio_t).mean()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.train_model.parameters(),1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        time.sleep(0.05)
                        continue
                    else:
                        raise
            else:
                pred_mouse_tanh,pred_mousebtn_logits,pred_keys_logits,pred_value=self.train_model.forward_train(states_t)
                mouse_l2=((pred_mouse_tanh-target_mouse_t)**2).mean(dim=1)
                btn_bce=F.binary_cross_entropy_with_logits(pred_mousebtn_logits,target_mousebtn_t,reduction="none").mean(dim=1)
                keys_bce=F.binary_cross_entropy_with_logits(pred_keys_logits,target_keys_t,reduction="none").mean(dim=1)
                value_target=returns_t.unsqueeze(1)
                val_mse=((pred_value-value_target)**2).mean(dim=1)
                sample_loss=(10.0*mouse_l2)+btn_bce+keys_bce+(VAL_LOSS_SCALE*val_mse)
                total_loss=(sample_loss*prio_t).mean()
                loss_mouse=(mouse_l2*prio_t).mean()
                loss_btn=(btn_bce*prio_t).mean()
                loss_keys=(keys_bce*prio_t).mean()
                loss_val=(val_mse*prio_t).mean()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.train_model.parameters(),1.0)
                self.optimizer.step()
            self.update_ema()
            self.train_steps+=1
            self.ready_ref["ready"]=True
            if (self.train_steps%LOG_PRINT_EVERY)==0:
                log(f"[TRAIN RL] step={self.train_steps} loss={float(total_loss.detach().cpu().item()):.4f} (mouse={float(loss_mouse.detach().cpu().item()):.4f} btn={float(loss_btn.detach().cpu().item()):.4f} keys={float(loss_keys.detach().cpu().item()):.4f} val={float(loss_val.detach().cpu().item()):.4f}) buffer={buf_len} bsz={batch_size}")
            if (self.train_steps%SAVE_INTERVAL_STEPS)==0:
                self.save_and_sync()
        self.save_and_sync()
        log("Trainer thread exiting.")
    def update_ema(self):
        with torch.no_grad():
            for p_ema,p in zip(self.ema_model.parameters(),self.train_model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data*(1.0-self.ema_decay))
    def get_dynamic_batch(self,buf_len):
        if self.device.type=="cuda":
            base=64
        else:
            base=32
        dyn=max(16,buf_len//200)
        return min(base,dyn)
    def save_and_sync(self):
        try:
            model_path=os.path.join(MODEL_DIR,"latest.pth")
            torch.save(self.ema_model.state_dict(),model_path)
            log(f"[TRAIN RL] model saved to {model_path}")
        except Exception as e:
            log(f"[TRAIN RL] save error: {e}")
        try:
            with self.policy_lock:
                self.policy_model.load_state_dict(self.ema_model.state_dict(),strict=False)
                self.policy_model.eval()
                self.policy_model.reset_memory()
            log("[TRAIN RL] policy_model synced")
        except Exception as e:
            log(f"[TRAIN RL] sync error: {e}")
    def sample_batch(self,batch_size):
        with self.replay_lock:
            if len(self.replay_buffer)<batch_size:
                return None
            buf_len=len(self.replay_buffer)
            start_index=max(0,buf_len-20000)
            idx_pool=list(range(start_index,buf_len))
            if len(idx_pool)<=batch_size:
                idxs=list(idx_pool)
            else:
                priorities=[]
                for i in idx_pool:
                    s=self.replay_buffer[i]
                    pr=float(np.float32(s[6]))
                    rec=(i-start_index+1)
                    priorities.append(max(1e-3,rec*pr))
                idxs=random.choices(idx_pool,weights=priorities,k=batch_size)
            samples=[self.replay_buffer[i] for i in idxs]
        state_list=[]
        mouse_list=[]
        mousebtn_list=[]
        keys_list=[]
        ret_list=[]
        prio_list=[]
        for sample in samples:
            frames_u8,ndx16,ndy16,mbtns_u8,keys_u8,ret16,prio16=sample
            state_list.append(frames_u8.astype(np.float32)/255.0)
            mouse_list.append([float(np.float32(ndx16)),float(np.float32(ndy16))])
            mousebtn_list.append(mbtns_u8.astype(np.float32))
            keys_list.append(keys_u8.astype(np.float32))
            ret_list.append(float(np.float32(ret16)))
            prio_list.append(float(np.float32(prio16)))
        states_np=np.stack(state_list,axis=0)
        mouse_np=np.stack(mouse_list,axis=0)
        mousebtn_np=np.stack(mousebtn_list,axis=0)
        keys_np=np.stack(keys_list,axis=0)
        ret_np=np.array(ret_list,dtype=np.float32)
        prio_np=np.array(prio_list,dtype=np.float32)
        states_t=torch.from_numpy(states_np).to(self.device).float()
        target_mouse_t=torch.from_numpy(mouse_np).to(self.device).float()
        target_mousebtn_t=torch.from_numpy(mousebtn_np).to(self.device).float()
        target_keys_t=torch.from_numpy(keys_np).to(self.device).float()
        returns_t=torch.from_numpy(ret_np).to(self.device).float()
        prio_t=torch.from_numpy(prio_np).to(self.device).float()
        return states_t,target_mouse_t,target_mousebtn_t,target_keys_t,returns_t,prio_t
class AutoPlayer:
    def __init__(self,explore_rate,max_mouse_delta):
        self.keyboard_ctl=keyboard.Controller()
        self.mouse_ctl=mouse.Controller()
        self.held_keys=set()
        self.held_mouse_left=False
        self.held_mouse_right=False
        self.key_to_pynput={}
        for k in KEY_LIST:
            if k=="space":
                self.key_to_pynput[k]=keyboard.Key.space
            elif k=="shift":
                self.key_to_pynput[k]=keyboard.Key.shift
            elif k=="ctrl":
                self.key_to_pynput[k]=keyboard.Key.ctrl
            else:
                self.key_to_pynput[k]=k
        self.max_mouse_delta=max_mouse_delta
        self.rng=random.Random()
        self.explore=explore_rate
        self.mouse_scale=1.2
    def release_all(self,suppress_ref):
        for k in list(self.held_keys):
            self._key_up(k,suppress_ref)
        self.held_keys.clear()
        if self.held_mouse_left:
            try:
                suppress_ref["suppress"]=True
                self.mouse_ctl.release(mouse.Button.left)
            except Exception:
                pass
            finally:
                suppress_ref["suppress"]=False
            self.held_mouse_left=False
        if self.held_mouse_right:
            try:
                suppress_ref["suppress"]=True
                self.mouse_ctl.release(mouse.Button.right)
            except Exception:
                pass
            finally:
                suppress_ref["suppress"]=False
            self.held_mouse_right=False
    def act_from_state(self,policy_model,device,stack_state_np,suppress_ref):
        with torch.no_grad():
            st=torch.from_numpy(stack_state_np).unsqueeze(0).to(device).float()/255.0
            if device.type=="cuda":
                if AMP_MODE=="new":
                    ctx=amp.autocast(device_type="cuda")
                else:
                    ctx=amp.autocast()
                with ctx:
                    mouse_tanh,mouse_btn_logits,key_logits,val_pred=policy_model.forward_step(st)
            else:
                mouse_tanh,mouse_btn_logits,key_logits,val_pred=policy_model.forward_step(st)
            mouse_move=mouse_tanh[0].cpu().numpy()
            dx_norm,dy_norm=float(mouse_move[0]),float(mouse_move[1])
            btn_probs=torch.sigmoid(mouse_btn_logits)[0].cpu().numpy()
            left_prob,right_prob=float(btn_probs[0]),float(btn_probs[1])
            key_probs=torch.sigmoid(key_logits)[0].cpu().numpy()
            val_score=float(torch.sigmoid(val_pred)[0].cpu().numpy()[0])
        dx=int(dx_norm*self.max_mouse_delta*self.mouse_scale)
        dy=int(dy_norm*self.max_mouse_delta*self.mouse_scale)
        if self.rng.random()<self.explore:
            dx+=self.rng.randint(-3,3)
            dy+=self.rng.randint(-3,3)
        try:
            suppress_ref["suppress"]=True
            self.mouse_ctl.move(dx,dy)
        except Exception:
            pass
        finally:
            suppress_ref["suppress"]=False
        want_left=(left_prob>0.5)
        want_right=(right_prob>0.5)
        if want_left and (not self.held_mouse_left):
            try:
                suppress_ref["suppress"]=True
                self.mouse_ctl.press(mouse.Button.left)
            except Exception:
                pass
            finally:
                suppress_ref["suppress"]=False
            self.held_mouse_left=True
        elif (not want_left) and self.held_mouse_left:
            try:
                suppress_ref["suppress"]=True
                self.mouse_ctl.release(mouse.Button.left)
            except Exception:
                pass
            finally:
                suppress_ref["suppress"]=False
            self.held_mouse_left=False
        if want_right and (not self.held_mouse_right):
            try:
                suppress_ref["suppress"]=True
                self.mouse_ctl.press(mouse.Button.right)
            except Exception:
                pass
            finally:
                suppress_ref["suppress"]=False
            self.held_mouse_right=True
        elif (not want_right) and self.held_mouse_right:
            try:
                suppress_ref["suppress"]=True
                self.mouse_ctl.release(mouse.Button.right)
            except Exception:
                pass
            finally:
                suppress_ref["suppress"]=False
            self.held_mouse_right=False
        keys_out_vec=np.zeros(len(KEY_LIST),dtype=np.float32)
        for i,keyname in enumerate(KEY_LIST):
            want_down=(key_probs[i]>0.5)
            if self.rng.random()<self.explore:
                want_down=bool(self.rng.randint(0,1))
            if want_down and (keyname not in self.held_keys):
                self._key_down(keyname,suppress_ref)
                self.held_keys.add(keyname)
                keys_out_vec[i]=1.0
            elif (not want_down) and (keyname in self.held_keys):
                self._key_up(keyname,suppress_ref)
                self.held_keys.remove(keyname)
                keys_out_vec[i]=0.0
            else:
                if keyname in self.held_keys:
                    keys_out_vec[i]=1.0
        conf_mouse=(abs(dx_norm)+abs(dy_norm))/2.0
        conf_btn=(max(left_prob,1.0-left_prob)+max(right_prob,1.0-right_prob))/2.0
        conf_keys=float(np.mean([max(p,1.0-p) for p in key_probs]))
        conf_val=val_score
        conf_total=(conf_mouse+conf_btn+conf_keys+conf_val)/4.0
        dx_clamped=max(-self.max_mouse_delta,min(self.max_mouse_delta,float(dx)))
        dy_clamped=max(-self.max_mouse_delta,min(self.max_mouse_delta,float(dy)))
        ndx=dx_clamped/self.max_mouse_delta
        ndy=dy_clamped/self.max_mouse_delta
        activity_mag=min(1.0,(abs(ndx)+abs(ndy))*0.5+float(np.mean(keys_out_vec))*0.5+float(want_left or want_right)*0.5)
        return {"ndx":ndx,"ndy":ndy,"mbtns":np.array([1.0 if want_left else 0.0,1.0 if want_right else 0.0],dtype=np.float32),"keys":keys_out_vec.astype(np.float32),"conf":float(CONF_BASE+(conf_total-CONF_BASE)*0.5),"activity":activity_mag}
    def _key_down(self,keyname,suppress_ref):
        kobj=self.key_to_pynput.get(keyname)
        if kobj is None:
            return
        try:
            suppress_ref["suppress"]=True
            self.keyboard_ctl.press(kobj)
        except Exception:
            pass
        finally:
            suppress_ref["suppress"]=False
    def _key_up(self,keyname,suppress_ref):
        kobj=self.key_to_pynput.get(keyname)
        if kobj is None:
            return
        try:
            suppress_ref["suppress"]=True
            self.keyboard_ctl.release(kobj)
        except Exception:
            pass
        finally:
            suppress_ref["suppress"]=False
def try_load_model(train_model,ema_model,policy_model,device,ready_ref):
    model_path=os.path.join(MODEL_DIR,"latest.pth")
    if os.path.isfile(model_path):
        try:
            sd=torch.load(model_path,map_location=device)
            train_model.load_state_dict(sd,strict=False)
            ema_model.load_state_dict(sd,strict=False)
            policy_model.load_state_dict(sd,strict=False)
            train_model.train()
            ema_model.eval()
            policy_model.eval()
            policy_model.reset_memory()
            ready_ref["ready"]=True
            log(f"[INIT] loaded existing model {model_path}")
        except Exception as e:
            log(f"[INIT] load fail: {e}")
def main():
    log("========== GameAI DeepRL starting ==========")
    exit_flag_ref={"exit":False}
    mode_ref={"mode":"HUMAN"}
    suppress_ref={"suppress":False}
    ready_ref={"ready":False}
    profile=get_hardware_profile()
    replay_maxlen=calc_max_replay(profile)
    explore_rate=calc_explore(profile)
    base_ch=calc_base_ch(profile)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        try:
            torch.backends.cudnn.benchmark=True
        except Exception:
            pass
    log(f"Using device: {device}")
    log(f"Replay buffer size: {replay_maxlen}")
    log(f"Model base channels: {base_ch}")
    cap=ScreenCapture()
    framestack=FrameStack(STACK_N)
    replay_buffer=deque(maxlen=replay_maxlen)
    replay_lock=threading.Lock()
    policy_model=GameBrain(num_keys=len(KEY_LIST),stack_n=STACK_N,base_ch=base_ch).to(device)
    policy_model.eval()
    ema_model=GameBrain(num_keys=len(KEY_LIST),stack_n=STACK_N,base_ch=base_ch).to(device)
    ema_model.eval()
    train_model=GameBrain(num_keys=len(KEY_LIST),stack_n=STACK_N,base_ch=base_ch).to(device)
    train_model.train()
    policy_model=maybe_compile(policy_model,False)
    train_model=maybe_compile(train_model,True)
    try_load_model(train_model,ema_model,policy_model,device,ready_ref)
    policy_lock=threading.Lock()
    inp=InputMonitor(KEY_LIST,exit_flag_ref,mode_ref,suppress_ref)
    trainer=TrainerThread(device=device,replay_buffer=replay_buffer,replay_lock=replay_lock,train_model=train_model,ema_model=ema_model,policy_model=policy_model,policy_lock=policy_lock,exit_flag_ref=exit_flag_ref,ready_ref=ready_ref)
    trainer.start()
    auto=AutoPlayer(explore_rate=explore_rate,max_mouse_delta=cap.dynamic_max_delta)
    log("Main control loop running. Play your game in the foreground.")
    log("Hands off for a few seconds and the AI will take over. Touch the controls to retake control. ESC to exit.")
    tick_interval=BASE_TICK_INTERVAL
    last_stat_t=time.time()
    loop_ticks=0
    pending=None
    running_return=0.0
    try:
        while not exit_flag_ref["exit"]:
            tick_start=time.time()
            state84_np=cap.grab_state84()
            framestack.push(state84_np)
            stack_state=framestack.get_stack()
            idle_sec=inp.get_idle_seconds()
            cur_mode=mode_ref["mode"]
            model_ready=ready_ref["ready"]
            if (cur_mode=="HUMAN") and (idle_sec>=IDLE_THRESHOLD) and model_ready:
                mode_ref["mode"]="AUTO"
                cur_mode="AUTO"
                policy_model.reset_memory()
                log(">>> Switched to AUTO mode.")
            elif cur_mode=="AUTO":
                if idle_sec<0.5:
                    mode_ref["mode"]="HUMAN"
                    cur_mode="HUMAN"
                    policy_model.reset_memory()
                    log(">>> Switched to HUMAN mode. Releasing keys.")
                    auto.release_all(suppress_ref)
            if (pending is not None):
                try:
                    diff_norm=float(np.mean(np.abs(state84_np.astype(np.float32)-pending["frame_gray"].astype(np.float32)))/255.0)
                except Exception:
                    diff_norm=0.0
                reward_tick=diff_norm+0.1*pending["activity"]+0.01
                running_return=reward_tick+(GAMMA*running_return)
                prio_val=max(0.1,reward_tick+0.1)
                frames_u8=pending["frames_u8"]
                ndx16=pending["ndx16"]
                ndy16=pending["ndy16"]
                mouse_btns_u8=pending["mouse_btns_u8"]
                keys_vec_u8=pending["keys_vec_u8"]
                ret16=np.float16(running_return)
                prio16=np.float16(prio_val)
                entry=(frames_u8,ndx16,ndy16,mouse_btns_u8,keys_vec_u8,ret16,prio16)
                with replay_lock:
                    replay_buffer.append(entry)
                pending=None
            if stack_state is not None:
                frames_u8=stack_state.copy()
            if cur_mode=="HUMAN":
                snap=inp.get_and_reset_action()
                if snap["any_active"]:
                    inp.human_hint()
                dx=float(snap["dx"])
                dy=float(snap["dy"])
                ndx=max(-auto.max_mouse_delta,min(auto.max_mouse_delta,dx))/auto.max_mouse_delta
                ndy=max(-auto.max_mouse_delta,min(auto.max_mouse_delta,dy))/auto.max_mouse_delta
                mouse_btns=np.array([snap["mouse_left"],snap["mouse_right"]],dtype=np.float32)
                keys_vec=snap["keys_vec"].astype(np.float32)
                ndx16=np.float16(ndx)
                ndy16=np.float16(ndy)
                mouse_btns_u8=mouse_btns.astype(np.uint8)
                keys_vec_u8=keys_vec.astype(np.uint8)
                activity_mag=1.0 if snap["any_active"] else 0.0
                pending={"frames_u8":frames_u8,"ndx16":ndx16,"ndy16":ndy16,"mouse_btns_u8":mouse_btns_u8,"keys_vec_u8":keys_vec_u8,"frame_gray":state84_np.copy(),"activity":activity_mag}
            elif (cur_mode=="AUTO") and model_ready and (stack_state is not None):
                with policy_lock:
                    auto_info=auto.act_from_state(policy_model,device,stack_state,suppress_ref)
                ndx16=np.float16(auto_info["ndx"])
                ndy16=np.float16(auto_info["ndy"])
                mouse_btns_u8=auto_info["mbtns"].astype(np.uint8)
                keys_vec_u8=auto_info["keys"].astype(np.uint8)
                activity_mag=auto_info["activity"]
                pending={"frames_u8":frames_u8,"ndx16":ndx16,"ndy16":ndy16,"mouse_btns_u8":mouse_btns_u8,"keys_vec_u8":keys_vec_u8,"frame_gray":state84_np.copy(),"activity":activity_mag}
            elapsed=time.time()-tick_start
            remaining=tick_interval-elapsed
            if remaining>0:
                time.sleep(remaining)
            tick_interval=0.9*tick_interval+0.1*max(0.01,elapsed)
            loop_ticks+=1
            now=time.time()
            if now-last_stat_t>5.0:
                fps=float(loop_ticks/(now-last_stat_t))
                loop_ticks=0
                last_stat_t=now
                log(f"[LOOP] mode={mode_ref['mode']} idle={idle_sec:.2f}s fps~{fps:.1f} buffer={len(replay_buffer)} ready={model_ready} delta={auto.max_mouse_delta:.1f}")
    except KeyboardInterrupt:
        log("KeyboardInterrupt -> exiting.")
    except Exception as e:
        log(f"Main loop exception: {e}")
    finally:
        exit_flag_ref["exit"]=True
        log("Stopping trainer and input monitor...")
        try:
            auto.release_all(suppress_ref)
        except Exception:
            pass
        try:
            inp.stop()
        except Exception:
            pass
        try:
            trainer.join()
        except Exception:
            pass
        log("========== GameAI DeepRL stopped ==========")
if __name__=="__main__":
    main()
