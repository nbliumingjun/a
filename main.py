import json
import time
import os
import platform
import subprocess
import ctypes
from ctypes import wintypes
from pathlib import Path
from threading import Event,Lock
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from pynput import mouse,keyboard
import pyautogui
from PIL import Image
import random
def 桌面路径():
    if platform.system()=="Windows":
        try:
            缓冲=ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            if ctypes.windll.shell32.SHGetFolderPathW(None,0,None,0,缓冲)==0 and 缓冲.value:
                return Path(缓冲.value).resolve()
        except Exception:
            pass
    主页=Path.home()
    备选=[主页/"Desktop",主页]
    for 路径 in 备选:
        if 路径.exists():
            return 路径.resolve()
    return 主页.resolve()
class 硬件环境:
    def __init__(self,根):
        self.根=根
        self.尺寸=pyautogui.size()
        self.核心=os.cpu_count() or 1
        self.内存=self.获取内存()
        self.显存=self.获取显存()
        self.调节路径=self.根/"hardware.json"
        if self.调节路径.exists():
            try:
                self.调节=json.loads(self.调节路径.read_text(encoding="utf-8"))
            except Exception:
                self.调节=self.默认调节()
        else:
            self.调节=self.默认调节()
            self.调节路径.write_text(json.dumps(self.调节,ensure_ascii=False),encoding="utf-8")
    def 获取内存(self):
        if platform.system()!="Windows":
            return 0
        try:
            class 状态(ctypes.Structure):
                _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
            信息=状态()
            信息.dwLength=ctypes.sizeof(状态)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(信息)):
                return 信息.ullTotalPhys
        except Exception:
            return 0
        return 0
    def 获取显存(self):
        if platform.system()!="Windows":
            return 0
        命令=["powershell","-NoProfile","-Command","Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty AdapterRAM"]
        try:
            输出=subprocess.run(命令,check=False,capture_output=True,text=True,timeout=5).stdout.strip().splitlines()
            数值=[int(行) for 行 in 输出 if 行.isdigit()]
            if 数值:
                return max(数值)
        except Exception:
            return 0
        return 0
    def 描述(self):
        内存值=self.内存//(1024**3) if self.内存 else 0
        显存值=self.显存//(1024**3) if self.显存 else 0
        return f"屏幕{self.尺寸.width}x{self.尺寸.height} 核心{self.核心} 内存{内存值}GB 显存{显存值}GB"
    def 默认参数(self):
        像素=max(self.尺寸.width*self.尺寸.height,1)
        核心=max(self.核心,1)
        内存值=self.内存 or 像素*self.调节["内存估算因子"]
        显存值=self.显存 or 像素*self.调节["显存估算因子"]
        轮询=int(max(self.调节["轮询范围"]["最小"],min(self.调节["轮询范围"]["最大"],self.调节["轮询范围"]["基准"]//max(1,int(核心*self.调节["轮询范围"]["核心权重"])))))
        持续=int(max(self.调节["记录时长范围"]["最小"],min(self.调节["记录时长范围"]["最大"],(内存值+显存值)//self.调节["记录时长范围"]["分母"])))
        宽度=int(max(self.调节["采样宽度范围"]["最小"],min(self.尺寸.width,self.调节["采样宽度范围"]["最大"])))
        高度=int(max(self.调节["采样高度范围"]["最小"],min(self.尺寸.height,self.调节["采样高度范围"]["最大"])))
        批量=int(max(self.调节["批量范围"]["最小"],min(self.调节["批量范围"]["最大"],核心*self.调节["批量范围"]["核心权重"])))
        轮数=int(max(self.调节["训练轮数范围"]["最小"],min(self.调节["训练轮数范围"]["最大"],(内存值+显存值)//self.调节["训练轮数范围"]["分母"])))
        缓冲=int(max(self.调节["缓冲范围"]["最小"],min(self.调节["缓冲范围"]["最大"],(内存值+显存值)//self.调节["缓冲范围"]["分母"])))
        学习率=float(max(self.调节["学习率范围"]["最小"],min(self.调节["学习率范围"]["最大"],1/(核心*self.调节["学习率范围"]["核心权重"]))))
        折扣=float(min(self.调节["折扣范围"]["最大"],max(self.调节["折扣范围"]["最小"],1-1/max(self.调节["折扣范围"]["分母"],核心*self.调节["折扣范围"]["核心权重"]))))
        探索衰减=float(min(self.调节["探索衰减范围"]["最大"],max(self.调节["探索衰减范围"]["最小"],1-1/max(self.调节["探索衰减范围"]["分母"],核心*self.调节["探索衰减范围"]["核心权重"]))))
        最小探索=float(max(self.调节["最小探索范围"]["最小"],min(self.调节["最小探索范围"]["最大"],1/(核心*self.调节["最小探索范围"]["核心权重"]))))
        同步=int(max(self.调节["同步范围"]["最小"],min(self.调节["同步范围"]["最大"],self.调节["同步范围"]["基准"]//max(1,int(核心*self.调节["同步范围"]["核心权重"])))))
        鼠标=int(max(self.调节["鼠标范围"]["最小"],min(self.调节["鼠标范围"]["最大"],self.调节["鼠标范围"]["基准"]//max(1,int(核心*self.调节["鼠标范围"]["核心权重"])))))
        帧跳=int(max(self.调节["帧跳范围"]["最小"],min(self.调节["帧跳范围"]["最大"],核心)))
        指纹=f"{self.尺寸.width}x{self.尺寸.height}_{核心}_{内存值}_{显存值}"
        return {"轮询毫秒":轮询,"记录时长秒":持续,"采样宽度":宽度,"采样高度":高度,"批量":批量,"训练轮数":轮数,"缓冲上限":缓冲,"学习率":学习率,"折扣因子":折扣,"探索衰减":探索衰减,"最小探索":最小探索,"同步毫秒":同步,"鼠标移动毫秒":鼠标,"帧跳":帧跳,"硬件指纹":指纹,"初始探索率":self.调节.get("初始探索率",1.0)}
    def 默认调节(self):
        return {"内存估算因子":4,"显存估算因子":2,"轮询范围":{"最小":1,"最大":20,"基准":1000,"核心权重":4},"记录时长范围":{"最小":90,"最大":1200,"分母":80000000},"采样宽度范围":{"最小":64,"最大":512},"采样高度范围":{"最小":64,"最大":512},"批量范围":{"最小":8,"最大":128,"核心权重":8},"训练轮数范围":{"最小":1,"最大":80,"分母":500000000},"缓冲范围":{"最小":1000,"最大":15000,"分母":5000000},"学习率范围":{"最小":1e-5,"最大":5e-4,"核心权重":1500},"折扣范围":{"最小":0.9,"最大":0.995,"分母":10,"核心权重":5},"探索衰减范围":{"最小":0.98,"最大":0.999,"分母":200,"核心权重":50},"最小探索范围":{"最小":0.02,"最大":0.2,"核心权重":5},"同步范围":{"最小":8,"最大":160,"基准":1000,"核心权重":2},"鼠标范围":{"最小":4,"最大":140,"基准":1000,"核心权重":6},"帧跳范围":{"最小":1,"最大":5},"初始探索率":1.0}
class 配置管理:
    def __init__(self,根,基础):
        self.根=根
        self.根.mkdir(parents=True,exist_ok=True)
        self.默认=基础
        self.配置路径=self.根/基础.get("配置文件","config.json")
        已有={}
        if self.配置路径.exists():
            try:
                已有=json.loads(self.配置路径.read_text(encoding="utf-8"))
            except Exception:
                已有={}
        合并={**基础,**已有}
        合并["停止键"]=已有.get("停止键",str(keyboard.Key.esc))
        合并.setdefault("动作文件",f"actions_{基础['硬件指纹']}.json")
        合并.setdefault("数据文件",f"dataset_{基础['硬件指纹']}.npz")
        合并.setdefault("模型文件",f"policy_{基础['硬件指纹']}.pt")
        合并.setdefault("克隆模型",f"clone_{基础['硬件指纹']}.pt")
        合并.setdefault("奖励模式","intensity")
        合并.setdefault("奖励区域",[0,0,1,1])
        合并.setdefault("奖励缩放",1.0)
        合并.setdefault("奖励平滑",0.0)
        合并.setdefault("克隆批量",基础.get("批量",32))
        合并.setdefault("克隆轮数",基础.get("训练轮数",1))
        合并.setdefault("克隆学习率",1e-3)
        合并.setdefault("克隆置信",0.8)
        合并.setdefault("自动执行探索率",0.0)
        合并.setdefault("初始探索率",基础.get("初始探索率",1.0))
        self.配置=合并
        self.配置路径.write_text(json.dumps(self.配置,ensure_ascii=False),encoding="utf-8")
    def 取(self,键):
        return self.配置[键]
    def 更新(self,键,值):
        self.配置[键]=值
        self.配置路径.write_text(json.dumps(self.配置,ensure_ascii=False),encoding="utf-8")
class 键助手:
    @staticmethod
    def 解析(文本):
        if 文本.startswith("Key."):
            名称=文本.split(".",1)[1]
            return getattr(keyboard.Key,名称)
        if len(文本)==1:
            return keyboard.KeyCode.from_char(文本)
        raise ValueError("停止键配置无效")
    @staticmethod
    def 标识(键):
        if isinstance(键,keyboard.KeyCode):
            return 键.char if 键.char is not None else str(键)
        return str(键)
    @staticmethod
    def 自动化值(键):
        if 键.startswith("Key."):
            return 键.split(".",1)[1]
        return 键
class 屏幕采集:
    def __init__(self,宽,高):
        self.宽=宽
        self.高=高
    def 截取(self):
        图=pyautogui.screenshot()
        if 图.size!=(self.宽,self.高):
            图=图.resize((self.宽,self.高),Image.BILINEAR)
        灰=图.convert("L")
        return np.asarray(灰,dtype=np.float32)/255.0
class 动作空间:
    def __init__(self,路径):
        self.路径=路径
        self.映射={}
        self.反向={}
        if self.路径.exists():
            try:
                数据=json.loads(self.路径.read_text(encoding="utf-8"))
                for 项 in 数据:
                    self.映射[项["签名"]]=项
                self._同步反向()
            except Exception:
                self.映射={}
                self.反向={}
    def _同步反向(self):
        self.反向={int(值["索引"]):值 for 值 in self.映射.values()}
    def 签名(self,动作):
        return json.dumps(动作,ensure_ascii=False,sort_keys=True)
    def 索引(self,动作):
        签=self.签名(动作)
        if 签 not in self.映射:
            索=len(self.映射)
            self.映射[签]={"索引":索,"事件":动作,"签名":签}
            self._同步反向()
        return self.映射[签]["索引"]
    def 事件(self,索引):
        项=self.反向.get(索引)
        return 项["事件"] if 项 else None
    def 保存(self):
        self.路径.write_text(json.dumps(list(self.映射.values()),ensure_ascii=False),encoding="utf-8")
    def 大小(self):
        return len(self.映射)
class 数据存储:
    def __init__(self,路径):
        self.路径=路径
        self.状态=None
        self.动作=None
        self.奖励=None
        self.下个=None
        self.终止=None
        if self.路径.exists():
            try:
                数据=np.load(self.路径,allow_pickle=False)
                self.状态=数据.get("states")
                self.动作=数据.get("actions")
                self.奖励=数据.get("rewards")
                self.下个=数据.get("nexts")
                self.终止=数据.get("dones")
            except Exception:
                self.状态=self.动作=self.奖励=self.下个=self.终止=None
    def 读取(self):
        if self.状态 is None:
            return []
        数量=self.状态.shape[0]
        return [(self.状态[i],int(self.动作[i]),float(self.奖励[i]),self.下个[i],float(self.终止[i])) for i in range(数量)]
    def 合并(self,状态集,动作集,奖励集,下个集,终止集):
        状态=np.stack(状态集).astype(np.float32)
        动作=np.array(动作集,dtype=np.int64)
        奖励=np.array(奖励集,dtype=np.float32)
        下个=np.stack(下个集).astype(np.float32)
        终止=np.array(终止集,dtype=np.float32)
        if self.状态 is None:
            self.状态=状态
            self.动作=动作
            self.奖励=奖励
            self.下个=下个
            self.终止=终止
        else:
            self.状态=np.concatenate([self.状态,状态],axis=0)
            self.动作=np.concatenate([self.动作,动作],axis=0)
            self.奖励=np.concatenate([self.奖励,奖励],axis=0)
            self.下个=np.concatenate([self.下个,下个],axis=0)
            self.终止=np.concatenate([self.终止,终止],axis=0)
        np.savez_compressed(self.路径,states=self.状态,actions=self.动作,rewards=self.奖励,nexts=self.下个,dones=self.终止)
class 奖励评估:
    def __init__(self,配置):
        self.配置=配置
        self.模式=self.配置.取("奖励模式")
        self.区域=list(self.配置.取("奖励区域"))
        if len(self.区域)!=4:
            self.区域=[0,0,1,1]
        self.区域=[max(0,min(1,float(x))) for x in self.区域]
        self.缩放=float(self.配置.取("奖励缩放"))
        self.平滑=float(self.配置.取("奖励平滑"))
        self.上一=None
    def 边界(self,图):
        高,宽=图.shape
        左=int(self.区域[0]*宽)
        上=int(self.区域[1]*高)
        右=int(max(self.区域[0],self.区域[2])*宽)
        下=int(max(self.区域[1],self.区域[3])*高)
        左=max(0,min(宽-1,左))
        右=max(左+1,min(宽,右))
        上=max(0,min(高-1,上))
        下=max(上+1,min(高,下))
        return 上,下,左,右
    def 指标(self,图):
        上,下,左,右=self.边界(图)
        区域=图[上:下,左:右]
        return float(np.mean(区域))
    def 重置(self):
        self.上一=None
    def 计算(self,当前,下一个):
        当前值=self.指标(当前)
        下值=self.指标(下一个)
        差=下值-当前值
        if self.上一 is not None and self.平滑>0:
            差=(差+self.上一*self.平滑)/(1+self.平滑)
        self.上一=差
        if self.模式=="intensity":
            return 差*self.缩放
        return self.缩放
class 重放缓冲:
    def __init__(self,容量):
        self.容量=容量
        self.状态=[]
        self.动作=[]
        self.奖励=[]
        self.下一个=[]
        self.终止=[]
    def 预填(self,数据):
        for 项 in 数据:
            self.添加(*项)
    def 添加(self,状态,动作,奖励,下一个,终止):
        if len(self.状态)>=self.容量:
            self.状态.pop(0)
            self.动作.pop(0)
            self.奖励.pop(0)
            self.下一个.pop(0)
            self.终止.pop(0)
        self.状态.append(状态)
        self.动作.append(动作)
        self.奖励.append(奖励)
        self.下一个.append(下一个)
        self.终止.append(终止)
    def 批量(self,大小):
        数量=min(len(self.状态),大小)
        索引=random.sample(range(len(self.状态)),数量)
        状态集=np.stack([self.状态[i] for i in 索引]).astype(np.float32)
        动作集=np.array([self.动作[i] for i in 索引],dtype=np.int64)
        奖励集=np.array([self.奖励[i] for i in 索引],dtype=np.float32)
        下个集=np.stack([self.下一个[i] for i in 索引]).astype(np.float32)
        终止集=np.array([self.终止[i] for i in 索引],dtype=np.float32)
        return 状态集,动作集,奖励集,下个集,终止集
    def 大小(self):
        return len(self.状态)
class 策略网络(nn.Module):
    def __init__(self,宽,高,动作数):
        super().__init__()
        通道=1
        基数=max(动作数*4,32)
        self.卷积=nn.Sequential(nn.Conv2d(通道,基数//2,kernel_size=5,stride=2,padding=2),nn.ReLU(),nn.Conv2d(基数//2,基数,kernel_size=3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(基数,基数,kernel_size=3,stride=2,padding=1),nn.ReLU())
        宽度=宽
        高度=高
        for _ in range(3):
            宽度=math.ceil(宽度/2)
            高度=math.ceil(高度/2)
        展平=基数*宽度*高度
        隐藏=max(256,动作数*32)
        self.全连=nn.Sequential(nn.Linear(展平,隐藏),nn.ReLU(),nn.Linear(隐藏,动作数))
    def forward(self,x):
        x=self.卷积(x)
        x=torch.flatten(x,1)
        return self.全连(x)
class 行为克隆器:
    def __init__(self,配置,动作空间):
        self.配置=配置
        self.动作空间=动作空间
        self.设备=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.模型=None
        self.优化器=None
        self.损失=nn.CrossEntropyLoss()
    def 同步(self):
        动作数=max(1,self.动作空间.大小())
        if self.模型 is None or self.模型.全连[-1].out_features!=动作数:
            self.模型=策略网络(self.配置.取("采样宽度"),self.配置.取("采样高度"),动作数).to(self.设备)
            self.优化器=optim.Adam(self.模型.parameters(),lr=self.配置.取("克隆学习率"))
    def 训练(self,状态序列,动作序列):
        if not 状态序列:
            return None
        self.同步()
        状态张=torch.from_numpy(np.stack(状态序列).astype(np.float32)).unsqueeze(1)
        动作张=torch.from_numpy(np.array(动作序列,dtype=np.int64))
        数据=TensorDataset(状态张,动作张)
        载体=DataLoader(数据,batch_size=self.配置.取("克隆批量"),shuffle=True)
        损失值=None
        for _ in range(self.配置.取("克隆轮数")):
            for 批状态,批动作 in 载体:
                批状态=批状态.to(self.设备)
                批动作=批动作.to(self.设备)
                预测=self.模型(批状态)
                损失=self.损失(预测,批动作)
                self.优化器.zero_grad()
                损失.backward()
                torch.nn.utils.clip_grad_norm_(self.模型.parameters(),1.0)
                self.优化器.step()
                损失值=float(损失.item())
        return 损失值
    def 预测(self,状态):
        if not self.就绪():
            return None,0.0
        张=torch.from_numpy(状态.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.设备)
        with torch.no_grad():
            输出=self.模型(张)
            概率=torch.softmax(输出,dim=1)
            信心,索引=torch.max(概率,dim=1)
        return int(索引.item()),float(信心.item())
    def 就绪(self):
        return self.模型 is not None
    def 保存(self,路径):
        if not self.就绪():
            return
        torch.save({"模型":self.模型.state_dict(),"动作数":self.动作空间.大小()},路径)
    def 加载(self,路径):
        if not 路径.exists():
            return False
        数据=torch.load(路径,map_location=self.设备)
        目标=数据.get("动作数",0)
        while self.动作空间.大小()<目标:
            self.动作空间.索引({"类型":"占位","值":None,"延迟":0})
        self.同步()
        self.模型.load_state_dict(数据["模型"])
        return True
class 强化学习器:
    def __init__(self,配置,动作空间):
        self.配置=配置
        self.动作空间=动作空间
        self.缓冲=重放缓冲(self.配置.取("缓冲上限"))
        self.设备=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.策略=None
        self.目标=None
        self.优化器=None
        self.折扣=self.配置.取("折扣因子")
        self.衰减=self.配置.取("探索衰减")
        self.最小探索=self.配置.取("最小探索")
        self.探索率=float(self.配置.取("初始探索率"))
        self.步数=0
    def 载入缓冲(self,数据):
        self.缓冲.预填(数据)
        if 数据:
            self.探索率=max(self.最小探索,self.探索率*(self.衰减**len(数据)))
    def 初始化网络(self):
        动作数=max(1,self.动作空间.大小())
        self.策略=策略网络(self.配置.取("采样宽度"),self.配置.取("采样高度"),动作数).to(self.设备)
        self.目标=策略网络(self.配置.取("采样宽度"),self.配置.取("采样高度"),动作数).to(self.设备)
        self.目标.load_state_dict(self.策略.state_dict())
        self.优化器=optim.Adam(self.策略.parameters(),lr=self.配置.取("学习率"))
    def 存储(self,状态,动作,奖励,下一个,终止):
        self.缓冲.添加(状态,动作,奖励,下一个,终止)
    def 训练(self):
        if self.策略 is None:
            self.初始化网络()
        if self.缓冲.大小()<self.配置.取("批量"):
            return None
        损失值=None
        for _ in range(self.配置.取("训练轮数")):
            状态集,动作集,奖励集,下个集,终止集=self.缓冲.批量(self.配置.取("批量"))
            状态张=torch.from_numpy(状态集).unsqueeze(1).to(self.设备)
            动作张=torch.from_numpy(动作集).to(self.设备)
            奖励张=torch.from_numpy(奖励集).to(self.设备)
            下个张=torch.from_numpy(下个集).unsqueeze(1).to(self.设备)
            终止张=torch.from_numpy(终止集).to(self.设备)
            当前值=self.策略(状态张).gather(1,动作张.view(-1,1)).squeeze(1)
            下个值=self.目标(下个张).max(1)[0]
            目标值=奖励张+(1-终止张)*self.折扣*下个值
            损失=nn.functional.smooth_l1_loss(当前值,目标值.detach())
            self.优化器.zero_grad()
            损失.backward()
            torch.nn.utils.clip_grad_norm_(self.策略.parameters(),1.0)
            self.优化器.step()
            if self.步数%10==0:
                self.目标.load_state_dict(self.策略.state_dict())
            self.步数+=1
            self.探索率=max(self.最小探索,self.探索率*self.衰减)
            损失值=损失.item()
        return 损失值
    def 选择动作(self,状态):
        if self.策略 is None:
            self.初始化网络()
        if random.random()<self.探索率:
            return random.randint(0,max(0,self.动作空间.大小()-1))
        张=torch.from_numpy(状态.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.设备)
        with torch.no_grad():
            价值=self.策略(张)
        return int(torch.argmax(价值,dim=1).item())
    def 保存(self,路径):
        if self.策略 is None:
            return
        torch.save({"模型":self.策略.state_dict(),"动作数":self.动作空间.大小(),"探索率":self.探索率},路径)
    def 加载(self,路径):
        if not 路径.exists():
            return False
        数据=torch.load(路径,map_location=self.设备)
        目标动作=数据.get("动作数",0)
        while self.动作空间.大小()<目标动作:
            self.动作空间.索引({"类型":"占位","值":None,"延迟":0})
        self.初始化网络()
        self.策略.load_state_dict(数据["模型"])
        self.目标.load_state_dict(self.策略.state_dict())
        self.探索率=数据.get("探索率",self.探索率)
        return True
class 事件采集器:
    def __init__(self,配置,动作空间,屏幕):
        self.配置=配置
        self.动作空间=动作空间
        self.屏幕=屏幕
        self.停止=Event()
        self.锁=Lock()
        self.轮询=self.配置.取("轮询毫秒")/1000
        self.停止键=键助手.解析(self.配置.取("停止键"))
        self.记录=[]
        self.时间=0
        self.帧跳=max(1,self.配置.取("帧跳"))
        self.计数=0
    def 键按(self,键):
        if 键==self.停止键:
            self.停止.set()
            return False
        return self.写入({"类型":"键盘按下","值":键助手.标识(键)})
    def 键抬(self,键):
        return self.写入({"类型":"键盘抬起","值":键助手.标识(键)})
    def 鼠移(self,x,y):
        return self.写入({"类型":"鼠标移动","值":[x,y]})
    def 鼠点(self,x,y,按钮,按下):
        return self.写入({"类型":"鼠标点击","值":{"位置":[x,y],"按钮":str(按钮),"按下":按下}})
    def 鼠滚(self,x,y,dx,dy):
        return self.写入({"类型":"鼠标滚轮","值":{"位置":[x,y],"水平":dx,"垂直":dy}})
    def 写入(self,信息):
        if self.停止.is_set():
            return False
        self.计数+=1
        if self.计数%self.帧跳!=0 and 信息["类型"]=="鼠标移动":
            return True
        当前=time.perf_counter()
        if not self.时间:
            self.时间=当前
        延迟=当前-self.时间
        self.时间=当前
        状态=self.屏幕.截取()
        with self.锁:
            self.记录.append({"事件":信息,"延迟":延迟,"状态":状态})
        return True
    def 采集(self):
        print("正在记录操作与屏幕状态，请开始游戏，按停止键结束。")
        self.记录=[]
        self.停止.clear()
        self.时间=0
        self.计数=0
        键监听=keyboard.Listener(on_press=self.键按,on_release=self.键抬)
        鼠监听=mouse.Listener(on_move=self.鼠移,on_click=self.鼠点,on_scroll=self.鼠滚)
        键监听.start()
        鼠监听.start()
        起始=time.perf_counter()
        限制=self.配置.取("记录时长秒")
        while not self.停止.is_set():
            if time.perf_counter()-起始>=限制:
                self.停止.set()
                break
            time.sleep(self.轮询)
        键监听.stop()
        鼠监听.stop()
        键监听.join()
        鼠监听.join()
        print("采集完成，准备训练。")
        return list(self.记录)
class 自动执行:
    def __init__(self,配置,动作空间,强化,屏幕,克隆):
        self.配置=配置
        self.动作空间=动作空间
        self.强化=强化
        self.屏幕=屏幕
        self.克隆=克隆
        self.停止=Event()
        self.监听=None
        self.同步=self.配置.取("同步毫秒")/1000
        self.鼠标=self.配置.取("鼠标移动毫秒")/1000
        self.停止键=键助手.解析(self.配置.取("停止键"))
        self.自动探索=float(self.配置.取("自动执行探索率"))
    def 监听停止(self):
        def 回调(键):
            if 键==self.停止键:
                self.停止.set()
                return False
        self.监听=keyboard.Listener(on_press=回调)
        self.监听.start()
    def 执行动作(self,事件):
        延迟=max(0.0,float(事件.get("延迟",0)))
        if 延迟>0:
            time.sleep(延迟)
        类型=事件["类型"]
        值=事件["值"]
        if 类型=="键盘按下":
            pyautogui.keyDown(键助手.自动化值(值))
        elif 类型=="键盘抬起":
            pyautogui.keyUp(键助手.自动化值(值))
        elif 类型=="鼠标移动":
            目标=self.变换坐标(值)
            pyautogui.moveTo(目标[0],目标[1],duration=self.鼠标)
        elif 类型=="鼠标点击":
            目标=self.变换坐标(值["位置"])
            按钮=值["按钮"].split(".")[-1]
            if 值["按下"]:
                pyautogui.mouseDown(x=目标[0],y=目标[1],button=按钮)
            else:
                pyautogui.mouseUp(x=目标[0],y=目标[1],button=按钮)
        elif 类型=="鼠标滚轮":
            目标=self.变换坐标(值["位置"])
            pyautogui.moveTo(目标[0],目标[1])
            if 值["垂直"]:
                pyautogui.scroll(int(值["垂直"]))
            if 值["水平"]:
                pyautogui.hscroll(int(值["水平"]))
    def 变换坐标(self,坐标):
        屏幕尺寸=pyautogui.size()
        if isinstance(坐标,list) and len(坐标)==2 and max(坐标)<=1:
            return [int(坐标[0]*屏幕尺寸.width),int(坐标[1]*屏幕尺寸.height)]
        return [int(坐标[0]),int(坐标[1])]
    def 开始(self):
        if self.动作空间.大小()==0:
            print("缺少动作数据，无法执行。")
            return
        self.监听停止()
        print("已进入自动策略执行模式，按停止键随时结束。")
        原探索=self.强化.探索率
        if 0<=self.自动探索<=1:
            self.强化.探索率=max(self.强化.最小探索,float(self.自动探索))
        try:
            while not self.停止.is_set():
                状态=self.屏幕.截取()
                动作索引=None
                if self.克隆 and self.克隆.就绪():
                    预测,信心=self.克隆.预测(状态)
                    if 预测 is not None and 信心>=float(self.配置.取("克隆置信")):
                        动作索引=预测
                if 动作索引 is None:
                    动作索引=self.强化.选择动作(状态)
                事件=self.动作空间.事件(动作索引)
                if not 事件:
                    continue
                self.执行动作(事件)
                time.sleep(self.同步)
        finally:
            self.强化.探索率=原探索
            if self.监听:
                self.监听.stop()
        print("自动执行结束。")
class 控制器:
    def __init__(self,外部):
        self.根=(桌面路径()/"GameAI").resolve()
        self.根.mkdir(parents=True,exist_ok=True)
        self.硬件=硬件环境(self.根)
        默认=self.硬件.默认参数()
        默认.update({k:v for k,v in 外部.items() if v is not None})
        self.配置=配置管理(self.根,默认)
        self.屏幕=屏幕采集(self.配置.取("采样宽度"),self.配置.取("采样高度"))
        self.动作=动作空间(self.根/self.配置.取("动作文件"))
        self.数据=数据存储(self.根/self.配置.取("数据文件"))
        self.奖励=奖励评估(self.配置)
        self.克隆=行为克隆器(self.配置,self.动作)
        self.强化=强化学习器(self.配置,self.动作)
        self.强化.载入缓冲(self.数据.读取())
        self.强化.加载(self.根/self.配置.取("模型文件"))
        self.克隆.加载(self.根/self.配置.取("克隆模型"))
    def 标准化(self,事件,尺寸):
        类型=事件["类型"]
        值=事件["值"]
        延迟=事件.get("延迟",0)
        if 类型=="鼠标移动":
            return {"类型":类型,"值":[max(0,min(1,值[0]/尺寸.width)),max(0,min(1,值[1]/尺寸.height))],"延迟":延迟}
        if 类型=="鼠标点击":
            位置=值["位置"]
            return {"类型":类型,"值":{"位置":[max(0,min(1,位置[0]/尺寸.width)),max(0,min(1,位置[1]/尺寸.height))],"按钮":值["按钮"],"按下":值["按下"]},"延迟":延迟}
        if 类型=="鼠标滚轮":
            位置=值["位置"]
            return {"类型":类型,"值":{"位置":[max(0,min(1,位置[0]/尺寸.width)),max(0,min(1,位置[1]/尺寸.height))],"水平":值["水平"],"垂直":值["垂直"]},"延迟":延迟}
        return {"类型":类型,"值":值,"延迟":延迟}
    def 训练(self,记录):
        if not 记录:
            return
        尺寸=pyautogui.size()
        状态序列=[]
        动作序列=[]
        奖励序列=[]
        下个序列=[]
        终止序列=[]
        self.奖励.重置()
        for i,项 in enumerate(记录):
            事件=项["事件"]
            标准=self.标准化({"类型":事件["类型"],"值":事件["值"],"延迟":项["延迟"]},尺寸)
            动作索引=self.动作.索引(标准)
            状态=项["状态"]
            if i+1<len(记录):
                下一个=记录[i+1]["状态"]
                终止=0.0
            else:
                下一个=状态
                终止=1.0
            奖励=self.奖励.计算(状态,下一个)
            状态序列.append(状态)
            动作序列.append(动作索引)
            奖励序列.append(奖励)
            下个序列.append(下一个)
            终止序列.append(终止)
        self.动作.保存()
        self.数据.合并(状态序列,动作序列,奖励序列,下个序列,终止序列)
        for s,a,r,n,d in zip(状态序列,动作序列,奖励序列,下个序列,终止序列):
            self.强化.存储(s,a,r,n,d)
        损失=self.强化.训练()
        if 损失 is not None:
            print(f"策略损失:{损失:.4f}")
        克隆损失=self.克隆.训练(状态序列,动作序列)
        if 克隆损失 is not None:
            print(f"克隆损失:{克隆损失:.4f}")
        self.强化.保存(self.根/self.配置.取("模型文件"))
        self.克隆.保存(self.根/self.配置.取("克隆模型"))
    def 运行(self):
        print(f"检测到{self.硬件.描述()}，已生成自适应配置。")
        print("程序会自动记录并训练策略，请在提示后立即游戏，按停止键结束。")
        采集器=事件采集器(self.配置,self.动作,self.屏幕)
        记录=采集器.采集()
        self.训练(记录)
        执行=自动执行(self.配置,self.动作,self.强化,self.屏幕,self.克隆)
        执行.开始()
        print("感谢体验，重新运行可继续强化模型。")
if __name__=="__main__":
    解析=argparse.ArgumentParser()
    解析.add_argument("--stop",dest="停止键",default=None)
    解析.add_argument("--duration",dest="记录时长秒",type=int,default=None)
    解析.add_argument("--resolution",dest="分辨率",default=None)
    解析.add_argument("--batch",dest="批量",type=int,default=None)
    解析.add_argument("--lr",dest="学习率",type=float,default=None)
    解析.add_argument("--sync",dest="同步毫秒",type=int,default=None)
    解析.add_argument("--mouse",dest="鼠标移动毫秒",type=int,default=None)
    解析.add_argument("--discount",dest="折扣因子",type=float,default=None)
    解析.add_argument("--epsilon",dest="最小探索",type=float,default=None)
    解析.add_argument("--decay",dest="探索衰减",type=float,default=None)
    参数=vars(解析.parse_args())
    自定义={}
    for 键,值 in 参数.items():
        if 值 is None:
            continue
        if 键=="分辨率":
            try:
                宽,高=[int(x) for x in 值.split("x")]
                自定义["采样宽度"]=max(32,宽)
                自定义["采样高度"]=max(32,高)
            except Exception:
                continue
        elif 键 in {"学习率","折扣因子","最小探索","探索衰减"}:
            自定义[键]=float(值)
        else:
            自定义[键]=值 if 键!="记录时长秒" else max(60,int(值))
    控制=控制器(自定义)
    控制.运行()
