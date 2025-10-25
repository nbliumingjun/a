import json
import time
import os
import platform
import subprocess
import ctypes
from ctypes import wintypes
from pathlib import Path
from threading import Event,Lock
from collections import deque,defaultdict
from pynput import mouse,keyboard
import pyautogui
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
    def __init__(self):
        self.尺寸=pyautogui.size()
        self.核心=os.cpu_count() or 1
        self.内存=self.获取内存()
        self.显存=self.获取显存()
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
    def 默认配置(self):
        像素=max(self.尺寸.width*self.尺寸.height,1)
        核心=max(self.核心,1)
        内存值=self.内存 or 像素*4
        显存值=self.显存 or 像素*2
        基准像素=max(像素,10**6)
        轮询=max(2,min(30,int(1000/max(1,核心*4))))
        等待=max(3,min(15,int((内存值+显存值)//基准像素)))
        记录=max(30,min(600,int((内存值+显存值)//(10**8))))
        同步=max(轮询,min(200,int(1000/max(1,核心*2))))
        鼠标=max(10,min(200,int(1000/max(1,核心*8))))
        序列=max(3,min(12,int(内存值//(10**9))+3))
        重复=max(1,min(5,int(max(1,核心//2))))
        样本=max(50,min(1000,int((内存值+显存值)//(10**8))))
        指纹=f"{self.尺寸.width}x{self.尺寸.height}_{核心}_{内存值}_{显存值}"
        return {"轮询毫秒":轮询,"停止键":"Key.esc","倒计时间隔秒":max(1,min(3,max(1,等待//3))),"等待启动秒":等待,"记录时长秒":记录,"数据文件":f"records_{指纹}.json","序列长度":序列,"最小重复次数":重复,"播放循环":True,"实时同步毫秒":同步,"鼠标移动持续毫秒":鼠标,"最大样本":样本,"硬件指纹":指纹}
class 配置:
    def __init__(self,根,默认):
        self.根=根
        self.路径=根/"config.json"
        if self.路径.exists():
            数据=json.loads(self.路径.read_text(encoding="utf-8"))
            if 数据.get("硬件指纹")!=默认["硬件指纹"]:
                self.数据=默认
            else:
                self.数据={**默认,**数据}
        else:
            self.数据=默认
        self.数据["停止键"]=默认["停止键"]
        self.路径.write_text(json.dumps(self.数据,ensure_ascii=False),encoding="utf-8")
    def 获取(self,键):
        return self.数据[键]
class 数据库:
    def __init__(self,根,文件,最大):
        self.根=根
        self.文件=文件
        self.路径=根/文件
        self.最大=最大
        if self.路径.exists():
            self.内容=json.loads(self.路径.read_text(encoding="utf-8"))
            if self.最大 and len(self.内容)>self.最大:
                self.内容=self.内容[-self.最大:]
                self.保存()
        else:
            self.内容=[]
    def 添加(self,记录):
        self.内容.append(记录)
        if self.最大 and len(self.内容)>self.最大:
            self.内容=self.内容[-self.最大:]
        self.保存()
    def 保存(self):
        self.路径.write_text(json.dumps(self.内容,ensure_ascii=False),encoding="utf-8")
    def 全部(self):
        return list(self.内容)
class 键工具:
    @staticmethod
    def 解析(文本):
        if 文本.startswith("Key."):
            名称=文本.split(".",1)[1]
            return getattr(keyboard.Key,名称)
        if len(文本)==1:
            return keyboard.KeyCode.from_char(文本)
        raise ValueError("无法解析停止键")
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
class 事件记录器:
    def __init__(self,配置,数据库):
        self.配置=配置
        self.数据库=数据库
        self.停止=Event()
        self.锁=Lock()
        self.事件=[]
        self.轮询秒=self.配置.获取("轮询毫秒")/1000
        self.停止键=键工具.解析(self.配置.获取("停止键"))
        self.开始时间=0
        self.最后时间=0
        self.倒计时间隔=self.配置.获取("倒计时间隔秒")
    def 标准化(self,事件列表,尺寸):
        宽,高=尺寸.width,尺寸.height
        转换=[]
        for 项 in 事件列表:
            新={"类型":项["类型"],"延迟":项.get("延迟",0)}
            值=项["值"]
            if 项["类型"]=="鼠标移动":
                新["值"]=self.坐标比率(值,宽,高)
            elif 项["类型"]=="鼠标点击":
                新["值"]={"位置":self.坐标比率(值["位置"],宽,高),"按钮":值["按钮"],"按下":值["按下"]}
            elif 项["类型"]=="鼠标滚轮":
                新["值"]={"位置":self.坐标比率(值["位置"],宽,高),"水平":值["水平"],"垂直":值["垂直"]}
            else:
                新["值"]=值
            转换.append(新)
        return 转换
    def 坐标比率(self,点,宽,高):
        return [round(max(0,min(点[0]/宽,1)),6),round(max(0,min(点[1]/高,1)),6)]
    def 键按下(self,键):
        if 键==self.停止键:
            self.停止.set()
            return False
        return self.记录({"类型":"键盘按下","值":键工具.标识(键)})
    def 键抬起(self,键):
        return self.记录({"类型":"键盘抬起","值":键工具.标识(键)})
    def 鼠标移动(self,x,y):
        return self.记录({"类型":"鼠标移动","值":[x,y]})
    def 鼠标点击(self,x,y,按钮,按下):
        return self.记录({"类型":"鼠标点击","值":{"位置":[x,y],"按钮":str(按钮),"按下":按下}})
    def 鼠标滚轮(self,x,y,dx,dy):
        return self.记录({"类型":"鼠标滚轮","值":{"位置":[x,y],"水平":dx,"垂直":dy}})
    def 记录(self,信息):
        if self.停止.is_set():
            return False
        此时=time.perf_counter()
        if not self.开始时间:
            self.开始时间=此时
            self.最后时间=此时
        延迟=此时-self.最后时间
        self.最后时间=此时
        with self.锁:
            self.事件.append({"类型":信息["类型"],"值":信息["值"],"延迟":延迟})
        return True
    def 采集(self):
        倒计时=self.配置.获取("等待启动秒")
        for 剩余 in range(倒计时,0,-1):
            print(f"将在{剩余}秒后开始记录，请开始准备游戏。")
            time.sleep(self.倒计时间隔)
        目标时长=self.配置.获取("记录时长秒")
        self.开始时间=0
        self.最后时间=0
        self.停止.clear()
        self.事件=[]
        键监听=keyboard.Listener(on_press=self.键按下,on_release=self.键抬起)
        鼠标监听=mouse.Listener(on_move=self.鼠标移动,on_click=self.鼠标点击,on_scroll=self.鼠标滚轮)
        键监听.start()
        鼠标监听.start()
        开始=time.perf_counter()
        while not self.停止.is_set():
            if time.perf_counter()-开始>=目标时长:
                self.停止.set()
                break
            time.sleep(self.轮询秒)
        键监听.stop()
        鼠标监听.stop()
        键监听.join()
        鼠标监听.join()
        尺寸=pyautogui.size()
        数据={"时间":time.time(),"事件":self.标准化(self.事件,尺寸),"尺寸":[尺寸.width,尺寸.height],"硬件指纹":self.配置.获取("硬件指纹")}
        self.数据库.添加(数据)
        print("键鼠记录已完成，开始训练自动化模型。")
        return 数据["事件"]
class 序列模型:
    def __init__(self,序列长度,最小重复次数):
        self.序列长度=序列长度
        self.最小重复次数=最小重复次数
        self.映射=defaultdict(list)
        self.起始=[]
    def 简化(self,事件):
        return (事件["类型"],self._值标识(事件["值"]))
    def _值标识(self,值):
        if isinstance(值,dict):
            return tuple(sorted((k,self._值标识(v)) for k,v in 值.items()))
        if isinstance(值,list):
            return tuple(self._值标识(v) for v in 值)
        return 值
    def 训练(self,样本):
        汇总=defaultdict(lambda:defaultdict(int))
        原始=defaultdict(dict)
        self.起始=[]
        for 记录 in 样本:
            事件序列=记录.get("事件",[])
            if not 事件序列:
                continue
            self.起始.append(事件序列[:self.序列长度])
            队列=deque(maxlen=self.序列长度)
            for 事件 in 事件序列:
                if len(队列)==self.序列长度:
                    上下文=tuple(self.简化(e) for e in 队列)
                    键=self.简化(事件)
                    if 键 not in 原始[上下文]:
                        原始[上下文][键]=事件
                    汇总[上下文][键]+=1
                队列.append(事件)
        映射={}
        for 上下文,计数 in 汇总.items():
            合格=[(原始[上下文][键],数量) for 键,数量 in 计数.items() if 数量>=self.最小重复次数]
            if 合格:
                合格.sort(key=lambda x:x[1],reverse=True)
                映射[上下文]=合格
        self.映射=defaultdict(list,映射)
    def 预测(self,上下文):
        结果=self.映射.get(上下文)
        if not 结果:
            return None
        return 结果[0][0]
    def 获取起始(self):
        if not self.起始:
            return []
        return max(self.起始,key=len)
class 自动玩家:
    def __init__(self,模型,循环,同步毫秒,停止键,鼠标移动持续毫秒):
        self.模型=模型
        self.循环=循环
        self.同步=同步毫秒/1000
        self.停止键=停止键
        self.鼠标持续=鼠标移动持续毫秒/1000
        self.停止=Event()
        self.监听=None
    def 启动监听(self):
        def 监控(键):
            if 键==self.停止键:
                self.停止.set()
                return False
        self.监听=keyboard.Listener(on_press=监控)
        self.监听.start()
    def 位置(self,比率):
        尺寸=pyautogui.size()
        return [min(max(int(round(比率[0]*尺寸.width)),0),尺寸.width-1),min(max(int(round(比率[1]*尺寸.height)),0),尺寸.height-1)]
    def 执行动作(self,事件):
        time.sleep(max(事件.get("延迟",0),0))
        类型=事件["类型"]
        值=事件["值"]
        if 类型=="键盘按下":
            pyautogui.keyDown(键工具.自动化值(值))
        elif 类型=="键盘抬起":
            pyautogui.keyUp(键工具.自动化值(值))
        elif 类型=="鼠标移动":
            坐标=self.位置(值)
            pyautogui.moveTo(坐标[0],坐标[1],duration=self.鼠标持续)
        elif 类型=="鼠标点击":
            坐标=self.位置(值["位置"])
            按钮=值["按钮"].split(".")[-1]
            if 值["按下"]:
                pyautogui.mouseDown(x=坐标[0],y=坐标[1],button=按钮)
            else:
                pyautogui.mouseUp(x=坐标[0],y=坐标[1],button=按钮)
        elif 类型=="鼠标滚轮":
            坐标=self.位置(值["位置"])
            pyautogui.moveTo(坐标[0],坐标[1])
            if 值["垂直"]:
                pyautogui.scroll(int(值["垂直"]))
            if 值["水平"]:
                pyautogui.hscroll(int(值["水平"]))
    def 播放(self):
        缓冲=deque(maxlen=self.模型.序列长度)
        初始=self.模型.获取起始()
        if not 初始:
            print("缺少训练数据，无法进行自动化播放。")
            return
        print("开始根据学习到的行为自动执行。按停止键结束自动化。")
        self.启动监听()
        while not self.停止.is_set():
            for 事件 in 初始:
                if self.停止.is_set():
                    break
                self.执行动作(事件)
                缓冲.append(事件)
            if self.停止.is_set():
                break
            while not self.停止.is_set():
                if len(缓冲)<self.模型.序列长度:
                    break
                上下文=tuple(self.模型.简化(e) for e in 缓冲)
                预测事件=self.模型.预测(上下文)
                if not 预测事件:
                    break
                self.执行动作(预测事件)
                缓冲.append(预测事件)
                time.sleep(self.同步)
            if not self.循环 or self.停止.is_set():
                break
        self.停止.set()
        if self.监听:
            self.监听.stop()
        print("自动化播放已结束。")
class 控制器:
    def __init__(self):
        self.根=(桌面路径()/"GameAI").resolve()
        self.根.mkdir(parents=True,exist_ok=True)
        self.硬件=硬件环境()
        默认=self.硬件.默认配置()
        self.配置=配置(self.根,默认)
        self.数据库=数据库(self.根,self.配置.获取("数据文件"),self.配置.获取("最大样本"))
        self.记录器=事件记录器(self.配置,self.数据库)
    def 运行(self):
        print(f"已根据{self.硬件.描述()}自动优化参数。")
        print("欢迎使用智能游戏助手。程序将自动记录您的操作并进行学习。按停止键可随时结束记录或自动化。")
        self.记录器.采集()
        模型=序列模型(self.配置.获取("序列长度"),self.配置.获取("最小重复次数"))
        数据集=self.数据库.全部()
        指纹=self.配置.获取("硬件指纹")
        样本=[记录 for 记录 in 数据集 if 记录.get("硬件指纹")==指纹]
        if not 样本:
            样本=数据集
        模型.训练(样本)
        玩家=自动玩家(模型,self.配置.获取("播放循环"),self.配置.获取("实时同步毫秒"),self.记录器.停止键,self.配置.获取("鼠标移动持续毫秒"))
        玩家.播放()
        print("感谢使用，您可以关闭程序或重新启动以再次训练。")
if __name__=="__main__":
    控制器().运行()
