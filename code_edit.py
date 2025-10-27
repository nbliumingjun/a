import tkinter as tk
from tkinter import filedialog,messagebox
import os,re
default_dir="E:\\游戏AI"
root=None
dir_entry=None
label_latest=None
label_next=None
patch_text=None
preview_text=None
status_label=None
def find_latest_file(folder):
    pattern=re.compile(r'^code(\d+)\.txt$',re.IGNORECASE)
    max_n=0
    latest_name=None
    try:
        items=os.listdir(folder)
    except Exception:
        items=[]
    for name in items:
        m=pattern.match(name)
        if m:
            n=int(m.group(1))
            if n>max_n:
                max_n=n
                latest_name=name
    return max_n,latest_name
def read_file(path):
    try:
        with open(path,'r',encoding='utf-8') as f:
            return f.read()
    except Exception:
        try:
            with open(path,'r',encoding='utf-8-sig') as f:
                return f.read()
        except Exception:
            return ""
def write_file(path,text):
    with open(path,'w',encoding='utf-8') as f:
        f.write(text)
def normalize_input(raw):
    if "diff --git" in raw:
        start=raw.find("diff --git")
        return raw[start:]
    return raw
def is_unified_diff(t):
    s=t.lstrip()
    return ('@@' in t) or s.startswith('---') or s.startswith('diff ')
def apply_unified_diff(orig_text,diff_text):
    lines=diff_text.splitlines()
    hunks=[]
    i=0
    while i<len(lines):
        l=lines[i]
        if l.startswith('@@'):
            header=l
            i+=1
            body=[]
            while i<len(lines) and not lines[i].startswith('@@'):
                if lines[i].startswith('--- ') or lines[i].startswith('+++ '):
                    i+=1
                    continue
                body.append(lines[i])
                i+=1
            hunks.append((header,body))
        else:
            i+=1
    regex=re.compile(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    orig_lines=orig_text.splitlines()
    res_lines=[]
    orig_index=0
    for header,body in hunks:
        m=regex.match(header)
        if not m:
            return diff_text
        a_start=int(m.group(1))
        if a_start<1:
            a_start=1
        copy_upto=a_start-1
        if copy_upto>orig_index:
            res_lines.extend(orig_lines[orig_index:copy_upto])
            orig_index=copy_upto
        for bline in body:
            if bline.startswith(' '):
                if orig_index<len(orig_lines):
                    res_lines.append(orig_lines[orig_index])
                else:
                    res_lines.append(bline[1:])
                orig_index+=1
            elif bline.startswith('-'):
                if orig_index<len(orig_lines):
                    orig_index+=1
            elif bline.startswith('+'):
                res_lines.append(bline[1:])
            elif bline.startswith('\\'):
                pass
            else:
                res_lines.append(bline)
    for rem in orig_lines[orig_index:]:
        res_lines.append(rem)
    return '\n'.join(res_lines)
def load_preview():
    folder=dir_entry.get().strip() or default_dir
    max_n,latest_name=find_latest_file(folder)
    content=""
    if latest_name:
        content=read_file(os.path.join(folder,latest_name))
    preview_text.config(state='normal')
    preview_text.delete('1.0','end')
    preview_text.insert('1.0',content)
    preview_text.config(state='disabled')
def refresh_info():
    folder=dir_entry.get().strip()
    if not folder:
        folder=default_dir
    os.makedirs(folder,exist_ok=True)
    max_n,latest_name=find_latest_file(folder)
    if max_n==0:
        label_latest.config(text="最新版本: (无)")
        label_next.config(text="将生成: code1.txt")
    else:
        label_latest.config(text="最新版本: "+latest_name)
        label_next.config(text="将生成: code"+str(max_n+1)+".txt")
    load_preview()
def browse_dir():
    folder=filedialog.askdirectory(initialdir=dir_entry.get() or default_dir)
    if folder:
        dir_entry.delete(0,'end')
        dir_entry.insert(0,folder)
        refresh_info()
def generate():
    folder=dir_entry.get().strip() or default_dir
    os.makedirs(folder,exist_ok=True)
    max_n,latest_name=find_latest_file(folder)
    orig_text=""
    if latest_name:
        orig_path=os.path.join(folder,latest_name)
        orig_text=read_file(orig_path)
    raw_input_text=patch_text.get('1.0','end-1c')
    if not raw_input_text.strip():
        messagebox.showwarning("提示","请输入git apply或补丁内容")
        return
    patch=normalize_input(raw_input_text)
    if is_unified_diff(patch):
        new_text=apply_unified_diff(orig_text,patch)
    else:
        new_text=patch
    new_index=max_n+1 if max_n>0 else 1
    new_name="code"+str(new_index)+".txt"
    new_path=os.path.join(folder,new_name)
    write_file(new_path,new_text)
    status_label.config(text="已创建 "+new_name+" ("+str(len(new_text.splitlines()))+" 行)")
    messagebox.showinfo("完成","新版本已保存为 "+new_name)
    patch_text.delete('1.0','end')
    refresh_info()
def main():
    global root,dir_entry,label_latest,label_next,patch_text,preview_text,status_label
    root=tk.Tk()
    root.title("程序版本生成器")
    bg_main="#f5f5f5"
    bg_card="#ffffff"
    accent="#4caf50"
    root.configure(bg=bg_main)
    root.geometry("900x600")
    top_frame=tk.Frame(root,bg=bg_main)
    top_frame.pack(fill="x",padx=10,pady=10)
    tk.Label(top_frame,text="工作目录:",bg=bg_main,font=("Microsoft YaHei",10)).grid(row=0,column=0,sticky="w")
    dir_entry=tk.Entry(top_frame,width=50,font=("Consolas",10))
    dir_entry.grid(row=0,column=1,sticky="we",padx=(5,5))
    dir_entry.insert(0,default_dir)
    browse_btn=tk.Button(top_frame,text="浏览...",command=browse_dir,font=("Microsoft YaHei",9),cursor="hand2")
    browse_btn.grid(row=0,column=2,padx=(0,5))
    refresh_btn=tk.Button(top_frame,text="刷新",command=refresh_info,font=("Microsoft YaHei",9),cursor="hand2")
    refresh_btn.grid(row=0,column=3)
    top_frame.grid_columnconfigure(1,weight=1)
    label_latest=tk.Label(top_frame,text="最新版本:",bg=bg_main,font=("Microsoft YaHei",10,"bold"),fg="#333")
    label_latest.grid(row=1,column=0,columnspan=2,sticky="w",pady=(8,0))
    label_next=tk.Label(top_frame,text="将生成:",bg=bg_main,font=("Microsoft YaHei",10),fg="#555")
    label_next.grid(row=1,column=2,columnspan=2,sticky="e",pady=(8,0))
    body_pane=tk.PanedWindow(root,orient="horizontal",sashrelief="groove",sashwidth=4,bg="#cccccc")
    body_pane.pack(fill="both",expand=True,padx=10,pady=(0,10))
    left_frame=tk.Frame(body_pane,bg=bg_card,bd=1,relief="solid")
    right_frame=tk.Frame(body_pane,bg=bg_card,bd=1,relief="solid")
    body_pane.add(left_frame)
    body_pane.add(right_frame)
    tk.Label(left_frame,text="在此粘贴git apply或补丁:",bg=bg_card,anchor="w",font=("Microsoft YaHei",10,"bold")).pack(fill="x",padx=10,pady=(10,5))
    patch_container=tk.Frame(left_frame,bg=bg_card)
    patch_container.pack(fill="both",expand=True,padx=10,pady=(0,10))
    patch_scroll=tk.Scrollbar(patch_container,orient="vertical")
    patch_text=tk.Text(patch_container,wrap="none",yscrollcommand=patch_scroll.set,font=("Consolas",10),bd=1,relief="solid",undo=True)
    patch_scroll.config(command=patch_text.yview)
    patch_text.pack(side="left",fill="both",expand=True)
    patch_scroll.pack(side="right",fill="y")
    tk.Label(right_frame,text="当前最新文件预览(只读):",bg=bg_card,anchor="w",font=("Microsoft YaHei",10,"bold")).pack(fill="x",padx=10,pady=(10,5))
    preview_container=tk.Frame(right_frame,bg=bg_card)
    preview_container.pack(fill="both",expand=True,padx=10,pady=(0,10))
    preview_scroll=tk.Scrollbar(preview_container,orient="vertical")
    preview_text=tk.Text(preview_container,wrap="none",yscrollcommand=preview_scroll.set,font=("Consolas",10),bd=1,relief="solid",state="disabled")
    preview_scroll.config(command=preview_text.yview)
    preview_text.pack(side="left",fill="both",expand=True)
    preview_scroll.pack(side="right",fill="y")
    bottom_frame=tk.Frame(root,bg=bg_main)
    bottom_frame.pack(fill="x",padx=10,pady=(0,10))
    generate_btn=tk.Button(bottom_frame,text="生成新版本",command=generate,font=("Microsoft YaHei",11,"bold"),bg=accent,fg="white",activebackground="#45a049",activeforeground="white",bd=0,padx=20,pady=8,cursor="hand2")
    generate_btn.pack(side="left")
    status_label=tk.Label(bottom_frame,text="",bg=bg_main,fg="#333",font=("Microsoft YaHei",9))
    status_label.pack(side="left",padx=10)
    refresh_info()
    root.mainloop()
if __name__=="__main__":
    main()
