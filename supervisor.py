"""监控专用 Web 进程；意外退出后等待 5 秒再启动，保留所有任务数据。"""
from __future__ import annotations
import logging
import os
import subprocess
import sys
import socket
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parent
logging.basicConfig(filename=ROOT/'supervisor.log',level=logging.INFO,encoding='utf-8',format='%(asctime)s %(message)s')


def main() -> None:
    """只管理由本监督器创建的 Web 及其已记录的模型子进程。"""
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',7865)) == 0:
            logging.error('端口 7865 已在使用，不重复启动应用')
            print('端口 7865 已被占用，未启动第二个实例')
            return
    while True:
        with (ROOT/'web-out.log').open('ab') as stdout, (ROOT/'web-error.log').open('ab') as stderr:
            child=subprocess.Popen([sys.executable,str(ROOT/'app.py')],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        logging.info('启动应用 pid=%s',child.pid)
        tracked={}
        try:
            while child.poll() is None:
                try:
                    for proc in psutil.Process(child.pid).children(recursive=True):
                        try:
                            if Path(proc.exe()).resolve()==(ROOT/'runtime_new'/'llama-server.exe').resolve():
                                tracked[proc.pid]=proc.create_time()
                        except (psutil.Error,OSError): pass
                except psutil.Error: pass
                time.sleep(2)
            code=child.returncode
            logging.warning('应用退出 pid=%s code=%s，5 秒后重启',child.pid,code)
            for pid,created in tracked.items():
                try:
                    proc=psutil.Process(pid)
                    if proc.create_time()==created and Path(proc.exe()).resolve()==(ROOT/'runtime_new'/'llama-server.exe').resolve():
                        proc.terminate();proc.wait(timeout=10)
                except (psutil.Error,OSError): pass
            time.sleep(5)
        except KeyboardInterrupt:
            logging.info('监督器由用户停止')
            if child.poll() is None: child.terminate()
            return


if __name__=='__main__':
    main()
