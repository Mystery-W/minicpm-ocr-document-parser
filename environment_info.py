"""收集本机运行环境摘要；仅供本地页面展示，不向外部发送。"""
from __future__ import annotations
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import psutil


def environment_summary(root: Path) -> dict:
    """报告独立环境、显存、内存和模型文件；缺失时明确返回不可用。"""
    packages = {}
    for name in ('fastapi','uvicorn','pymupdf','Pillow','paddlepaddle','paddlex','minicpm-ocr','pypdfium2','psutil'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    gpu=[]; error=None
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=5,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for line in result.stdout.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)==5:
                gpu.append(dict(name=parts[0],driver=parts[1],memory_total_mib=parts[2],memory_used_mib=parts[3],utilization_percent=parts[4]))
    except (OSError,subprocess.SubprocessError) as exc:
        error=f'无法读取 NVIDIA 状态：{type(exc).__name__}'
    mem=psutil.virtual_memory(); disk=shutil.disk_usage(root)
    files={'OCR 模型':root/'models'/'MiniCPM-OCR-1_0-Q8_0.gguf','视觉投影':root/'models'/'mmproj-model-f16-4x.gguf','版面模型':root/'models'/'PP-DocLayoutV3_infer'/'inference.pdiparams','推理服务':root/'runtime_new'/'llama-server.exe'}
    system=platform.platform()
    if sys.platform == 'win32':
        version=sys.getwindowsversion()
        system=f"Windows {'11' if version.build >= 22000 else '10'} build {version.build} ({platform.machine()})"
    return {'system':system,'python':platform.python_version(),'python_executable':sys.executable,
            'environment_prefix':sys.prefix,'layout_device':'CPU','ocr_device_requested':'NVIDIA GPU（实际取决于后端加载）',
            'ram_total_gib':round(mem.total/2**30,2),'ram_available_gib':round(mem.available/2**30,2),'ram_used_percent':mem.percent,
            'disk_free_gib':round(disk.free/2**30,2),'gpu':gpu,'gpu_error':error,
            'files':{label:{'available':path.is_file(),'path':str(path)} for label,path in files.items()},'packages':packages}
