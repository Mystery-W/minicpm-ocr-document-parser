"""下载固定版本的官方模型和 Windows CUDA 推理程序，不修改全局环境。

所有下载先写入 .part，支持 HTTP Range 断点续传；仅 SHA-256 匹配才安装。
解包时拒绝路径穿越，已有模型及运行时不会被覆盖。首次运行需要网络。
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import quote
import requests

ROOT=Path(__file__).resolve().parent
DOWNLOADS=ROOT/'downloads'
MODEL_REPO='https://huggingface.co/openbmb/MiniCPM-OCR-1.0-gguf/resolve/main/'
MODEL_MIRROR='https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-gguf/resolve/master/'
RELEASE='https://github.com/ggml-org/llama.cpp/releases/download/b11972/'
ASSETS=[
 ('MiniCPM-OCR-1_0-Q8_0.gguf',811598432,'791486a476bdacb48acb3f31aa3a18dbb5be87e723a4074bcd13a37d61c852e2'),
 ('mmproj-model-f16-4x.gguf',901017536,'ceed82261b3b1380c3165f8b37c10117a50093be1362cbfeeb692fa036444682'),
 ('PP-DocLayoutV3_infer.tar',132014080,'98b9bac88c80f6bc0fda7e0bfc2cae180020371c0b2edbb1eb498a70ace751b1'),
 ('llama-b11972-bin-win-cuda-12.4-x64.zip',279764740,'2c8b6e789650f836291e7b1732dc6c11f6b3a1cf86440fc19d14caa2f55237bd'),
 ('cudart-llama-bin-win-cuda-12.4-x64.zip',391443627,'8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6')]


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def download(url: str, path: Path, size: int, sha: str) -> None:
    """继续未完成下载；不信任长度不符或哈希错误的结果。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size==size and digest(path)==sha:
            print(f'已校验，跳过：{path.name}',flush=True);return
        raise RuntimeError(f'现有文件校验失败，保留原文件：{path}')
    part=path.with_name(path.name+'.part')
    for attempt in range(4):
        try:
            start=part.stat().st_size if part.exists() else 0
            if start>size: raise ValueError('临时文件超过预期大小，请检查后手动处理')
            if start==size:
                if digest(part)!=sha: raise ValueError('临时文件 SHA-256 不匹配，已保留供检查')
                part.replace(path);return
            headers={'Range':f'bytes={start}-'} if start else {}
            with requests.get(url,headers=headers,stream=True,timeout=(30,180)) as r:
                r.raise_for_status()
                if start and r.status_code==206:
                    if not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise ValueError('续传范围不匹配')
                    mode='ab'
                elif r.status_code==200: mode='wb'
                else: raise ValueError(f'意外的下载响应：{r.status_code}')
                print(f'下载 {path.name}（已有 {start/2**20:.0f} MiB）',flush=True)
                with part.open(mode) as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk:f.write(chunk)
            if part.stat().st_size!=size:raise IOError('下载尚未完成，将继续')
            if digest(part)!=sha:raise ValueError('SHA-256 不匹配，临时文件已保留')
            part.replace(path);print(f'校验通过：{path.name}',flush=True);return
        except (requests.RequestException,IOError) as exc:
            if attempt==3:raise RuntimeError(f'{path.name} 下载失败：{type(exc).__name__}；可重新运行继续') from exc
            time.sleep(3)


def safe_zip(archive: Path, dest: Path) -> None:
    """仅解压普通文件到临时目录，拒绝绝对路径及父目录。"""
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name=item.filename.replace('\\','/')
            target=(dest/name).resolve()
            if not target.is_relative_to(dest.resolve()) or ':' in name or name.startswith('/'):
                raise ValueError(f'不安全的压缩路径：{name}')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:raise ValueError('不接受符号链接')
        z.extractall(dest)


def main() -> None:
    parser=argparse.ArgumentParser(description='下载官方 OCR、版面模型和 Windows CUDA 推理程序')
    parser.add_argument('--model-source',choices=['huggingface','modelscope'],default='huggingface')
    args=parser.parse_args();DOWNLOADS.mkdir(exist_ok=True)
    model_dir=ROOT/'models';model_dir.mkdir(exist_ok=True)
    for name,size,sha in ASSETS[:2]:
        base=MODEL_REPO if args.model_source=='huggingface' else MODEL_MIRROR
        download(base+quote(name),model_dir/name,size,sha)
    layout=model_dir/'PP-DocLayoutV3_infer'
    required=('inference.pdiparams','inference.json','inference.yml')
    if layout.exists():
        if not all((layout/x).is_file() and (layout/x).stat().st_size>0 for x in required):raise RuntimeError('版面模型目录不完整，已保留，请手动检查')
        print('版面模型目录已存在，跳过（未重新校验已有解包文件）',flush=True)
    else:
        name,size,sha=ASSETS[2];path=DOWNLOADS/name
        download('https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/'+name,path,size,sha)
        with tempfile.TemporaryDirectory(dir=model_dir,prefix='.layout-') as tmp:
            stage=Path(tmp)/'model';stage.mkdir();found=set()
            with tarfile.open(path) as tar:
                for item in tar:
                    parts=Path(item.name).parts
                    if item.issym() or item.islnk() or '..' in parts or item.name.startswith(('/', '\\')):raise ValueError('不安全的 tar 条目')
                    if item.isfile() and Path(item.name).name in required:
                        name=Path(item.name).name
                        if name in found:raise ValueError('重复模型文件')
                        with tar.extractfile(item) as src,(stage/name).open('wb') as dst:shutil.copyfileobj(src,dst)
                        found.add(name)
            if found!=set(required):raise ValueError('版面模型缺少文件')
            stage.rename(layout)
    runtime=ROOT/'runtime_new'
    if runtime.exists():
        if not (runtime/'llama-server.exe').is_file():raise RuntimeError('运行时目录不完整，已保留，请手动检查')
        print('推理运行时已存在，跳过（未重新校验已有文件）',flush=True)
    else:
        archives=[]
        for name,size,sha in ASSETS[3:]:
            target=DOWNLOADS/name;download(RELEASE+name,target,size,sha);archives.append(target)
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.runtime-') as tmp:
            stage=Path(tmp)/'runtime';stage.mkdir()
            for archive in archives:safe_zip(archive,stage)
            if not (stage/'llama-server.exe').is_file():raise RuntimeError('运行时压缩包缺少 llama-server.exe')
            stage.rename(runtime)
    print('资源准备完成；请启动应用。',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'资源准备失败：{exc}',file=sys.stderr);raise SystemExit(1)
