"""本地多文件、逐页 OCR 服务。文件和每页结果独立保存，便于中断后继续。"""
from __future__ import annotations
import base64
import atexit
import io
import json
import logging
import os
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import re
import subprocess
import threading
import time
import uuid
import tempfile
import zipfile
from urllib.parse import quote
from pathlib import Path
from typing import Any
from pydantic import BaseModel

import pymupdf as fitz
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from PIL import Image, ImageOps
from safe_io import atomic_text, atomic_bytes, PauseRequested, install_context, check_pause, retryable, wait_retry, read_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
MODEL_DIR = ROOT / 'models'
SERVER = 'http://127.0.0.1:18499'
FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
logging.basicConfig(filename=ROOT / 'app.log', level=logging.INFO, encoding='utf-8', format='%(asctime)s %(levelname)s %(message)s')
app = FastAPI(title='MiniCPM 文档解析')
lock = threading.Lock()
active: set[str] = set()
pause_events: dict[str, threading.Event] = {}
volatile: dict[str, dict[str, Any]] = {}
batch_running = False
process: subprocess.Popen | None = None

def stop_owned_model() -> None:
    """关闭由本应用启动的模型进程，不触碰其他服务。"""
    if process and process.poll() is None:
        process.terminate()

atexit.register(stop_owned_model)


def folder(doc_id: str) -> Path:
    """只接受应用生成的 UUID，避免通过路径参数访问其他文件。"""
    if not re.fullmatch(r'[0-9a-f]{32}', doc_id):
        raise HTTPException(404, '文件不存在')
    p = DATA / doc_id
    if not p.is_dir():
        raise HTTPException(404, '文件不存在')
    return p


def read_meta(p: Path) -> dict[str, Any]:
    return json.loads(read_text(p / 'meta.json'))


def write_meta(p: Path, meta: dict[str, Any]) -> None:
    """原子替换状态文件，避免异常退出时留下半个 JSON。"""
    atomic_text(p / 'meta.json', json.dumps(meta, ensure_ascii=False, indent=2))


def page_count(path: Path) -> int:
    if path.suffix.lower() == '.pdf':
        with fitz.open(path) as doc:
            if doc.needs_pass: raise ValueError('PDF 有密码，请先另存为未加密副本')
            return len(doc)
    with Image.open(path) as im:
        return getattr(im, 'n_frames', 1)


def image_page(p: Path, index: int) -> bytes:
    """按需渲染单页，限制最长边降低显存压力；原文件不改动。"""
    m = read_meta(p); source = p / m['stored']; count = m['pages']
    if not 1 <= index <= count: raise HTTPException(404, '页码超出范围')
    if source.suffix.lower() == '.pdf':
        with fitz.open(source) as doc:
            page = doc[index - 1]
            scale = min(2.0, 2400 / max(page.rect.width, page.rect.height))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            return pix.tobytes('png')
    with Image.open(source) as im:
        im.seek(index - 1)
        image = ImageOps.exif_transpose(im.convert('RGB'))
        image.thumbnail((2400, 2400))
        buf = io.BytesIO(); image.save(buf, format='PNG'); return buf.getvalue()


def start_model() -> str:
    """启动项目专属服务；若其他服务已占端口则拒绝误用。"""
    global process
    if process and process.poll() is None:
        return '模型服务启动中或已运行'
    try:
        requests.get(SERVER + '/health', timeout=1)
        raise RuntimeError('18499 端口已被其他服务占用，请先释放端口')
    except requests.exceptions.ConnectionError:
        pass
    except requests.exceptions.Timeout:
        raise RuntimeError('18499 端口无响应，请检查端口占用')
    binaries = list((ROOT / 'runtime_new').rglob('llama-server.exe'))
    model = MODEL_DIR / 'MiniCPM-OCR-1_0-Q8_0.gguf'
    projector = MODEL_DIR / 'mmproj-model-f16-4x.gguf'
    if not binaries or not model.exists() or not projector.exists():
        raise RuntimeError('缺少 llama-server 或模型文件，请查看下载日志')
    log = (ROOT / 'model-server.log').open('ab')
    args = [str(binaries[0]), '-m', str(model), '--mmproj', str(projector), '--host', '127.0.0.1', '--port', '18499', '-c', '8192', '-ngl', '999', '--parallel', '1', '--alias', 'minicpm-ocr']
    process = subprocess.Popen(args, cwd=str(binaries[0].parent), stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    log.close()
    return '已启动模型服务，正在加载；GPU 实际使用请查看模型日志'


def work_folder(p: Path, meta: dict[str, Any]) -> Path:
    """新版和旧版使用不同目录，保留历史结果和源文件。"""
    if meta.get('mode') != 'layout_v2':return p
    if not meta.get('work_dir'):return p/'layout_v2'
    target=Path(meta['work_dir'])
    if not target.is_absolute() or target.name != 'layout_v2':raise RuntimeError('中间目录配置无效')
    parent=target.parent
    if parent.name != 'minicpm-ocr-'+meta['id'] or parent.is_symlink() or target.is_symlink():raise RuntimeError('中间目录所有权检查失败')
    marker=parent/'.owner.json'
    if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')).get('doc_id')!=meta['id']:raise RuntimeError('中间目录缺少所有权标记')
    return target


def make_final_archive(p: Path, m: dict[str, Any], output: Path) -> None:
    """检查每页与所有图片后，保存合并 ZIP；原始逐页结果保持独立。"""
    final = output / 'result_complete.zip'
    if final.exists():
        with zipfile.ZipFile(final) as z:
            if z.testzip() is not None or 'document.md' not in z.namelist():
                raise RuntimeError('已有完整结果包损坏，未覆盖，请人工检查')
        return
    parts=[]; assets=[]
    for i in range(1,m['pages']+1):
        check_pause()
        page=output/f'page_{i:05d}.md'; manifest=output/f'page_{i:05d}.assets.json'
        if not page.is_file() or not manifest.is_file(): raise RuntimeError(f'第 {i} 页或图片记录缺失，无法合并')
        parts.append(f'<!-- 第 {i} 页 -->\n\n'+page.read_text(encoding='utf-8'))
        for rel in json.loads(manifest.read_text(encoding='utf-8')):
            if not re.fullmatch(r'images/page_\d{5}_region_\d{3,5}\.png',rel) or not (output/rel).is_file():
                raise RuntimeError(f'第 {i} 页图片缺失：{rel}')
            assets.append(rel)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('document.md','\n\n'.join(parts))
        for rel in dict.fromkeys(assets):
            check_pause()
            z.write(output/rel,rel)
    data=buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if z.testzip() is not None: raise RuntimeError('完整结果包校验失败')
    atomic_bytes(final,data)


def run_doc(doc_id: str) -> None:
    """按区域安全保存；临时占用每 5 秒重试，暂停在区域边界生效。"""
    from layout_engine import parse_page
    from checkpoints import checkpoint
    p=folder(doc_id)
    event=pause_events.setdefault(doc_id,threading.Event())
    def notify(message: str) -> None:
        volatile[doc_id]={'status':'retrying','phase':message}
    install_context(event,notify)
    try:
        m=read_meta(p); output=work_folder(p,m); output.mkdir(parents=True,exist_ok=True)
        for n in range(1,m['pages']+1):
            check_pause()
            out=output/f'page_{n:05d}.md'
            if out.exists(): continue
            def progress(phase: str) -> None:
                check_pause()
                volatile[doc_id]={'status':'running','current_page':n,'phase':phase}
                state=read_meta(p);state.update(current_page=n,phase=phase);write_meta(p,state)
            while True:
                try:
                    progress('准备页面')
                    content,assets=parse_page(image_page(p,n),output,n,SERVER,MODEL_DIR/'PP-DocLayoutV3_infer',progress,bool(m.get('exclude_headers',True)))
                    atomic_text(output/f'page_{n:05d}.assets.json',json.dumps(assets,ensure_ascii=False))
                    atomic_text(out,content)
                    checkpoint(output,n,int(m.get('segment_pages',50)),int(m.get('ram_limit',90)),by_page=m.get('segment_by_page',True),by_memory=m.get('segment_by_memory',True))
                    logging.info('版面识别完成 %s 第 %s 页',doc_id,n)
                    break
                except (OSError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as exc:
                    if isinstance(exc, requests.exceptions.HTTPError):
                        if exc.response is None or exc.response.status_code not in (429, 502, 503, 504): raise
                    elif not isinstance(exc,(requests.exceptions.ConnectionError,requests.exceptions.Timeout)) and not retryable(exc): raise
                    if isinstance(exc,requests.exceptions.ConnectionError) and (not process or process.poll() is not None):
                        try: start_model()
                        except Exception as restart_error: logging.warning('模型服务重启等待：%s',restart_error)
                    wait_retry(exc)
        while True:
            try:
                check_pause()
                checkpoint(output,m['pages'],int(m.get('segment_pages',50)),int(m.get('ram_limit',90)),final=True,by_page=m.get('segment_by_page',True),by_memory=m.get('segment_by_memory',True))
                volatile[doc_id]={'status':'running','phase':'合并完整结果'}
                make_final_archive(p,m,output)
                m=read_meta(p);m.update(status='completed',error='',current_page=None,phase='完成');write_meta(p,m)
                break
            except OSError as exc:
                if not retryable(exc): raise
                wait_retry(exc)
    except PauseRequested:
        # 状态文件落盘本身可能受锁定，暂停标志仍保留供重启恢复。
        event.clear()
        install_context(threading.Event(),lambda message: volatile.__setitem__(doc_id,{'status':'pausing','phase':message}))
        m=read_meta(p);m.update(status='paused',error='',phase='已暂停');write_meta(p,m)
        logging.info('用户暂停任务 %s',doc_id)
    except Exception as exc:
        logging.exception('版面识别失败 %s',doc_id)
        m=read_meta(p);m.update(status='failed',error=str(exc),phase='失败');write_meta(p,m)
    finally:
        volatile.pop(doc_id,None)
        with lock: active.discard(doc_id)


def start_layout_job(p: Path, exclude_headers: bool, segment_pages: int, ram_limit: int, segment_by_page: bool | None = None, segment_by_memory: bool | None = None, work_root: str | None = None) -> None:
    """首次执行切换到独立目录；续跑必须使用同一过滤选项。"""
    m = read_meta(p)
    if m.get('layout_started') and m.get('exclude_headers', True) != exclude_headers:
        raise HTTPException(409, '续跑时不能改变页眉页码选项；请重新导入创建新任务')
    # 旧任务未保存开关时沿用旧行为；新任务默认不生成临时分段。
    if work_root and work_root.strip():
        base=Path(work_root.strip()).expanduser()
        if not base.is_absolute() or not base.is_dir():raise HTTPException(400,'中间文件目录必须是已存在的绝对路径')
        base=base.resolve()
        previous=m.get('work_root')
        if m.get('layout_started') and (not previous or Path(previous).resolve()!=base):
            raise HTTPException(409,'已有任务不能更改中间文件目录；请重新导入创建任务')
        if not m.get('layout_started'):
            owner=base/('minicpm-ocr-'+m['id'])
            if owner.is_symlink():raise HTTPException(400,'中间目录不能是符号链接')
            if owner.exists():
                marker=owner/'.owner.json'
                if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')).get('doc_id')!=m['id']:raise HTTPException(409,'目标任务目录已存在且不属于该任务')
            else:
                owner.mkdir()
                atomic_text(owner/'.owner.json',json.dumps({'doc_id':m['id']}))
            (owner/'layout_v2').mkdir(exist_ok=True)
            m['work_root']=str(base);m['work_dir']=str(owner/'layout_v2')
    previous_default = bool(m.get('layout_started'))
    if segment_by_page is None: segment_by_page = m.get('segment_by_page', previous_default)
    if segment_by_memory is None: segment_by_memory = m.get('segment_by_memory', previous_default)
    m.update(mode='layout_v2', layout_started=True, exclude_headers=exclude_headers, segment_pages=segment_pages, ram_limit=ram_limit, segment_by_page=segment_by_page, segment_by_memory=segment_by_memory, status='running', error='', current_page=None, phase='准备中')
    write_meta(p,m)


@app.get('/', response_class=HTMLResponse)
def home() -> str:
    return (ROOT / 'index.html').read_text(encoding='utf-8')


@app.post('/api/model/start')
def model_start() -> dict[str, str]:
    try: return {'message': start_model()}
    except Exception as exc: raise HTTPException(400, str(exc))


@app.post('/api/model/stop')
def model_stop() -> dict[str, str]:
    """仅停止本应用启动的模型服务，运行任务时避免中途终止。"""
    with lock:
        if active or batch_running: raise HTTPException(409, '文件仍在处理中，暂不能停止模型')
        if process and process.poll() is None:
            process.terminate()
            return {'message': '已停止模型，正在释放显存'}
    return {'message': '模型未运行'}


@app.get('/api/model')
def model_status() -> dict[str, Any]:
    try:
        r = requests.get(SERVER + '/health', timeout=2)
        owned = bool(process and process.poll() is None)
        detail = ('模型已就绪' if r.ok else '模型正在加载，请稍候') if owned else '18499 端口由其他服务占用，本应用尚未启动模型'
        return {'ready': r.ok and owned, 'detail': detail}
    except requests.RequestException:
        detail = '模型进程已退出，请查看 model-server.log' if process and process.poll() is not None else '模型服务未就绪，请启动并等待加载'
        return {'ready':False, 'detail':detail}


@app.get('/api/environment')
def environment() -> dict[str, Any]:
    """仅向本机浏览器返回环境信息，不上传本地路径或统计数据。"""
    from environment_info import environment_summary
    result=environment_summary(ROOT)
    result['model_service']=model_status()
    return result


@app.post('/api/upload')
def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    results=[]
    for f in files:
        name = Path(f.filename or '').name; suffix = Path(name).suffix.lower()
        if suffix not in FORMATS:
            results.append({'name':name, 'error':'不支持该格式'}); continue
        doc_id=uuid.uuid4().hex; p=DATA/doc_id; p.mkdir(); target=p/('source'+suffix)
        try:
            with target.open('wb') as stream:
                while chunk := f.file.read(1024*1024): stream.write(chunk)
            count=page_count(target)
            if count < 1: raise ValueError('文件没有可读取的页面')
            write_meta(p, {'id':doc_id,'name':name,'stored':target.name,'pages':count,'status':'pending','error':'','mode':'layout_v2','exclude_headers':True})
            results.append({'id':doc_id,'name':name,'pages':count})
        except Exception as exc:
            logging.exception('上传失败: %s', name)
            results.append({'name':name,'error':str(exc)})
            (p/'error.txt').write_text(str(exc),encoding='utf-8')
        finally: f.file.close()
    return {'results':results}


@app.get('/api/docs')
def docs() -> list[dict[str, Any]]:
    result=[]
    for p in DATA.iterdir():
        if not (p/'meta.json').exists(): continue
        try:
            m=read_meta(p); output=work_folder(p,m); m['done']=sum((output / f'page_{i:05d}.md').is_file() for i in range(1,m['pages']+1))
            if m.get('cleanup_status') and (output/'result_complete.zip').is_file(): m['done']=m['pages']
            if m['id'] in volatile: m.update(volatile[m['id']])
            if (p/'pause.request').exists() and m['id'] not in active: m['status']='paused'
            m['segments'] = json.loads((output/'segments.json').read_text(encoding='utf-8')) if not m.get('cleanup_status') and (output/'segments.json').is_file() else []
            m['can_export'] = m['status'] == 'completed' and m['done'] == m['pages'] and m['id'] not in active
            if m['status'] in ('running','retrying') and m['id'] not in active: m['status']='interrupted'
            result.append(m)
        except Exception: logging.exception('读取记录失败 %s',p)
    return sorted(result, key=lambda x:x['name'].lower())


@app.post('/api/docs/{doc_id}/run')
def run(doc_id: str, exclude_headers: bool = True, segment_pages: int = 50, ram_limit: int = 90, segment_by_page: bool | None = None, segment_by_memory: bool | None = None, work_root: str | None = None) -> dict[str,str]:
    if not 1 <= segment_pages <= 1000 or not 50 <= ram_limit <= 100: raise HTTPException(400, '分段页数或内存阈值超出范围')
    p=folder(doc_id)
    if read_meta(p).get('cleanup_status'):raise HTTPException(409,'该任务的中间文件已清理；如需重新识别，请重新导入')
    if not model_status()['ready']: raise HTTPException(400,'模型服务未就绪')
    with lock:
        if active or batch_running: raise HTTPException(409,'已有文件正在处理，请等待完成后继续')
        pause_events.setdefault(doc_id,threading.Event()).clear()
        (p/'pause.request').unlink(missing_ok=True)
        active.add(doc_id)
    try:
        start_layout_job(p, exclude_headers, segment_pages, ram_limit, segment_by_page, segment_by_memory, work_root)
        threading.Thread(target=run_doc,args=(doc_id,),daemon=True).start()
    except Exception:
        with lock: active.discard(doc_id)
        raise
    return {'message':'已开始'}


@app.post('/api/docs/{doc_id}/pause')
def pause_doc(doc_id: str) -> dict[str,str]:
    """标记暂停请求；当前区域完成或重试等待时安全退出。"""
    p=folder(doc_id)
    with lock:
        if doc_id not in active: raise HTTPException(409,'当前文件没有正在运行的任务')
        event=pause_events.setdefault(doc_id,threading.Event())
        event.set()
        (p/'pause.request').touch()
        volatile[doc_id]={'status':'pausing','phase':'正在等待当前区域安全保存'}
    return {'message':'暂停请求已发送，当前区域处理后暂停'}


def batch_worker(ids: list[str], exclude_headers: bool, segment_pages: int, ram_limit: int, segment_by_page: bool | None, segment_by_memory: bool | None, work_root: str | None) -> None:
    """同一批次串行处理，单个文件失败不阻止其余文件。"""
    global batch_running
    try:
        for doc_id in ids:
            with lock: active.add(doc_id)
            p = folder(doc_id)
            try:
                if (p/'pause.request').exists():
                    with lock: active.discard(doc_id)
                    continue
                pause_events.setdefault(doc_id,threading.Event()).clear()
                start_layout_job(p, exclude_headers, segment_pages, ram_limit, segment_by_page, segment_by_memory, None if read_meta(p).get('layout_started') else work_root)
                run_doc(doc_id)
            except Exception as exc:
                logging.exception('批处理启动失败 %s', doc_id)
                m=read_meta(p);m.update(status='failed',error=str(exc),phase='失败');write_meta(p,m)
                with lock: active.discard(doc_id)
    finally:
        with lock: batch_running = False


@app.post('/api/batch')
def batch(exclude_headers: bool = True, segment_pages: int = 50, ram_limit: int = 90, segment_by_page: bool | None = None, segment_by_memory: bool | None = None, work_root: str | None = None) -> dict[str, str]:
    if not 1 <= segment_pages <= 1000 or not 50 <= ram_limit <= 100: raise HTTPException(400, '分段页数或内存阈值超出范围')
    global batch_running
    if not model_status()['ready']: raise HTTPException(400, '模型服务未就绪')
    with lock:
        if active or batch_running: raise HTTPException(409, '已有任务正在处理')
        ids = [d['id'] for d in docs() if d['done'] < d['pages'] and d['status'] != 'paused']
        if not ids: return {'message': '没有未完成文件'}
        batch_running = True
        threading.Thread(target=batch_worker, args=(ids, exclude_headers, segment_pages, ram_limit, segment_by_page, segment_by_memory, work_root), daemon=True).start()
    return {'message': f'已加入 {len(ids)} 个文件，将依次处理'}


def archive_data(output: Path, total: int) -> tuple[list[str], dict[str, bytes]]:
    """校验原始完整包及页序；清理后用它恢复预览和再次导出。"""
    with zipfile.ZipFile(output / 'result_complete.zip') as z:
        if z.testzip() is not None: raise RuntimeError('原始完整 ZIP 校验失败')
        text=z.read('document.md').decode('utf-8')
        markers=list(re.finditer(r'(?m)^<!-- 第 (\d+) 页 -->\r?\n\r?\n',text))
        if [int(x.group(1)) for x in markers] != list(range(1,total+1)):
            raise RuntimeError('原始完整 ZIP 页码不完整或存在歧义')
        pages=[]
        for j,match in enumerate(markers):
            value=text[match.end():markers[j+1].start() if j+1<len(markers) else len(text)]
            if j+1<len(markers) and value.endswith('\n\n'): value=value[:-2]
            pages.append(value)
        assets={name:z.read(name) for name in z.namelist() if re.fullmatch(r'images/page_\d{5}_region_\d{3,5}\.png',name)}
        for page in pages:
            for rel in re.findall(r'!\[[^\]]*\]\((images/[^)]+)\)',page):
                if rel not in assets: raise RuntimeError('完整 ZIP 缺少图片：'+rel)
        return pages,assets


@app.get('/api/docs/{doc_id}/page/{page}')
def page_data(doc_id: str, page: int, formatted: bool = True) -> dict[str, Any]:
    p=folder(doc_id); m=read_meta(p)
    if not 1 <= page <= m['pages']: raise HTTPException(404,'页码超出范围')
    output=work_folder(p,m); text=output/f'page_{page:05d}.md'
    assets_file=output/f'page_{page:05d}.assets.json'
    assets=json.loads(assets_file.read_text(encoding='utf-8')) if text.exists() and assets_file.exists() else []
    content=text.read_text(encoding='utf-8') if text.exists() else ''
    recovered=False
    if m.get('cleanup_status'):
        pages,archive_assets=archive_data(output,m['pages']);content=pages[page-1];recovered=True
        assets=[x for x in archive_assets if x.startswith(f'images/page_{page:05d}_')]
    if content and formatted:
        from document_format import normalize_page
        content,_=normalize_page(content)
    return {'text':content, 'done':text.exists() or recovered, 'assets':assets}


@app.get('/api/docs/{doc_id}/image/{page}')
def page_image(doc_id: str, page: int) -> Response:
    return Response(image_page(folder(doc_id),page),media_type='image/png')


@app.get('/api/docs/{doc_id}/asset/{name}')
def asset(doc_id: str, name: str) -> FileResponse:
    p=folder(doc_id); m=read_meta(p)
    if not re.fullmatch(r'page_\d{5}_region_\d{3,5}\.png',name): raise HTTPException(404,'图片不存在')
    img=work_folder(p,m)/'images'/name
    if not img.is_file():
        if m.get('cleanup_status'):
            _,assets=archive_data(work_folder(p,m),m['pages']); data=assets.get('images/'+name)
            if data is not None:return Response(data,media_type='image/png')
        raise HTTPException(404,'图片不存在')
    return FileResponse(img, media_type='image/png')


def formatted_zip(output: Path, start: int, end: int, include_page_markers: bool, merge_cross_page: bool, output_format: str = "zip", archived: tuple[list[str], dict[str, bytes]] | None = None) -> Response:
    """从原始逐页结果派生 ZIP；确认所有页面和图片存在，不覆盖原始文件。"""
    from document_format import assemble_document
    pages=[];assets=[]
    for i in range(start,end+1):
        page=output/f'page_{i:05d}.md'; manifest=output/f'page_{i:05d}.assets.json'
        if archived is not None:
            pages.append(archived[0][i-1]);assets.extend(x for x in archived[1] if x.startswith(f'images/page_{i:05d}_'));continue
        if not page.is_file() or not manifest.is_file(): raise HTTPException(409,f'第 {i} 页或图片记录缺失，无法导出')
        pages.append(page.read_text(encoding='utf-8'))
        for rel in json.loads(manifest.read_text(encoding='utf-8')):
            if not re.fullmatch(r'images/page_\d{5}_region_\d{3,5}\.png',rel) or not (output/rel).is_file():
                raise HTTPException(409,f'第 {i} 页图片缺失或路径无效')
            assets.append(rel)
    document,report=assemble_document(pages,include_page_markers,merge_cross_page,start_page=start)
    report.update({'page_start':start,'page_end':end,'warning':'跨页合并是启发式推断，须对照原文复核；页码为 PDF 页面序号。'})
    if output_format in ('md','txt','pdf'):
        from text_export import text_only_markdown, plain_text, text_pdf
        markdown=text_only_markdown(document)
        if output_format=='md':data=markdown.encode('utf-8');media='text/markdown; charset=utf-8'
        elif output_format=='txt':data=plain_text(markdown).encode('utf-8');media='text/plain; charset=utf-8'
        else:data=text_pdf(plain_text(markdown));media='application/pdf'
        return Response(data,media_type=media,headers={'Content-Disposition':f'attachment; filename="pages_{start}-{end}.{output_format}"'})
    if output_format!='zip':raise HTTPException(400,'不支持的导出格式')
    archive=tempfile.TemporaryFile()
    try:
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('document.md',document)
            z.writestr('processing_report.json',json.dumps(report,ensure_ascii=False,indent=2))
            for rel in dict.fromkeys(assets):
                if archived is None:z.write(output/rel,rel)
                else:z.writestr(rel,archived[1][rel])
        archive.seek(0)
        def chunks():
            try:
                while data := archive.read(1024*1024):yield data
            finally:archive.close()
        return StreamingResponse(chunks(),media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="pages_{start}-{end}.zip"'})
    except Exception:
        archive.close();raise


@app.get('/api/docs/{doc_id}/segments/{start}/{end}')
def segment_download(doc_id: str, start: int, end: int, include_page_markers: bool = True, merge_cross_page: bool = False) -> Response:
    """导出已提交的临时分段，格式化不改变原始结果。"""
    from checkpoints import read_segments
    p=folder(doc_id);m=read_meta(p);output=work_folder(p,m)
    if not any(x['start']==start and x['end']==end for x in read_segments(output)):raise HTTPException(404,'临时分段不存在')
    return formatted_zip(output,start,end,include_page_markers,merge_cross_page)


@app.get('/api/docs/{doc_id}/export')
def export(doc_id: str, include_page_markers: bool = True, merge_cross_page: bool = False, formatted: bool = True, output_format: str = 'zip') -> Response:
    """完整文件才可导出，默认应用可复核的格式规范化。"""
    p=folder(doc_id);m=read_meta(p);output=work_folder(p,m)
    if m['status']!='completed' or doc_id in active:raise HTTPException(409,'文档尚未完整处理，不能下载')
    files=[output/f'page_{i:05d}.md' for i in range(1,m['pages']+1)]
    archived=None
    if m.get('cleanup_status'):
        try:archived=archive_data(output,m['pages'])
        except Exception as exc:raise HTTPException(409,f'无法从原始完整 ZIP 恢复：{exc}')
    elif any(not f.is_file() for f in files):raise HTTPException(409,'文档缺少已完成页面，不能下载')
    if m.get('mode') != 'layout_v2':
        if output_format not in ('zip','md'):raise HTTPException(400,'旧版任务仅支持 Markdown；请重新导入后使用新版处理')
        raw_pages=[f.read_text(encoding='utf-8') for f in files]
        if formatted:
            from document_format import assemble_document
            text,_=assemble_document(raw_pages,include_page_markers,merge_cross_page)
        else:
            text='\n\n'.join(f'<!-- 第 {i} 页 -->\n\n'+content for i,content in enumerate(raw_pages,1))
        return Response(text,media_type='text/markdown; charset=utf-8',headers={'Content-Disposition':f'attachment; filename="{doc_id}.md"'})
    original=output/'result_complete.zip'
    if not formatted:
        if not original.is_file():raise HTTPException(404,'未找到原始完整结果包')
        return FileResponse(original,media_type='application/zip',filename=f'{doc_id}_original.zip')
    return formatted_zip(output,1,m['pages'],include_page_markers,merge_cross_page,output_format,archived)


class SaveOptions(BaseModel):
    """本地另存参数：输出目录由用户提供，禁止覆盖已有文件。"""
    directory: str
    output_format: str = 'zip'
    include_page_markers: bool = True
    merge_cross_page: bool = False
    cleanup: bool = False


@app.post('/api/docs/{doc_id}/save')
async def save_local(doc_id: str, options: SaveOptions) -> dict[str, Any]:
    """先保存并校验完整输出，再按用户选择清除已知中间文件。"""
    directory=Path(options.directory.strip()).expanduser()
    if not directory.is_absolute() or not directory.is_dir():raise HTTPException(400,'另存目录必须是已存在的绝对路径')
    if options.output_format not in ('zip','md','txt','pdf'):raise HTTPException(400,'不支持的导出格式')
    if options.cleanup and options.output_format!='zip':raise HTTPException(400,'清理中间文件必须先另存完整 ZIP')
    p=folder(doc_id);m=read_meta(p);output=work_folder(p,m)
    if m.get('mode')!='layout_v2':raise HTTPException(400,'旧版任务不支持本机另存；请重新导入后使用新版处理')
    with lock:
        if active or batch_running:raise HTTPException(409,'文件处理中，无法另存或清理')
        if m['status']!='completed':raise HTTPException(409,'文档尚未完整处理')
        active.add(doc_id)
    try:
        # export 使用 active 作为运行保护，另存期间持锁占位后直接调用构建函数。
        archived=None
        if m.get('cleanup_status'):archived=archive_data(output,m['pages'])
        response=formatted_zip(output,1,m['pages'],options.include_page_markers,options.merge_cross_page,options.output_format,archived)
        target=directory / (Path(m['name']).stem + '_OCR_' + uuid.uuid4().hex[:8] + '.' + options.output_format)
        try:
            with target.open('xb') as f:
                if isinstance(response,StreamingResponse):
                    async for chunk in response.body_iterator:f.write(chunk)
                else:f.write(response.body)
                f.flush();os.fsync(f.fileno())
            if options.output_format=='zip':
                with zipfile.ZipFile(target) as z:
                    if z.testzip() is not None or 'document.md' not in z.namelist() or 'processing_report.json' not in z.namelist():raise RuntimeError('另存 ZIP 校验失败')
                    report=json.loads(z.read('processing_report.json'))
                    if report.get('page_start')!=1 or report.get('page_end')!=m['pages']:raise RuntimeError('另存 ZIP 页数校验失败')
            elif target.stat().st_size==0:raise RuntimeError('导出文件为空')
        except Exception:
            # 不删除不完整文件，避免误删；返回路径便于检查。
            logging.exception('另存失败，可能存在不完整文件 %s',target)
            raise HTTPException(500,f'另存或校验失败，请检查文件：{target}')
        cleanup_message='已保留中间文件'
        if options.cleanup:
            # 清理前独立检查原始完整包，确保后续预览及重新导出仍有依据。
            try:archive_data(output,m['pages'])
            except Exception as exc:raise HTTPException(409,f'已另存到 {target}，但原始 ZIP 校验失败，未清理：{exc}')
            m['cleanup_status']='in_progress';write_meta(p,m)
            removed=0;errors=[]
            patterns=[r'page_\d{5}\.md',r'page_\d{5}\.assets\.json',r'page_\d{5}\.regions\.json',r'page_\d{5}\.regions\.tmp',r'segments\.json']
            candidates=[x for x in output.iterdir() if x.is_file() and not x.is_symlink() and any(re.fullmatch(pattern,x.name) for pattern in patterns)]
            for sub,pattern in [('images',r'page_\d{5}_region_\d{3,5}\.png'),('segments',r'pages_\d{5}_\d{5}\.md')]:
                d=output/sub
                if d.is_dir() and not d.is_symlink():candidates.extend(x for x in d.iterdir() if x.is_file() and not x.is_symlink() and re.fullmatch(pattern,x.name))
            for item in candidates:
                try:item.unlink();removed+=1
                except OSError as exc:errors.append(f'{item}: {exc}')
            m['cleanup_status']='partial' if errors else 'complete';write_meta(p,m)
            cleanup_message=f'已清理 {removed} 个中间文件' + (f'；{len(errors)} 个未能清理，请查看日志' if errors else '')
            for error in errors:logging.warning('清理失败 %s',error)
        return {'path':str(target),'message':cleanup_message}
    finally:
        with lock:active.discard(doc_id)


def watchdog() -> None:
    """每 5 秒恢复意外结束的运行任务；用户暂停和永久错误不会自动启动。"""
    while True:
        try:
            if not active and not batch_running:
                for p in DATA.iterdir():
                    if not (p/'meta.json').is_file() or (p/'pause.request').exists(): continue
                    m=read_meta(p)
                    if m.get('status') not in ('running','retrying'): continue
                    if not model_status()['ready']:
                        if not process or process.poll() is not None:
                            try: start_model()
                            except Exception as exc: logging.warning('恢复等待模型：%s',exc)
                        break
                    doc_id=m['id']
                    with lock:
                        if active or batch_running or (p/'pause.request').exists(): break
                        active.add(doc_id)
                        pause_events.setdefault(doc_id,threading.Event()).clear()
                        threading.Thread(target=run_doc,args=(doc_id,),daemon=True).start()
                        logging.warning('自动恢复中断任务 %s',doc_id)
                    break
        except Exception:
            logging.exception('检查中断任务失败，5 秒后重试')
        threading.Event().wait(5)


@app.on_event('startup')
def start_watchdog() -> None:
    threading.Thread(target=watchdog,daemon=True,name='ocr-watchdog').start()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=7865)
