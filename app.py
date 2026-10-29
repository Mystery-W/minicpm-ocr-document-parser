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
    return p / 'layout_v2' if meta.get('mode') == 'layout_v2' else p


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


def start_layout_job(p: Path, exclude_headers: bool, segment_pages: int, ram_limit: int, segment_by_page: bool | None = None, segment_by_memory: bool | None = None) -> None:
    """首次执行切换到独立目录；续跑必须使用同一过滤选项。"""
    m = read_meta(p)
    if m.get('layout_started') and m.get('exclude_headers', True) != exclude_headers:
        raise HTTPException(409, '续跑时不能改变页眉页码选项；请重新导入创建新任务')
    # 旧任务未保存开关时沿用旧行为；新任务默认不生成临时分段。
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
            if m['id'] in volatile: m.update(volatile[m['id']])
            if (p/'pause.request').exists() and m['id'] not in active: m['status']='paused'
            m['segments'] = json.loads((output/'segments.json').read_text(encoding='utf-8')) if (output/'segments.json').is_file() else []
            m['can_export'] = m['status'] == 'completed' and m['done'] == m['pages'] and m['id'] not in active
            if m['status'] in ('running','retrying') and m['id'] not in active: m['status']='interrupted'
            result.append(m)
        except Exception: logging.exception('读取记录失败 %s',p)
    return sorted(result, key=lambda x:x['name'].lower())


@app.post('/api/docs/{doc_id}/run')
def run(doc_id: str, exclude_headers: bool = True, segment_pages: int = 50, ram_limit: int = 90, segment_by_page: bool | None = None, segment_by_memory: bool | None = None) -> dict[str,str]:
    if not 1 <= segment_pages <= 1000 or not 50 <= ram_limit <= 100: raise HTTPException(400, '分段页数或内存阈值超出范围')
    p=folder(doc_id)
    if not model_status()['ready']: raise HTTPException(400,'模型服务未就绪')
    with lock:
        if active or batch_running: raise HTTPException(409,'已有文件正在处理，请等待完成后继续')
        pause_events.setdefault(doc_id,threading.Event()).clear()
        (p/'pause.request').unlink(missing_ok=True)
        active.add(doc_id)
    try:
        start_layout_job(p, exclude_headers, segment_pages, ram_limit, segment_by_page, segment_by_memory)
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


def batch_worker(ids: list[str], exclude_headers: bool, segment_pages: int, ram_limit: int, segment_by_page: bool | None, segment_by_memory: bool | None) -> None:
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
                start_layout_job(p, exclude_headers, segment_pages, ram_limit, segment_by_page, segment_by_memory)
                run_doc(doc_id)
            except Exception as exc:
                logging.exception('批处理启动失败 %s', doc_id)
                m=read_meta(p);m.update(status='failed',error=str(exc),phase='失败');write_meta(p,m)
                with lock: active.discard(doc_id)
    finally:
        with lock: batch_running = False


@app.post('/api/batch')
def batch(exclude_headers: bool = True, segment_pages: int = 50, ram_limit: int = 90, segment_by_page: bool | None = None, segment_by_memory: bool | None = None) -> dict[str, str]:
    if not 1 <= segment_pages <= 1000 or not 50 <= ram_limit <= 100: raise HTTPException(400, '分段页数或内存阈值超出范围')
    global batch_running
    if not model_status()['ready']: raise HTTPException(400, '模型服务未就绪')
    with lock:
        if active or batch_running: raise HTTPException(409, '已有任务正在处理')
        ids = [d['id'] for d in docs() if d['done'] < d['pages'] and d['status'] != 'paused']
        if not ids: return {'message': '没有未完成文件'}
        batch_running = True
        threading.Thread(target=batch_worker, args=(ids, exclude_headers, segment_pages, ram_limit, segment_by_page, segment_by_memory), daemon=True).start()
    return {'message': f'已加入 {len(ids)} 个文件，将依次处理'}


@app.get('/api/docs/{doc_id}/page/{page}')
def page_data(doc_id: str, page: int) -> dict[str, Any]:
    p=folder(doc_id); m=read_meta(p)
    if not 1 <= page <= m['pages']: raise HTTPException(404,'页码超出范围')
    output=work_folder(p,m); text=output/f'page_{page:05d}.md'
    assets_file=output/f'page_{page:05d}.assets.json'
    assets=json.loads(assets_file.read_text(encoding='utf-8')) if text.exists() and assets_file.exists() else []
    return {'text':text.read_text(encoding='utf-8') if text.exists() else '', 'done':text.exists(), 'assets':assets}


@app.get('/api/docs/{doc_id}/image/{page}')
def page_image(doc_id: str, page: int) -> Response:
    return Response(image_page(folder(doc_id),page),media_type='image/png')


@app.get('/api/docs/{doc_id}/asset/{name}')
def asset(doc_id: str, name: str) -> FileResponse:
    p=folder(doc_id); m=read_meta(p)
    if not re.fullmatch(r'page_\d{5}_region_\d{3,5}\.png',name): raise HTTPException(404,'图片不存在')
    img=work_folder(p,m)/'images'/name
    if not img.is_file(): raise HTTPException(404,'图片不存在')
    return FileResponse(img, media_type='image/png')


@app.get('/api/docs/{doc_id}/segments/{start}/{end}')
def segment_download(doc_id: str, start: int, end: int) -> Response:
    """下载明确标识页码范围的已提交临时分段，包含关联图片。"""
    from checkpoints import read_segments
    p=folder(doc_id); m=read_meta(p); output=work_folder(p,m)
    record=next((x for x in read_segments(output) if x['start']==start and x['end']==end), None)
    if record is None: raise HTTPException(404,'临时分段不存在')
    md=output/'segments'/record['file']
    if not md.is_file(): raise HTTPException(409,'临时分段文件缺失')
    archive=tempfile.TemporaryFile()
    try:
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            z.write(md,'document.md')
            for i in range(start,end+1):
                manifest=output/f'page_{i:05d}.assets.json'
                if not manifest.is_file(): raise HTTPException(409,'分段图片记录缺失')
                for rel in json.loads(manifest.read_text(encoding='utf-8')):
                    if not re.fullmatch(r'images/page_\d{5}_region_\d{3,5}\.png',rel) or not (output/rel).is_file(): raise HTTPException(409,'分段图片缺失')
                    z.write(output/rel,rel)
        archive.seek(0)
        def chunks():
            try:
                while data := archive.read(1024*1024): yield data
            finally: archive.close()
        return StreamingResponse(chunks(),media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="temporary_pages_{start}_{end}.zip"'})
    except Exception:
        archive.close(); raise


@app.get('/api/docs/{doc_id}/export')
def export(doc_id: str) -> Response:
    """只允许所有页面成功后导出，图片与 Markdown 打包为 ZIP。"""
    p=folder(doc_id); m=read_meta(p); output=work_folder(p,m)
    if m['status'] != 'completed' or doc_id in active:
        raise HTTPException(409, '文档尚未完整处理，不能下载')
    files=[output/f'page_{i:05d}.md' for i in range(1,m['pages']+1)]
    if any(not f.is_file() for f in files):
        raise HTTPException(409, '文档缺少已完成页面，不能下载')
    final=output/'result_complete.zip'
    if m.get('mode') == 'layout_v2' and final.is_file():
        filename=quote(Path(m['name']).stem + '_解析.zip')
        return FileResponse(final,media_type='application/zip',headers={'Content-Disposition':f"attachment; filename=ocr.zip; filename*=UTF-8''{filename}"})
    pieces=[f'<!-- 第 {i} 页 -->\n\n'+f.read_text(encoding='utf-8') for i,f in enumerate(files,1)]
    result='\n\n'.join(pieces)
    if m.get('mode') != 'layout_v2':
        return Response(result,media_type='text/markdown; charset=utf-8',headers={'Content-Disposition':f'attachment; filename="{doc_id}.md"'})
    archive=tempfile.TemporaryFile()
    try:
      with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('document.md',result)
        for i in range(1,m['pages']+1):
            manifest=output/f'page_{i:05d}.assets.json'
            if not manifest.is_file():
                raise HTTPException(409,f'第 {i} 页缺少图片记录，不能下载')
            for rel in json.loads(manifest.read_text(encoding='utf-8')):
                if not re.fullmatch(r'images/page_\d{5}_region_\d{3,5}\.png',rel):
                    raise HTTPException(409,'图片记录无效，不能下载')
                image=output/rel
                if not image.is_file():
                    raise HTTPException(409,f'图片缺失：{rel}')
                z.write(image,rel)
    except Exception:
        archive.close(); raise
    archive.seek(0)
    def chunks():
        try:
            while data := archive.read(1024*1024): yield data
        finally: archive.close()
    filename=quote(Path(m['name']).stem + '_解析.zip')
    return StreamingResponse(chunks(), media_type='application/zip', headers={'Content-Disposition':f"attachment; filename=ocr.zip; filename*=UTF-8''{filename}"})


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
