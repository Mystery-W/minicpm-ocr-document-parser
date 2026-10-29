"""版面识别：使用官方 PP-DocLayoutV3 检测器和 MiniCPM 分类提示词。

识别区域逐个运行，避免在小显存 GPU 上并发；仅当所有区域成功才提交页面。
图片区域保留裁剪结果，供预览和随完整文档一起导出。
"""
from __future__ import annotations
import base64
import io
import hashlib
import json
from pathlib import Path
from typing import Callable
import requests
from PIL import Image
from safe_io import atomic_bytes, atomic_text, check_pause

_detector = None


def detector(model_dir: Path):
    """延迟加载 CPU 版面模型，并在同一应用进程中复用。"""
    global _detector
    if _detector is None:
        from minicpmvocr.layouts.ppdoclayout3_paddlex import PPDocLayoutV3PaddleXDetector
        config = {'model_dir': str(model_dir), 'device': 'cpu', 'threshold': 0.3, 'nms': True,
                  'use_chart_recognition': True, 'use_seal_recognition': True}
        instance = PPDocLayoutV3PaddleXDetector(config)
        instance.start()
        _detector = instance
    return _detector


def recognize(img: Image.Image, label: str, server: str, depth: int = 0) -> str:
    """对区域使用官方标签对应的提示词，拒绝空输出和截断输出。"""
    from minicpmvocr.prompts import DEFAULT_PROMPT_MAP, get_element_cfg
    from minicpmvocr.postprocess.otsl.converter import convert_otsl_to_html
    spec = get_element_cfg(DEFAULT_PROMPT_MAP, label)
    if spec is None:
        raise RuntimeError(f'不支持的区域类型: {label}')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    payload = {'model': 'minicpm-ocr', 'messages': [{'role':'user', 'content': [
        {'type':'image_url', 'image_url': {'url':'data:image/png;base64,' + encoded}},
        {'type':'text', 'text':spec['prompt']}]}], 'temperature':0,
        'max_tokens':min(int(spec['max_new_tokens']), 4096)}
    response = requests.post(server + '/v1/chat/completions', json=payload, timeout=900)
    response.raise_for_status()
    choice = response.json()['choices'][0]
    if choice.get('finish_reason') in ('length', 'max_tokens'):
        # 文本过长时只在中部的空白行分割；表格/公式不自动拆，以免破坏结构。
        splittable = {'text','content','abstract','reference','reference_content','vertical_text','vision_footnote','algorithm','footnote','aside_text'}
        if label in splittable and depth < 4 and img.height >= 160:
            import numpy as np
            pixels = np.asarray(img.convert('L'))
            white = (pixels >= 220).mean(axis=1)
            lo, hi = int(img.height * .30), int(img.height * .70)
            candidates = [i for i in range(lo,hi) if white[i] >= .995]
            if candidates:
                cut = min(candidates,key=lambda i:abs(i-img.height/2))
                upper = recognize(img.crop((0,0,img.width,cut)), label, server, depth+1)
                lower = recognize(img.crop((0,cut,img.width,img.height)), label, server, depth+1)
                return upper + '\n\n' + lower
        raise RuntimeError(f'{label} 区域达到输出上限，且无法安全分割；已保存前面区域，请人工检查')
    content = choice['message'].get('content')
    if not isinstance(content,str) or not content.strip():
        raise RuntimeError(f'{label} 区域识别结果为空，未保存此页')
    content = content.strip()
    if label in ('table','chart') and any(x in content for x in ('<fcel>','<ecel>','<nl>')):
        content = convert_otsl_to_html(content)
    return content


def parse_page(page_png: bytes, output: Path, number: int, server: str, model_dir: Path,
               progress: Callable[[str], None], exclude_headers: bool = True) -> tuple[str, list[str]]:
    """检测、按阅读顺序识别单页，返回 Markdown 和相对图片路径。"""
    image = Image.open(io.BytesIO(page_png)).convert('RGB')
    progress('加载版面模型')
    worker = detector(model_dir)
    progress('检测版面')
    detections = worker.detect([image])[0]
    blocks = worker.crop(image, detections)
    if not blocks:
        # 只对几乎纯白的页面认定为空白；扫描噪声较多时必须人工复核。
        sample = image.convert('L'); sample.thumbnail((300,300))
        hist = sample.histogram(); dark = sum(hist[:235]) / max(1, sum(hist))
        if dark < 0.001: return '<!-- 空白页：版面检测未发现内容 -->', []
        raise RuntimeError('未检测到版面区域；非空白页需人工检查')
    pieces: list[str] = []
    assets: list[str] = []
    checkpoint = output / f'page_{number:05d}.regions.json'
    completed = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}
    image_dir = output / 'images'
    for index, block in enumerate(blocks, 1):
        check_pause()
        label = block.get('label', 'text')
        if exclude_headers and label in ('header','footer','number','header_image','footer_image'):
            continue
        progress(f'识别区域 {index}/{len(blocks)}（{label}）')
        array = block.get('image')
        if array is None:
            continue  # SDK 标识为已合并到前一区域
        key = str(index) + ':' + hashlib.sha256(array.tobytes()).hexdigest()
        if key in completed and completed[key].get('label') == label:
            record = completed[key]
            if all((output / item).is_file() for item in record.get('assets', [])):
                pieces.extend(record['pieces']); assets.extend(record.get('assets', [])); continue
        local_pieces: list[str] = []
        local_assets: list[str] = []
        crop = Image.fromarray(array).convert('RGB')
        if crop.width == 0 or crop.height == 0:
            raise RuntimeError(f'区域 {index} 裁剪为空')
        # 图片、图表和印章均保留可独立查看的裁剪原图。
        if label in ('image','header_image','footer_image','chart','seal'):
            image_dir.mkdir(parents=True,exist_ok=True)
            name = f'page_{number:05d}_region_{index:03d}.png'
            path = image_dir / name
            buf = io.BytesIO()
            crop.save(buf, format='PNG')
            atomic_bytes(path, buf.getvalue())
            rel = 'images/' + name
            local_assets.append(rel)
            local_pieces.append(f'![{label}]({rel})')
        if label not in ('image','header_image','footer_image'):
            text = recognize(crop, label, server)
            if label == 'doc_title': text = '# ' + text
            elif label == 'paragraph_title': text = '## ' + text
            elif label == 'display_formula' and not text.startswith('$$'): text = '$$\n' + text + '\n$$'
            local_pieces.append(text)
        completed[key] = {'label': label, 'pieces': local_pieces, 'assets': local_assets}
        atomic_text(checkpoint, json.dumps(completed, ensure_ascii=False))
        pieces.extend(local_pieces); assets.extend(local_assets)
    if not pieces:
        return '<!-- 页面仅含已过滤的页眉或页码 -->', []
    return '\n\n'.join(pieces), assets
