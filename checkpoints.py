"""按连续页生成可恢复的临时分段；原始每页文件仍是最终结果的依据。"""
from __future__ import annotations
import json
from pathlib import Path
import psutil
from safe_io import atomic_text


def read_segments(output: Path) -> list[dict]:
    """读取已提交的分段索引；不存在时表示尚无分段。"""
    path = output / 'segments.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []


def checkpoint(output: Path, end: int, interval: int, ram_percent: int, final: bool = False, *, by_page: bool = True, by_memory: bool = True) -> bool:
    """页间检查保存条件，连续高内存时最少间隔 5 页避免产生大量小文件。"""
    if not by_page and not by_memory: return False
    segments = read_segments(output)
    previous = int(segments[-1]['end']) if segments else 0
    length = end - previous
    if length <= 0: return False
    high_memory = by_memory and psutil.virtual_memory().percent >= ram_percent
    page_due = by_page and length >= interval
    if not (final or page_due or (high_memory and length >= 5)): return False
    pages = [output / f'page_{i:05d}.md' for i in range(previous + 1, end + 1)]
    if any(not f.is_file() for f in pages): raise RuntimeError('临时分段存在缺页，未保存')
    dir = output / 'segments'; dir.mkdir(exist_ok=True)
    name = f'pages_{previous+1:05d}_{end:05d}.md'
    text = '\n\n'.join(f'<!-- 第 {i} 页 -->\n\n'+p.read_text(encoding='utf-8') for i,p in enumerate(pages,previous+1))
    atomic_text(dir/name,text)
    record={'start':previous+1,'end':end,'file':name,'trigger':'完成' if final else ('页数' if page_due else '系统内存')}
    segments.append(record)
    atomic_text(output/'segments.json',json.dumps(segments,ensure_ascii=False,indent=2))
    return True
