"""非破坏性的 Markdown 清理和可选跨页合并。

只转换高置信度的目录行和独立数字上角标，原始 OCR 页文件始终不变。
跨页合并使用保守文本规则，不适合替代人工校对。
"""
from __future__ import annotations
import re
from typing import Sequence

PAGE_END = re.compile(r'(?:\d{1,4}|[ivxlcdm]{1,8})\s*$', re.IGNORECASE)
TOC_TERM = re.compile(r'\b(?:lecture|chapter|part|appendix|section|index|bibliography)\b|第[一二三四五六七八九十百零\d]+[章节篇部分]|目录|目次|\.{3,}|…{2,}',re.IGNORECASE)
SUP = re.compile(r'\\\(\s*\^\{\s*(\d{1,4})\s*\}\s*\\\)|(?<!\\)\(\s*\^\{\s*(\d{1,4})\s*\}\s*\)')
SPECIAL_START = re.compile(r'^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||!\[|<table\b|</?div\b|\$\$|\\\[|<sup>|\\\(\s*\^\{|\[\^)',re.IGNORECASE)


def _toc_block(block: str) -> bool:
    """至少六行，绝大多数行以页码结尾，且多行带目录特征。"""
    lines=[line.strip() for line in block.splitlines() if line.strip()]
    if len(lines)<6 or any(x.startswith(('```','~~~','|','<table')) for x in lines):return False
    ends=sum(bool(PAGE_END.search(x)) for x in lines)
    hints=sum(bool(TOC_TERM.search(x)) for x in lines)
    return ends/len(lines)>=0.75 and hints>=3


def normalize_page(text: str) -> tuple[str, dict[str, int]]:
    """保留段落和代码块，只修复高置信度目录换行及独立数字角标。"""
    counts={'toc_blocks':0,'superscripts':0}
    # 分离 fenced code，代码内容不应被后处理。
    chunks=re.split(r'(```[^\n]*\n[\s\S]*?```|~~~[^\n]*\n[\s\S]*?~~~)',text)
    for i in range(0,len(chunks),2):
        blocks=re.split(r'(\n[ \t]*\n)',chunks[i])
        for j in range(0,len(blocks),2):
            block=blocks[j]
            if _toc_block(block):
                lines=block.split('\n')
                for k in range(len(lines)-1):
                    if lines[k].strip() and lines[k+1].strip() and not lines[k].endswith('  '):
                        lines[k]=lines[k].rstrip()+'  '
                block='\n'.join(lines);counts['toc_blocks']+=1
            # 避开行内代码；只匹配完整的数字角标表达式。
            spans=re.split(r'(`+[^`\n]*`+)',block)
            for k in range(0,len(spans),2):
                spans[k],n=SUP.subn(lambda m:f'<sup>{m.group(1) or m.group(2)}</sup>',spans[k]);counts['superscripts']+=n
            blocks[j]=''.join(spans)
        chunks[i]=''.join(blocks)
    return ''.join(chunks),counts


def can_merge(previous: str, following: str) -> bool:
    """只合并明显未完句且下一页以小写字母或汉字续接的普通段落。"""
    if not previous.strip() or not following.strip():return False
    left=re.split(r'\n\s*\n',previous.strip())[-1].strip()
    right=re.split(r'\n\s*\n',following.strip())[0].strip()
    if len(left)<20 or len(right)<20 or _toc_block(left) or _toc_block(right):return False
    if SPECIAL_START.match(left) or SPECIAL_START.match(right):return False
    if any(x in left or x in right for x in ('```','~~~','<table','</table>','$$','![','|')):return False
    if re.search(r'[.!?。！？:：;；]([\"\'”’）\)]*)\s*$',left):return False
    if re.search(r'[-‐‑–—]\s*$',left):return False
    return bool(re.match(r'^[a-z\u4e00-\u9fff]',right))


def assemble_document(pages: Sequence[str], include_page_markers: bool = True, merge_cross_page: bool = False, start_page: int = 1) -> tuple[str,dict]:
    """按 PDF 页顺序组装；合并边界时标记放在段内，确保可追溯。"""
    parts=[];previous=''; merges=[];toc=[];sup=0
    for offset,raw in enumerate(pages):
        index=start_page+offset
        text,stats=normalize_page(raw);sup+=stats['superscripts']
        if stats['toc_blocks']:toc.append(index)
        marker=f'【PDF 第 {index} 页】'
        merge=bool(offset>0 and merge_cross_page and can_merge(previous,text))
        if offset==0:
            parts.append((marker+'\n\n' if include_page_markers else '')+text)
        elif merge:
            # 可见页标放在同一段内，既能定位也不把续接段拆成两段。
            parts[-1]=parts[-1].rstrip()+' '+((marker+' ') if include_page_markers else '')+text.lstrip()
            merges.append({'from_page':index-1,'to_page':index})
        else:
            parts.append(((marker+'\n\n') if include_page_markers else '')+text)
        previous=text
    return '\n\n'.join(parts),{'pages':len(pages),'toc_pages':toc,'superscripts_reformatted':sup,'merged_boundaries':merges,
                   'include_page_markers':include_page_markers,'merge_cross_page':merge_cross_page}
