"""纯文字导出：Markdown、TXT、重排 PDF；不修改 OCR 原始结果。"""
from __future__ import annotations
import html
import re
import textwrap
from html.parser import HTMLParser
import pymupdf


class TableText(HTMLParser):
    """把识别出的 HTML 表格转换成可读的纯文字行。"""
    def __init__(self) -> None:
        super().__init__(); self.parts: list[str] = []
    def handle_data(self, data: str) -> None:
        self.parts.append(data)
    def handle_starttag(self, tag: str, attrs: list[tuple[str,str | None]]) -> None:
        if tag in ('td','th') and self.parts:self.parts.append(' | ')
        elif tag=='tr' and self.parts:self.parts.append('\n')
    def text(self) -> str:
        return ''.join(self.parts).strip()


def text_only_markdown(text: str) -> str:
    """文字版保留标题、表格和公式，明确标记被省略的插图。"""
    return re.sub(r'!\[([^\]]*)\]\(images/[^)]+\)',lambda m:'[插图已省略：'+(m.group(1) or 'image')+']',text)


def plain_text(text: str) -> str:
    """去掉常见 Markdown 标记；数学公式保留原始 LaTeX。"""
    text=text_only_markdown(text)
    def table(match: re.Match[str]) -> str:
        parser=TableText();parser.feed(match.group());return parser.text()
    text=re.sub(r'<table\b[\s\S]*?</table>',table,text,flags=re.I)
    text=re.sub(r'<sup>(\d+)</sup>',r'[^\1]',text)
    text=re.sub(r'^\s{0,3}#{1,6}\s+', '',text,flags=re.M)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',text)
    text=re.sub(r'(?m)\s{2,}$','',text)
    return html.unescape(text)


def text_pdf(text: str) -> bytes:
    """生成可选择、可搜索的纯文字 PDF；图像和原文排版不复刻。

    内置 CJK 字体支持常见中文和拉丁字母；生僻字符和复杂数学符号需复核。
    """
    doc=pymupdf.open(); page=None; y=0
    try:
        for source_line in text.splitlines():
            for line in (textwrap.wrap(source_line,width=46,break_long_words=True,break_on_hyphens=False,replace_whitespace=False,drop_whitespace=False) if source_line else ['']):
                if page is None or y>800:
                    page=doc.new_page(width=595,height=842);y=42
                if line.strip():page.insert_text((42,y),line,fontname='china-s',fontsize=10)
                y+=13
        if not len(doc):doc.new_page()
        return doc.tobytes(garbage=3,deflate=True)
    finally:doc.close()
