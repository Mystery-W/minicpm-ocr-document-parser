"""格式修复不改原始结果；跨页只合并可信的普通段落边界。"""
import unittest
from document_format import normalize_page,assemble_document,can_merge

class FormatTests(unittest.TestCase):
    def test_toc_single_newlines_become_hard_breaks(self):
        s='## Contents\n\n'+'\n'.join(f'Lecture {n} topic {n+10}' for n in range(1,7))
        out,info=normalize_page(s);self.assertEqual(info['toc_blocks'],1);self.assertEqual(out.count('  \n'),5);self.assertNotIn('  \n',s)
    def test_toc_hard_breaks_unchanged(self):
        s='  \n'.join(f'Lecture {n} topic {n+10}' for n in range(1,7));self.assertEqual(normalize_page(s)[0],s)
    def test_superscript_and_code(self):
        s=r'Footnote \( ^{1} \) and `\( ^{2} \)` and \(x^{3}\)';out,info=normalize_page(s)
        self.assertIn('<sup>1</sup>',out);self.assertIn(r'`\( ^{2} \)`',out);self.assertIn(r'\(x^{3}\)',out);self.assertEqual(info['superscripts'],1)
    def test_cross_page_and_markers(self):
        a='A previous paragraph.\n\nThis paragraph continues across the page by'
        b='continuing with lowercase text in the next page.\n\nA new paragraph remains separate.'
        self.assertTrue(can_merge(a,b))
        for show in (False,True):
            merged,info=assemble_document([a,b],show,True)
            self.assertEqual(info['merged_boundaries'],[{'from_page':1,'to_page':2}])
            self.assertIn('by '+('【PDF 第 2 页】 ' if show else '')+'continuing',merged)
            self.assertEqual('【PDF 第 2 页】' in merged,show)
            self.assertIn('\n\nA new paragraph remains separate.',merged)
        self.assertFalse(can_merge('This is a completed sentence.','another paragraph begins here with enough words.'))
        self.assertFalse(can_merge('This list has an unfinished item','- another bullet starting on a new page.'))
    def test_segment_starts_on_original_pdf_page(self):
        out,report=assemble_document(['Text'],True,False,start_page=20)
        self.assertIn('【PDF 第 20 页】',out);self.assertEqual(report['pages'],1)

if __name__=='__main__':unittest.main()
