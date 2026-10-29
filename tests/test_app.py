"""验证分段开关、完整性门禁、恢复标志和本机环境接口。"""
import io,json,tempfile,threading,unittest,uuid,zipfile
from pathlib import Path
from unittest.mock import patch
import app,checkpoints,layout_engine
from fastapi.testclient import TestClient

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.old=app.DATA;app.DATA=self.root
    def tearDown(self):
        app.DATA=self.old;self.temp.cleanup()
    def test_optional_segmentation(self):
        class Memory:percent=95
        with patch('checkpoints.psutil.virtual_memory',return_value=Memory()):
            for by_page,by_memory in ((False,False),(True,False),(False,True),(True,True)):
                p=self.root/str(uuid.uuid4());p.mkdir()
                for n in range(1,6):(p/f'page_{n:05d}.md').write_text(str(n),encoding='utf-8')
                done=checkpoints.checkpoint(p,5,5,90,by_page=by_page,by_memory=by_memory)
                self.assertEqual(done,by_page or by_memory)
                if done:self.assertEqual(checkpoints.read_segments(p)[0]['trigger'],'页数' if by_page else '系统内存')
                else:self.assertFalse(checkpoints.checkpoint(p,5,5,90,final=True,by_page=False,by_memory=False))
    def test_complete_and_partial(self):
        doc=uuid.uuid4().hex;p=self.root/doc;p.mkdir();out=p/'layout_v2';out.mkdir()
        m={'id':doc,'name':'sample.pdf','stored':'source.pdf','pages':2,'status':'running','mode':'layout_v2','exclude_headers':True,'layout_started':True,'segment_by_page':False,'segment_by_memory':False,'segment_pages':50,'ram_limit':90}
        app.write_meta(p,m);client=TestClient(app.app)
        self.assertEqual(client.get(f'/api/docs/{doc}/export').status_code,409)
        app.pause_events[doc]=threading.Event();app.active.add(doc)
        def fake(data,output,n,server,model_dir,progress,exclude):progress('test');return f'page {n}',[]
        with patch.object(app,'image_page',return_value=b'x'),patch.object(layout_engine,'parse_page',fake):app.run_doc(doc)
        self.assertEqual(app.read_meta(p)['status'],'completed');self.assertFalse((out/'segments.json').exists())
        with zipfile.ZipFile(out/'result_complete.zip') as z:self.assertEqual(z.read('document.md').decode().count('<!-- 第 '),2)
        self.assertEqual(client.get(f'/api/docs/{doc}/export').status_code,200)
        (out/'page_00002.md').unlink();self.assertEqual(client.get(f'/api/docs/{doc}/export').status_code,409)
    def test_export_options_and_raw_preservation(self):
        doc=uuid.uuid4().hex;p=self.root/doc;p.mkdir();out=p/'layout_v2';out.mkdir()
        app.write_meta(p,{'id':doc,'name':'test.pdf','stored':'source.pdf','pages':2,'status':'completed','mode':'layout_v2'})
        pages=['A paragraph that continues across the page by', 'continuing with lowercase text on the following page.']
        for n,text in enumerate(pages,1):
            (out/f'page_{n:05d}.md').write_text(text,encoding='utf-8');(out/f'page_{n:05d}.assets.json').write_text('[]',encoding='utf-8')
        client=TestClient(app.app)
        for marker in (True,False):
            for merge in (True,False):
                response=client.get(f'/api/docs/{doc}/export',params={'include_page_markers':marker,'merge_cross_page':merge})
                self.assertEqual(response.status_code,200)
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    text=z.read('document.md').decode();report=json.loads(z.read('processing_report.json'))
                    self.assertEqual('【PDF 第 2 页】' in text,marker)
                    self.assertEqual(bool(report['merged_boundaries']),merge)
        self.assertEqual([(out/f'page_{n:05d}.md').read_text(encoding='utf-8') for n in (1,2)],pages)

    def test_environment_api(self):
        response=TestClient(app.app).get('/api/environment');self.assertEqual(response.status_code,200);self.assertIn('python',response.json());self.assertIn('gpu',response.json())

if __name__=='__main__':unittest.main()
