import io, json, tempfile, unittest, uuid, zipfile
from pathlib import Path
from fastapi.testclient import TestClient
import app
import pymupdf

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.old=app.DATA;app.DATA=self.root/'data';app.DATA.mkdir()
        self.id=uuid.uuid4().hex;self.p=app.DATA/self.id;self.p.mkdir();self.out=self.p/'layout_v2';self.out.mkdir()
        self.meta={'id':self.id,'name':'中文 sample.pdf','stored':'source.pdf','pages':2,'status':'completed','mode':'layout_v2','layout_started':True}
        app.write_meta(self.p,self.meta);(self.p/'source.pdf').write_bytes(b'source')
        for n in (1,2):
            (self.out/f'page_{n:05d}.md').write_text(f'第 {n} 页 Chinese and English',encoding='utf-8')
            (self.out/f'page_{n:05d}.assets.json').write_text('[]',encoding='utf-8')
        app.make_final_archive(self.p,self.meta,self.out);self.client=TestClient(app.app)
    def tearDown(self):app.DATA=self.old;self.temp.cleanup()
    def test_formats_cleanup_and_recovery(self):
        for fmt in ('md','txt','pdf','zip'):
            r=self.client.get(f'/api/docs/{self.id}/export',params={'output_format':fmt});self.assertEqual(r.status_code,200)
            if fmt=='pdf':
                with pymupdf.open(stream=r.content,filetype='pdf') as pdf:self.assertIn('Chinese', ''.join(p.get_text() for p in pdf))
        bad=self.client.post(f'/api/docs/{self.id}/save',json={'directory':str(self.root),'output_format':'txt','cleanup':True});self.assertEqual(bad.status_code,400)
        r=self.client.post(f'/api/docs/{self.id}/save',json={'directory':str(self.root),'output_format':'zip','cleanup':True});self.assertEqual(r.status_code,200,r.text)
        self.assertTrue(Path(r.json()['path']).is_file());self.assertTrue((self.p/'source.pdf').is_file());self.assertTrue((self.out/'result_complete.zip').is_file());self.assertFalse((self.out/'page_00001.md').exists())
        self.assertEqual(self.client.get('/api/docs').json()[0]['done'],2)
        self.assertIn('Chinese',self.client.get(f'/api/docs/{self.id}/page/1').json()['text'])
        for fmt in ('md','txt','pdf','zip'):
            r=self.client.get(f'/api/docs/{self.id}/export',params={'output_format':fmt});self.assertEqual(r.status_code,200,r.text if fmt=='md' else '')
    def test_invalid_raw_archive_never_cleans(self):
        (self.out/'result_complete.zip').write_bytes(b'bad')
        r=self.client.post(f'/api/docs/{self.id}/save',json={'directory':str(self.root),'cleanup':True});self.assertEqual(r.status_code,409);self.assertTrue((self.out/'page_00001.md').exists())
    def test_external_directory(self):
        m=self.meta.copy();m['layout_started']=False;app.write_meta(self.p,m)
        app.start_layout_job(self.p,True,50,90,False,False,str(self.root))
        m=app.read_meta(self.p);self.assertTrue(app.work_folder(self.p,m).is_dir())
        with self.assertRaises(Exception):app.start_layout_job(self.p,True,50,90,False,False,str(self.root/'data'))

if __name__=='__main__':unittest.main()
