"""下载校验和解包安全检查使用本地模拟，不连接互联网。"""
import hashlib,io,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import setup_assets

class Response:
    def __init__(self,data,start=0):self.data=data;self.status_code=206 if start else 200;self.headers={'Content-Range':f'bytes {start}-{len(data)-1}/{len(data)}'};self.start=start
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_content(self,size):yield self.data[self.start:]

class Tests(unittest.TestCase):
    def test_resume_and_checksum(self):
        data=b'123456789'; sha=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.bin';(Path(d)/'test.bin.part').write_bytes(data[:4])
            def get(url,headers,**kw):return Response(data,int(headers['Range'][6:-1]) if headers else 0)
            with patch('setup_assets.requests.get',side_effect=get):setup_assets.download('https://example.invalid',p,len(data),sha)
            self.assertEqual(p.read_bytes(),data)
    def test_zip_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'bad.zip'
            with zipfile.ZipFile(a,'w') as z:z.writestr('../escape.txt','x')
            with self.assertRaises(ValueError):setup_assets.safe_zip(a,root/'safe')
            self.assertFalse((root/'escape.txt').exists())

if __name__=='__main__':unittest.main()
