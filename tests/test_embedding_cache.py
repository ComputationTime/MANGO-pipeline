import sys,tempfile,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflow/scripts'))
import embedding_cache as c

class CacheTests(unittest.TestCase):
 def test_target_change_reuses_but_antigen_or_tensor_change_invalidates(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'test.pt';p.write_bytes(b'original')
   row={'id':'x','split':'test','resolved_ag_seq':'ACD','resolved_L_seq':'VL'}
   index={'test/x':{'antigen_identity':c.antigen_identity(row),'tensor_sha256':c.digest(p)}}
   self.assertTrue(c.verified_record(index,dict(row,resolved_L_seq='DIQ'),p,[]))
   self.assertFalse(c.verified_record(index,dict(row,resolved_ag_seq='AAA'),p,[]))
   p.write_bytes(b'changed')
   self.assertFalse(c.verified_record(index,row,p,[]))
 def test_code_and_config_change_invalidate_index(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'impl.py';p.write_text('original')
   (Path(tmp)/'audited_cache.json').write_text(json.dumps({'signature':'v1','implementation':{str(p):c.digest(p)},'records':{'x':1}}))
   self.assertEqual(c.read_index(tmp,'v1'),{'x':1})
   self.assertEqual(c.read_index(tmp,'v2'),{})
   p.write_text('new')
   self.assertEqual(c.read_index(tmp,'v1'),{})

 def test_newer_explicit_weights_invalidate_audited_tensor(self):
  import os
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'tensor.pt';p.write_bytes(b'original')
   weight=Path(tmp)/'weights.pt';weight.write_bytes(b'weights')
   row={'id':'x','split':'test'}
   index={'test/x':{'antigen_identity':c.antigen_identity(row),'tensor_sha256':c.digest(p)}}
   timestamp=p.stat().st_mtime_ns
   os.utime(weight,ns=(timestamp-1,timestamp-1))
   self.assertTrue(c.verified_record(index,row,p,['records.csv',weight]))
   os.utime(weight,ns=(timestamp+1,timestamp+1))
   self.assertFalse(c.verified_record(index,row,p,['records.csv',weight]))
