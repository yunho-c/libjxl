#!/usr/bin/env python3
"""Checks for same-call attribution, paired scaling, and frozen provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cjxl_gjxl_paired_breakdown as paired


class PairedBreakdownTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'bin').mkdir();(self.root/'bin/capture').write_bytes(b'fixture')
        source=self.root/'input.pfm';source.write_bytes(b'input fixture')
        self.job=dict(case_id='00-e04',image_id='a',resolution_class='test',width=10,height=10,
            input_path=str(source),input_sha256=paired.digest(source),effort=4,setting='Q80',distance=1.9)
        self.config=dict(schema_version=1,capture_kind='paired-same-call-v1',label='Fixture',
            source_revision='fixture',samples=2,warmups=0,cpu_threads=8,collect_final_score=False,
            frozen_sha256={'bin/capture':paired.digest(self.root/'bin/capture')},
            binary_sha256=paired.digest(self.root/'bin/capture'),images=[self.job],jobs=[self.job],
            efforts=[4],settings=['Q80'])
        self.path=self.root/'config.json';self.write(self.path,self.config)
        self.case=self.root/'cases/00-e04';self.attempt=self.case/'attempt-000';self.attempt.mkdir(parents=True)
        p=dict(total=100,input_preparation=10,quantization_pipeline=40,codestream_encoding=45,
            input_geometry_and_storage=1,input_color_transform=1,input_matrix_scale_stats=1,
            input_resident_preparation=4,input_quantization_preparation=2,codestream_validation=1,
            codestream_dc_tokenization=4,codestream_ac_tokenization=10,codestream_entropy_optimization=10,
            codestream_section_writing=10,codestream_assembly=5)
        self.timings=dict(capture_kind='paired-same-call-v1',source_width=10,source_height=10,
            rows=[dict(sample_index=i,mode=mode,complete_call_nanoseconds=110 if mode=='profiled' else 90,
                       committed_submissions=3,byte_equal=True,summary_equal=True,phase_nanoseconds=p)
                  for i in range(2) for mode in ('ordinary','profiled')])
        stage=dict(stage_id='butteraugli.test',begin_timestamp=100,end_timestamp=130,
                   gpu_nanoseconds=30,timestamp_valid=True,dispatches=[])
        self.gpu=dict(schema_version=4,execution_path='production-aligned-resident-v1',mode='stage',
            gpu_aq='fully-resident',collect_final_score=False,distance=1.9,warmups=0,sample_count=2,
            workloads=[dict(source_width=10,source_height=10,samples=[dict(sample_index=i,
                capabilities=dict(timestamp_counter=True,stage_boundary=True),
                submissions=[dict(stages=[copy.deepcopy(stage)])]) for i in range(2)])])
        self.command=dict(returncode=0,interference=[],argv=['capture','--input',str(source),
            '--effort','4','--cpu-threads','8','--samples','2','--warmups','0','--distance','1.9'])
        self.sync()

    def tearDown(self):
        plt.close('all')

    def write(self,path,data):
        path.write_text(json.dumps(data))

    def sync(self):
        for name,data in [('gpu',self.gpu),('paired',self.timings),('command',self.command)]:
            self.write(self.attempt/(name+'.json'),data)
        self.write(self.case/'complete.json',dict(attempt='attempt-000',
            command_sha256=paired.digest(self.attempt/'command.json'),
            summary={n+'_sha256':paired.digest(self.attempt/(n+'.json')) for n in ('paired','gpu')}))

    def load(self):
        return paired.load_profiles(self.path)

    def test_additive_complete_call_and_explicit_residuals(self):
        report=self.load();flat=report['means'].query("kind == 'flat'").set_index('stage')
        self.assertAlmostEqual(flat.ms.sum()*1e6,110)
        self.assertAlmostEqual(flat.loc['GPU: Butteraugli comparison','ms']*1e6,30)
        self.assertAlmostEqual(flat.loc[paired.PIPELINE_REMAINDER,'ms']*1e6,10)
        self.assertAlmostEqual(flat.loc[paired.OUTER,'ms']*1e6,10)
        self.assertNotIn(paired.legacy.UNRESOLVED_PIPELINE,flat.index)

    def test_scaled_view_uses_each_paired_total(self):
        self.timings['rows'][2]['complete_call_nanoseconds']=180
        self.sync();report=self.load()
        s=report['samples'].query("kind == 'scaled'").groupby('sample_index').ms.sum()
        self.assertAlmostEqual(s[0]*1e6,90);self.assertAlmostEqual(s[1]*1e6,180)
        self.assertAlmostEqual(report['means'].query("kind == 'scaled'").ms.sum()*1e6,135)

    def test_counter_sum_cannot_exceed_pipeline(self):
        for s in self.gpu['workloads'][0]['samples']:
            s['submissions'][0]['stages'][0].update(end_timestamp=150,gpu_nanoseconds=50)
        self.sync();report=self.load()
        self.assertTrue(report['means'].empty);self.assertEqual(len(report['rejected']),1)

    def test_verified_empty_indirect_stage_is_zero_work(self):
        for s in self.gpu['workloads'][0]['samples']:
            s['submissions'][0]['stages'].append(dict(stage_id='frontend.ac_strategy.empty',
                timestamp_valid=False,begin_timestamp=0,end_timestamp=0,gpu_nanoseconds=0,
                dispatches=[dict(kind='indirect_threadgroups',grid=[0,1,1])]))
        self.sync();self.assertTrue((self.load()['coverage'].status=='complete').all())

    def test_unexplained_missing_timestamp_rejects_case(self):
        self.gpu['workloads'][0]['samples'][0]['submissions'][0]['stages'][0].update(
            timestamp_valid=False,begin_timestamp=0,end_timestamp=0,gpu_nanoseconds=0)
        self.sync();self.assertTrue(self.load()['means'].empty)

    def test_missing_pair_fails_closed(self):
        self.timings['rows'].pop();self.sync()
        with self.assertRaisesRegex(ValueError,'Incomplete paired'):self.load()

    def test_duplicate_gpu_sample_fails_closed(self):
        self.gpu['workloads'][0]['samples'][1]['sample_index']=0;self.sync()
        with self.assertRaisesRegex(ValueError,'duplicate GPU'):self.load()

    def test_hash_and_command_mismatches_fail_closed(self):
        (self.attempt/'paired.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.load()
        self.command['argv'][self.command['argv'].index('--effort')+1]='5';self.sync()
        with self.assertRaisesRegex(ValueError,'declared setting'):self.load()

    def test_missing_case_stays_incomplete(self):
        (self.case/'complete.json').unlink()
        report=self.load();self.assertTrue(report['means'].empty)
        self.assertTrue((report['coverage'].status=='incomplete').all())

    def test_render_never_starts_collection(self):
        with mock.patch('subprocess.Popen',side_effect=AssertionError('No collection')):
            report=paired.generate_breakdowns(self.path,self.root/'plots',formats=('png',))
        self.assertEqual(len(report['figures']['test'].axes),1)
        self.assertTrue((self.root/'plots/gjxl-stage-breakdown-test.png').is_file())

    def test_notebook_dispatches_to_same_call_helper(self):
        import cjxl_runtime_characterization_notebook as notebook
        with mock.patch('subprocess.Popen',side_effect=AssertionError('No collection')):
            report=notebook.generate_gjxl_stage_breakdowns(
                self.path,self.root/'notebook',formats=(),scaled=True)
        self.assertAlmostEqual(report['means'].query("kind == 'scaled'").ms.sum()*1e6,90)

    def test_notebook_missing_manifest_and_legacy_scaling(self):
        import cjxl_runtime_characterization_notebook as notebook
        with mock.patch('subprocess.Popen',side_effect=AssertionError('No collection')):
            self.assertIsNone(notebook.generate_gjxl_stage_breakdowns(self.root/'missing',self.root/'out'))
        self.write(self.path,{'schema_version':2})
        with self.assertRaisesRegex(ValueError,'requires paired'):
            notebook.generate_gjxl_stage_breakdowns(self.path,self.root/'out',scaled=True)


if __name__ == '__main__':
    unittest.main()
