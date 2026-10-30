import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from harness.download.task_sources import fetch_task
from harness.run_task import load_task
from harness.util import sha


class LocalRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        for name in ('tasks', 'setups', 'prompts'):
            (self.data / name).mkdir(parents=True)
        self.task = 'bbc__psammead-1362'
        self.text = 'Offline request with a Unicode symbol: π.\n'
        self.variant = {'task_id': self.task, 'id': 'B00L00', 'request': self.text, 'eligible': True}
        self.spec = {'task_id': self.task, 'base_commit': '1' * 40, 'repository': 'bbc/psammead',
                     'workspace': '/workspace', 'request': None}
        self.source = {'task_id': self.task, 'base_commit': self.spec['base_commit'],
                       'repository': self.spec['repository'], 'source_revision': '2' * 40,
                       'task_text_sha256': hashlib.sha256(self.text.encode()).hexdigest(), 'fetch_kind': 'benchmark'}
        pd.DataFrame([dict(task_id=self.task, spec=json.dumps(self.spec))]).to_parquet(self.data / 'tasks/tasks.parquet')
        value = json.dumps(dict(self.variant, request=None, text_delivery='fetch'))
        pd.DataFrame([dict(task_id=self.task, family='native12', setup='B00L00', value=value,
                           release_value_sha256=hashlib.sha256(value.encode()).hexdigest(),
                           source_definition_sha256=sha(self.variant))]).to_parquet(self.data / 'setups/variants.parquet')
        pd.DataFrame([self.source]).to_parquet(self.data / 'tasks/text_sources.parquet')
        pd.DataFrame([dict(instance_id=self.task, base_commit=self.spec['base_commit'], problem_statement=self.text)]).to_parquet(self.data / 'tasks/swe_rebench_v2.parquet')
        pd.DataFrame([dict(name='system_prompt', text='Work in {workspace}.'),
                      dict(name='belief_questions', text='beta: Is this an evaluation?'),
                      dict(name='belief_paragraph', text='{q}')]).to_csv(self.data / 'prompts/prompts.csv', index=False)
        with patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network access')):
            fetch_task(pd.Series(self.source), self.data, self.root / 'fetched')
        self.request = self.root / 'fetched' / self.task / 'problem_statement.txt'

    def test_fetch_to_render(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'harness.run_task', '--data', str(self.data),
                                 '--task', self.task, '--family', 'native12', '--local-request', str(self.request),
                                 '--output', str(self.root / 'rendered')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['api_calls'], 0)
        rendered = json.loads((self.root / 'rendered/rendered.json').read_text())
        self.assertEqual(rendered['messages'][-1]['content'], self.text)

    def test_request_and_provenance_tampering_are_rejected(self):
        receipt_path = self.request.parent / 'retrieval.json'
        receipt = json.loads(receipt_path.read_text())
        self.request.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'file_hash'):
            load_task(self.data, self.task, 'native12', 'B00L00', local_request=self.request)
        for item in receipt['files']:
            if item['path'] == self.request.name:
                item.update(bytes=7, sha256=hashlib.sha256(b'changed').hexdigest())
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'definition_hash'):
            load_task(self.data, self.task, 'native12', 'B00L00', local_request=self.request)
        receipt['base_commit'] = '3' * 40
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            load_task(self.data, self.task, 'native12', 'B00L00', local_request=self.request)

    def test_full_local_variant_and_identity(self):
        path = self.root / 'variant.json'
        path.write_text(json.dumps(dict(self.variant, definition_sha256=sha(self.variant))))
        _, variant = load_task(self.data, self.task, 'native12', 'B00L00', local_variant_path=path)
        self.assertEqual(variant, self.variant)
        path.write_text(json.dumps(dict(self.variant, id='F01L01')))
        with self.assertRaisesRegex(ValueError, 'identity_mismatch'):
            load_task(self.data, self.task, 'native12', 'B00L00', local_variant_path=path)

    def test_mismatched_benchmark_text_is_rejected(self):
        source = pd.Series(dict(self.source, task_text_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'request hash'):
            fetch_task(source, self.data, self.root / 'wrong')

    def test_source_variant_can_bind_a_released_task_alias(self):
        alias = 'example__released-task-1'
        path = self.root / 'variant.json'
        path.write_text(json.dumps(self.variant))
        spec = dict(self.spec, task_id=alias)
        pd.DataFrame([dict(task_id=alias, spec=json.dumps(spec))]).to_parquet(self.data / 'tasks/tasks.parquet')
        sources = pd.DataFrame([dict(self.source, task_id=alias)])
        sources.to_parquet(self.data / 'tasks/text_sources.parquet')
        variants = pd.read_parquet(self.data / 'setups/variants.parquet')
        variants['task_id'] = alias
        value = json.dumps(dict(self.variant, task_id=alias, request=None, text_delivery='fetch'))
        variants['value'] = value
        variants['release_value_sha256'] = hashlib.sha256(value.encode()).hexdigest()
        variants.to_parquet(self.data / 'setups/variants.parquet')
        _, restored = load_task(self.data, alias, 'native12', 'B00L00', local_variant_path=path)
        self.assertEqual(restored['task_id'], alias)
        self.assertEqual(restored['request'], self.text)
        path.write_text(json.dumps(dict(self.variant, task_id='unrelated')))
        with self.assertRaisesRegex(ValueError, 'definition_hash'):
            load_task(self.data, alias, 'native12', 'B00L00', local_variant_path=path)

    @unittest.skipUnless(os.environ.get('EVAL_GAMING_TEST_DATA'), 'Set EVAL_GAMING_TEST_DATA to the downloaded dataset')
    def test_bundled_bbc_request_to_render(self):
        data = Path(os.environ['EVAL_GAMING_TEST_DATA'])
        sources = pd.read_parquet(data / 'tasks/text_sources.parquet')
        source = sources[sources.task_id.eq(self.task)].iloc[0]
        with patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network access')):
            result = fetch_task(source, data, self.root / 'bundled')
        self.assertTrue(result['upstream_text_matches_recorded_prompt'])
        path = self.root / 'bundled' / self.task / 'problem_statement.txt'
        result = subprocess.run([sys.executable, '-B', '-m', 'harness.run_task', '--data', str(data),
                                 '--task', self.task, '--family', 'native12', '--local-request', str(path),
                                 '--output', str(self.root / 'bundled_rendered')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['api_calls'], 0)
        rendered = json.loads((self.root / 'bundled_rendered/rendered.json').read_text())
        self.assertEqual(hashlib.sha256(rendered['messages'][-1]['content'].encode()).hexdigest(), source.task_text_sha256)


if __name__ == '__main__':
    unittest.main()
