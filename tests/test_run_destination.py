import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from harness.actor import run_actor
from harness.runtime import Runtime, claim_output
from harness.run_task import main


def inventory(root):
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob('*') if path.is_file()}


class RunDestinationTests(unittest.TestCase):
    def test_existing_run_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            (output / 'exchanges/actor_0000').mkdir(parents=True)
            (output / 'summary.json').write_text('{"end_status":"natural_completion"}\n')
            (output / 'actor_final_context.json').write_text('[{"role":"assistant","content":"done"}]\n')
            (output / 'exchanges/actor_0000/response.json').write_bytes(b'original response\n')
            before = inventory(output)
            with patch('harness.runtime.Transport.exchange', side_effect=AssertionError('unexpected API call')):
                with self.assertRaises(FileExistsError):
                    Runtime(output, SimpleNamespace(), 'same-run')
            self.assertEqual(before, inventory(output))

    def test_actor_retry_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Runtime(Path(temporary) / 'run', SimpleNamespace(), 'example')
            runtime.start = Mock()
            runtime.close = Mock(return_value={'state': 'closed'})
            runtime.checkpoint = Mock(return_value={'complete': True})
            runtime.exchange = Mock(return_value={'choices': [{'finish_reason': 'stop',
                                                              'message': {'role': 'assistant', 'content': 'done'}}]})
            protocol = SimpleNamespace(name='offline', request=lambda *args: {},
                                       check_response=lambda response, messages: response, replay=lambda message: message)
            arguments = ({'task_id': 'example', 'workspace': '/workspace', 'actor_env': []},
                         {'id': 'B00L00', 'request': 'example'},
                         {'system': '{workspace}', 'tools': []},
                         {'run_id': 'example', 'seed': 1, 'stage': 'full', 'continue_to_end': True}, runtime)
            result = run_actor(*arguments, protocol=protocol)
            self.assertEqual(result['end_status'], 'natural_completion')
            before = inventory(runtime.output)
            with self.assertRaises(FileExistsError):
                run_actor(*arguments, protocol=protocol)
            self.assertEqual(before, inventory(runtime.output))
            self.assertEqual(runtime.exchange.call_count, 1)

    def test_cli_retry_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'rendered'
            argv = ['run_task', '--task', 'example', '--family', 'native12', '--output', str(output)]
            with patch('sys.argv', argv), patch('harness.run_task.load_task', return_value=({'workspace': '/workspace'}, {})), \
                    patch('harness.run_task.contexts', return_value=([{'role': 'user', 'content': 'example'}], {})):
                main()
                before = inventory(output)
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(before, inventory(output))
            self.assertEqual(json.loads((output / 'rendered.json').read_text())['messages'][0]['content'], 'example')

    def test_atomic_claim_has_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'new'
            barrier = threading.Barrier(2)

            def claim():
                barrier.wait()
                try:
                    claim_output(output)
                    return True
                except FileExistsError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sorted(pool.map(lambda _: claim(), range(2))), [False, True])

    def test_empty_directory_and_symlink_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'empty'
            target.mkdir()
            link = Path(temporary) / 'alias'
            link.symlink_to(target, target_is_directory=True)
            for output in (target, link):
                with self.assertRaises(FileExistsError):
                    claim_output(output)
            self.assertEqual(list(target.iterdir()), [])

    @unittest.skipUnless(os.environ.get('EVAL_GAMING_TEST_DATA'), 'Set EVAL_GAMING_TEST_DATA to the downloaded dataset')
    def test_setup_reproduction_uses_the_runtime_claim(self):
        from harness.setup_summary import reproduce
        result = reproduce(Path(os.environ['EVAL_GAMING_TEST_DATA']))
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['api_calls'], 0)
        self.assertEqual(result['mock_execution']['turn_limit']['calls'], 128)


if __name__ == '__main__':
    unittest.main()
