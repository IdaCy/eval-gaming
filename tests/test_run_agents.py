import csv
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from harness.validate_runs import RUN_AGENT_MODELS, validate


class RunAgentTests(unittest.TestCase):
    def test_unrecorded_rows_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / 'setups').mkdir()
            (data / 'runs/example').mkdir(parents=True)
            with (data / 'setups/agents.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['agent'])
                writer.writeheader()
                writer.writerows({'agent': model} for model in sorted(set(RUN_AGENT_MODELS.values())))
            rows = [{'run_id': 'run_1', 'run_set': 'example', 'agent': 'deepseek-flash', 'status': 'natural_completion'},
                    {'run_id': 'run_2', 'run_set': 'example', 'agent': 'task__B00L00__deepseek-flash__d0', 'status': 'not_recorded'}]
            path = data / 'runs/example/part-00000.parquet'
            (data / 'runs/index.json').write_text(json.dumps({'run_sets': [{'run_set': 'example', 'rows': 2,
                'shards': [{'path': 'runs/example/part-00000.parquet'}]}]}))
            pq.write_table(pa.Table.from_pylist(rows), path)
            report = validate(data)
            self.assertFalse(report['passed'])
            self.assertEqual(report['errors'][0]['run_id'], 'run_2')
            self.assertEqual(report['run_rows'], 2)
            rows[1]['agent'] = 'deepseek-flash'
            pq.write_table(pa.Table.from_pylist(rows), path)
            report = validate(data)
            self.assertTrue(report['passed'])
            self.assertEqual(report['run_rows'], 2)
            self.assertEqual(report['statuses']['not_recorded'], 1)


if __name__ == '__main__':
    unittest.main()
