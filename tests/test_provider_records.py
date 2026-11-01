from collections import Counter
import json
import unittest

from harness import project_provider_records as p


class ProviderRecordTests(unittest.TestCase):
    def assert_projected(self, value, retained='unchanged source'):
        transcript = json.dumps([{'role': 'assistant', 'content': 'Keep my evaluated reasoning.', 'reasoning': 'visible text'},
                                 {'role': 'tool', 'content': value}])
        counts = Counter()
        result = p.clean(transcript, counts)
        decoded = json.loads(result)
        self.assertEqual(decoded[0], json.loads(transcript)[0])
        self.assertEqual(decoded[1]['role'], 'tool')
        self.assertIn(p.MARKER, decoded[1]['content'])
        self.assertIn(retained, decoded[1]['content'])
        self.assertFalse(p.fixture_fragment(decoded[1]['content']))
        again = Counter()
        self.assertEqual(p.clean(result, again), result)
        self.assertFalse(again)
        self.assertTrue(counts)
        return decoded[1]['content']

    def test_complete_and_escaped_events(self):
        event = {'type': 'response.created', 'response': {'id': 'resp_' + 'a' * 40, 'object': 'response',
                 'output': [{'id': 'rs_' + 'b' * 40, 'type': 'reasoning', 'summary': [], 'encrypted_content': '[redacted]'}]}}
        for depth in range(4):
            value = json.dumps(event)
            for _ in range(depth):
                value = json.dumps(value)
            with self.subTest(depth=depth):
                self.assert_projected('fixture.json: "body": ' + value + '\nunchanged source')

    def test_escaped_truncated_events_and_reasoning(self):
        snippets = [
            'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_' + 'a' * 40,
            '{"id":"resp_' + 'a' * 40 + '","object":"response","output":[{"type":"reasoning","summary":',
            '{\n  "type": "reasoning",\n  "content": [],\n  "encrypted_content": "[redacted]',
            '{"type":"response.created',
            '{"type":"reasoning',
            '{"object":"response',
        ]
        for snippet in snippets:
            for depth in range(4):
                value = snippet
                for _ in range(depth):
                    value = json.dumps(value)[1:-1]
                with self.subTest(depth=depth, snippet=snippet[:20]):
                    self.assert_projected('unchanged source\nfixtures/example.json:9: "body": "' + value + '\n[Tool output truncated; use a narrower command to inspect further.]')

    def test_multiline_truncated_record(self):
        value = 'unchanged source\n{\n  "id": "resp_' + 'a' * 40 + '",\n  "object": "response",\n  "output": [\n    {"type": "reasoning", "summary": [\n'
        result = self.assert_projected(value)
        self.assertNotIn('"output"', result)
        self.assert_projected('unchanged source\n  "type":\n    "reasoning",\n  "summary": [\n')

    def test_symbolic_record_fragments(self):
        for value in ["fixture.php:73: 'encrypted_content' => 'gAAAAA-encrypted',",
                      "fixture.php:75: 'type' => 'reasoning',", '{"type":"reasoning"}',
                      '{"item_id":"rs_example"}', '{"response_id":"resp_example"}',
                      r'fixture.json: \u0022type\u0022:\u0022response.created\u0022']:
            with self.subTest(value=value):
                self.assert_projected(value + '\nunchanged source')

    def test_ordinary_source_and_model_text_stay_identical(self):
        call_id = 'chatcmpl-tool-' + 'a' * 30
        value = json.dumps([{'role': 'assistant', 'content': 'I should reason carefully.', 'reasoning': 'visible text',
                             'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command":"pwd"}'}}]},
                            {'role': 'tool', 'content': 'response.status_code == 200\nunchanged source\nreasoning about tests',
                             'tool_call_id': call_id, 'object': {'nested': []}}])
        counts = Counter()
        self.assertEqual(p.clean(value, counts), value)
        self.assertFalse(counts)

    def test_long_unrelated_escape_runs(self):
        value = 'response implementation\n' + '\\' * 100000 + 'x\nunchanged source'
        counts = Counter()
        self.assertEqual(p.clean(value, counts), value)
        self.assertFalse(counts)
        self.assert_projected(value + '\n{"type":"response.created","response":')


if __name__ == '__main__':
    unittest.main()
