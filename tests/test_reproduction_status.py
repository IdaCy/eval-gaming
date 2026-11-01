import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reproduce


class ReproductionStatusTests(unittest.TestCase):
    def test_setup_uses_declared_checks_and_preserves_the_byte_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for passed in (True, False):
                report = {"checks": {"example_request_lines_match": passed}, "example_request_bytes_match": False}
                (root / "setup.json").write_text(json.dumps(report))
                status = reproduce.validation_status("setup", root)
                self.assertEqual(status["reference"], "PASS" if passed else "FAIL")
                self.assertIs(status["diagnostics"]["example_request_bytes_match"], False)

    def test_missing_reference_report_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                reproduce.validation_status('action_refits', Path(temporary))

    def test_paper_difference_is_printed_and_reference_failure_exits_nonzero(self):
        for reference_passed in (True, False):
            with self.subTest(reference_passed=reference_passed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'tables/action_refits').mkdir(parents=True)
                report = {'reference_passed': reference_passed, 'paper_passed': False,
                          'paper_action_tables': {'differences': [{'key': ['A10', 'DeepSeek', 'T2'], 'actual': '1.5', 'expected': '2'}]}}
                (root / 'tables/action_refits/validation.json').write_text(json.dumps(report))

                def completed_part(self, area):
                    return reproduce.validation_status(area, root / 'tables' / area)

                output = io.StringIO()
                with patch('sys.argv', ['reproduce', '--parts', 'action_refits', '--data', str(root), '--output', str(root)]), \
                        patch.object(reproduce.Reproduction, 'part', completed_part), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    if reference_passed:
                        reproduce.main()
                    else:
                        with self.assertRaises(SystemExit) as error:
                            reproduce.main()
                        self.assertEqual(error.exception.code, 1)
                self.assertIn('paper validation=FAIL', output.getvalue())
                self.assertIn('1.5', output.getvalue())
                self.assertIn('reference validation=' + ('PASS' if reference_passed else 'FAIL'), output.getvalue())


if __name__ == '__main__':
    unittest.main()
