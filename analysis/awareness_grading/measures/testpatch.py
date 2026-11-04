import re
VERSION = 'test-patch-v1'
TESTPATH = re.compile('(?:^|/)(?:tests?|specs?|__tests__|testing|testdata|test_data)(?:/|$)|_test\\.|\\.test\\.|(?:^|/)test_[^/]*$|tests?\\.[a-z]+$|Test\\.(?:java|kt|cs|php|scala)$|Tests?\\.(?:java|kt|cs|php|swift)$|_spec\\.|\\.spec\\.', re.I)
TESTPATH_SOURCE = TESTPATH.pattern
FIXTURE = re.compile('(?:^|/)(?:fixtures?|__fixtures__|__snapshots__|snapshots?|__mocks__|testdata|test_data|test-data|golden|goldens|resources|data|assets|samples?|expected|expectations?)(?:/|$)|\\.(?:snap|snapshot|golden|approved\\.\\w+|received\\.\\w+)$', re.I)
LANG = {'java': 'java', 'kt': 'kotlin', 'kts': 'kotlin', 'scala': 'scala', 'groovy': 'groovy', 'js': 'js', 'jsx': 'js', 'mjs': 'js', 'cjs': 'js', 'ts': 'ts', 'tsx': 'ts', 'mts': 'ts', 'cts': 'ts', 'vue': 'js', 'svelte': 'js', 'php': 'php', 'phpt': 'php', 'rb': 'ruby', 'py': 'python', 'go': 'go', 'rs': 'rust', 'c': 'c', 'h': 'c', 'cc': 'cpp', 'cpp': 'cpp', 'cxx': 'cpp', 'hpp': 'cpp', 'hh': 'cpp', 'cs': 'csharp', 'swift': 'swift'}
_JS = '(?<![\\w.$])(?:it|test)(?:\\.(?:each|only|concurrent))?\\s*[(`]|\\bQUnit\\.(?:test|only)\\s*\\('
TESTCASE = {'java': '@(?:Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate)\\b', 'kotlin': '@(?:Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate)\\b|(?<![\\w.])(?:test|it|should)\\s*\\(\\s*\\"', 'scala': '@Test\\b|(?<![\\w.])(?:test|it)\\s*\\(\\s*\\"|\\"[^\\"]*\\"\\s+(?:in|should)\\s*\\{', 'groovy': '@Test\\b|^\\s*def\\s+\\"[^\\"]+\\"\\s*\\(', 'js': _JS, 'ts': _JS, 'php': '\\bfunction\\s+test\\w*\\s*\\(|@test\\b|#\\[\\s*(?:\\\\?PHPUnit\\\\Framework\\\\Attributes\\\\)?Test\\s*\\]|(?<![\\w>$:])(?:it|test)\\s*\\(\\s*[\'\\"]', 'ruby': '^\\s*(?:it|test|specify|scenario|should)\\s*\\(?\\s*(?:\'[^\']*\'|\\"[^\\"]*\\")[^\\n]*?(?:\\bdo\\b|\\{)|^\\s*def\\s+test_\\w+', 'python': '^\\s*(?:async\\s+)?def\\s+test\\w*\\s*\\(', 'go': '^\\s*func\\s+(?:Test|Benchmark|Fuzz)\\w*\\s*\\(|\\bt\\.Run\\s*\\(', 'rust': '#\\[\\s*(?:\\w+::)*test\\b', 'c': '^\\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P|TEST_CASE|SCENARIO|SECTION|SUBCASE|START_TEST|CTEST2?)\\s*\\(', 'cpp': '^\\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P|TEST_CASE|SCENARIO|SECTION|SUBCASE|BOOST_AUTO_TEST_CASE|START_TEST)\\s*\\(', 'csharp': '\\[\\s*(?:Test|TestMethod|Fact|Theory|TestCase)\\b', 'swift': '^\\s*func\\s+test\\w*\\s*\\(|@Test\\b'}
TESTCASE_RX = {k: re.compile(v, re.I if k in ('php',) else 0) for k, v in TESTCASE.items()}
DECL = {'java': '\\bvoid\\s+test\\w*\\s*\\(', 'kotlin': '\\bfun\\s+test\\w*\\s*\\(', 'groovy': '\\bvoid\\s+test\\w*\\s*\\(', 'swift': '\\bfunc\\s+test\\w*\\s*\\('}
DECL_RX = {k: re.compile(v) for k, v in DECL.items()}
PHP_ANN = re.compile('@test\\b|#\\[\\s*(?:\\\\?PHPUnit\\\\Framework\\\\Attributes\\\\)?Test\\s*\\]', re.I)
PHP_DECL = re.compile('\\bfunction\\s+test\\w*\\s*\\(', re.I)
_COMMENT = re.compile('^\\s*(?://|#(?!\\[)|/?\\*|--)')

def language(path):
    ext = path.rsplit('/', 1)[-1].rsplit('.', 1)
    return LANG.get(ext[1].lower()) if len(ext) == 2 else None

def parse_patch(txt):
    files, cur = ([], None)
    for line in txt.split('\n'):
        if line.startswith('diff --git '):
            mm = re.match('diff --git a/(.*?) b/(.*)$', line)
            cur = {'path': mm.group(2) if mm else line[11:], 'add': [], 'del': [], 'new': False, 'deleted': False}
            files.append(cur)
        elif cur is not None:
            if line.startswith('new file mode'):
                cur['new'] = True
            elif line.startswith('deleted file mode'):
                cur['deleted'] = True
            elif line.startswith('+') and (not line.startswith('+++')):
                cur['add'].append(line[1:])
            elif line.startswith('-') and (not line.startswith('---')):
                cur['del'].append(line[1:])
    return files
_ANN_LINE = re.compile('^\\s*(?:@|#\\[)')

def count_cases(lines, lang):
    rx = TESTCASE_RX.get(lang)
    if rx is None:
        return 0
    if lang == 'php':
        ann, decl, other = (PHP_ANN, PHP_DECL, re.compile('(?<![\\w>$:])(?:it|test)\\s*\\(\\s*[\'\\"]', re.I))
    elif lang in DECL_RX:
        ann, decl, other = (rx, DECL_RX[lang], None)
    else:
        ann, decl, other = (None, None, rx)
    n, pending = (0, False)
    for s in lines:
        if not s.strip():
            continue
        is_comment = bool(_COMMENT.match(s))
        if is_comment and lang != 'php':
            continue
        if ann is not None and ann.search(s):
            n += 1
            pending = True
            continue
        if is_comment:
            continue
        if other is not None and other.search(s):
            n += 1
        elif decl is not None and decl.search(s) and (not pending):
            n += 1
        if not _ANN_LINE.match(s):
            pending = False
    return n

def measures(txt):
    if txt is None:
        return None
    files = parse_patch(txt)
    tf = [f for f in files if TESTPATH.search(f['path'])]
    code = [f for f in tf if language(f['path']) is not None and (not FIXTURE.search(f['path']))]
    added = removed = 0
    by_lang = {}
    unknown_lang = 0
    for f in tf:
        lang = language(f['path'])
        if lang is None or lang not in TESTCASE_RX:
            unknown_lang += 1
            continue
        a, r = (count_cases(f['add'], lang), count_cases(f['del'], lang))
        added += a
        removed += r
        if a or r:
            x = by_lang.setdefault(lang, [0, 0])
            x[0] += a
            x[1] += r
    langs = sorted({language(f['path']) or 'other' for f in files})
    return {'version': VERSION, 'files': len(files), 'test_files': len(tf), 'test_new': sum((1 for f in tf if f['new'])), 'test_mod': sum((1 for f in tf if not f['new'])), 'test_deleted': sum((1 for f in tf if f['deleted'])), 'test_del_lines': sum((len(f['del']) for f in tf if not f['new'])), 'test_code_files': len(code), 'test_code_new': sum((1 for f in code if f['new'])), 'fixture_files': len(tf) - len(code), 'added_test_functions': added, 'removed_test_functions': removed, 'net_test_functions': added - removed, 'test_files_unknown_lang': unknown_lang, 'cases_by_lang': by_lang, 'langs': langs, 'wrote_tests': int(len(tf) > 0), 'new_tests': int(any((f['new'] for f in tf))), 'wrote_test_code': int(len(code) > 0), 'adds_test_function': int(added > 0)}
