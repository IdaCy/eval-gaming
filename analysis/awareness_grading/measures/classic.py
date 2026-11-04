import io
import json
import os
import re
import types
TEST = re.compile('(pytest|py\\.test|phpunit|rspec|rake\\s+(test|spec)|bundle\\s+exec\\s+(rspec|rake)|\\bmvn\\b[^\\n;|&]*\\b(test|verify)\\b|gradlew?\\b[^\\n;|&]*\\b(test|check)\\b|\\bnpm\\s+(run\\s+)?test|\\byarn\\s+(run\\s+)?test|\\bpnpm\\s+(run\\s+)?test|\\bjest\\b|\\bvitest\\b|\\bmocha\\b|\\bgo\\s+test\\b|\\bcargo\\s+test\\b|\\bctest\\b|\\bant\\s+test\\b|\\bcomposer\\s+(run\\s+)?test|node\\s+--test|\\bminitest\\b|ruby\\s+-I)', re.I)
TP = re.compile('(^|/)(tests?|spec|specs|__tests__|testing|fixtures?)(/|$)|(_test|_spec|\\.test|\\.spec|Test|Tests|TestCase|Spec|IT)\\.[A-Za-z0-9]+$')
DOC = re.compile('(\\.md$|\\.rst$|\\.adoc$|\\.txt$|(^|/)docs?(/|$)|changelog|changes|history|news)', re.I)

def patch_stats(p):
    o = dict(p_files=0, p_src_files=0, p_test_files=0, p_doc_files=0, p_add=0, p_del=0, p_src_add=0, p_src_del=0, p_test_add=0, p_test_del=0, p_doc_add=0, p_new_files=0, p_comment_add=0)
    if not os.path.exists(p):
        return o
    cur = None
    for line in open(p, errors='replace'):
        if line.startswith('diff --git '):
            m = re.match('diff --git a/(.*) b/(.*)', line.rstrip('\n'))
            cur = m.group(2) if m else ''
            o['p_files'] += 1
            kind = 'test' if TP.search(cur) else 'doc' if DOC.search(cur) else 'src'
            o['p_' + kind + '_files'] += 1
        elif line.startswith('new file mode'):
            o['p_new_files'] += 1
        elif line.startswith('+++') or line.startswith('---'):
            continue
        elif line.startswith('+'):
            o['p_add'] += 1
            if cur is not None:
                kind = 'test' if TP.search(cur) else 'doc' if DOC.search(cur) else 'src'
                o['p_' + kind + '_add'] += 1
                if kind == 'src' and re.match('\\+\\s*(//|#|/\\*|\\*)', line):
                    o['p_comment_add'] += 1
        elif line.startswith('-'):
            o['p_del'] += 1
            if cur is not None:
                kind = 'test' if TP.search(cur) else 'doc' if DOC.search(cur) else 'src'
                if kind in ('src', 'test'):
                    o['p_' + kind + '_del'] += 1
    return o

def context_counts(ctx):
    cmds = []
    nread = 0
    final = ''
    for m in ctx:
        if m.get('role') == 'assistant':
            c = m.get('content')
            if isinstance(c, str) and c.strip():
                final = c
            elif isinstance(c, list):
                t = ' '.join((x.get('text', '') for x in c if isinstance(x, dict) and x.get('type') in ('text', 'output_text')))
                if t.strip():
                    final = t
            for tc in m.get('tool_calls') or []:
                fn = tc.get('function') or {}
                name = fn.get('name', '')
                a = fn.get('arguments', '')
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {'command': a}
                cmd = a.get('command') if isinstance(a, dict) else ''
                if isinstance(cmd, list):
                    cmd = ' '.join(map(str, cmd))
                if name == 'read' or (isinstance(a, dict) and 'path' in a and (not cmd)):
                    nread += 1
                else:
                    cmds.append(cmd or '')
    tests = [i for i, c in enumerate(cmds) if TEST.search(c)]
    return {'test_runs': len(tests), 'final_report': len(final)}

def patch_counts(text, testpath=None, docpath=None):
    namespace = dict(patch_stats.__globals__)
    namespace['os'] = types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda path: True))
    namespace['open'] = lambda path, errors=None: io.StringIO(text)
    if testpath is not None:
        namespace['TP'] = testpath
    if docpath is not None:
        namespace['DOC'] = docpath
    parser = types.FunctionType(patch_stats.__code__, namespace, patch_stats.__name__)
    return parser('in_memory_patch')
