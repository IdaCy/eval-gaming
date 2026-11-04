import ast
import json
from pathlib import PurePosixPath
import re

def message_text(message):
    value = message.get('content')
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ' '.join((p.get('text', '') for p in value if isinstance(p, dict) and p.get('type') in ('text', 'output_text')))
    return ''

def final_message_length(messages):
    for message in reversed(messages):
        if message.get('role') == 'assistant':
            text = message_text(message)
            if text.strip():
                return len(text)
    return 0

def patch_path(value):
    value = value.split('\t', 1)[0]
    if value.startswith('"'):
        value = ast.literal_eval(value)
    return value[2:] if value.startswith(('a/', 'b/')) else value

def added_files(patch):
    result, current, last_added = ([], None, False)
    for line in patch.splitlines(keepends=True):
        if line.startswith('diff --git '):
            current = {'path': None, 'new': False, 'added': []}
            result.append(current)
            last_added = False
        elif current is not None:
            if line.startswith('new file mode ') or line.rstrip('\r\n') == '--- /dev/null':
                current['new'] = True
            elif line.startswith('+++ '):
                current['path'] = patch_path(line[4:].rstrip('\r\n'))
                last_added = False
            elif line.startswith('+'):
                current['added'].append(line[1:])
                last_added = True
            elif line.startswith('\\ No newline at end of file'):
                if last_added and current['added']:
                    current['added'][-1] = current['added'][-1].removesuffix('\n')
                last_added = False
            else:
                last_added = False
    return [r for r in result if r['path'] and r['path'] != '/dev/null']

def report_file_length(files):
    return sum((len(''.join(row['added'])) for row in files if row['new'] and PurePosixPath(row['path']).suffix.lower() in ('.md', '.rst', '.txt') and (not row['path'].startswith('data/'))))

def measure(messages, patch):
    final = final_message_length(messages)
    files = report_file_length(added_files(patch))
    return {"report_length_raw": final + files, "report_length_final_only_raw": final, "report_file_chars": files}
