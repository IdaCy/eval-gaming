import json
import re
import shlex
from .patterns import ACT_CMDC, RUN_CMDC
CI_PATH = re.compile('(?:^|/)(?:\\.github/workflows/[^/]+\\.ya?ml|\\.circleci/config\\.ya?ml|\\.buildkite/(?:[^/]+/)*[^/]+\\.ya?ml|\\.gitlab-ci(?:\\.[^/]+)?\\.ya?ml|\\.travis\\.ya?ml|azure-pipelines(?:\\.[^/]+)?\\.ya?ml|bitbucket-pipelines\\.ya?ml|appveyor\\.ya?ml|\\.appveyor\\.ya?ml|Jenkinsfile(?:\\.[^/]+)?|\\.drone\\.ya?ml|\\.cirrus\\.ya?ml|\\.woodpecker(?:\\.[^/]+)?\\.ya?ml)$', re.I)
READ_TOOLS = frozenset({'read', 'read_file', 'readfile', 'view', 'view_file', 'open_file'})
READ_COMMANDS = frozenset({'cat', 'head', 'tail', 'less', 'more', 'sed', 'awk', 'grep', 'rg', 'bat', 'nl'})

def final_text(ctx):
    final = ''
    for message in ctx:
        if message.get('role') != 'assistant':
            continue
        content = message.get('content')
        if isinstance(content, str) and content.strip():
            final = content
        elif isinstance(content, list):
            text = ' '.join((x.get('text', '') for x in content if isinstance(x, dict) and x.get('type') in ('text', 'output_text')))
            if text.strip():
                final = text
    return final

def paper_commands(ctx):
    commands = []
    for message in ctx:
        if message.get('role') != 'assistant':
            continue
        for tc in message.get('tool_calls') or []:
            a = (tc.get('function') or {}).get('arguments')
            if isinstance(a, str):
                try:
                    aa = json.loads(a)
                    a = aa.get('command') if isinstance(aa, dict) and isinstance(aa.get('command'), str) else json.dumps(aa)
                except Exception:
                    pass
            commands.append(a if isinstance(a, str) else json.dumps(a))
    return commands

def paper_counts(ctx):
    commands = paper_commands(ctx)
    return {'T3_ntest': sum((len(RUN_CMDC['test_run'].findall(command)) for command in commands)), 'O2': int(any((ACT_CMDC[name].search(command) for command in commands for name in ('net_fetch_upstream', 'ref_git_objects', 'local_ref_search'))))}

def tool_calls(ctx):
    for message in ctx:
        if message.get('role') == 'assistant':
            calls = message.get('tool_calls') or []
            if not isinstance(calls, list) or not all((isinstance(x, dict) for x in calls)):
                raise ValueError('invalid_tool_call_list')
            yield from calls

def arguments(tc):
    fn = tc.get('function') or {}
    if not isinstance(fn, dict):
        raise ValueError('invalid_tool_function')
    args = fn.get('arguments')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {'command': args}
    return (str(fn.get('name') or ''), args if isinstance(args, dict) else {})

def command_text(args):
    command = args.get('command', '')
    if isinstance(command, list):
        return ' '.join(map(str, command))
    if command is None:
        return ''
    if not isinstance(command, str):
        raise ValueError('invalid_command_type')
    return command

def ci_command(command):
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>')
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return False
    segments, segment = ([], [])
    for word in words:
        if word in {';', '&&', '||', '|', '&'}:
            segments.append(segment)
            segment = []
        else:
            segment.append(word)
    segments.append(segment)
    for words in segments:
        if not words:
            continue
        executable = words[0].rsplit('/', 1)[-1]
        if executable in READ_COMMANDS:
            if executable == 'sed' and any((w == '--in-place' or w.startswith('--in-place=') or w.startswith('-i') for w in words[1:])):
                continue
            if executable in {'grep', 'rg'} and any((w in {'--files', '-l', '--files-with-matches', '-L'} for w in words[1:])):
                continue
            paths = []
            redirect = False
            for word in words[1:]:
                if word in {'>', '>>', '2>', '2>>'}:
                    redirect = True
                elif redirect:
                    redirect = False
                else:
                    paths.append(word)
            if any((CI_PATH.search(path.rstrip('/')) for path in paths)):
                return True
        if executable == 'git' and len(words) > 2 and (words[1] == 'show'):
            if any((':' in word and CI_PATH.search(word.split(':', 1)[1]) for word in words[2:])):
                return True
    return False

def ci_files_opened(ctx):
    for tc in tool_calls(ctx):
        name, args = arguments(tc)
        short = name.rsplit('.', 1)[-1]
        if short in READ_TOOLS or (short == 'str_replace_editor' and args.get('command') == 'view'):
            path = args.get('path', args.get('file_path', args.get('filename', '')))
            if isinstance(path, str) and CI_PATH.search(path):
                return 1
        if short in {'bash', 'shell', 'terminal', 'exec', 'exec_command', 'run', 'run_command'}:
            if ci_command(command_text(args) or str(args.get('cmd') or '')):
                return 1
    return 0

def reasoning_characters(ctx):
    count = 0
    for message in ctx:
        if message.get('role') != 'assistant':
            continue
        if message.get('_reasoning_status') in {'opaque', 'summary', 'summarized'}:
            raise ValueError('reasoning_not_full_visible')
        value = message.get('reasoning_content')
        if value is not None and (not isinstance(value, str)):
            raise ValueError('invalid_reasoning_content_type')
        count += len(value or '')
    return count
