"""Replace exact fetch-only task text while preserving other record content."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

MARKER = '[task text: fetch with harness/download]'
FIELDS = ('messages', 'final_patch', 'final_answer', 'grading_result')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def request_strings(value, history=False):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {'request', 'problem_statement', 'body'} or (history and key == 'content'):
                if isinstance(child, str) and child.strip():
                    yield child
            elif key == 'history':
                yield from request_strings(child, history=True)
            elif isinstance(child, (dict, list)) and key != 'files':
                yield from request_strings(child, history=history)
    elif isinstance(value, list):
        for child in value:
            yield from request_strings(child, history=history)


def fragments(value):
    """Keep full strings and substantial prose paragraphs, excluding fenced code."""
    yield value
    plain = re.sub(r'(?ms)^\s*(```|~~~).*?^\s*\1[^\n]*$', '', value)
    for paragraph in re.split(r'\n\s*\n', plain):
        paragraph = paragraph.strip()
        if len(paragraph) >= 40 and len(re.findall(r'\b[A-Za-z]+\b', paragraph)) >= 6:
            yield paragraph


def make_corpus(source_root, text_sources):
    index = pq.read_table(text_sources).to_pylist()
    tasks = {row['task_id']: set() for row in index}
    pins = []
    sources = [('tasks/tasks.parquet', 'spec'), ('setups/variants.parquet', 'value'),
               ('tables/steering/tasks.parquet', 'spec_json'), ('tables/steering/variants.parquet', 'variant_json'),
               ('tables/steering/evaluation_prompts.parquet', 'messages_json')]
    for relative, column in sources:
        path = source_root / relative
        if not path.is_file():
            continue
        pins.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': file_digest(path)})
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['task_id', column]):
            for row in batch.to_pylist():
                if row['task_id'] not in tasks or not row[column]:
                    continue
                value = json.loads(row[column])
                strings = (m['content'] for m in value if m.get('role') == 'user' and isinstance(m.get('content'), str)) if column == 'messages_json' else request_strings(value)
                for text in strings:
                    tasks[row['task_id']].update(x for x in fragments(text) if x and x != MARKER)
    missing = [task for task, strings in tasks.items() if not strings]
    if missing:
        raise ValueError('Task source text is unavailable: ' + ', '.join(missing))
    return {'version': 1, 'marker': MARKER, 'sources': pins,
            'text_sources_sha256': file_digest(text_sources),
            'tasks': {task: sorted(strings) for task, strings in sorted(tasks.items())}}


class Matcher:
    def __init__(self, strings, acceleration=True):
        needles = set()
        for value in strings:
            for text in {value, value.replace('\r\n', '\n'), value.replace('\r\n', '\n').replace('\n', '\r\n')}:
                if text and text != MARKER:
                    needles.add(text)
                    needles.add(json.dumps(text, ensure_ascii=False)[1:-1])
                    needles.add(json.dumps(text, ensure_ascii=True)[1:-1])
        self.automaton = None
        if acceleration:
            try:
                import ahocorasick
            except ImportError:
                pass
            else:
                self.automaton = ahocorasick.Automaton()
                for text in sorted(needles):
                    self.automaton.add_word(text, len(text))
                self.automaton.make_automaton()
        if self.automaton is None:
            self.pattern = re.compile('(?=(' + '|'.join(re.escape(x) for x in sorted(needles, key=lambda x: (-len(x), x))) + '))')

    def spans(self, value):
        if self.automaton is not None:
            spans = sorted((end + 1 - length, end + 1) for end, length in self.automaton.iter(value))
        else:
            spans = [(m.start(), m.start() + len(m.group(1))) for m in self.pattern.finditer(value)]
        merged = []
        for left, right in spans:
            if merged and left < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        return merged

    def replace(self, value, counts, location):
        spans = self.spans(value)
        if not spans:
            return value
        counts['occurrences'] += len(spans)
        counts['matched_utf8_bytes'] += sum(len(value[a:b].encode('utf-8')) for a, b in spans)
        counts['locations:' + location] += len(spans)
        pieces, position = [], 0
        for left, right in spans:
            pieces.extend((value[position:left], MARKER))
            position = right
        pieces.append(value[position:])
        return ''.join(pieces)


def project_value(value, matcher, counts, location):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            projected_key = matcher.replace(key, counts, location + '.<key>') if isinstance(key, str) else key
            if projected_key in result:
                raise ValueError('Task text projection would merge distinct object keys')
            result[projected_key] = project_value(child, matcher, counts, location + '.' + str(key))
        return result
    if isinstance(value, list):
        return [project_value(child, matcher, counts, location + '[]') for child in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith(('{', '[', '"')):
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            pass
        else:
            if isinstance(parsed, (dict, list, str)) and parsed != value:
                result = project_value(parsed, matcher, counts, location + '::json')
                return json.dumps(result, ensure_ascii=False) if result != parsed else value
    return matcher.replace(value, counts, location)


class Projection:
    def __init__(self, corpus, acceleration=True):
        if corpus['marker'] != MARKER:
            raise ValueError('Unexpected marker')
        self.sources = corpus['tasks']
        self.matchers = {}
        self.acceleration = acceleration

    def row(self, row):
        result = dict(row)
        task = row.get('task', row.get('task_id'))
        counts = Counter()
        result['task_text_withheld'] = task in self.sources
        if task not in self.sources:
            return result, counts
        if task not in self.matchers:
            self.matchers[task] = Matcher(self.sources[task], self.acceleration)
        matcher = self.matchers[task]
        for field in FIELDS:
            if field not in row:
                continue
            result[field] = project_value(row[field], matcher, counts, field)
            if result[field] != row[field]:
                counts['fields:' + field] += 1
        if counts['occurrences']:
            counts['rows_changed'] += 1
        return result, counts


def initialize(corpus_path, input_root, output_root):
    global PROJECTOR, INPUT_ROOT, OUTPUT_ROOT, CORPUS_SHA256
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    PROJECTOR = Projection(json.loads(Path(corpus_path).read_text()))
    CORPUS_SHA256 = file_digest(corpus_path)
    INPUT_ROOT, OUTPUT_ROOT = Path(input_root), Path(output_root)


def project_file(relative):
    source, target = INPUT_ROOT / relative, OUTPUT_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    if 'task_text_withheld' not in schema.names:
        schema = schema.append(pa.field('task_text_withheld', pa.bool_()))
    temporary = target.with_name(target.name + '.partial')
    counts = Counter()
    task_counts = Counter()
    changed_ids = []
    unmatched_ids = []
    with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
        for batch in parquet.iter_batches(batch_size=32):
            output = []
            for row in batch.to_pylist():
                value, changes = PROJECTOR.row(row)
                counts.update(changes)
                counts['rows'] += 1
                if value['task_text_withheld']:
                    counts['restricted_rows'] += 1
                    if not changes['occurrences'] and row.get('messages') not in ('[]', None):
                        unmatched_ids.append(row['run_id'])
                verified, remaining = PROJECTOR.row(value)
                if verified != value or remaining:
                    raise ValueError('Projection left matching task text')
                if any(value[key] != child for key, child in row.items() if key not in FIELDS and key != 'task_text_withheld'):
                    raise ValueError('Projection changed unrelated metadata')
                if changes['occurrences']:
                    task_counts[value['task']] += 1
                    changed_ids.append(value['run_id'])
                output.append(value)
            writer.write_table(pa.Table.from_pylist(output, schema=schema))
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed while being read')
    temporary.replace(target)
    return {'path': str(relative), 'corpus_sha256': CORPUS_SHA256,
            'input_bytes': before.st_size, 'output_bytes': target.stat().st_size,
            'input_sha256': file_digest(source) if source != target else None, 'sha256': file_digest(target),
            'counts': dict(counts), 'changed_rows_by_task': dict(task_counts), 'changed_run_ids': changed_ids,
            'restricted_rows_without_matches': unmatched_ids, 'idempotence_and_metadata_verified': True}


def project_and_release(relative):
    result = project_file(relative)
    PROJECTOR.matchers.clear()
    import gc
    gc.collect()
    pa.default_memory_pool().release_unused()
    if os.name == 'posix':
        import ctypes
        library = ctypes.CDLL(None)
        if hasattr(library, 'malloc_trim'):
            library.malloc_trim(0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    build = modes.add_parser('build')
    build.add_argument('--source-hf', type=Path, required=True)
    build.add_argument('--text-sources', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    project = modes.add_parser('project')
    project.add_argument('--corpus', type=Path, required=True)
    project.add_argument('--input', type=Path, required=True)
    project.add_argument('--output', type=Path, required=True)
    project.add_argument('--report', type=Path, required=True)
    project.add_argument('--workers', type=int, default=os.cpu_count())
    project.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.mode == 'build':
        corpus = make_corpus(args.source_hf, args.text_sources)
        args.output.write_text(json.dumps(corpus, ensure_ascii=False) + '\n')
        print(json.dumps({'tasks': len(corpus['tasks']), 'strings': sum(map(len, corpus['tasks'].values())), 'sha256': file_digest(args.output)}))
        return
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be separate directories')
    files = sorted(args.input.rglob('*.parquet'), key=lambda p: -p.stat().st_size)
    results = []
    args.report.parent.mkdir(parents=True, exist_ok=True)
    detail = args.report.parent / (args.report.stem + '_files')
    detail.mkdir(exist_ok=True)
    pending = []
    corpus_sha256 = file_digest(args.corpus)
    for path in files:
        relative = path.relative_to(args.input)
        receipt = detail / (digest(str(relative)) + '.json')
        if args.resume and receipt.is_file():
            item = json.loads(receipt.read_text())
            target = args.output / relative
            if (item.get('corpus_sha256') == corpus_sha256 and target.is_file() and file_digest(target) == item['sha256']
                    and file_digest(path) == item['input_sha256']):
                results.append(item)
                continue
        pending.append(relative)
    print(json.dumps({'resumed': len(results), 'remaining': len(pending)}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(args.corpus), str(args.input), str(args.output))) as pool:
        futures = [pool.submit(project_and_release, relative) for relative in pending]
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            (detail / (digest(item['path']) + '.json')).write_text(json.dumps(item) + '\n')
            print(json.dumps({'complete': len(results), 'total': len(files), 'path': item['path'], 'counts': item['counts']}), flush=True)
    report = {'corpus_sha256': file_digest(args.corpus), 'code_sha256': file_digest(__file__),
              'counts': dict(sum((Counter(x['counts']) for x in results), Counter())),
              'files': sorted(results, key=lambda x: x['path'])}
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
