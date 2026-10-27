"""Remove recorded provider response objects and opaque encrypted payloads."""

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

MARKER = '[redacted]'
FIELDS = ('messages', 'final_patch', 'final_answer', 'grading_result')
IDENTITY = re.compile(r'(?:resp_|rs_)[A-Za-z0-9]{20,}|(?:chatcmpl-|cmpl-)[A-Za-z0-9_-]{16,}')
PAYLOAD = re.compile(r'(?<![A-Za-z0-9_+/=-])gAAAA[A-Za-z0-9_+/=-]{27,}')
TOKEN = re.compile(r'["{]')
OBJECT_START = re.compile(r'\{\s*"')
DECODER = json.JSONDecoder()
QUOTE = r'(?<!\\)(?:\\*["\x27]|\\+u002[27])'
SPACE = r'(?:\s|\\+[nrt])*'
SEPARATOR = SPACE + r'(?::|=>)' + SPACE
VALUE_END = r'(?:' + QUOTE + r'|(?=(?:\r?\n|\\+[nr]|$)))'
FIXTURE_FRAGMENT = re.compile(
    r'\bevent' + SPACE + r':' + SPACE + r'response\.[a-zA-Z0-9_.]+'
    + r'|' + QUOTE + r'(?:type|object)' + QUOTE + SEPARATOR + QUOTE
    + r'(?:response(?:\.[a-zA-Z0-9_.]+)?|reasoning|chat\.completion(?:\.chunk)?|text_completion)' + VALUE_END
    + r'|' + QUOTE + r'encrypted_content' + QUOTE + SEPARATOR
    + r'|' + QUOTE + r'(?:id|item_id|response_id)' + QUOTE + SEPARATOR + QUOTE
    + r'(?:resp_|rs_|chatcmpl-|cmpl-)[A-Za-z0-9_-]+')
JSON_LINE = re.compile(r'^\s*(?:[+-]\s*)?(?:["\x27{}\[\]]|(?:true|false|null|[0-9.-]+)\s*[,\]}])')


def fixture_fragment(value):
    return any(word in value for word in ('response', 'reasoning', 'encrypted_content', 'resp_', 'rs_', 'cmpl-')) and FIXTURE_FRAGMENT.search(value) is not None


def remove_fixture_fragments(value, counts):
    """Replace undecodable record lines, including escaped and truncated snippets."""
    if not fixture_fragment(value):
        return value
    lines = value.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    selected = set()
    for match in FIXTURE_FRAGMENT.finditer(value):
        first = bisect_right(offsets, match.start()) - 1
        last = bisect_right(offsets, match.end() - 1) - 1
        selected.update(range(first, last + 1))
    for index in sorted(selected):
        line = lines[index]
        if JSON_LINE.match(line):
            left, right = index - 1, index + 1
            while left >= 0 and JSON_LINE.match(lines[left]):
                selected.add(left)
                left -= 1
            while right < len(lines) and JSON_LINE.match(lines[right]) and not lines[right].startswith('[Tool output truncated;'):
                selected.add(right)
                right += 1
    result = []
    for index, line in enumerate(lines):
        if index in selected:
            if index - 1 not in selected:
                result.append(MARKER)
                counts['raw_fixture_fragments'] += 1
            if index + 1 not in selected:
                result.append('\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else '')
        else:
            result.append(line)
    return ''.join(result)


def identity_marker(value):
    if IDENTITY.fullmatch(value):
        return True
    return any(quote+prefix in value for quote in ('"', "'") for prefix in ('resp_', 'rs_', 'chatcmpl-', 'cmpl-')) and IDENTITY.search(value) is not None


def payload_matches(value):
    position = value.find('gAAAA')
    while position >= 0:
        match = PAYLOAD.match(value, position)
        if match:
            yield match
        position = value.find('gAAAA', position+5)


def has_markers(value):
    return isinstance(value, str) and (identity_marker(value) or fixture_fragment(value) or next(payload_matches(value), None) is not None)


def recorded_object(value):
    if not isinstance(value, dict):
        return False
    kind = value.get('type')
    if isinstance(kind, str) and (kind == 'response' or kind.startswith('response.')):
        return True
    if kind == 'reasoning':
        return True
    if 'encrypted_content' in value:
        return True
    if value.get('object') in ('response', 'chat.completion', 'chat.completion.chunk', 'text_completion'):
        return True
    identified = any(isinstance(value.get(key), str) and value[key].startswith(('resp_', 'rs_', 'chatcmpl-', 'cmpl-'))
                     for key in ('id', 'item_id', 'response_id'))
    return identified and (set(value) <= {'id', 'item_id', 'response_id'} or bool({'output', 'choices'} & value.keys()))


def clean(value, counts):
    if recorded_object(value):
        counts['raw_response_objects'] += 1
        counts['encrypted_payloads'] += sum(1 for _ in payload_matches(json.dumps(value)))
        return MARKER
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            new_key = clean(key, counts)
            if new_key in result:
                raise ValueError('Projection would merge object keys')
            result[new_key] = clean(child, counts)
        return result
    if isinstance(value, list):
        return [clean(child, counts) for child in value]
    if not has_markers(value):
        return value
    stripped = value.strip()
    if stripped.startswith(('{', '[', '"')):
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            pass
        else:
            if isinstance(parsed, (dict, list, str)) and parsed != value:
                result = clean(parsed, counts)
                return json.dumps(result, ensure_ascii=False) if result != parsed else value
    replacements = []
    position = 0
    scan_tokens = identity_marker(value) or fixture_fragment(value)
    while scan_tokens:
        match = TOKEN.search(value, position)
        if match is None:
            break
        start = match.start()
        if value[start] == '{' and not OBJECT_START.match(value, start):
            position = start+1
            continue
        try:
            parsed, end = DECODER.raw_decode(value, start)
        except (ValueError, RecursionError):
            position = start + 1
            continue
        if recorded_object(parsed):
            replacement = clean(parsed, counts)
        elif isinstance(parsed, str) and parsed != value and has_markers(parsed):
            replacement = clean(parsed, counts)
        else:
            position = end if isinstance(parsed, str) else start + 1
            continue
        if replacement != parsed:
            replacements.append((start, end, json.dumps(replacement, ensure_ascii=False)))
            position = end
        else:
            position = start + 1
    for match in payload_matches(value):
        if any(left <= match.start() and match.end() <= right for left, right, _ in replacements):
            continue
        replacements.append((match.start(), match.end(), MARKER))
        counts['encrypted_payloads'] += 1
    result = value
    for left, right, replacement in sorted(replacements, reverse=True):
        result = result[:left] + replacement + result[right:]
    return remove_fixture_fragments(result, counts)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def project_file(source, target, report_path):
    source, target, report_path = map(Path, (source, target, report_path))
    parquet = pq.ParquetFile(source)
    edits, counts, ids = {}, Counter(), []
    ordinal = 0
    for batch in parquet.iter_batches(batch_size=16, columns=['run_id', *FIELDS], use_threads=False):
        for row in batch.to_pylist():
            changed, delta = {}, Counter()
            for field in FIELDS:
                value = row[field]
                if has_markers(value):
                    result = clean(value, delta)
                    if result != value:
                        changed[field] = result
                        counts['fields:'+field] += 1
                        again = Counter()
                        assert clean(result, again) == result and not again
            counts.update(delta)
            if changed:
                edits[ordinal] = (changed, sum(delta.values()))
                ids.append(row['run_id'])
            ordinal += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    if edits:
        temporary = target.with_suffix('.partial')
        offset = 0
        with pq.ParquetWriter(temporary, parquet.schema_arrow, compression='zstd', compression_level=3) as writer:
            for batch in parquet.iter_batches(batch_size=128, use_threads=False):
                rows = batch.to_pylist()
                for row in rows:
                    if offset in edits:
                        changed, number = edits[offset]
                        row.update(changed)
                        row['redacted_fields'] += number
                    offset += 1
                writer.write_table(pa.Table.from_pylist(rows, schema=parquet.schema_arrow))
        temporary.replace(target)
    else:
        if target.exists():
            target.unlink()
        os.link(source, target)
    result = {'source': str(source), 'target': str(target), 'rows': ordinal, 'changed_rows': len(edits),
              'counts': dict(counts), 'changed_run_ids': ids, 'input_sha256': checksum(source),
              'output_sha256': checksum(target), 'bytes': target.stat().st_size}
    report_path.write_text(json.dumps(result)+'\n')
    summary = {k:v for k,v in result.items() if k != 'changed_run_ids'}
    del edits, parquet
    gc.collect()
    pa.default_memory_pool().release_unused()
    library = ctypes.CDLL(None)
    if hasattr(library, 'malloc_trim'):
        library.malloc_trim(0)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Use separate input and output directories')
    (args.reports/'files').mkdir(parents=True, exist_ok=True)
    paths = sorted((p for p in args.input.glob('*/*.parquet') if p.parent.name != '_field_chunks'), key=lambda p:-p.stat().st_size)
    results, pending = [], []
    for path in paths:
        relative = path.relative_to(args.input)
        receipt = args.reports/'files'/(hashlib.sha256(str(relative).encode()).hexdigest()+'.json')
        destination = args.output/relative
        if args.resume and receipt.exists():
            record = json.loads(receipt.read_text())
            if destination.exists() and checksum(path) == record['input_sha256'] and checksum(destination) == record['output_sha256']:
                results.append(record)
                continue
        pending.append(path)
    print(json.dumps({'resumed':len(results), 'remaining':len(pending)}),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        jobs = [pool.submit(project_file, str(path), str(args.output/path.relative_to(args.input)),
                            str(args.reports/'files'/(hashlib.sha256(str(path.relative_to(args.input)).encode()).hexdigest()+'.json'))) for path in pending]
        for future in as_completed(jobs):
            record = future.result()
            results.append(record)
            print(json.dumps({'complete':len(results), 'total':len(paths), **record}), flush=True)
    report = {'files': results, 'counts': dict(sum((Counter(r['counts']) for r in results), Counter())),
              'changed_rows': sum(r['changed_rows'] for r in results), 'passed': True,
              'code_sha256': checksum(Path(__file__))}
    (args.reports/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
