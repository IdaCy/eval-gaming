"""Restore large patch fields from verified UTF-8 chunks."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def iter_field_chunks(root, run_id, field, manifest=None):
    root = Path(root)
    manifest = manifest or json.loads((root / 'runs/field_manifest.json').read_text())
    records = [record for record in manifest['fields'] if record['run_id'] == run_id and record['field'] == field]
    if len(records) != 1:
        raise ValueError('Expected one field manifest entry')
    record = records[0]
    digest = hashlib.sha256()
    size, expected_index = 0, 0
    for relative in manifest['files']:
        path = root / relative
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            for row in batch.to_pylist():
                if row['run_id'] != run_id or row['field'] != field:
                    continue
                raw = row['text'].encode('utf-8')
                if row['chunk_index'] != expected_index:
                    raise ValueError('Missing, duplicated, or unordered chunk')
                if len(raw) != row['utf8_bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
                    raise ValueError('Chunk integrity mismatch')
                expected_index += 1
                size += len(raw)
                digest.update(raw)
                yield row['text']
    if expected_index != record['chunk_count'] or size != record['utf8_bytes'] or digest.hexdigest() != record['sha256']:
        raise ValueError('Reconstructed field integrity mismatch')


def restore_row(row, root, manifest=None):
    root = Path(root)
    manifest = manifest or json.loads((root / 'runs/field_manifest.json').read_text())
    result = dict(row)
    for record in manifest['fields']:
        if record['run_id'] == row['run_id']:
            result[record['field']] = ''.join(iter_field_chunks(root, row['run_id'], record['field'], manifest))
    return result
