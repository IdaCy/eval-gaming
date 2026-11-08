"""Parse JSON probability answers, including optional Markdown fences."""
import json

import math

import re

FENCE = re.compile(r'^\s*```(?:json)?\s*(.*?)\s*```\s*$', re.S)

def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate_json_key')
        out[key] = value
    return out

def parse(text):
    """Return (probability, reason, status). status: ok, blank, unparseable or out_of_range."""
    if not isinstance(text, str) or not text.strip():
        return None, None, 'blank'
    body = text.strip()
    fenced = FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        data = json.loads(body, object_pairs_hook=unique_object)
    except ValueError:
        return None, None, 'unparseable'
    if not isinstance(data, dict) or 'probability' not in data:
        return None, None, 'unparseable'
    value = data['probability']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, None, 'unparseable'
    if not 0 <= value <= 1:
        return None, None, 'out_of_range'
    reason = data.get('reason')
    return float(value), reason if isinstance(reason, str) else None, 'ok'
