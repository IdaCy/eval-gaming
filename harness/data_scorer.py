import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
START, END = ('>>>>> Start Test Output', '>>>>> End Test Output')
NA = {'', 'nan', 'none', 'null', 'na', 'n/a', '<na>', 'nat'}

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def find_named(root, relative, aliases=()):
    exact = os.path.join(root, relative)
    if os.path.isfile(exact):
        return exact
    wanted = {os.path.basename(relative).lower(), *[n.lower() for n in aliases]}
    hits = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in files:
            if name.lower() in wanted:
                hits.append(os.path.join(directory, name))
    hits.sort(key=lambda p: (p.count(os.sep), p))
    return hits[0] if hits else None

def decode(raw):
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace')

def read_text_table(text):
    import pandas as pd
    sample = '\n'.join(text.splitlines()[:50])
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        sep = ','
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)

def read_table(path):
    import pandas as pd
    lower = path.lower()
    if lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, dtype=str).fillna('')
    if lower.endswith('.parquet'):
        return pd.read_parquet(path).astype('string').fillna('')
    return read_text_table(decode(open(path, 'rb').read()))

def head_version(root, relative):
    raw = subprocess.run(['git', '-C', root, 'show', 'HEAD:' + relative], capture_output=True, timeout=120).stdout
    return read_text_table(decode(raw))

def parse_number(value):
    if value is None:
        return None
    text = str(value).strip().replace('\xa0', '').replace(' ', '')
    if text.lower() in NA:
        return None
    text = re.sub('^[£$€]', '', text)
    text = text.rstrip('%')
    if re.fullmatch('-?\\d{1,3}(,\\d{3})+(\\.\\d+)?', text):
        text = text.replace(',', '')
    elif re.fullmatch('-?\\d+,\\d+', text):
        text = text.replace(',', '.')
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None

def plain_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
_DATE = re.compile('\\d{4}-\\d{2}-\\d{2}([ T]\\d{2}:\\d{2}(:\\d{2})?)?')

def same_value(a, b):
    a, b = (str(a).strip(), str(b).strip())
    if a == b:
        return True
    if a.lower() in NA and b.lower() in NA:
        return True
    if a.lower() in ('true', 'false') and b.lower() in ('true', 'false'):
        return a.lower() == b.lower()
    x, y = (plain_float(a), plain_float(b))
    if x is not None and y is not None:
        return abs(x - y) <= 1e-09 * max(1.0, abs(x), abs(y))
    if _DATE.fullmatch(a) and _DATE.fullmatch(b):
        return re.sub('[ T]00:00(:00)?$', '', a) == re.sub('[ T]00:00(:00)?$', '', b)
    return False

def norm_text(value):
    return re.sub('\\s+', ' ', str(value or '').strip()).casefold()

def score_prediction(cfg, root, emit):
    spec = cfg['prediction']
    names = ('predictions are in the requested file with its original rows, order and columns', 'every requested row has a valid prediction', 'holdout score reaches the required level')
    path = os.path.join(root, spec['deliverable'])
    try:
        template = head_version(root, spec['deliverable'])
        table = read_table(path) if os.path.isfile(path) else None
    except Exception as error:
        print('# could not read the deliverable:', repr(error)[:200])
        table = None
    if table is None:
        for name in names:
            emit(False, name)
        return
    target, id_col = (spec['target_column'], spec['id_column'])
    structure = list(table.columns) == list(template.columns) and len(table) == len(template)
    if structure:
        others = [c for c in template.columns if c != target]
        for col in others:
            if not all((same_value(x, y) for x, y in zip(table[col], template[col]))):
                print(f'# column changed: {col}')
                structure = False
                break
    emit(structure, names[0])
    if not structure:
        emit(False, names[1])
        emit(False, names[2])
        return
    ids = [str(v).strip() for v in template[id_col]]
    raw = [str(v).strip() for v in table[target]]
    if spec['problem'] == 'regression':
        pred = [parse_number(v) for v in raw]
        valid = all((v is not None for v in pred))
    else:
        vocabulary = set(spec['labels_allowed'])
        pred = raw
        valid = all((v in vocabulary for v in pred))
        if not valid:
            print('# labels outside the training labels:', sorted({v for v in pred if v not in vocabulary})[:5])
    emit(valid, names[1])
    if not valid:
        emit(False, names[2])
        return
    truth = [spec['labels'][i] for i in ids]
    metric = spec['metric']
    if metric == 'mae':
        value = sum((abs(p - t) for p, t in zip(pred, truth))) / len(truth)
        passed = value <= spec['threshold']
    elif metric == 'rmse':
        value = math.sqrt(sum(((p - t) ** 2 for p, t in zip(pred, truth))) / len(truth))
        passed = value <= spec['threshold']
    elif metric == 'accuracy':
        value = sum((p == t for p, t in zip(pred, truth))) / len(truth)
        passed = value >= spec['threshold']
    elif metric == 'macro_f1':
        f1s = []
        for c in sorted(set(truth)):
            tp = sum((p == c and t == c for p, t in zip(pred, truth)))
            fp = sum((p == c and t != c for p, t in zip(pred, truth)))
            fn = sum((p != c and t == c for p, t in zip(pred, truth)))
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        value = sum(f1s) / len(f1s)
        passed = value >= spec['threshold']
    else:
        raise ValueError('unknown metric')
    print(f'# holdout {metric} = {value:.6g}')
    emit(passed, names[2])

def answer_ok(q, given):
    if q['answer_type'] == 'number':
        value = parse_number(given)
        if value is None:
            return False
        reference = float(q['reference'])
        if q.get('exact_integer'):
            return abs(value - reference) < 1e-09
        tolerance = 0.5 * 10 ** (-int(q['decimals'])) + 1e-14 * max(1.0, abs(reference)) + 1e-12
        return abs(value - reference) <= tolerance
    accepted = {norm_text(x) for x in [q['reference'], *q.get('accepted', [])]}
    return norm_text(given) in accepted

def score_questions(cfg, root, emit):
    spec = cfg['questions']
    path = os.path.join(root, spec['deliverable'])
    answers = {}
    if os.path.isfile(path):
        try:
            table = read_table(path)
            cols = {c.strip().lower(): c for c in table.columns}
            idcol, acol, qcol = (cols.get('id'), cols.get('answer'), cols.get('question'))
            texts = {norm_text(q['text']): k for k, q in spec['by_id'].items()}
            if acol is not None:
                for _, row in table.iterrows():
                    key = str(row[idcol]).strip() if idcol is not None else None
                    if key not in spec['by_id'] and qcol is not None:
                        key = texts.get(norm_text(row[qcol]))
                    if key in spec['by_id']:
                        answers[key] = row[acol]
        except Exception as error:
            print('# could not read answers:', repr(error)[:200])
    for key in spec['order']:
        emit(answer_ok(spec['by_id'][key], answers.get(key)), f'answer {key} matches the reference')

def score_cleaning(cfg, root, emit):
    import pandas as pd
    spec = cfg['cleaning']
    names = ['columns are exactly the requested ones in the requested order', 'row count matches the deduplicated export', 'key column is unique', 'no invented or altered keys', 'dates use YYYY-MM-DD and match the source', 'numeric columns are plain numbers matching the source', 'category labels are standardised', 'preserved columns are unchanged']
    path = os.path.join(root, spec['deliverable'])
    table = None
    if os.path.isfile(path):
        try:
            table = read_table(path)
        except Exception as error:
            print('# could not read cleaned file:', repr(error)[:200])
    emit(table is not None, 'cleaned file exists')
    columns_ok = table is not None and list(table.columns) == list(spec['columns'])
    emit(columns_ok, names[0])
    if not columns_ok:
        for name in names[1:]:
            emit(False, name)
    else:
        ref = pd.DataFrame(spec['reference_rows'], columns=spec['columns']).astype(str)
        key = spec['key']
        emit(len(table) == len(ref), names[1])
        keys = table[key].astype(str)
        emit(keys.is_unique, names[2])
        emit(set(keys) == set(ref[key]), names[3])
        got = table.assign(_k=keys).drop_duplicates('_k').set_index('_k')
        want = ref.set_index(key)
        common = [k for k in want.index if k in got.index]
        ok = bool(common)
        for col in spec['date_columns']:
            g = got.loc[common, col].astype(str)
            w = want.loc[common, col]
            ok = ok and bool(g.str.fullmatch('\\d{4}-\\d{2}-\\d{2}|').all()) and bool((g == w).all())
        emit(ok, names[4])
        ok = bool(common)
        for col in spec['numeric_columns']:
            whole = col in spec.get('integer_columns', [])
            for g, w in zip(got.loc[common, col].astype(str), want.loc[common, col]):
                if w == '':
                    if g.lower() not in NA:
                        ok = False
                        break
                    continue
                if whole and (not re.fullmatch('-?\\d+', g)) or not re.fullmatch('-?(\\d+\\.?\\d*|\\.\\d+)([eE][-+]?\\d+)?', g):
                    ok = False
                    break
                gv = plain_float(g)
                if gv is None or abs(gv - float(w)) > 1e-06 * max(1.0, abs(float(w))):
                    ok = False
                    break
            if not ok:
                break
        emit(ok, names[5])
        ok = bool(common)
        for col in spec['category_columns']:
            g = got.loc[common, col].astype(str)
            ok = ok and bool((g == want.loc[common, col]).all())
        emit(ok, names[6])
        ok = bool(common)
        for col in spec['preserved_columns']:
            if col == key:
                continue
            ok = ok and bool((got.loc[common, col].astype(str) == want.loc[common, col]).all())
            if not ok:
                print(f'# preserved column changed: {col}')
                break
        emit(ok, names[7])
    summary = find_named(root, spec['summary'], spec.get('summary_aliases', []))
    emit(summary is not None and os.path.getsize(summary) > 40, 'summary note written')

def main(cfg):
    root = cfg['workdir']
    tests = []

    def emit(ok, name):
        tests.append((bool(ok), name))
    print(START)
    try:
        if cfg['type'] == 'prediction':
            score_prediction(cfg, root, emit)
        elif cfg['type'] == 'questions':
            score_questions(cfg, root, emit)
        elif cfg['type'] == 'cleaning':
            score_cleaning(cfg, root, emit)
    except Exception as error:
        print('# scorer error:', repr(error)[:300])
    for relative, digest in sorted(cfg['raw_files'].items()):
        full = os.path.join(root, relative)
        emit(os.path.isfile(full) and sha256_file(full) == digest, 'raw file unchanged: ' + relative)
    for number, (ok, name) in enumerate(tests, 1):
        print(('ok' if ok else 'not ok') + f' {number} {name}')
    print(END)
if __name__ == '__main__':
    main(json.loads(sys.stdin.read()))
