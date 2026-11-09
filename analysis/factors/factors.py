"""Input element counts for the experimental setup conditions."""
BITS = {'sys_eval_statement': None, 'ws_evaluations': None, 'ws_benchmarks': None, 'ws_tasks': None, 'ws_sandbox': None, 'ws_projects': None, 'bench_suffix_heading': None, 'bench_word_headings': None, 'eval_config_file': None, 'eval_config_locked_note': None, 'shell_announces_eval': None, 'grader_config_locked': None, 'ps_heading': None, 'bname_request': None, 'bname_file': None, 'bname_path': None, 'bexamples_file': None, 'grading_files': None, 'answer_key': None, 'answer_key_in_workspace': None, 'check_disabled': None, 'headings6': None, 'typo_level': None, 'lowercase_level': None, 'punct_level': None, 'rewrite_any': None, 'casual_opening': None, 'formality_level': None, 'hedge_level': None, 'hedge_permission_stop': None, 'prior_turns': None, 'conv_clarification': None, 'conv_correct_describe': None, 'conv_correct_scope': None, 'work_notes_n': None, 'note_checks': None}
F07_WORDS = {2: 1, 3: 2, 4: 3, 5: 5}

def code_condition(cond, domain='swe'):
    b = {k: 0 for k in BITS}
    if cond in ('B00L00', 'nan', '', None):
        return b
    if cond == 'E01':
        b['sys_eval_statement'] = 1
        return b
    fam, lev = (cond[:3], int(cond[4:6]))
    if fam == 'F01':
        b['typo_level'] = lev
    elif fam == 'F02':
        b['lowercase_level'] = lev
    elif fam == 'F03':
        if lev >= 1:
            b['rewrite_any'] = 1
            b['formality_level'] = lev
            b['casual_opening'] = int(lev == 1)
    elif fam == 'F04':
        b['punct_level'] = lev
    elif fam == 'F05':
        b['prior_turns'] = 1
        b['conv_clarification'] = int(lev == 3)
        b['conv_correct_describe'] = int(lev == 4)
        b['conv_correct_scope'] = int(lev == 5)
    elif fam == 'F06':
        b['hedge_level'] = lev
        b['hedge_permission_stop'] = int(lev == 5)
    elif fam == 'F07':
        b['headings6'] = 1
        if lev >= 1:
            b['bench_suffix_heading'] = 1
        b['bench_word_headings'] = F07_WORDS.get(lev, 0)
    elif fam == 'F08':
        b[{1: 'ws_projects', 2: 'ws_tasks', 3: 'ws_benchmarks', 4: 'ws_evaluations', 5: 'ws_sandbox'}[lev]] = 1
    elif fam == 'F09':
        if lev in (1, 2, 3, 4):
            b['eval_config_file'] = 1
        b['eval_config_locked_note'] = int(lev == 1)
        b['shell_announces_eval'] = int(lev in (3, 4))
        b['grader_config_locked'] = int(lev == 4)
        b['grading_files'] = int(lev == 5)
    elif fam == 'F10':
        b['work_notes_n'] = lev
        b['note_checks'] = int(lev >= 3)
    elif fam == 'F11':
        b['ps_heading'] = 1
        b['bname_request'] = int(lev >= 2)
        b['bname_file'] = int(lev >= 3)
        b['bname_path'] = int(lev >= 4)
        b['ws_benchmarks'] = int(lev >= 4)
        b['bexamples_file'] = int(lev >= 5)
    elif fam == 'F12':
        b['answer_key'] = 1
        b['answer_key_in_workspace'] = int(lev == 2)
    elif fam == 'F13':
        b['check_disabled'] = 1
    else:
        raise ValueError(cond)
    return b
ROLE_GROUPS = {'occasion': ['sys_eval_statement', 'ws_evaluations', 'ws_benchmarks', 'ws_tasks', 'ws_sandbox', 'bench_suffix_heading', 'bench_word_headings', 'eval_config_file', 'eval_config_locked_note', 'shell_announces_eval', 'grader_config_locked'], 'identity': ['ps_heading', 'bname_request', 'bname_file', 'bname_path', 'bexamples_file'], 'protocol': ['grading_files'], 'access_key': ['answer_key', 'answer_key_in_workspace'], 'access_check': ['check_disabled'], 'realism': ['typo_level', 'lowercase_level', 'punct_level', 'rewrite_any', 'casual_opening', 'formality_level', 'hedge_level'], 'conversation': ['prior_turns', 'conv_clarification'], 'instruction': ['headings6', 'hedge_permission_stop', 'conv_correct_describe', 'conv_correct_scope', 'note_checks'], 'continuity': ['work_notes_n'], 'control': ['ws_projects']}
