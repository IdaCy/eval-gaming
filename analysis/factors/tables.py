"""Collect the appendix tables and the variance accounting from computed results."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['llm2_ambiguity','llm2_information_missing','llm2_needs_upstream_context','llm2_expected_behaviour_clear','llm2_fix_obvious_from_issue','llm2_fix_difficulty','fix_log_lines_changed','fix_files','tp_test_funcs_added','llm2_test_difficulty','llm2_needs_test_infrastructure','llm2_behaviour_testable','llm2_test_location_obvious','orig_line_chars_mean','orig_hex_like','orig_checkboxes','orig_version_numbers','orig_stack_lines','task_age_days','tree_log_bytes','repo_tasks_in_corpus']

def read(root,name):return json.loads((root/name).read_text())
def interval(m,point='rd_pp',lo='rd_lo',hi='rd_hi'):
    return {'estimate':m[point],'lo':m[lo],'hi':m[hi]}

def collect(results,out):
    out.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(results/'task_features_screen_v2.csv')
    f=f[f.feature.isin(FEATURES)].copy();f['order']=f.feature.map({k:i for i,k in enumerate(FEATURES)})
    f['lo']=f.b_per_sd-1.96*f.se;f['hi']=f.b_per_sd+1.96*f.se;f['n_mark']=(f.q>=.1)|~f.same_sign
    f=f.sort_values(['order','outcome'])
    f[['feature','outcome','b_per_sd','lo','hi','q','same_sign','n_mark','tasks']].to_csv(out/'table_A16.csv',index=False)
    rows=[]
    for design in ['C2','C3','C4']:
        for outcome in ['judged','notests']:
            records=read(results,'validation/'+design+'_'+outcome+'.json')
            r=next(r for r in records if r.get('target','all')=='all')
            key='frac_transfer' if design=='C4' else 'frac_roles'
            rows.append({'test':design,'outcome':outcome,'estimate':100*r[key],'lo':100*r[key+'_ci'][0],'hi':100*r[key+'_ci'][1]})
    for outcome in ['G_rmask','G_tests']:
        records=read(results,'transfer/'+outcome+'.json')
        for r in records:
            if r['agent']=='DeepSeek' and r['block']=='main':
                t=r['real_minus_invented'];rows.append({'test':'new_tasks_real_minus_invented','outcome':outcome,'estimate':t['obs_pp'],'lo':t['obs_lo'],'hi':t['obs_hi'],'predicted':t['pred_pp']})
        if outcome=='G_rmask':
            small=[c for r in records for c in r['cells'] if c['condition'] in ['E01','F07L05','F11L01','F16L03']]
            rows.append({'test':'new_tasks_small_changes_range','outcome':outcome,'observed_min':min(c['obs_diff_pp'] for c in small),'observed_max':max(c['obs_diff_pp'] for c in small),'predicted_min':min(c['pred_diff_pp'] for c in small),'predicted_max':max(c['pred_diff_pp'] for c in small)})
    for r in read(results,'new_cues.json'):
        for cell,c in r['cells'].items():
            if cell!='C1':rows.append({'test':'new_cues','outcome':'judged','model':r['agent'],'condition':cell,'estimate':c['obs_diff'],'lo':c['obs_diff_lo'],'hi':c['obs_diff_hi'],'predicted':r['before_run_differences'][cell],'predicted_after_calibration':c['diff']})
        rows.append({'test':'new_cues_gap_closed','outcome':'judged','model':r['agent'],'estimate':r['share_gap_closed'],'lo':r['share_lo'],'hi':r['share_hi']})
    pd.DataFrame(rows).to_csv(out/'table_A15.csv',index=False)
    additional=read(results,'additional_factors.json');summary=read(results,'task_feature_summary.json');training=read(results,'training.json');tr=read(results,'transcript_factors.json')
    rows=[]
    for r in read(results,'age_slopes.json'):rows.append({'factor':'age','model':'DeepSeek','outcome':r['outcome'],'estimate':r['slope'],'lo':r['lo'],'hi':r['hi'],'tasks':r['tasks']})
    for r in additional['task_groups']:rows.append({'factor':'task_group','model':'DeepSeek','estimate':r['rate'],**r})
    rows.append({'factor':'recent_task_name_effect','model':'DeepSeek','outcome':'notests',**interval(additional['recent_issues'])})
    for r in training['checkpoint_effects']:
        if (r['cue']=='F11L03' and r['lineage']=='all') or (r['cue']=='F09L05' and r['lineage']=='tmax9b'):rows.append({'factor':'checkpoint_effect','estimate':r['effect'],**r})
    for key in ['training_start','training_peak','strict_evaluation_peak']:rows.append({'factor':key,'estimate':training[key]})
    for r in training['hidden_test_beliefs']:rows.append({'factor':'hidden_test_belief','estimate':100*r['rate'],**r})
    for r in tr['first_evaluation_trigger']:rows.append({'factor':'evaluation_trigger','estimate':r['percent'],**r})
    for r in tr['memory']:rows.append({'factor':'memory','estimate':r['percent'],**r})
    pd.DataFrame(rows).to_csv(out/'table_A17.csv',index=False)
    rows=[]
    for r in pd.read_csv(results/'run_level_screen.csv').to_dict('records'):rows.append({'factor':r['factor'],'estimate':100*r['diff'],'lo':100*r['lo'],'hi':100*r['hi']})
    rows.append({'factor':'whole_issue_rewording',**interval(additional['whole_issue_rewording'])})
    rows.append({'factor':'small_rewordings',**additional['small_rewordings']})
    rows.append({'factor':'commit_date',**interval(additional['commit_date'],lo='normal_lo',hi='normal_hi')})
    rows.append({'factor':'recall',**interval(additional['recall'],point='rho',lo='lo',hi='hi')})
    rows.append({'factor':'prediction_all_features','estimate':100*summary['judged']['out_of_repo_R2_all_features']})
    rating_r2=summary['judged']['by_family[n,corr,R2]']['task: rating by rater models'][2]
    rows.append({'factor':'prediction_ratings','estimate':100*rating_r2})
    variance=read(results,'variance.json');v=next(r for r in variance if r['scope']=='small_changes')
    rows.append({'factor':'small_changes_variance','estimate':v['shares_pct']['condition'],'lo':v['condition_share_ci_pct'][0],'hi':v['condition_share_ci_pct'][1]})
    pd.DataFrame(rows).to_csv(out/'table_A18.csv',index=False)
    s=next(r for r in variance if r['scope']=='pooled')['shares_pct'];task=s['task'];identified_setup=s['condition']+s['agent_condition'];identified_task=task*rating_r2
    total=identified_setup+identified_task
    accounting={'model':s['agent'],'task':task,'setup':s['condition']+s['agent_condition']+s['task_condition'],'setup_common':s['condition'],'setup_by_model':s['agent_condition'],'setup_by_task':s['task_condition'],'identified_setup':identified_setup,'unidentified_setup':s['task_condition'],'ratings':identified_task,'other_task':task-identified_task,'identified_total':total,'unidentified_total':100-total,'ratings_R2_used':rating_r2}
    (out/'variance_accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=OUTPUT);p.add_argument('--output',type=Path,default=OUTPUT / 'tables');a=p.parse_args();collect(a.results,a.output)
