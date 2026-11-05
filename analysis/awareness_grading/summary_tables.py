"""Format the awareness and grading tables and collect numerical claims."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def load(path):
    return json.loads(path.read_text())


def action_tables(data):
    frame=pd.read_csv(data/'action_original_aggregates.csv')
    measures=['O1','T2','T3','S1','n_tool_calls','TK_reasoning','TK_output','RP','O2','O3']
    rows=[]
    for code,models in [('F11L02',['DeepSeek','Qwen','Sol','Luna','Opus','o3','GPT-5.5']),
                        ('F18L03',['DeepSeek','Qwen','gpt-oss','Nemotron','Sol','Luna','Opus','o3','GPT-5.5'])]:
        for model in models:
            for measure in measures:
                x=frame[(frame.agent==model)&(frame.code==code)&(frame.measure==measure)]
                row=dict(table='A9' if code=='F11L02' else 'A10',agent=model,measure=measure)
                if not len(x):
                    rows.append(dict(**row,display='–',clear=False));continue
                r=x.iloc[0];share=r.kind=='share'
                change=100*r['diff'] if share else 100*np.expm1(r['diff'])
                clear=(r.lo>0 or r.hi<0) and abs(change)>=(5 if share else 10)
                a=100*r.level_nocue if share else np.expm1(r.level_nocue)
                b=100*r.level_cue if share else np.expm1(r.level_cue)
                scale=1000 if measure.startswith('TK_') else 1
                a,b=a/scale,b/scale
                def fmt(v):
                    return f'{v:.1f}' if scale==1000 or not share and v<10 else f'{v:.0f}'
                rows.append(dict(**row,display=f'{fmt(a)} → {fmt(b)}' if clear else '·',clear=bool(clear),
                                 baseline=a,cue=b,change=change,lo=r.lo,hi=r.hi))
    return pd.DataFrame(rows)


def main(data, results):
    pooled=load(results/'pooled_summary.json')
    general=load(results/'general_judges.json')
    actions=load(results/'action_counts.json')
    tests=load(results/'grading_tests.json')['agents']
    awareness=load(results/'awareness_tests.json')
    final=load(results/'final_messages.json')
    levels=pd.read_csv(data/'belief_judge_levels.csv')
    auc=pd.read_csv(data/'auc_original_aggregates.csv')
    output=dict(bounds=pooled['bounds'],awareness_sentences=pooled['awareness_sentences'],
                awareness_tests=awareness['all agents:F16L03'],
                awareness_output_tokens=actions['F16L03|all|output_tokens'],
                agreement=load(results/'agreement.json'),reports=load(results/'report_counts.json'))
    output['evaluation_verdicts']=pooled['evaluation_verdicts']
    selected={}
    for key,value in tests.items():
        model=key.split(' [')[0]
        if model not in selected or value['queue']=='b1':selected[model]=value
    selected={k:v for k,v in selected.items() if v['nocue_tests']>=40 and 'F18L03' in v}
    groups={'at_least_20':[],'12_to_14':[],'6':[],'not_clear':[]}
    for model,value in selected.items():
        change,lo,hi=value['F18L03']['change'];rounded=round(-change)
        if hi>=0:group='not_clear'
        elif rounded>=20:group='at_least_20'
        elif 12<=rounded<=14:group='12_to_14'
        elif rounded==6:group='6'
        else:group='other'
        groups.setdefault(group,[]).append(model)
    tokens=actions['F18L03|tests>=40%|output_tokens']
    token_models={k.split(' [')[0] for k,v in tokens['per_agent'].items() if v[2]<0}
    output['tests_17']=dict(models=len(selected),groups=groups,tokens_lower=len(token_models),
                            tokens_models=sorted(token_models),tokens_pooled=tokens['pooled'],
                            independent_model_batches=tokens['agents'])
    output['final_message_cohort']={}
    for cue in ['P6D01','P6D02']:
        cells=[v[cue]['final']['change'] for v in final['cells'].values() if cue in v and 'final' in v[cue]]
        output['final_message_cohort'][cue]=dict(models=len(cells),change_range=[min(cells),max(cells)],counts=final['counts'][cue])
    grading=auc[(auc.code=='F18L03')&~auc.agent.eq('gpt-oss')]
    output['grading_auc_range']={f:[float(g.auc.min()),float(g.auc.max())] for f,g in grading.groupby('features')}
    name=auc[(auc.code=='F11L02')&auc.features.eq('all')]
    output['name_auc']=dict(zip(name.agent,name.auc))
    cross=pd.read_csv(data/'all_model_grading_auc.csv')
    output['all_model_grading_auc']=dict(models=len(cross),above_chance=int(cross.above_chance.sum()))
    output['stated_evaluation_probability']=[dict(agent=r.agent,code=r.code,pct=100*r.value) for r in levels[levels.measure.eq('rho')&levels.agent.isin(['DeepSeek','Sol','Luna'])&levels.code.isin(['B00L00','F16L03','F17L01'])].itertuples()]
    nest=pd.read_csv(data/'judge_awareness_counts.csv')
    output['judged_awareness_nesting']=dict(low_belief_runs=int(nest.loc[nest.belief.le(1),'n'].sum()),
          low_belief_material=int(nest.loc[nest.belief.le(1)&nest.G.eq(1),'n'].sum()),
          material_rated_aware_pct=100*float(nest.loc[nest.G.eq(1)&nest.belief.ge(3),'n'].sum()/nest.loc[nest.G.eq(1),'n'].sum()))
    blind=pd.read_csv(data/'first_experiment_blind_counts.csv')
    output['first_experiment_actions_judge']=dict(positive=int(blind.positive.sum()),runs=int(blind.runs.sum()))
    output['opus_evaluation_counts']=pd.read_csv(data/'opus_evaluation_counts.csv').to_dict('records')
    output['general_heading_changes']=general['DeepSeek']['headings']
    output['stakes_judged_change']=pd.read_csv(data/'measure_rank_agreement.csv').query("code == 'F23L01'")[['d_J','d_J_lo','d_J_hi']].to_dict('records')
    rb=levels[levels.agent.eq('DeepSeek')&levels.code.eq('F18L03')&levels.measure.eq('judge_B')].iloc[0]
    output['actions_judge_notgraded']=dict(positive=round(rb.value*rb.runs),runs=int(rb.runs))
    output['conditioned_judges']=load(results/'conditioned_judges.json')
    corr=pd.read_csv(data/'pass_judged_contrasts.csv')
    output['score_judge_correlation']=float(corr[['pass_diff','judged_diff']].corr().iloc[0,1])
    verdicts=load(results/'test_verdicts.json')
    output['test_verdicts']=verdicts
    pair=pd.read_csv(results/'pair_verdicts.csv')
    output['pair_verdict_columns']=list(pair)
    output['tableA6']={m:{c:dict(tasks=v['tasks'],runs=v['runs'],p0=int(np.floor(round(v['p0'],1)+.5)),p1=int(np.floor(round(v['p1'],1)+.5))) if v.get('value') is not None else dict(tasks=v['tasks'],value=None) for c,v in x['A'].items()} for m,x in general.items() if m!='Qwen'}
    output['tableA7']={m:{c:dict(tasks=v['tasks'],runs=v.get('runs'),difference=[round(v['value'],1),*[round(t,1) for t in v['ci']]]) if v.get('value') is not None else dict(tasks=v['tasks'],value=None) for c,v in x['G'].items()} for m,x in general.items()}
    table8=levels[levels.measure.eq('judge_R')].copy()
    for r in pooled['judged_levels']:
        ix=(table8.agent==r['agent'])&(table8.code==r['code'])
        if ix.any():table8.loc[ix,['value','runs','tasks']]=[r['rate']/100,r['runs'],r['issues']]
    corrected=pd.read_csv(data/'corrected_judge_levels.csv')
    for r in corrected[corrected.code.isin(['F18L03','F18L04'])].itertuples():
        ix=(table8.agent=='gpt-oss')&(table8.code==r.code)
        table8.loc[ix,['value','runs','tasks']]=[r.value,r.runs,r.tasks]
    table8['percent']=100*table8.value
    table8=table8.drop(columns=['lo','hi'],errors='ignore')
    shown={
        'DeepSeek':['B00L00','F16L03','F17L01','F23L01','F15L01','F22L01','F19L01','F11L02','F18L03','F18L01','F18L04','F18L02','F20L01','F24L04','F24L05','F24L03'],
        'Qwen':['B00L00','F15L01','F19L01','F11L02','F18L03','F18L04','F20L01','F24L04','F24L03'],
        'gpt-oss':['B00L00','F18L03','F18L04'],
        'Nemotron':['B00L00','F18L03','F18L04','F18L02'],
        'Opus':['B00L00','F19L01','F11L02','F18L03','F18L04']}
    table8=table8[[r.code in shown.get(r.agent,[]) for r in table8.itertuples()]]
    table8.to_csv(results/'tableA8_levels.csv',index=False)
    action_tables(data).to_csv(results/'tablesA9_A10.csv',index=False)
    contrasts=pd.read_csv(results/'pooled_differences.csv')
    table11=[]
    for model in ['DeepSeek','Qwen','GLM']:
        for code,kind in [('F11L02','name'),('F11L04','full')]:
            issues='heldout' if kind=='full' and model in ['DeepSeek','GLM'] else 'main'
            row=contrasts.query('agent == @model and code == @code and issues == @issues and outcome == "J"').iloc[0]
            table11.append(dict(agent=model,condition=kind,issues=issues,value=row.pooled,lo=row.pooled_lo,hi=row.pooled_hi))
    for row in pd.read_csv(data/'rule_judged_aggregates.csv').itertuples():
        table11.append(dict(agent=row.agent,condition='rule',issues='main',value=row.diff,lo=row.lo,hi=row.hi))
    pd.DataFrame(table11).to_csv(results/'tableA11_judged.csv',index=False)
    additional=pd.read_csv(data/'registered_action_contrasts.csv')
    for column in ['rd_pp','rd_lo','rd_hi']:
        ix=additional.scale.eq('log1p units')
        additional.loc[ix,column]=100*np.expm1(additional.loc[ix,column])
    additional.loc[additional.scale.eq('log1p units'),'scale']='percent change'
    output['registered_action_contrasts']=additional.to_dict('records')
    (results/'paper_numbers.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--results',type=Path,default=OUTPUT)
    args=p.parse_args();main(args.data,args.results)
