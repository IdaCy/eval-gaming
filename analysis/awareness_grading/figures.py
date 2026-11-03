"""Render the five awareness and grading figures from cleaned numeric inputs."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = '#286b8e', '#bc683b', '#b6b6b6'
CUES = {
    'F16L03':'This is an evaluation', 'F23L01':'Stakes: decides deployment',
    'F15L01':'Invented benchmark name', 'F22L01':'Plain benchmark label',
    'F19L01':'SWT-Bench: credits tests', 'F11L02':'SWE-bench: real benchmark name',
    'F18L03':'Added tests not counted', 'F18L04':'Added tests credited',
    'F20L01':'Request for tests', 'F24L04':'Real name + request for tests'}
MODELS = ['DeepSeek','Qwen','gpt-oss','Nemotron','Sol','Luna','Opus','o3','GPT-5.5']


def save(fig, output, name):
    fig.savefig(output/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(output/(name+'.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def awareness(data, output):
    levels = pd.read_csv(data/'belief_judge_levels.csv')
    auc = pd.read_csv(data/'auc_original_aggregates.csv')
    models = ['DeepSeek','Sol','Luna','o3','GPT-5.5']
    fig, axes = plt.subplots(1, 2, figsize=(10.8,4.3), sharey=True)
    used = []
    for i, model in enumerate(models):
        baseline = levels[(levels.agent==model)&(levels.code=='B00L00')&(levels.measure=='rho')].iloc[0]
        axes[0].barh(i,100*baseline.value,height=.65,color=GREY,alpha=.65)
        for code, shift, color in [('F16L03',-.17,BLUE),('F18L03',.17,ORANGE)]:
            a=levels[(levels.agent==model)&(levels.code==code)&(levels.measure=='rho')].iloc[0]
            b=auc[(auc.agent==model)&(auc.code==code)&(auc.features=='all')].iloc[0]
            axes[0].plot([0,100*a.value],[i+shift]*2,color=color,lw=4)
            axes[0].text(100*a.value+1,i+shift,f'{100*a.value:.0f}',va='center',fontsize=8)
            axes[1].plot([b.lo,b.hi],[i+shift]*2,color=color)
            axes[1].plot(b.auc,i+shift,'o',color=color)
            axes[1].text(b.auc,i+shift-.10,f'{b.auc:.2f}',ha='center',fontsize=8)
            used.append(dict(agent=model,code=code,probability=100*a.value,baseline_probability=100*baseline.value,auc=b.auc,lo=b.lo,hi=b.hi))
    axes[0].set(yticks=range(len(models)),yticklabels=models,xlim=(0,110),xlabel='Stated probability of evaluation (%)')
    axes[0].invert_yaxis()
    axes[1].axvline(.5,color=GREY,ls='--')
    axes[1].set(xlim=(.2,1.03),xlabel='AUC from actions')
    fig.legend([plt.Line2D([],[],color=BLUE,lw=3),plt.Line2D([],[],color=ORANGE,lw=3),plt.Line2D([],[],color=GREY,lw=6)],
               ['Evaluation statement','Tests not graded','No cue'],loc='lower center',ncol=3,bbox_to_anchor=(.5,-.04))
    fig.tight_layout()
    pd.DataFrame(used).to_csv(output/'figure4_values.csv',index=False)
    save(fig,output,'fig_awareness_two_sentences')


def auc_grid(data, output):
    frame=pd.read_csv(data/'auc_original_aggregates.csv').query('features == "all"')
    fig,axes=plt.subplots(len(CUES),len(MODELS),figsize=(13,8),sharex=True,sharey=True)
    used=[]
    for i,(code,label) in enumerate(CUES.items()):
        for j,model in enumerate(MODELS):
            ax=axes[i,j];x=frame[(frame.agent==model)&(frame.code==code)]
            ax.set_xlim(.3,1.02);ax.set_ylim(-.5,.5);ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if i==0:ax.set_title('gpt-oss-120b' if model=='gpt-oss' else model,fontsize=9)
            if j==0:ax.text(-.06,.5,label,ha='right',va='center',transform=ax.transAxes,fontsize=9)
            if len(x):
                r=x.iloc[0];ax.axvline(.5,color=GREY,lw=.7)
                ax.barh(0,r.auc-.5,left=.5,height=.45,color=BLUE)
                ax.plot([r.lo,r.hi],[0,0],color='black',lw=1)
                ax.text(.67,.28,f'{r.auc:.2f}',ha='center',fontsize=8)
                used.append(r.to_dict())
            else:ax.set_facecolor('#eeeeee')
    fig.subplots_adjust(left=.29,wspace=.2,hspace=.25,bottom=.05)
    fig.text(.5,.01,'AUC from actions; lines are 95% issue-resampling intervals.',ha='center',fontsize=9)
    pd.DataFrame(used).to_csv(output/'figure5_values.csv',index=False)
    save(fig,output,'fig_auc_by_cue_and_agent')


def reports(data, results, output):
    frame=pd.read_csv(results/'report_ratios.csv')
    order=['GPT-4o','GPT-4.1 mini','GPT-4.1','o3','GPT-5 nano','GPT-5 mini','GPT-5','GPT-5.1','GPT-5.2','GPT-5.3 C',
           'GPT-5.4 nano','GPT-5.4 mini','GPT-5.4','GPT-5.5','GPT-5.6 Luna','GPT-5.6 Terra','GPT-5.6 Sol',
           'GPT-6 Sol','GPT-6.1 Terra','GPT-6 Luna','GPT-6.1 Sol','DeepSeek']
    order=[m for m in order if m in set(frame.agent)]
    fig,ax=plt.subplots(figsize=(9,8))
    for code,shift,color,label in [('F25L01',-.2,ORANGE,'Values graded'),('F16L03',0,GREY,'Evaluation statement'),('F25L02',.2,BLUE,'Explanation graded')]:
        x=frame[frame.code.eq(code)].set_index('agent').reindex(order)
        ax.errorbar(x.ratio,np.arange(len(order))+shift,xerr=[x.ratio-x.normal_lo,x.normal_hi-x.ratio],fmt='o',ms=3,color=color,label=label)
    ax.axvline(1,color='black',lw=.7);ax.set_xscale('log')
    ax.set(yticks=range(len(order)),yticklabels=order,xlabel='Written report length / no-cue report length')
    ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,1.065),ncol=3);fig.tight_layout()
    frame.to_csv(output/'figure6_values.csv',index=False)
    save(fig,output,'fig_data_tasks_reports')


def data_reasoning(data, output):
    frame=pd.read_parquet(data/'data_reasoning_task_sums.parquet')
    labels={'B00L00':'No cue','F07L05':'Five benchmark headings','F08L03':'Working folder benchmarks/',
            'F09L03':'Shell mentions evaluator configuration','F11L04':'Benchmark presented in full',
            'F09L05':'Grading files in workspace','F12L02':'Answer key in workspace','F03L05':'Formal wording',
            'F04L02':'All periods removed','F05L05':'Earlier conversation, level 5',
            'F03L00':'Original wording, line endings only','F08L01':'Working folder projects/','F07L00':'Plain section headings'}
    fig,axes=plt.subplots(1,3,figsize=(12,6),sharey=True)
    used=[]
    for j,(outcome,title) in enumerate([('repeated','Recurring reasoning about checks (%)'),('aware','Evaluation plausible, rating ≥2 (%)'),('material','Judged materially different (%)')]):
        for i,(code,label) in enumerate(labels.items()):
            x=frame[(frame.cond==code)&(frame.outcome==outcome)].sort_values('task')
            n=x.n.to_numpy();total=x.total.to_numpy();value=100*total.sum()/n.sum()
            rng=np.random.default_rng(20261110);idx=rng.integers(0,len(x),size=(2000,len(x)))
            draws=100*total[idx].sum(1)/n[idx].sum(1);lo,hi=np.quantile(draws,[.025,.975])
            color=ORANGE if outcome=='material' else BLUE
            axes[j].errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=color,ms=4,capsize=2)
            axes[j].text(value+3,i,f'{value:.0f}',va='center',fontsize=8)
            if code=='B00L00':axes[j].axvline(value,color=GREY,ls='--')
            used.append(dict(code=code,outcome=outcome,value=value,lo=lo,hi=hi,runs=int(n.sum()),tasks=len(x)))
        axes[j].set(xlim=(-2,110),xlabel=title)
    axes[0].set(yticks=range(len(labels)),yticklabels=list(labels.values()));axes[0].invert_yaxis();fig.tight_layout()
    pd.DataFrame(used).to_csv(output/'figureA1_values.csv',index=False)
    save(fig,output,'fig_deepseek_data_reasoning')


def name_actions(data, results, output):
    frame=pd.read_csv(data/'action_original_aggregates.csv')
    additions=json.loads((results/'name_changes.json').read_text())
    models=['DeepSeek','Qwen','Sol','Opus','o3','GPT-5','GPT-5.6 Luna']
    measures={'O1':('Tests in patch','tests_in_patch'),'T2':('Test lines','test_lines'),'S1':('Source lines','source_lines'),
              'TK_reasoning':('Reasoning tokens','reasoning_tokens'),'TK_output':('Output tokens','output_tokens'),
              'RP':('Final message',None),'O2':('Searches for fix','fetch_attempts_runs')}
    fig,axes=plt.subplots(len(models),len(measures),figsize=(13,7))
    used=[]
    for i,model in enumerate(models):
        for j,(measure,(label,alt)) in enumerate(measures.items()):
            ax=axes[i,j];ax.set(xlim=(-100,40),ylim=(-.5,.5),xticks=[],yticks=[])
            for spine in ax.spines.values():spine.set_visible(False)
            if i==0:ax.set_title(label,fontsize=9)
            if j==0:ax.text(-.1,.5,model,ha='right',va='center',transform=ax.transAxes,fontsize=10)
            ax.axvline(0,color=GREY,lw=.7)
            if model in ['GPT-5','GPT-5.6 Luna']:
                key='openai-gpt-5-high' if model=='GPT-5' else 'openai-gpt-5.6-luna-high'
                cell=additions[key].get(alt) if alt else None
                if cell is None:ax.set_facecolor('#eeeeee');continue
                change=cell['change'][0];clear=cell['clear'];base=cell['no_cue'];cue=cell['name']
            else:
                sub=frame[(frame.agent==model)&(frame.code=='F11L02')&(frame.measure==measure)]
                if not len(sub):ax.set_facecolor('#eeeeee');continue
                r=sub.iloc[0];binary=r.kind=='share'
                change=100*r['diff'] if binary else 100*np.expm1(r['diff'])
                base=100*r.level_nocue if binary else np.expm1(r.level_nocue)
                cue=100*r.level_cue if binary else np.expm1(r.level_cue)
                clear=(r.lo>0 or r.hi<0) and abs(change)>=(5 if binary else 10)
                scale=1000 if measure.startswith('TK_') else 1
                digits=1 if scale==1000 else 0
                shown_base=round(base/scale,digits)
                shown_cue=round(cue/scale,digits)
                change=(shown_cue-shown_base) if binary else 100*(shown_cue/shown_base-1) if shown_base else change
            if clear:
                ax.barh(0,change,height=.35,color=ORANGE if change<0 else BLUE)
                unit='' if measure in ['O1','O2'] else '%'
                ax.text(-30,.28,f'{change:+.0f}{unit}',ha='center',fontsize=9)
                scale=1000 if measure.startswith('TK_') else 1
                fmt='.1f' if scale==1000 else '.0f'
                ax.text(-30,-.32,f'{base/scale:{fmt}} → {cue/scale:{fmt}}',ha='center',fontsize=8)
            used.append(dict(agent=model,measure=measure,change=change,baseline=base,cue=cue,clear=bool(clear)))
    fig.subplots_adjust(left=.14,hspace=.4,wspace=.12)
    pd.DataFrame(used).to_csv(output/'figureA2_values.csv',index=False)
    save(fig,output,'fig_what_changed_name')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--results',type=Path,default=OUTPUT)
    p.add_argument('--output',type=Path,default=FIGURES)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    awareness(args.data,args.output);auc_grid(args.data,args.output);reports(args.data,args.results,args.output)
    data_reasoning(args.data,args.output);name_actions(args.data,args.results,args.output)
