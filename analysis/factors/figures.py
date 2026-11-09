"""Plot the fitted input weights, variance shares, and task-age rates."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import os
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(OUTPUT / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

TASK_LABELS={'llm2_ambiguity':'the issue is ambiguous','llm2_feature_request':'it is a feature request','llm2_fix_scope':'the fix is wide in scope','llm2_fix_difficulty':'the fix is difficult','llm2_test_difficulty':'a test is hard to write','llm2_information_missing':'information is missing','llm2_needs_test_infrastructure':'test infrastructure is needed','llm2_needs_upstream_context':'needs context from the upstream project','llm2_test_location_obvious':'the place for a test is obvious','llm2_domain_specialised':'needs specialised knowledge','llm2_version_or_regression':'it is about a version or a regression','llm2_reproduction_given':'steps to reproduce are given','llm2_fix_obvious_from_issue':'the fix is obvious from the issue','llm2_expected_behaviour_clear':'the expected behaviour is clear','llm2_behaviour_testable':'the behaviour is testable','llm2_known_upstream_feel':'reads like a known upstream issue'}
SETUP_LABELS={'id_known':'name of a benchmark the model can know','protocol':'grading files in the workspace','occasion':'signals of an evaluation','id_harness':'harness or folder signals','instruction':'section headings','control':'controls','id_format':"the benchmark's format",'realism':'realism of the request','continuity':'work notes','conversation':'earlier conversation'}

def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/(name+'.png'),bbox_inches='tight',dpi=160)
    plt.close(fig)

def input_factors(results,out):
    w=pd.read_csv(results/'formula/final/final_weights_G_rmask_category.csv')
    f=json.loads((results/'formula/final/final_G_rmask.json').read_text())['versions']['category']
    rows=[]
    for kind,labels,color in [('setup',SETUP_LABELS,'#cc7a29'),('task',TASK_LABELS,'#287b55')]:
        d=w[w.input.isin(labels)].sort_values('weight',ascending=False)
        for r in d.itertuples():rows.append((labels[r.input],r.weight,r.lo_two_way,r.hi_two_way,color,kind))
    a=f['alpha'];base=a['DeepSeek|deepinfra']
    for model in ['Qwen','GLM']:rows.append((f'{model} relative to DeepSeek',a[model+'|one']-base,np.nan,np.nan,'#7953a3','model'))
    fig,ax=plt.subplots(figsize=(9.0,9.4));ys=np.arange(len(rows))[::-1]
    for y,(label,pt,lo,hi,color,kind) in zip(ys,rows):
        if np.isfinite(lo):ax.plot([lo,hi],[y,y],color=color,lw=1.6)
        ax.scatter(pt,y,s=27,c=color,zorder=3)
    ax.axvline(0,color='.65',lw=.8,zorder=0)
    ax.set_yticks(ys,[r[0] for r in rows],fontsize=9)
    ax.set_xlabel('Change in judged gaming per factor step (log-odds)')
    ax.set_xlim(-1.0,2.45);ax.grid(axis='x',alpha=.15)
    ax.legend(handles=[Line2D([0],[0],marker='o',color=c,label=l,lw=1.5) for c,l in [('#cc7a29','Environment'),('#287b55','Task rating, per SD'),('#7953a3','Model')]],loc='lower right',frameon=False,fontsize=9)
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.tight_layout();save(fig,out,'fig_input_factors')
    pd.DataFrame(rows,columns=['label','point','lo','hi','color','kind']).drop(columns='color').to_csv(out/'input_factors.csv',index=False)

def variance(results,out):
    rec={r['scope']:r for r in json.loads((results/'variance.json').read_text())}
    scopes=['pooled','DeepSeek','GLM','Qwen','gpt-oss-120b']
    labels=['The four models together','DeepSeek V4.1 Flash','GLM-5.3-Flash','Qwen3.8-27B','gpt-oss-120b']
    parts=['agent','task','condition','agent_condition','task_condition']
    colors=['#7953a3','#287b55','#d38b34','#efbd7b','#f6dcba']
    legend=['Which model','Which task','Setup, common','Setup, by model','Setup, by task']
    fig,ax=plt.subplots(figsize=(10,3.5));left=np.zeros(len(scopes));ys=np.arange(len(scopes))[::-1];records=[]
    for part,color,name in zip(parts,colors,legend):
        vals=np.array([rec[s]['shares_pct'].get(part,0) for s in scopes])
        ax.barh(ys,vals,left=left,color=color,label=name,height=.62)
        for y,l,v,s in zip(ys,left,vals,scopes):
            if v>=7:ax.text(l+v/2,y,f'{v:.0f}%',ha='center',va='center',fontsize=10,color='white' if part in ['agent','task'] else '#30251b')
            records.append({'scope':s,'component':part,'percent':float(v)})
        left+=vals
    ax.set_yticks(ys,labels);ax.set_xlim(0,100);ax.set_xlabel('Share of differences in judged gaming (%)')
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout();save(fig,out,'fig_differences_per_agent');pd.DataFrame(records).to_csv(out/'variance_shares.csv',index=False)

def age(results,data,out):
    f=pd.read_csv(results/'age_bins.csv');bins=pd.read_parquet(data/'age_bins.parquet').sort_values('label')
    labels=['2022','2023','2024\nJan–Jun','2024\nJul–Dec','2025\nJan–Jun','2025\nJul–Dec','2026\nJan–Mar','2026\nApr–Jun']
    fig,ax=plt.subplots(figsize=(9.5,4.4))
    slopes={r['outcome']:r for r in json.loads((results/'age_slopes.json').read_text())}
    for outcome,color,offset,name in [('O9','#287b55',-.06,'Names the benchmark'),('O2a','#cc7a29',.06,'Attempts to fetch the fix')]:
        d=f[f.outcome.eq(outcome)].set_index('age_bin').reindex(bins.label);x=np.arange(len(d))+offset
        ax.errorbar(x,d.rate,yerr=[d.rate-d.lo,d.hi-d.rate],fmt='o-',capsize=3,color=color,label=f'{name}; slope +{slopes[outcome]["slope"]:.2f}')
    ax.set_xticks(np.arange(8),labels);ax.set_ylim(0,100);ax.set_ylabel('Runs (%)')
    ax.set_xlabel("Date the issue's fix was merged (93 issues)");ax.legend(frameon=False,loc='best',fontsize=9)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save(fig,out,'fig_issue_age')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--results',type=Path,default=OUTPUT);parser.add_argument('--output',type=Path,default=FIGURES)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    input_factors(args.results,args.output);variance(args.results,args.output);age(args.results,args.data,args.output)
