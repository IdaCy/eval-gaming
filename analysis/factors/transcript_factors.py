"""Summarize categorical transcript annotations without transcript text."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=DATA)
parser.add_argument('--output',type=Path,default=OUTPUT / 'transcript_factors.json')
args=parser.parse_args()
f=pd.read_parquet(args.data/'trigger_counts.parquet')
f['denominator']=f.groupby('model').n.transform('sum')
f['percent']=100*f.n/f.denominator
wanted={'DeepSeek':['none_spontaneous'],'GLM':['recalled_upstream'],'Qwen':['recalled_upstream'],'gpt-oss-120b':['missing_dependency_or_tool','network_or_install_failure']}
f=f[[t in wanted[m] for m,t in zip(f.model,f.trigger)]]
m=pd.read_parquet(args.data/'memory_counts.parquet');m['percent']=100*m.k/m.n
result={'first_evaluation_trigger':f.to_dict('records'),'memory':m.to_dict('records')}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
