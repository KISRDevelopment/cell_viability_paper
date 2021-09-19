import pandas as pd

paths = [
    { 
        "path" : '../generated-data/task_yeast_smf_30',
        "name" : "S. cerevisiae",
        "bins" : ['L', 'R', 'N']
    },
    { 
        "path" : '../generated-data/task_pombe_smf',
        "name" : "S. pombe",
        "bins" : ['L', 'R', 'N']
    },
    { 
        "path" : '../generated-data/task_human_smf',
        "name" : "H. sapiens",
        "bins" : ['L', 'R', 'N']
    },
    { 
        "path" : '../generated-data/task_dro_smf',
        "name" : "D. melanogaster",
        "bins" : ['L', 'R', 'N']
    },
    { 
        "path" : '../generated-data/task_human_smf_ca_ma_v',
        "name" : "H. sapiens (CA vs MO vs V)",
        "bins" : ['CA', 'MO', 'V']
    },
    { 
        "path" : '../generated-data/task_dro_smf_ca_ma_v',
        "name" : "D. melanogaster (CA vs MO vs V)",
        "bins" : ['CA', 'MO', 'V']
    },
]

writer = pd.ExcelWriter('../tables/table8_smf_tasks.xlsx')
for p in paths:
    df = pd.read_csv(p['path'])
    if p['name'] == 'S. cerevisiae':
        df['gene'] = [s.split('  ')[0] for s in df['gene']]
    
    df['class'] = [p['bins'][b] for b in df['bin'].astype(int)]
    df = df[['gene', 'class']]
    df.to_excel(writer, sheet_name=p['name'], index=False)
writer.save()