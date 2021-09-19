import pandas as pd

paths = [
    { 
        "path" : '../generated-data/task_yeast_gi_hybrid',
        "name" : "S. cerevisiae (Hybrid)",
        "bins" : ['-', 'N', '+', 'S']
    },
    { 
        "path" : '../generated-data/task_yeast_gi_costanzo',
        "name" : "S. cerevisiae (Costanzo)",
        "bins" : ['-', 'N', '+', 'S']
    },
    { 
        "path" : '../generated-data/task_pombe_gi',
        "name" : "S. pombe",
        "bins" : ['-', 'N', '+', 'S']
    },
    { 
        "path" : '../generated-data/task_human_gi',
        "name" : "H. sapiens",
        "bins" : ['-', 'N', '+', 'S'],
    },
    { 
        "path" : '../generated-data/task_dro_gi',
        "name" : "D. melanogaster",
        "bins" : ['-', 'N', '+', 'S'],
    }
]

writer = pd.ExcelWriter('../tables/table9_gi_tasks.xlsx')
for p in paths:
    df = pd.read_csv(p['path'])
    if  'S. cerevisiae' in p['name']:
        df['a'] = [s.split('  ')[0] for s in df['a']]
        df['b'] = [s.split('  ')[0] for s in df['b']]
    
    ix = df['bin'] != 1
    df = df[ix]
    df['class'] = [p['bins'][b] for b in df['bin'].astype(int)]
    df = df[['a', 'b', 'class']]
    df.to_excel(writer, sheet_name=p['name'], index=False)
writer.save()