import pandas as pd 
import numpy as np 

def merge_excels(files_names, output_path, reading_kw_args={}, writing_kw_args={}):

    # merge everything into one file
    writer = pd.ExcelWriter(output_path)
    for results_file, sheet_name in files_names:
        df = pd.read_excel(results_file, **reading_kw_args)
        df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
        df.to_excel(writer, sheet_name=sheet_name, **writing_kw_args)
    writer.save()