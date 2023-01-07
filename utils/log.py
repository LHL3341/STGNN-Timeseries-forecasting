import pandas as pd
import os
import openpyxl
def excel_logs(f_path,config,performence):
    df_ = pd.merge(config,performence,right_index=True,left_index=True)
    if not os.path.exists(f_path):
        df_.to_excel(f_path,index=False,sheet_name='Sheet1')
    else:
        df = pd.read_excel(f_path,engine='openpyxl',sheet_name='Sheet1')
        df = df.append(df_)
        with pd.ExcelWriter(f_path, engine='openpyxl') as writer:
            df.to_excel(writer,index=False,sheet_name='Sheet1')