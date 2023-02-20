import pandas as pd
import os
import openpyxl
import matplotlib.pyplot as plt

def excel_logs(f_path,config,performence):
    df_ = pd.merge(config,performence,right_index=True,left_index=True)
    if not os.path.exists(f_path):
        df_.to_excel(f_path,index=False,sheet_name='Sheet1')
    else:
        df = pd.read_excel(f_path,engine='openpyxl',sheet_name='Sheet1')
        df = df.append(df_)
        with pd.ExcelWriter(f_path, engine='openpyxl') as writer:
            df.to_excel(writer,index=False,sheet_name='Sheet1')
            
def plot_loss(t_mse,t_mae,v_mse,v_mae):
    plt.figure(figsize=(12,10),dpi=80)
    length = len(t_mse)
    plt.subplot(2,1,1)
    plt.plot(range(length),t_mse,'b*-',range(length),v_mse,'y^-')
    plt.title('mse')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','val'])
    plt.grid(True,linestyle=':')
    plt.subplot(2,1,2)
    plt.plot(range(length),t_mae,'b*-',range(length),v_mae,'y^-')
    plt.title('mae')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','val'])
    plt.grid(True,linestyle=':')
    plt.show()
    
def plot_pred(x,gt,pd):
    plt.figure(figsize=(12,10),dpi=80)
    n = len(gt)
    for i in range(n):
        plt.subplot(n,1,i+1)
        length = len(gt[i])
        plt.plot(range(length+12),x[i]+gt[i],'b*-',range(12,length+12),pd[i],'y^-')
        plt.title(f'variable_{i}')
        plt.xlabel('timestamp')
        plt.ylabel('value')
        plt.legend(['groud_truth','predict'])
        plt.grid(True,linestyle=':')
    plt.show()

def plot_graph():
    pass