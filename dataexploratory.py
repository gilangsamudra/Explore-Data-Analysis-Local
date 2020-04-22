# Bismillah
# Bagian 1 - Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compare_values(act_col, sat_col):
    act_vals = []
    sat_vals = []
    # Buat List dulu agar bisa dicek
    for a_val in act_col:
        act_vals.append(a_val)
    for s_val in sat_col:
        sat_vals.append(s_val)
        
    print('Values in ACT only: ')
    for val_a in act_vals:
        if (val_a not in sat_vals):
            print(val_a)
    print('--------------------')
    print('Values in SAT only: ')
    for val_s in sat_vals:
        if (val_s not in sat_vals):
            print(val_s)


# Bagian 2 - Load the data
sat_17 = pd.read_csv('D:/Phyton Code/Contoh dari Github/\
sat_act_analysis-master/data/sat_2017.csv')

sat_18 = pd.read_csv('D:/Phyton Code/Contoh dari Github/\
sat_act_analysis-master/data/sat_2018.csv')

act_17 = pd.read_csv('D:/Phyton Code/Contoh dari Github/\
sat_act_analysis-master/data/act_2017.csv')

act_18 = pd.read_csv('D:/Phyton Code/Contoh dari Github/\
sat_act_analysis-master/data/act_2018.csv')

# Exploring the data and cleaning corrupted data
print('SAT 2017 shape = ', sat_17.shape)
print('SAT 2018 shape = ', sat_18.shape)
print('ACT 2017 shape = ', act_17.shape)
print('ACT 2018 shape = ', act_18.shape)

act_18['State'].value_counts()
act_18[act_18['State'] == 'Maine']

# drop the incorrect data
act_18.drop(act_18.index[52], inplace=True)
act_18 = act_18.reset_index(drop=True)
act_18.shape

compare_values(act_17['State'], sat_17['State'])
compare_values(act_18['State'], sat_18['State'])

act_17[act_17['State'] == 'National']
act_17.drop(act_17.index[0], inplace=True)
act_17 = act_17.reset_index(drop=True)
act_17.shape

act_18[act_18['State'] == 'National']
act_18.drop(act_18.index[23], inplace=True)
act_18 = act_18.reset_index(drop=True)
act_18.shape

act_18['State'].replace({'Washington, D.C.': 'District of Columbia'},
                        inplace=True)

# final check of consistency
print("FINAL CHECK ACT DATA \n")
compare_values(act_17['State'], sat_17['State'])
print("FINAL CHECK SAT DATA \n")
compare_values(act_18['State'], sat_18['State'])

# Membandingkan nama kolom dalam setiap data dengan menggunakan atribut
# (.columns)
print('SAT 2017 column names = ', sat_17.columns, "\n")
print('SAT 2018 column names = ', sat_18.columns, "\n")
print('ACT 2017 column names = ', act_17.columns, "\n")
print('ACT 2018 column names = ', act_18.columns, "\n")

# removing unecessary columns unsing (.drop()) method
sat_17.drop(columns=['Evidence-Based Reading and Writing', 'Math'], inplace=True)
sat_18.drop(columns=['Evidence-Based Reading and Writing', 'Math'], inplace=True)
act_17.drop(columns=['English', 'Math', 'Reading', 'Science'], inplace=True)

# check again
print('SAT 2017 column names = ', sat_17.columns, "\n")
print('SAT 2018 column names = ', sat_18.columns, "\n")
print('ACT 2017 column names = ', act_17.columns, "\n")
print('ACT 2018 column names = ', act_18.columns, "\n")

print(sat_17.isnull().sum())
