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


def fix_participation(column):
    return column.apply(lambda cells: cells.strip('%'))


def convert_to_float(exam_df):
    features = [col for col in exam_df.columns if col != 'State']
    exam_df[features] = exam_df[features].astype(float)
    return exam_df


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

print('SAT 2017 Missing Data:', "\n", sat_17.isnull().sum(), '\n')
print('SAT 2018 Missing Data:', "\n", sat_18.isnull().sum(), '\n')
print('ACT 2017 Missing Data:', "\n", act_17.isnull().sum(), '\n')
print('ACT 2018 Missing Data:', "\n", act_18.isnull().sum(), '\n')

print('SAT 2017 Data Type:', "\n", sat_17.dtypes, '\n')
print('SAT 2018 Data Type:', "\n", sat_18.dtypes, '\n')
print('ACT 2017 Data Type:', "\n", act_17.dtypes, '\n')
print('ACT 2018 Data Type:', "\n", act_18.dtypes, '\n')

# Fix the participation type
sat_17['Participation'] = fix_participation(sat_17['Participation'])
sat_18['Participation'] = fix_participation(sat_18['Participation'])
act_17['Participation'] = fix_participation(act_17['Participation'])
act_18['Participation'] = fix_participation(act_18['Participation'])

# convert to float type
sat_17 = convert_to_float(sat_17)
sat_18 = convert_to_float(sat_18)
act_18 = convert_to_float(act_18)

# remove corrupted character
act_17['Composite'] = act_17['Composite'].apply(lambda x: x.strip('x'))

# convert again to float type
act_17 = convert_to_float(act_17)

# rename the columns
new_act_17_cols = {
    'State': 'state',
    'Participation': 'act_participation_17',
    'Composite': 'act_composite_17'}
act_17.rename(columns=new_act_17_cols, inplace=True)

new_act_18_cols = {
    'State': 'state',
    'Participation': 'act_participation_18',
    'Composite': 'act_composite_18'}
act_18.rename(columns=new_act_18_cols, inplace=True)

new_sat_17_cols = {
    'State': 'state',
    'Participation': 'sat_participation_17',
    'Total': 'sat_score_17'}
sat_17.rename(columns=new_sat_17_cols, inplace=True)

new_sat_18_cols = {
    'State': 'state',
    'Participation': 'sat_participation_18',
    'Total': 'sat_score_18'}
sat_18.rename(columns=new_sat_18_cols, inplace=True)

# sort the data
sat_17.sort_values(by=['state'], inplace=True)
sat_18.sort_values(by=['state'], inplace=True)
act_17.sort_values(by=['state'], inplace=True)
act_18.sort_values(by=['state'], inplace=True)

# reset the index
sat_17 = sat_17.reset_index(drop=True)
sat_18 = sat_18.reset_index(drop=True)
act_17 = act_17.reset_index(drop=True)
act_18 = act_18.reset_index(drop=True)

df1 = pd.merge(sat_17, sat_18, left_index=True, on='state', how='outer')
df2 = pd.merge(act_17, act_18, left_index=True, on='state', how='outer')
df = pd.merge(df1, df2, left_index=True, on='state', how='outer')
data = [sat_17, sat_18, act_17, act_18]
fd = pd.concat(data, join='inner', axis=1)

# Plotting
plt.figure(figsize = (15,10))
plt.title('SAT and ACT Correlation Heatmap', fontsize = 16);

# Mask to remove redundancy from the heatmap.
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax = 1, cmap = "coolwarm",  annot = True);
plt.savefig('heatmap.png')

plt.figure(figsize = (8,6))
features = ['sat_participation_17', 'sat_participation_18', 'act_participation_17', 'act_participation_18']
plt.title('SAT and ACT Participation Rate Correlations', fontsize = 16);
mask = np.zeros_like(df[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df[features].corr(), mask=mask, vmin=-1, vmax = 1, cmap = "coolwarm",  annot = True);
plt.savefig('heatmap01.png')

plt.figure(figsize = (8,6))
features = ['sat_score_17', 'sat_score_18', 'act_composite_17', 'act_composite_18']
plt.title('Average SAT Score vs Average ACT Composite Score Correlations', fontsize = 16);
mask = np.zeros_like(df[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df[features].corr(), mask=mask, vmin=-1, vmax = 1, cmap = "coolwarm",  annot = True);
plt.savefig('heatmap02.png')

# Boxplots comparing the average participation rates of the 2017 ACT, 2018 ACT, 2017 SAT, and 2018 SAT.
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12,8))

sns.boxplot(df.sat_participation_17, ax = ax[0,0], orient="h", color = 'orange').set(
    xlabel='', title='SAT Participation Rates 2017');

sns.boxplot(df.sat_participation_18, ax = ax[0,1], orient="h", color = 'orange').set(
    xlabel='', title='SAT Participation Rates 2018');

sns.boxplot(df.act_participation_17, ax = ax[1,0], orient="h", color= 'pink').set(
    xlabel='', title='ACT Participation Rates 2017');

sns.boxplot(df.act_participation_18, ax = ax[1,1], orient="h", color = 'pink').set(
    xlabel='', title='ACT Participation Rates 2018');

plt.tight_layout()

plt.savefig('boxplot.png');

plt.figure(figsize = (15,8))

# SAT Participation Rates 2017 histogram
plt.subplot(1,2,1) 
sns.distplot(df.sat_participation_17, kde=False,bins=8);

plt.title('SAT Participation Rates 2017 Distribution', fontsize=16)
plt.xlabel('Participation Rate', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.xlim(0, 101)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ACT Participation Rates 2017 histogram
plt.subplot(1,2,2) 
sns.distplot(df.act_participation_17, kde=False, bins=8);

plt.title('ACT Participation Rates 2017 Distribution', fontsize=16)
plt.xlabel('Participation Rate', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.xlim(0, 101)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('histo01.png');

plt.figure(figsize = (15,8))

# SAT Participation Rates 2018 histogram
plt.subplot(1,2,1) 
sns.distplot(df.sat_participation_18, kde=False, bins=8);

plt.title('SAT Participation Rates 2018 Distribution', fontsize=16);
plt.xlabel('Participation Rate', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(0, 101)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ACT Participation Rates 2018 histogram
plt.subplot(1,2,2) 
sns.distplot(df.act_participation_18,kde=False,bins=8);
plt.title('ACT Participation Rates 2018 Distribution', fontsize=16);
plt.xlabel('Participation Rate', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(0, 101)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('histo02.png');
