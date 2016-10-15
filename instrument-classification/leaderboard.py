import glob
import numpy as np
import pandas as pd

def load_df(file_name):
    model_id = file_name.split('/')[4]
    df = pd.read_csv(file_name)
    df.rename(columns={'Unnamed: 0': 'split'}, inplace=True)
    df['model_id'] = model_id
    if 'auc' not in df.columns:
        df['auc'] = np.nan
    df = df[['model_id', 'split', 'loss', 'error', 'auc']]
    return df

files = sorted(glob.glob('data/working/single-notes-2000/models/*/evaluation/final_metrics.csv'))
df = pd.concat([load_df(f) for f in files])

df_valid = df[df['split'] == 'valid']
df_valid.set_index('model_id', inplace=True)

print('Ranking on the validation split metrics.\n')
print('Lowest error:')
print(df_valid.sort_values('error', ascending=True)[:5])

print('\nHighest AUC:')
print(df_valid.sort_values('auc', ascending=False)[:5])

print('\nRank of the last few models by error, total:', len(df_valid))
print(df_valid.rank()['error'].astype(int).iloc[-5:])
