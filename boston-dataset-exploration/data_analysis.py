import logging
import matplotlib
# non-interactive plotting - just outputs the images and doesn't open the window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston

def dataset_to_dataframe(dataset, target_name):
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df[target_name] = dataset.target
    return df

def print_structure(dataset, file):
    logging.debug('Analyzing dataset structure')
    print('Number of instances:', dataset.data.shape[0], file=file)
    print('Number of attributes:', dataset.data.shape[1], file=file)
    print('Attribute names:', ', '.join(dataset.feature_names), file=file)

def summarize_distributions(df, file):
    logging.debug('Summarizing attribute distributions')
    print('Attribute distribution summary:', file=file)
    # pd.set_option('display.width', 200)
    desc = df.describe().T
    desc['mode'] = df.mode().ix[0]
    print(desc, file=file)
    # print(df.describe().T[['count','mean','std','min','50%','max']], file=file)

    missing_counts = pd.isnull(df).sum()
    if missing_counts.any():
        print('Missing values:', file=file)
        print(missing_counts, file=file)
    else:
        print('Missing values: NONE', file=file)

def print_correlations(df, file):
    logging.debug('Analyzing attribute pairwise correlations')
    print("Pearson's correlation:", file=file)
    pearson = df.corr(method='pearson')
    print(pearson, file=file)
    print("Spearman's correlation:", file=file)
    spearman = df.corr(method='spearman')
    print(spearman, file=file)
    
    def predictivity(correlations):
        best = correlations.ix[-1]
        best.sort(ascending=False)
        return best
    
    print('Attribute-target correlations (Pearson):', file=file)
    print(predictivity(pearson), file=file)
    print('Attribute-target correlations (Spearman):', file=file)
    print(predictivity(spearman), file=file)
    
    print('Important attribute correlations (Pearson):', file=file)
    attrs = pearson.iloc[:-1,:-1] # all except target
    # only important correlations and not auto-correlations
    important_corrs = (attrs[abs(attrs) > 0.5][attrs != 1.0]) \
        .unstack().dropna().to_dict()
    unique_important_corrs = pd.DataFrame(
        list(set([(tuple(sorted(key)), important_corrs[key]) \
        for key in important_corrs])), columns=['attribute pair', 'correlation'])
    unique_important_corrs.sort('correlation', ascending=False, inplace=True)
    print(unique_important_corrs, file=file)

def attribute_correlations(df, img_file='attr_correlations.png'):
    logging.debug('Plotting attribute pairwise correlations')
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = sns.blend_palette(['#00008B', '#6A5ACD', '#F0F8FF',
        '#FFE6F8', '#C71585', '#8B0000'], as_cmap=True)
    sns.corrplot(df, annot=False, sig_stars=False,
        diag_names=False, cmap=cmap, ax=ax)
    fig.tight_layout()
    fig.savefig(img_file)
    plt.close(fig)

def attribute_histograms(df, real_cols, int_cols):
    def plot_hist(col, func):
        file = 'dist_{}.png'.format(col)
        logging.debug('histogram: %s', file)
        fig = plt.figure()
        func(col)
        fig.tight_layout()
        fig.savefig(file)
        plt.close(fig)

    def plot_real(col):
        sns.distplot(df[col], kde=False)
    
    def plot_int(col):
        plt.bar(*list(zip(*df[col].value_counts().items())))
        plt.xlabel(col)
    
    logging.debug('Plotting attribute histograms')
    
    for col in real_cols:
        plot_hist(col, plot_real)

    for col in int_cols:
        plot_hist(col, plot_int)        

def pairwise_scatter_matrix(df, img_file='pairwise_scatter_matrix.png'):
    logging.debug('Plotting pairwise scatter matrix')
    grid = sns.pairplot(df)
    grid.savefig(img_file)
    plt.close()

def pairwise_joint_plots(df, cols):
    logging.debug('Plotting pairwise joint distributions')
    cols = sorted(cols)
    for colA, colB in [(a,b) for a in cols for b in cols if a < b]:
        file = 'joint_{}_{}.png'.format(colA, colB)
        logging.debug('joint plot: %s', file)
        fig = plt.figure()
        sns.jointplot(df[colA], df[colB], kind='hex')
        plt.savefig(file)
        plt.close()

def make_report(dataset, df, report_file_name='data_analysis_report.txt'):
    report_file = open(report_file_name, 'w')
    
    print_structure(dataset, report_file)
    summarize_distributions(df, report_file)
    print_correlations(df, report_file)

    logging.info('Report is in file: %s', report_file_name)    

def visualize(df, int_cols):
    sns.set(style='darkgrid')
    
    int_cols = set(int_cols)
    real_cols = set(df.columns) - int_cols

    attribute_correlations(df)
    attribute_histograms(df, real_cols, int_cols)
    pairwise_joint_plots(df, real_cols)
    pairwise_scatter_matrix(df)

if __name__ == '__main__':
    log_format='%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=log_format, level=logging.DEBUG)
    
    # load data
    
    boston = load_boston()

    df = dataset_to_dataframe(boston, target_name='MEDV')
    
    report_dir = 'report'
    os.makedirs(report_dir, exist_ok=True)
    os.chdir(report_dir)

    make_report(boston, df)

    visualize(df, int_cols=['CHAS', 'RAD'])
   
    logging.debug('Done')
