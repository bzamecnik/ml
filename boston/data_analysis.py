from sklearn.datasets import load_boston
import pandas as pd

def dataset_to_dataframe(dataset, target_name):
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df[target_name] = boston.target
    return df

def print_structure(dataset):
    print("number of instances:", dataset.data.shape[0])
    print("number of attributes:", dataset.data.shape[1])
    print("attribute names:", ', '.join(dataset.feature_names))

def summarize_distributions(df):
    print("attribute distribution summary:")
    # pd.set_option('display.width', 200)
    desc = df.describe().T
    desc['mode'] = df.mode().ix[0]
    print(desc)
    # print(df.describe().T[['count','mean','std','min','50%','max']])

    missing_counts = pd.isnull(df).sum()
    if missing_counts.any():
        print("missing values:")
        print(missing_counts)
    else:
        print("missing values: NONE")

def print_correlations(df):
    print("Pearson's correlation:")
    pearson = df.corr(method='pearson')
    print(pearson)
    print("Spearman's correlation:")
    spearman = df.corr(method='spearman')
    print(spearman)
    
    def predictivity(correlations):
        best = correlations.ix[-1]
        best.sort(ascending=False)
        return best
    
    print("Attribute-target correlations (Pearson):")
    print(predictivity(pearson))
    print("Attribute-target correlations (Sparman):")
    print(predictivity(spearman))
    
    print("Important attribute correlations (Pearson):")
    attrs = pearson.iloc[:-1,:-1] # all except target
    # only important correlations and not auto-correlations
    important_corrs = (attrs[abs(attrs) > 0.5][attrs != 1.0]) \
        .unstack().dropna().to_dict()
    unique_important_corrs = pd.DataFrame(
        list(set([(tuple(sorted(key)), important_corrs[key]) \
        for key in important_corrs])), columns=['attribute pair', 'correlation'])
    unique_important_corrs.sort('correlation', ascending=False, inplace=True)
    print(unique_important_corrs)

def attribute_histograms(df):
    pass # TODO

def pairwise_scatter_plots(df):
    pass # TODO

if __name__ == '__main__':
    boston = load_boston()

    df = dataset_to_dataframe(boston, target_name='MEDV')
    
    print_structure(boston)
    summarize_distributions(df)
    print_correlations(df)

    attribute_histograms(df)
    pairwise_scatter_plots(df)
