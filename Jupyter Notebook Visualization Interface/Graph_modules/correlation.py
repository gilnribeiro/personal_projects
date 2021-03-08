import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def variable_correlation_heatmap(df, dependent_variable):
    '''
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable
    '''
    plt.figure(figsize=(8, 10))
    sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable),
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1)
    plt.show()

    
def dskc_visualization_correlation_plot(df, vmin, vmax, cmap, annot, cbar, xticklabels,yticklabels):


    #Create a correlation matrix
    corrMatrix = df.corr()
    plt.figure(figsize=(10, 5))
    #Create a heatmap of the Correlation Matrix
    ax = sns.heatmap(corrMatrix, vmin=vmin, vmax=vmax, annot=annot, cmap=cmap, cbar=cbar,
                    xticklabels=xticklabels, yticklabels=yticklabels)

    #Plot the graph
    plt.show(ax)
    


def dskc_visualization_pca_plot(df, features, target, column, n_components, marker, title):


    # Standardize the data
    # PCA is effected by scale so you need to scale the features in your data before applying PCA

    features = features
    target = target

    # Separating out the features
    x = df[features].values

    # Separating out the target
    y = df[target].values


    #Dimensionality Reduction
    pca = PCA(n_components = n_components)
    reduced = pca.fit_transform(x)

    #Plot
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 7))
    for values in df[column].unique():
        mask = df[column] == values
        plt.plot(reduced[mask, 0], reduced[mask, 1], marker, label=values)
    plt.title(title)
    plt.legend()