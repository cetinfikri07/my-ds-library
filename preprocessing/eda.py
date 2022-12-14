import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


#############################################
# GENERAL
#############################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """


    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat




#############################################
# CATEGORICAL
#############################################

def cat_summary(dataframe, col_name,target,sort_by = "Ratio"):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        "Target Mean" : dataframe.groupby(col_name)[target].mean()}).sort_values(by = sort_by,ascending=False))


#############################################
# NUMERICAL
#############################################

def num_summary(dataframe, numerical_col,quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99], plot=False):
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.ylabel("Count")
        plt.title(numerical_col)
        plt.show()
    return dataframe[numerical_col].describe(quantiles).T



#############################################
# TARGET
#############################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def duplicates(dataframe):
    return f"{dataframe.duplicated().sum()}, {np.round(100 * dataframe.duplicated().sum() / len(dataframe), 1)}%"

def pie_charts(dataframe,col):
    plt.figure(figsize = (8,5))
    explode = [0.1 for x in range(dataframe[col].nunique())]
    dataframe[col].value_counts().\
                        plot.pie(explode = explode,
                                autopct='%1.1f%%',
                                shadow = True,
                                textprops = {"fontsize" : 17}).\
                                set_title('Distribution')
    plt.show()

def all_pie_charts(dataframe,categorical_cols,nrow,ncol,figsize = (10,30)):
    fig = plt.figure(figsize=figsize)
    for i,col in enumerate(categorical_cols):
        # Left column
        ax = fig.add_subplot(nrow,ncol,i+1)
        explode = [0.1 for x in range(dataframe[col].nunique())]
        palette_color = sns.color_palette('hls', 8)
        sizes = dataframe[col].value_counts().values
        labels = dataframe[col].dropna().unique()
        plt.pie(sizes,
                explode=explode,
                labels=labels, colors=palette_color,
                autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax.set_title(col)

    fig.tight_layout()
    plt.show()

    return fig


def heatmap_missing(dataframe,figsize = (10,4)):
    na_cols = dataframe.columns[dataframe.isna().any()].to_list()
    plt.figure(figsize=figsize)
    sns.heatmap(dataframe[na_cols].isna().T,cmap = "summer")
    plt.title("heatmap of missing values")

def heatmap_jointdist(dataframe,xcol,target_col,figsize=(10,4)):
    '''
    Plots heatmap of joint distribution of two categorical variable

            Parameters:
                    xcol (str): Categorical col in pandas dataframe
                    target_col (str): Categorical col in pandas dataframe

            Returns: Joint distribution
    '''
    joint_dist = dataframe.groupby([xcol, target_col]).size().unstack().fillna(0)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(joint_dist.T, cmap="coolwarm", annot=True, fmt="g")

    return joint_dist

def count_plot_jointdist(dataframe,xcol,target_col,plot=True):
    '''
    Plots joint distribution countplot of two categorical variable

            Parameters:
                    xcol (str): Categorical col in pandas dataframe
                    target_col (str): Categorical col in pandas dataframe

            Returns: joint distribution
    '''
    joint_dist = dataframe.groupby([xcol,target_col])[target_col].size().unstack().fillna(0)
    if plot:
        sns.countplot((joint_dist > 0).sum(axis=1))

    return joint_dist

def plot_log_transformed(dataframe,transform_cols,nrow,ncol,figsize=(12,20)):
    fig = plt.figure(figsize=figsize)
    for i,col in enumerate(transform_cols):
        #Right plot
        plt.subplot(nrow,ncol,2*i+1)
        sns.histplot(dataframe[col],binwidth=100)
        plt.ylim([0, 200])
        plt.title(f"{col} original")
        #Left plot
        plt.subplot(nrow,ncol,2*i+2)
        sns.histplot(np.log(1+dataframe[col]),color = "C1")
        plt.ylim([0, 200])
        plt.title(f"{col} log transformed")

    fig.tight_layout()
    plt.show()

def visualize_pca_3d(dataframe):
    pca = PCA(n_components=3)
    components = pca.fit_transform(dataframe)
    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=y[:-1], size=0.1*np.ones(len(dataframe)), opacity = 1,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        width=800, height=500
        )
    fig.show()
