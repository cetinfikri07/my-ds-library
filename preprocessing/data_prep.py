import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def outlier_tresholds(dataframe,col_name,q1 = 0.01, q3 = 0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_tresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_tresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_tresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_tresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def fillna_with_median_by_groups(dataframe,na_col,groups:list):
    na_rows = dataframe.loc[dataframe[na_col].isna(),na_col].index
    dataframe.loc[dataframe[na_col].isna(), na_col] = \
    dataframe.groupby(groups)[na_col] \
        .transform(lambda x: x.fillna(x.median()))[na_rows]

    return na_rows

def fillna_with_mean_by_groups(dataframe,na_col,groups:list):
    na_rows = dataframe.loc[dataframe[na_col].isna(),na_col].index
    dataframe.loc[dataframe[na_col].isna(), na_col] = \
    dataframe.groupby(groups)[na_col] \
        .transform(lambda x: x.fillna(x.mean()))[na_rows]

    return na_rows


def fillna_with_mode_by_groups(dataframe,na_col,groups:list):
    na_bef = dataframe[na_col].isna().sum()
    # Select na rows indices
    na_rows = dataframe.loc[dataframe[na_col].isna(),na_col].index
    # fill na values with mode by groups
    dataframe.loc[dataframe[na_col].isna(),na_col] = \
    dataframe.groupby(groups)[na_col].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows]

    print(f'{na_col} missing values before :{na_bef}')
    print(f'{na_col} missing values after :{dataframe[na_col].isna().sum()}')

    return na_rows


def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Scale and encode with sklearn pipelines
def numerical_transformer(numerical_cols):
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    return numerical_transformer


def categorical_transformer(categorical_cols):
    # One-hot encode categorical data
    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=False))])
    return categorical_transformer


def columns_transformer(dataframe, num_transformer, cat_transformer, numerical_cols, categorical_cols):
    ct = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)],
        remainder='passthrough')

    return ct


def apply_log_transfrom(dataframe,transform_cols):
    for col in transform_cols:
        dataframe[col] = np.log(1+dataframe[col])

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}).sort_values(by = "RATIO",ascending = False), end="\n\n\n")


def rare_encoder(dataframe, rare_perc, cat_cols):
    # 1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    # rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
    # eğer 1'den büyük ise rare cols listesine alınıyor.
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe







