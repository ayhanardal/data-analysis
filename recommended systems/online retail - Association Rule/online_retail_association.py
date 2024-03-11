
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, int(up_limit)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe = dataframe[(dataframe['Price'] > 0)]
    replace_with_thresholds(dataframe, 'Quantity')
    replace_with_thresholds(dataframe, 'Price')
    return dataframe
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice','StockCode'])['Quantity'].sum().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).map(
            lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name =  dataframe[dataframe['StockCode'] == stock_code][['Description']].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country='France'):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_items = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric='support', min_threshold=0.01)
    return rules

df = retail_data_prep(pd.read_excel('online_retail_II.xlsx'))
rules = create_rules(df)
rules.to_csv('ayhan.csv') 