
###############################################################
# Online Retail Customer Lifetime Value
###############################################################

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)

def create_clv_c(dataframe, profit = 0.1):
    # Veriyi Hazırlama
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]
    dataframe = dataframe[(dataframe['Quantity']>0)]
    dataframe.dropna(inplace=True)
    dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['Price']
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction','total_unit','total_price']

    # avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']

    # purchase_frequency
    cltv_c['purchase_frequency'] = cltv_c['total_transaction'] / cltv_c.shape[0]

    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit

    # customer value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c['purchase_frequency'])

    # cltv
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

    # segment
    cltv_c['segment'] = pd.qcut(cltv_c['cltv'], 4, labels=['D','C','B','A'])

    return cltv_c

online_retail_clv = create_clv_c(pd.read_excel('online_retail_II.xlsx'))
online_retail_clv.to_csv('online_retail_cltv.csv', index=False)