
###############################################################
# Online Retail Customer Lifetime Value Predict
###############################################################

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def create_cltv_p(dataframe, month = 3):
    # Data Preparation
    dataframe.dropna(inplace = True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]
    dataframe = dataframe[(dataframe['Quantity']>0)]
    dataframe = dataframe[(dataframe['Price']>0)]
    replace_with_thresholds(dataframe, 'Quantity')
    replace_with_thresholds(dataframe, 'Price')
    dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['Price']
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg({
        'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max()-InvoiceDate.min()).days,
                        lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
        'Invoice': lambda  Invoice: Invoice.nunique(),
        'TotalPrice': lambda TotalPrice: TotalPrice.sum()
    })

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency','T','frequency','monetary']
    cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']
    cltv_df = cltv_df[(cltv_df['frequency']>1)]
    cltv_df['recency'] = cltv_df['recency'] / 7
    cltv_df['T'] = cltv_df['T'] / 7

    # BG-NBD Modelinin Kurulumu
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df['expected_purc_1_week'] = bgf.predict(1,
                                                    cltv_df['frequency'],
                                                    cltv_df['recency'],
                                                    cltv_df['T'])

    cltv_df['expected_purc_1_month'] = bgf.predict(4,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df['expected_purc_3_month'] = bgf.predict(12,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    # GammaGamma Modelinin Kurulumu
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df['expected_avarage_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # BG-NBD ve GG Modeli ile CLTV'nin hesaplanması
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,
                                       freq = 'W',
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, how='left', on='Customer ID')
    cltv_final['segment'] = pd.qcut(cltv_final['clv'], 4, labels = ['D','C','B','A'])

    return cltv_final

cltv_pred = create_cltv_p(pd.read_excel('online_retail_II.xlsx'))
cltv_pred.to_csv('online_retail_cltv_pred.csv', index=False)