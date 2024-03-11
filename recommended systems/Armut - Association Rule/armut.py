
import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

def create_rules(df):

    df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
    df["CreateDate"] = pd.to_datetime(df["CreateDate"])
    df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
    df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
    invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).map(
        lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))


    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] 


rules = create_rules(pd.read_csv('armut_data.csv'))

print(arl_recommender(rules,"2_0", 4))

