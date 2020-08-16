import pandas as pd

def prediction_adapter(request):
    data=request.get_json()
    return pd.DataFrame(data, index=[0])


def predict_all_transaction_adapter(request):
    data=request.get_json()
    return data['customer_id']


def update_adapter(request):
    data=request.get_json()
    transaction = pd.DataFrame(data, index=[0])
    new_cat_name = transaction[['new_category_name']].to_dict('records')[0]['new_category_name']
    transaction.drop(['new_category_name'], axis=1, inplace=True)
    return transaction, new_cat_name