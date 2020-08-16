from flask import Flask
from flask_cors import CORS
from flask import Flask ,jsonify,request
import json
import os
from Categorization import Categorization
from Category_unification import Category_unification
from Model_update import Model_update
import Adapters as adapter
from DB_management import DB_management
from Customer import Customer
dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
CORS(app)


@app.route('/',methods=['GET','POST'])
def all_categories():
    return jsonify( {"about":'Hello World!'})



@app.route('/predict_transaction',methods=['GET','POST'])
def predict_transaction():
    if(request.method=='POST'):
        transaction = adapter.prediction_adapter(request)
        result = categorization.transaction_predict(transaction)
        db_access_obj.save_category_matching(db_access_obj,'matching_category' ,result)
        return jsonify( {"result":result})


@app.route('/predict_transactions',methods=['GET','POST'])
def predict_transactions():
    if(request.method=='POST'):
        transaction = adapter.predict_all_transaction_adapter(request)
        prepared_transactions = customer.prepare_transactions(1504846)
        result = categorization.transaction_predict(transaction)
        db_access_obj.save_category_matching(db_access_obj,'matching_category' ,result)
        return jsonify( {"result":result})



@app.route('/update_model',methods=['GET','POST'])
def update_model():
    if(request.method=='POST'):
        transaction, new_category = adapter.update_adapter(request)
        res = model_update.transaction_update(transaction, new_category, category_unification.categories,
                            category_unification.existing_categories_lemmatization, 
                            category_name = None, customer_id='', model_path='')
        return jsonify( {"result":res})


if __name__ == '__main__':

    db_access_obj = DB_management('mongodb://localhost:27017/', 'uib_customers_categories')
    categorization = Categorization(db_access_obj)
    filename = 'nlu.pickle'
    category_unification = Category_unification(db_access_obj, filename)
    model_update = Model_update(db_access_obj, category_unification)
    customer = Customer()
    app.run(debug=True)




