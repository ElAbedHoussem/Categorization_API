import pandas as pd
from pymongo import MongoClient
import pickle
import requests
from textblob.classifiers import DecisionTreeClassifier
import re
import time
from paths_global_vars import categorical_model_path, subcategorical_model_path, matching_category_collection_name

class Model_update():

    def __init__(self, db_access_obj, category_unification_obj):
        ## Connect to the BD to save results
        self.db_connection = db_access_obj.db
        self.db_manager_obj = db_access_obj
        self.category_unification_obj = category_unification_obj

    def load_model(self, customer_id='', path='./models/category_model/', model_name = 'categorical_classifier'):
        '''
        description:
            This method is used to load any model type : categorical model, shopping model , food model ....
            by default; it loads the categorical model 
        input:
            customer_id : the id of user, because for each user we have specific models, and 
                        those models are saved under a directory that have the id of the customer
            path: path to customers models 
            model_name: by default is categorical_classifier, that's mean that by default we load the categorical
                        model and not a sub categorical model
        output:
            clf: classifier model
        '''
        with open(path+customer_id+model_name+".pickle","rb") as f :
            clf=pickle.load(f)
        return clf

    def transaction_preparation(self, transaction):
        '''
        Description:
            this method prepare the transaction to pass it to the classifier
        Input:
            transaction: A string with is the combinaison of Transaction label, Activity label and mechant name
        Output:
            Prepared string for the training
        '''
        transaction  = transaction.Transaction_label+' '+transaction.LIB_ACTIVITE+' '+transaction.Merchant_name
        """Strip numbers from the given string"""
        return re.sub("[^A-Z ]", "", transaction[0]) 

    def transaction_update(self, transaction, new_category, categories,
                       existing_categories_lemmatization, 
                       category_name = None, customer_id='', model_path=''):
        '''
        Description:
            This method is used to do the update of a category, this function is called when a customer try
            to change an existing category or to add a new category
        Input:
            transaction : DF row, that contains all transaction informations
            new_category : the new category tha t the customer has typped 
            categories: list of existing categories
            existing_categories_lemmatization: list of existing categories after lemmatization
            category_name:  if it is None that means that user try to add or update a category. If it is different
            of None that means the customer try to update or add a sub_category
            customer_id: int, which is the used_id and he is used to point to the correct directory in which models exist
            model_path : String, it is the path of all models for all customers
        Output:
            result : Boolean: True: update succesfuly, False : an error appears
        '''

        document = self.category_unification_obj.categorization(new_category, 
                                                                self.category_unification_obj.categories,
                                                                self.category_unification_obj.existing_categories_lemmatization,
                                                                None)
        prepared_transation  = self.transaction_preparation(transaction)        
        ## The update will be done on the categorical( general) model
        if category_name == None: 
            classif = self.load_model(path= categorical_model_path) 
        ## The update will be done on a subcategory model
        else:
            classif = self.load_model(path= subcategorical_model_path, model_name = category_name)
        classif.update([(prepared_transation, document['category_details']['sub_category_name'])])
        #print(document['category_details']['sub_category_name'])
        self.db_manager_obj.save_category_matching(self.db_connection, matching_category_collection_name, document)        
        return document








'''
from DB_management import DB_management
print('ok1')
db_access_obj = DB_management('mongodb://localhost:27017/', 'uib_customers_categories')
filename = 'nlu.md'
category_unification_obj = Category_unification(db_access_obj, filename)
print('ok2')

mu = Model_update(db_access_obj, category_unification_obj)
print('ok3')



import pandas as pd
transactions=pd.read_csv("data/customers_acounts_transactions_operation_revenu5.csv",delimiter=",")
columns = transactions.columns 
transaction = transactions.loc[2]
frame = {}
for index, elem in enumerate(transaction):
    frame[transaction.index[index]] = elem
transaction  = pd.DataFrame(frame, columns = columns, index=[0])
print('ok4')


new_category = 'shell'

mu.transaction_update(transaction, new_category, category_unification_obj.categories,
                       category_unification_obj.existing_categories_lemmatization, 
                       category_name = None, customer_id='', model_path='')
print('ok5')
'''
