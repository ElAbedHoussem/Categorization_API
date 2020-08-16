import pandas as pd
from pymongo import MongoClient
import pickle
import requests
from textblob.classifiers import DecisionTreeClassifier
import re
from Category_unification import Category_unification
 
class Categorization():

    def __init__(self, db_access_obj):
        ## Connect to the BD to save results
        self.db_connection = db_access_obj.db
        self.db_manager_obj = db_access_obj

    def load_model(self, customer_id='', path='./models/', model_name = 'categorical_classifier'):
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

    def transaction_predict(self, transaction, customer_id=1, transaction_id = 1, model_path=''):
        '''
        Description:
            This method is used to do the predict the category for a given transaction
        Input:
            transaction : DF row, that contains all transaction informations
            customer_id: int, which is the used_id and he is used to point to the correct directory in which models exist
            model_path : String, it is the path of all models for all customers
        Output:
            category_name : The caterogy of the transaction
            sub_category_name: The subcategory of the transaction
        '''
        clf = self.load_model(path='./models/category_model/') ## Load cateorical_model (General model)
        prepared_transation  = self.transaction_preparation(transaction)
        category_name = clf.classify(prepared_transation)
        #return category_name
        sub_cat_clf = self.load_model(path = './models/sub_categories_models/', model_name = category_name)
        sub_category_name = sub_cat_clf.classify(prepared_transation)
        return  self.db_manager_obj.define_dcocument_format_to_save(2, transaction_id,
                                                '', '',
                                                category_name, sub_category_name)

    def transactions_predict(self, transactions, customer_id=1, transaction_id = 1, model_path=''):
        '''
        Description:
            This method is used to do the predict the category for a list of transactions
        Input:
            transactions : DF rows, that contains all transactions informations
            customer_id: int, which is the used_id and he is used to point to the correct directory in which models exist
            model_path : String, it is the path of all models for all customers
        Output:
            categories_names : Categories of transactions
            sub_categories_names: Subcategories of transactions
        '''

        clf = self.load_model(path='./models/category_model/') ## Load cateorical_model (General model)
        result = []
        for transaction in transactions:
            prepared_transation  = self.transaction_preparation(transaction)
            category_name = clf.classify(prepared_transation)
            #return category_name
            sub_cat_clf = self.load_model(path = './models/sub_categories_models/', model_name = category_name)
            sub_category_name = sub_cat_clf.classify(prepared_transation)
            result.append(self.db_manager_obj.define_dcocument_format_to_save(2, transaction_id,
                                                    '', '',
                                                    category_name, sub_category_name))
        return result    

    def extractor(self, doc):
        '''
        Description:
            Extract delimeters from a given string
        Input:
            doc(string): the string to split
        Output:
            features(dict): dict of tokens 
        '''
        tokens = self.split_by_multiple_delims(doc, [' ', '/','-'])
        features = {}
        for token in tokens:
            if token == "":
                continue
            features[token] = True
        return features

    def split_by_multiple_delims(self, string, delims):
        '''
        Description:
            Split the given string by the list of delimiters given
        Input:
            string(string): the string to split
            delims(list): list of delimiters
        Output:
        splited string 
        '''
        regexp = "|".join(delims)
        return re.split(regexp, string)


'''
from DB_management import DB_management
db_access_obj = DB_management('mongodb://localhost:27017/', 'uib_customers_categories')
cat = Categorization(db_access_obj)
transactions=pd.read_csv("data/customers_acounts_transactions_operation_revenu5.csv",delimiter=",")
columns = transactions.columns 
transaction = transactions.loc[2]
frame = {}
for index, elem in enumerate(transaction):
    frame[transaction.index[index]] = elem
transaction  = pd.DataFrame(frame, columns = columns, index=[0])
doc = cat.transaction_predict(transaction)
print(doc)
db_access_obj.save_category_matching(db_access_obj,'matching_category' ,doc)

'''






