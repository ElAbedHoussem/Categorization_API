import pandas as pd 
import numpy as np
from textblob.classifiers import DecisionTreeClassifier
import re
import glob
import pickle

def read_data(path):
    train_data=pd.read_csv(path,delimiter=",")
    train_data = train_data[~train_data.CAT.str.startswith('Income')]
    return train_data

def Create_subcategory_form_categorical_field(train_data): 
    train_data ['SUB_CAT'] = ''
    for index, row in train_data.iterrows():
        train_data.at[index, 'LIB'] = row['LIB'][1:]
        
        cat = row['CAT'].split('-')
        train_data.at[index, 'CAT'] = cat[0]
        train_data.at[index, 'SUB_CAT'] = cat[1]  
    return train_data

def Create_the_data_file_for_categorical_classifier(train_data):
    train_data.drop(['SUB_CAT'],axis=1).to_csv('data/separated_cat_houssem/categorical_classifier.csv', index= False)
    unique_cat =  np.unique(train_data.CAT)
    return unique_cat

def Create_the_data_file_for_each_subcategorical_classifier(unique_cat, train_data):
    for cat in unique_cat :
        temp_df = train_data[train_data.CAT == cat]
        temp_df[['LIB','SUB_CAT']].rename(columns={'SUB_CAT': 'CAT'}).to_csv('data/separated_cat_houssem/'+cat+'.csv', index= False)

## Call the method that call the classifier
def general_gues(classifier, sen,lib,act,com,asking=False):
    g, classifier =  guess(classifier, sen,lib,act,com)
    return g, classifier

## Eliminate numbers from a string
def strip_numbers(s):
    """Strip numbers from the given string"""
    return re.sub("[^A-Z ]", "", s)

def split_by_multiple_delims(string, delims):
    """Split the given string by the list of delimiters given"""
    regexp = "|".join(delims)

    return re.split(regexp, string)

def guess(classifier, sen,lib,act,com):
    """guess category of transaction giving the lib and merchant """
    stripped_text = strip_numbers(lib+' '+act+' '+com)
    #stripped_text=self.clean(stripped_text)
    g=classifier.classify(stripped_text)
    return g, classifier

def extractor(doc):
    """Extract tokens from a given string"""
    tokens = split_by_multiple_delims(doc, [' ', '/','-'])

    features = {}

    for token in tokens:
        if token == "":
            continue
        features[token] = True

    return features

def strip_numbers(s):
    """Strip numbers from the given string"""
    return re.sub("[^A-Z ]", "", s)

def split_by_multiple_delims(string, delims):
    """Split the given string by the list of delimiters given"""
    regexp = "|".join(delims)

    return re.split(regexp, string)
  
def get_training(df):
    """Get training data for the classifier, consisting of tuples of
    (text, category)"""
    train = []
    subset = df[df['CAT'] != '']
    for i in subset.index:
        row = subset.iloc[i]
        new_desc = strip_numbers(row['LIB'])
        train.append( (new_desc, row['CAT']) )

    return train

def load_all_transactions_data(path):
    transactions=pd.read_csv(path,delimiter=",")
    return transactions

def create_and_save_models(transactions,path):
    ## Use the same name of the data file( which have the name of the category) as model name
    files_names = glob.glob(path+"/*.csv")
    files_names = [f.split('/')[2][:-4] for f in files_names]
    ## Fetch data files
    for index, file in enumerate(files_names):
        print(file)
        ## Load data for a specific category
        prev_data=pd.read_csv(path+file+".csv",delimiter=",")
        ## Create instance of DecisionTreeClassifier
        clf = DecisionTreeClassifier(get_training(prev_data), extractor)
        ## Train the model
        for index, row in transactions.iterrows():
            # For each transaction we try to update the model
            ## format is : Transaction_label,LIB_ACTIVITE , Merchant_name
            g, clf = general_gues(clf, row['Transaction_type'],row['Merchant_name'],row['LIB_ACTIVITE'],row['Merchant_name'],asking=False)
        ## Save the model 
        if file == 'categorical_classifier':
            with open("models/category_model/"+file+".pickle","wb") as f :
                pickle.dump(clf,f)
        else:
            with open("models/sub_categories_models/"+file+".pickle","wb") as f :
                pickle.dump(clf,f) 

def create_personal_model(path):
    prev_data=pd.read_csv("data/separated_cat_houssem/categorical_classifier.csv",delimiter=",")
    clf = DecisionTreeClassifier(get_training(prev_data), extractor)
    with open(path+"Personal.pickle","wb") as f :
        pickle.dump(clf,f)   

def load_model( customer_id='', path='./models/category_model/', model_name = 'categorical_classifier'):
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



'''
train_data = read_data("data/alldata3.csv")
train_data = Create_subcategory_form_categorical_field(train_data)
unique_cat = Create_the_data_file_for_categorical_classifier(train_data)
Create_the_data_file_for_each_subcategorical_classifier(unique_cat, train_data)
transactions = load_all_transactions_data("data/customers_acounts_transactions_operation_revenu5.csv")
create_and_save_models(transactions, "data/separated_cat_houssem/")
create_personal_model("models/category_model/")
create_personal_model("models/sub_categories_models/")

'''