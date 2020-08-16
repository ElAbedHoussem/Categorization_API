import pandas as pd 
import numpy as np
from textblob.classifiers import DecisionTreeClassifier
import re
import glob
import pickle
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class Prepar_Data():
    
    def __init__(self):
        self.train_data=pd.read_csv("data/alldata3.csv",delimiter=",")
        self.train_data = self.train_data[~self.train_data.CAT.str.startswith('Income')]
    
    def create_sub_cat_from_cat(self):
        ## Create subcategory form categorical feeld 
        self.train_data ['SUB_CAT'] = ''
        for index, row in self.train_data.iterrows():
            self.train_data.at[index, 'LIB'] = row['LIB'][1:]
            
            cat = row['CAT'].split('-')
            self.train_data.at[index, 'CAT'] = cat[0]
            self.train_data.at[index, 'SUB_CAT'] = cat[1]  

    def create_data_for_categorical_classifier(self):
        self.train_data.drop(['SUB_CAT'],axis=1).to_csv('separated_cat_houssem/categorical_classifier.csv', index= False)

    def create_data_foreach_subcategorical_classifier(self):
        unique_cat =  np.unique(self.train_data.CAT)
        for cat in unique_cat :
            temp_df = self.train_data[self.train_data.CAT == cat]
            temp_df[['LIB','SUB_CAT']].rename(columns={'SUB_CAT': 'CAT'}).to_csv('separated_cat_houssem/'+cat+'.csv', index= False)

    ## Call the method that call the classifier
    def general_gues(self, classifier, sen,lib,act,com,asking=False):
        g, classifier =  self.guess(classifier, sen,lib,act,com)
        return g, classifier

    ## Eliminate numbers from a string
    def strip_numbers(self, s):
        """Strip numbers from the given string"""
        return re.sub("[^A-Z ]", "", s)

    def split_by_multiple_delims(self, string, delims):
        """Split the given string by the list of delimiters given"""
        regexp = "|".join(delims)

        return re.split(regexp, string)

    def guess(self, classifier, sen,lib,act,com):
        """guess category of transaction giving the lib and merchant """
        stripped_text = self.strip_numbers(lib+' '+act+' '+com)
        #stripped_text=self.clean(stripped_text)
        g=classifier.classify(stripped_text)
        return g, classifier

    def extractor(self, doc):
        """Extract tokens from a given string"""
        tokens = self.split_by_multiple_delims(doc, [' ', '/','-'])

        features = {}

        for token in tokens:
            if token == "":
                continue
            features[token] = True

        return features

    def split_by_multiple_delims(self, string, delims):
        """Split the given string by the list of delimiters given"""
        regexp = "|".join(delims)

        return re.split(regexp, string)
    
    def get_training(self, df):
        """Get training data for the classifier, consisting of tuples of
        (text, category)"""
        train = []
        subset = df[df['CAT'] != '']
        for i in subset.index:
            row = subset.iloc[i]
            new_desc = self.strip_numbers(row['LIB'])
            train.append( (new_desc, row['CAT']) )

        return train

    def load_all_transactions(self):
        transactions=pd.read_csv("data/customers_acounts_transactions_operation_revenu5.csv",delimiter=",")
        return transactions

    def create_models(self,transactions):
        ## Use the same name of the data file( which have the name of the category) as model name
        files_names = glob.glob("separated_cat_houssem/*.csv")
        files_names = [f.split('/')[1][:-4] for f in files_names]
        ## Fetch data files
        for index, file in enumerate(files_names):
            print(file)
            ## Load data for a specific category
            prev_data=pd.read_csv("separated_cat_houssem/"+file+".csv",delimiter=",")
            ## Create instance of DecisionTreeClassifier
            clf = DecisionTreeClassifier(self.get_training(prev_data), self.extractor)
            ## Train the model
            for index, row in transactions.iterrows():
                # For each transaction we try to update the model
                ## format is : Transaction_label,LIB_ACTIVITE , Merchant_name
                g, clf = self.general_gues(clf, row['Transaction_type'],row['Merchant_name'],row['LIB_ACTIVITE'],row['Merchant_name'],asking=False)
            ## Save the model 
            with open("models/"+file+".pickle","wb") as f :
                pickle.dump(clf,f) 




'''prepare_data = Prepar_Data()
prepare_data.create_sub_cat_from_cat()
prepare_data.create_data_for_categorical_classifier()
prepare_data.create_data_foreach_subcategorical_classifier()
transactions = prepare_data.load_all_transactions()
prepare_data.create_models(transactions)
'''
