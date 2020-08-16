import pandas as pd
from string import punctuation
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from textblob import TextBlob
from googletrans import Translator
import spacy
from pymongo import MongoClient
import pickle, gzip
import requests
from textblob.classifiers import DecisionTreeClassifier
import time
from paths_global_vars import nul_model_url, categories
 
class Category_unification():

    def __init__(self, db_access_obj, filename):
        
        ## Connect to the BD to save results
        self.db = db_access_obj.db
        self.db_manager_obj = db_access_obj

        ## NLU categorization URL model, this api give us the have to consume the nlu model 
        self.model_url = nul_model_url
        ## Prepare stop_words SpellChecker for the correction.
        ##  If we want to to add new language we have to add here the soptword
        ##  and spellchecker methods assocated to thios language
        self.english_stopwords = set(stopwords.words("english"))
        self.french_stopwords = set(stopwords.words("french"))
        self.english_spell = SpellChecker() 
        self.french_spell = SpellChecker(language='fr')
        self.nlp = spacy.load('en_core_web_sm')
        ## Reference categories
        self.categories = categories
        
        ## 
        self.filename = filename
        self.intents_of_categories = self.load_intents(self.filename)
        ### This take some time ~6s
        self.existing_categories_lemmatization = self.lemmatize_original_categories(self.categories)

    #WARNING !!!! :  This method will be deleted, because we will use only the NLU model
    def load_intents(self, file_name):
        '''
        Description:
            This method load intents that are used to train the nlu model to compare if the given 
            category matches with any existing category
        Input:
            file_name(string) : file that contains intents
        Output:
            intents_of_categories(List of dicts): List that contains dicts
            each dict contains all intents for each sub-category
        '''
        with open(file_name,"rb") as f :
            return pickle.load(f)

    def language_detection(self, text):
        '''
        Description:
            This method detect the language of a given string.
            It is used to choose which STOPWORDS and WORDCORRECTION methods must be used
        Input:
            text(string): text to detect the origine language
        Output:
        string: indicates the origin language of this text (fr, en , it ....) 
            
        '''
        #b = TextBlob(text)
        #return b.detect_language()
        translator = Translator()
        return translator.detect(text).lang

    def split_clean_text(self, text,text_language):
        '''
        Description:
            This method used to do the spell correction foreach word in a text for a given language
        Input:
            text(string): text in which we will apply the correction
            text_language(string): the used language to choose the appropriate spell corrector
        Output:
            tokens(string): text after applying spell correction
        '''
        tokens = word_tokenize(text.lower())
        switcher = { 
            'fr': [self.french_spell.correction(word) for word in tokens], 
            'en': [self.english_spell.correction(word) for word in tokens], 
        } 
        tokens = switcher.get(text_language, [self.english_spell.correction(word) for word in tokens])

        #tokens = [spell.correction(word) for word in tokens]
        tokens = ' '.join(tokens)
        return tokens

    def translate_words(self, words, language,text_language):
        '''
        Description:
            This method is used delete punctuations, stop word and then do the tradiction to english
        Input:
            words(list of strings): list that contains all the text words
            language(string): the language to which we will convert all the text, in general it will be english
            text_language(string): the text language, this is used to choose the STOPWORDS  method to use
        Output:
            result(list of strings): list of translated words 
        '''
        result = []
        words = word_tokenize(words.lower())
        words = [word for word in words if word not in punctuation]
        switcher = { 
            'fr': [word for word in words if word not in self.french_stopwords], 
            'en': [word for word in words if word not in self.english_stopwords], 
        } 
        words = switcher.get(text_language, [word for word in words if word not in self.english_stopwords])

        #words = [ TextBlob(word) for word in words]
        translator = Translator()
        for word in words:
            try:
                result.append(translator.translate(word, dest=language).text)
            except:
                result.append(str(word))
        #return str(TextBlob(words).translate(to='en'))
        return result
        
    def lemmatization(self, sentence):
        '''
        Description:
            This method is used to return each word of a given sentance to here origin
        Input:
            sentence(list of string): list of words of a sentence
        Output:
            word_lemmatizer(string): text after lemmatization
        '''
        #words = ' '.join(words)
        sentence = ' '.join(sentence)
        word_lemmatizer = self.nlp(sentence) 
        #word_lemmatizer = [nlp(i) for i in sentence]
        word_lemmatizer = [i.lemma_ for i in word_lemmatizer]
        
        
        #lemmatizer = WordNetLemmatizer() 
        #word_lemmatizer = [lemmatizer.stem(i) for i in sentence]
        #word_lemmatizer = lemmatizer.lemmatize(sentence, pos="n")
        return word_lemmatizer

    def lemmatize_original_categories(self, categories):
        '''
        Description:
            This method do the lemmatization of all existing categories and sub-categories
        Input:
            categories(list of strings): list that contains all categories and subcategories
        Output:
            categories_lemmatization(list of strings): list that conains all categories after lemmatization
        '''
        categories_lemmatization = []
        for category in categories:
            category_language = self.language_detection(category)   
            category_translation = self.translate_words(category,'en', category_language)
            category_translation = ' '.join(category_translation)
            word_lemmatizer = self.nlp(category_translation) 
            word_lemmatizer = [i.lemma_ for i in word_lemmatizer]
            word_lemmatizer = ' '.join(word_lemmatizer)
            categories_lemmatization.append(word_lemmatizer)
        return categories_lemmatization   

    #WARNING !!!! :  This method will be deleted, because we will use only the NLU model
    def verify_new_caegory_in_existing_categories(self, new_cat, categories_lemmatized,original_categories, user_id = 1, transaction_id=1):
        '''
        Description:
            This method verify if the given category matchs with any existing lemmatized category or sub-category.
            In this case we return a dict
        Input:
            new_cat(string): the user given category
            categories_lemmatized(list of strings): list that contains all lemmatized categories and sub-categories
            original_categories(list of strings): list that contains all categories and sub-categories
            user_id(int): the user id 
            transaction_id(int): the transaction id
            
        Output:
            existing_categories_result(list that contain at most one dict): at most one dict
            that contains the user_id, the transaction_id, the user given category, the category with it we do the matching
            and the reference category
        '''
        existing_categories_result = []
        
        for index, category in enumerate(categories_lemmatized):
            for word in new_cat:
                word_exist = category.find(word)
                ## the customer category is found under existing categories
                if word_exist!=-1:                
                    existing_categories_result.append(
                                        self.db_manager_obj.define_dcocument_format_to_save(user_id, transaction_id,
                                                                        ' '.join(new_cat), category,
                                                                        original_categories[index], ''
                                                        ))
                    break
        return existing_categories_result        

    #WARNING !!!! :  This method will be deleted, because we will use only the NLU model
    def search_category_in_intents(self, new_cat, intents_of_categories,original_categories, user_id = 1, transaction_id=1):
        '''
        Description:
            This method verify if the given category matchs with any existing lemmatized category or sub-category.
            In this case we return a dict
        Input:
            new_cat(string): the user given category
            categories_lemmatized(list of strings): list that contains all lemmatized categories and sub-categories
            original_categories(list of strings): list that contains all categories and sub-categories
            user_id(int): the user id 
            transaction_id(int): the transaction id
            
        Output:
            existing_categories_result(list that contain at most one dict): at most one dict
            that contains the user_id, the transaction_id, the user given category, the category with it we do the matching
            and the reference category
        '''
        for index, intent in enumerate(intents_of_categories):
            for word in new_cat:
                if word in intent:
                    return  self.db_manager_obj.define_dcocument_format_to_save(user_id, transaction_id,
                                                            ' '.join(new_cat), word,
                                                            original_categories[index], '')
        return None    

    def categorization(self, customer_category, exiting_categories,lemmatized_exiting_categories, intent_list= None, user_id = 1, transaction_id=1):
        '''
        Description:
        Input:
        Output:
        '''
        text_language = self.language_detection(customer_category)
        cleaned_category = self.split_clean_text(customer_category, text_language)
        translated_category = self.translate_words(cleaned_category,'en', text_language)
        lemmatized_category = self.lemmatization(translated_category)
        ## search if the category exists under existing categories after lemmatization
        founded_category = self.verify_new_caegory_in_existing_categories(
                                                            lemmatized_category,
                                                            lemmatized_exiting_categories, 
                                                            exiting_categories)
        ## In the case where we don't found the category under the existing categories, 
        ## we search under the existing intents using nlu model
        if len(founded_category) ==0:
            print('The category is not found under reference categories, we will use nlu categorization model')
            
            
            ## Fetch all lemmitazed categories, for each one we will pass it to the nlop model and 
            ## try to test if the confidence is higher than 0.7 
            for limma  in lemmatized_category:
                founded_category  = self.nlu_categorization(self.model_url, limma)         
                #founded_category = search_category_in_intents(category_lemmatization, intents_of_categories,  categories)
                ## if the category confidence is lower than 0.7, we will categorize the
                ## transaction under "Personal" category, esle  we will pick this category as the reference category
                if (founded_category['confidence']>=0.7):
                    print("Category is classified correctly using nlu categorization model under the category : {}".format(founded_category['name']))
                    return  self.db_manager_obj.define_dcocument_format_to_save(user_id, transaction_id,
                                                            customer_category, limma,
                                                            founded_category['name'], '')
                    
        ## the categhory is found after lemmatization 
        else:
            print("problem solved with lemmatization")
            founded_category['customer_category'] = customer_category
            #print(founded_category)
            return founded_category

        print("category is not found after lemmatization and and even by the nlu model, this transaction will be categorised under Personal category")
        return  self.db_manager_obj.define_dcocument_format_to_save(2, transaction_id,
                                                customer_category, '',
                                                'Personal', 'Personal')

    ### We this method in case where the taped category does not match with any reference category
    def nlu_categorization(self, model_url, category):
        '''
        Description: 
            This method consume the nlu model using his API
        Input:
            model_url(string): the nlu model url
            category(string): the word to categorize
        Output:
            most_accurate_intent(string): the category that matches with the highest probability with the given word
        '''
        params = '{"text":"'+category+'"}'
        r = requests.post(model_url,data=params)
        most_accurate_intent= r.json()['intent']
        if(most_accurate_intent['confidence'])<0.7:
            most_accurate_intent['name'] = 'Personal'
        return most_accurate_intent

    ## This method is used to give a ranking of the taped category
    def nlu_categorization_ranking(self, model_url, category):
        '''
        Description: 
            This method return the list of all categories that matches with a given word
        Input:
            model_url(string): the nlu model url
            category(string): the word to categorize
        Output:
            cateroy_matching(list of strings): categories that matches with the given word
        '''
        cateroy_matching = []
        params = '{"text":"'+category+'"}'
        r = requests.post(model_url,data=params)
        intents_ranking= r.json()['intent_ranking']
        for intent in intents_ranking:
            #print(intent['name'], intent['confidence'])
            cateroy_matching.append((intent))
        return cateroy_matching







'''
from DB_management import DB_management
db_access_obj = DB_management('mongodb://localhost:27017/', 'uib_customers_categories')
filename = "nlu.pickle"
clf = Category_unification(db_access_obj, filename)
test_cat = 'mazzout'
document = clf.categorization(test_cat, clf.categories,clf.existing_categories_lemmatization, clf.intents_of_categories)
print(document)
db_access_obj.save_category_matching(db_access_obj, 'matching_category', document)
'''