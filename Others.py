## Convert nlu.md to nlu.pickle after apply some transformations
import pickle
#WARNING !!!! :  This method will be deleted, because we will use only the NLU model
def load_intents(file_name):
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
    list_of_categories = []
    dict_of_categories_intents = {}
    intents_of_categories = [] 
    with open(file_name, 'r') as fileinput:
        for line in fileinput:
            ## Get all subcategories names, save them and ceate an elemen in a dict 
            ##that has the same name as the subcategory
            if line.startswith('##') :
                cat = line.replace('## intent:', '').replace('\n', '')
                dict_of_categories_intents[cat]  = [] 
                list_of_categories.append(cat)
            else:
                ## Save all possible words different toempty line under the dict
                if line != "\n":
                    sub_cat = line.replace('\n', '').replace('- ', '')
                    dict_of_categories_intents[list_of_categories[len(list_of_categories)-1]].append(sub_cat)
    
    # return a dict 
    #return dict_of_categories_intents
    
    for tab in dict_of_categories_intents:
        intents_of_categories.append(dict_of_categories_intents.get(tab))
        
    with open("nlu.pickle","wb") as f :
        pickle.dump(intents_of_categories,f)
load_intents('nlu.md')
