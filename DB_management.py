from pymongo import MongoClient

class DB_management():

    def __init__(self, url, db_name):
        self.instance = None
        #url = 'mongodb://localhost:27017/'
        #db_name = 'uib_customers_categories'
        self.db = self.singleton(url,db_name, user_name=None, password=None)
    
    def singleton(self, url,db_name, user_name=None, password=None):
        '''
        Description: 
            this method is used to create a unique instance of the used DB
        Input:
            url: url of the DB
            db_name: database name
            user_name: user name to connect to the db
            passowrd: Password to access to the DB 
        Output:
            DB object that points to the given db_name
        '''
        if self.instance is None:
            try:
                self.instance = MongoClient(url)
                if user_name!= None:
                    self.instance.the_database.authenticate(user_name, password)
            except (pymongo.errors.ConnectionFailure, e):
                print ("Could not connect to server: %s" % e)
        db = self.instance[db_name]
        return db

    def define_dcocument_format_to_save(self, user_id, transaction_id, customer_category, 
                                    matcher_category, original_category, sub_category_name):
        '''
        Description: 
            This function is used to define the format to save it into the DB. We separate this method in case we want to change 
            the format of the doc to save, we just change it once in this method
        Input:
            user_id: the custmer  ID
            transaction_id: Transaction Id
            customer_category: THe category that the customer has tappe it 
            matcher_category: The category used to do the matching
            original_category: Reference gategory
            sub_category_name: Reference subgategory
        Output: 
            Dict in a specific format 
        '''
        return {'customer_id' : user_id,
                'transaction_id': transaction_id,
                'category_details':
                    {'customer_category' : customer_category,
                        'matcher_category'  : matcher_category,
                        'original_category' : original_category,
                        'sub_category_name' :sub_category_name                   
                        }
                }
    
    def save_category_matching(self, db_connection, collection, document):
        '''
        Description: 
            save a document in a DB
        Input:
            db_connection(object): DB connection object
            collection(string): the collection name
            document(dict): th data dict to save
        Output:
            None
        '''
        matching_category = self.db.matching_category
        result = matching_category.update_one({'customer_id': document['customer_id']}, { "$set": document}, upsert=True)

        #result = matching_category.insert_one(document)
        if result.acknowledged == True:
            print("Saved document in DB ")
        else:
            print("Error with saving category in DB")
        
        return result.acknowledged      