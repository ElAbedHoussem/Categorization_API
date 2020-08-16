class Customer():
    def __init__(self, db_connection=None):
        ## db_connection: Connection to the Bd that contains all transactions (RDB, Kafka ...)
        self.db_connection = db_connection

    def get_all_transactions(self, customer_id):
        ## TODO: Get all transactions of a specific customer using customer_id 
        self.customer_id = customer_id

    def get_transaction(self, trnasaction_id):
        ## Return transaction details using the transaction ID
        ## TODO: Complete this function
        self.customer_id = customer_id

    def prepare_transactions(self, customer_id):
        ## TODO: This function prepare transaction for prediction.
        ## We must have The preparation phases from SAHAR
        transactions = self.get_all_transactions(customer_id)
        self.customer_id = customer_id