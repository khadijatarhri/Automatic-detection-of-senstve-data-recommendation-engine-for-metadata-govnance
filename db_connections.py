import pymongo

url = "mongodb://mongodb:27017"

client = pymongo.MongoClient(url)


db = client["csv_anonymizer_db"]