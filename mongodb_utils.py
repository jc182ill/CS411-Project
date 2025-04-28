from pymongo import MongoClient
from typing import Dict, List, Optional
import logging

class MongoDBUtils:
    def __init__(self, uri: str, db_name: str = "AcademicWorld"):
        self.client = None
        self.db = None
        self.uri = uri
        self.db_name = db_name

    def connect(self) -> bool:
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            return True
        except Exception as e:
            logging.error(f"MongoDB Connection Error: {str(e)}")
            return False

    def aggregate(self, collection: str, pipeline: List[Dict]) -> Optional[List[Dict]]:
        """Execute aggregation pipeline"""
        try:
            return list(self.db[collection].aggregate(pipeline))
        except Exception as e:
            logging.error(f"Aggregation Error: {str(e)}")
            return None

    def find(self, collection: str, query: Dict, projection: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Execute find query"""
        try:
            return list(self.db[collection].find(query, projection))
        except Exception as e:
            logging.error(f"Find Operation Error: {str(e)}")
            return None

    def close(self) -> None:
        """Close connection"""
        if self.client:
            self.client.close()

# Example Usage:
# mongo = MongoDBUtils('mongodb://localhost:27017')
# mongo.connect()
# pipeline = [{"$match": {"venue": "AcademicWorld Conference"}}]
# results = mongo.aggregate("publications", pipeline)
# mongo.close()
