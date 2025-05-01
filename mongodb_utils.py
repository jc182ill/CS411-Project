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
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            return True
        except Exception as e:
            logging.error(f"MongoDB Connection Error: {str(e)}")
            return False

    def find(self, collection: str, query: Dict, projection: Optional[Dict] = None) -> Optional[List[Dict]]:
        try:
            return list(self.db[collection].find(query, projection))
        except Exception as e:
            logging.error(f"Find Operation Error: {str(e)}")
            return None

    def upsert_override(self, collection: str, entity_id: int, updates: Dict):
        try:
            updates['approved'] = False  # mark as pending approval
            return self.db[collection].update_one(
                {"entity_id": entity_id},
                {"$set": updates},
                upsert=True
            )
        except Exception as e:
            logging.error(f"Upsert Error: {str(e)}")
            return None

    def delete_override(self, collection: str, entity_id: int):
        try:
            return self.db[collection].delete_one({"entity_id": entity_id})
        except Exception as e:
            logging.error(f"Delete Error: {str(e)}")
            return None

    def approve_override(self, collection: str, entity_id: int):
        try:
            return self.db[collection].update_one(
                {"entity_id": entity_id},
                {"$set": {"approved": True}}
            )
        except Exception as e:
            logging.error(f"Approval Error: {str(e)}")
            return None

    def get_pending_overrides(self, collection: str) -> List[Dict]:
        try:
            return list(self.db[collection].find({"approved": False}))
        except Exception as e:
            logging.error(f"Retrieval Error: {str(e)}")
            return []
    
    def close(self):
        if self.client:
            self.client.close()
