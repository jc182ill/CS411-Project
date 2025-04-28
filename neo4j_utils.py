from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging

class Neo4jUtils:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = None
        self.uri = uri
        self.user = user
        self.password = password

    def connect(self) -> bool:
        """Establish Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            return True
        except Exception as e:
            logging.error(f"Neo4j Connection Error: {str(e)}")
            return False

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Execute Cypher query"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logging.error(f"Cypher Query Error: {str(e)}")
            return None

    def close(self) -> None:
        """Close driver connection"""
        if self.driver:
            self.driver.close()

# Example Usage:
# neo4j = Neo4jUtils("bolt://localhost:7687", "neo4j", "password")
# neo4j.connect()
# results = neo4j.execute_query(
#     "MATCH (a:Author)-[:AUTHORED]->(p:Publication) RETURN a.name, COUNT(p) AS publications ORDER BY publications DESC LIMIT 10"
# )
# neo4j.close()
