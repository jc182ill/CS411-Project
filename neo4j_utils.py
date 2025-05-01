from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging

class Neo4jUtils:
    def __init__(self, db_name: str = "academicworld"):
        self.driver = None
        self.db_name = db_name
        self.uri = "bolt://localhost:7687"  # Default academic DB port
        
        # Security recommendation: Store credentials in environment variables
        self.auth = (
            os.getenv("NEO4J_USER") or NEO4J_CONFIG.get('auth', ('neo4j', ''))[0],
            os.getenv("NEO4J_PASSWORD") or NEO4J_CONFIG.get('auth', ('', 'cs411_fallback'))[1]
        )

    def connect(self) -> bool:
        """Establish connection with automatic credential discovery"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=15  # Prevent long hanging connections
            )
            # Verify actual connectivity
            with self.driver.session(database=self.db_name) as session:
                session.run("RETURN 1 AS test")
            return True
        except Exception as e:
            logging.error(f"Connection failed: {str(e)}")
            self._suggest_credentials_resolution(e)
            return False

    def execute_query(self, query: str, params: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Execute query with database context"""
        if not self.driver:
            logging.error("Connection not established")
            return None

        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(query, params or {})
                return [self._sanitize_record(record) for record in result]
        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            return None

    def _sanitize_record(self, record) -> Dict:
        """Convert Neo4j types to Python native types"""
        return {key: self._convert_value(value) for key, value in record.items()}

    def _convert_value(self, value):
        """Handle Neo4j-specific data types"""
        if isinstance(value, list):
            return [self._convert_value(item) for item in value]
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        return value

    def _suggest_credentials_resolution(self, error):
        """Provide contextual help for common academic DB issues"""
        if "authentication" in str(error).lower():
            print("""
            Common academic DB credentials:
            - Username: 'neo4j' (default)
            - Password: Often course-specific (e.g., 'cs411spring2024')
            
            Check course materials or contact TA for current credentials
            """)

    def close(self) -> None:
        """Safely close connection"""
        if self.driver:
            try:
                self.driver.close()
            except Exception as e:
                logging.warning(f"Cleanup error: {str(e)}")

# Example Usage:
# neo4j = Neo4jUtils("bolt://localhost:7687", "neo4j", "password")
# neo4j.connect()
# results = neo4j.execute_query(
#     "MATCH (a:Author)-[:AUTHORED]->(p:Publication) RETURN a.name, COUNT(p) AS publications ORDER BY publications DESC LIMIT 10"
# )
# neo4j.close()
