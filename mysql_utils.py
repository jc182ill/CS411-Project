import pymysql
import logging
from typing import Optional, List, Dict

class MySQLUtils:
    def __init__(self, host: str, user: str, password: str, db: str):
        self.connection = None
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        
    def connect(self) -> bool:
        """Establish MySQL connection"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.db,
                cursorclass=pymysql.cursors.DictCursor
            )
            return True
        except Exception as e:
            logging.error(f"MySQL Connection Error: {str(e)}")
            return False

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[Dict]]:
        """Execute SQL query and return results"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
                result = cursor.fetchall()
                self.connection.commit()
                return result
        except Exception as e:
            logging.error(f"Query Execution Error: {str(e)}")
            return None

    def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()

# Example Usage:
# mysql = MySQLUtils('localhost', 'user', 'password', 'AcademicWorld')
# mysql.connect()
# results = mysql.execute_query("SELECT * FROM publications WHERE year > 2020")
# mysql.close()
