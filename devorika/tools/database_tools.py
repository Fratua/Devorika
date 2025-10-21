"""
Database Tools for Devorika
Comprehensive database operations for SQL, NoSQL, migrations, and ORM.
"""

import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
import subprocess
from .base import Tool


class SQLQueryTool(Tool):
    """
    Execute SQL queries on various databases (SQLite, PostgreSQL, MySQL).
    """

    name = "sql_query"
    description = "Execute SQL queries and get results"

    def execute(self, query: str, database: str, db_type: str = "sqlite",
                **connection_params) -> Dict[str, Any]:
        """
        Execute SQL query.

        Args:
            query: SQL query to execute
            database: Database name or path
            db_type: Database type (sqlite, postgresql, mysql)
            **connection_params: Additional connection parameters

        Returns:
            Dict with query results
        """
        try:
            if db_type == "sqlite":
                return self._execute_sqlite(query, database)
            elif db_type == "postgresql":
                return self._execute_postgresql(query, database, connection_params)
            elif db_type == "mysql":
                return self._execute_mysql(query, database, connection_params)
            else:
                return {"error": f"Unsupported database type: {db_type}"}

        except Exception as e:
            return {"error": f"Query execution failed: {str(e)}"}

    def _execute_sqlite(self, query: str, database: str) -> Dict[str, Any]:
        """Execute SQLite query."""
        try:
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(query)

            # Check if query returns data
            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                return {
                    'success': True,
                    'rows': results,
                    'count': len(results),
                    'database': database
                }
            else:
                conn.commit()
                return {
                    'success': True,
                    'rows_affected': cursor.rowcount,
                    'database': database
                }

        except sqlite3.Error as e:
            return {"error": f"SQLite error: {str(e)}"}
        finally:
            if conn:
                conn.close()

    def _execute_postgresql(self, query: str, database: str,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PostgreSQL query."""
        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                dbname=database,
                user=params.get('user', 'postgres'),
                password=params.get('password', ''),
                host=params.get('host', 'localhost'),
                port=params.get('port', 5432)
            )

            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query)

            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                return {
                    'success': True,
                    'rows': rows,
                    'count': len(rows),
                    'database': database
                }
            else:
                conn.commit()
                return {
                    'success': True,
                    'rows_affected': cursor.rowcount,
                    'database': database
                }

        except ImportError:
            return {"error": "psycopg2 not installed. Install with: pip install psycopg2-binary"}
        except Exception as e:
            return {"error": f"PostgreSQL error: {str(e)}"}
        finally:
            if 'conn' in locals():
                conn.close()

    def _execute_mysql(self, query: str, database: str,
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MySQL query."""
        try:
            import mysql.connector

            conn = mysql.connector.connect(
                database=database,
                user=params.get('user', 'root'),
                password=params.get('password', ''),
                host=params.get('host', 'localhost'),
                port=params.get('port', 3306)
            )

            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)

            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                return {
                    'success': True,
                    'rows': rows,
                    'count': len(rows),
                    'database': database
                }
            else:
                conn.commit()
                return {
                    'success': True,
                    'rows_affected': cursor.rowcount,
                    'database': database
                }

        except ImportError:
            return {"error": "mysql-connector not installed. Install with: pip install mysql-connector-python"}
        except Exception as e:
            return {"error": f"MySQL error: {str(e)}"}
        finally:
            if 'conn' in locals():
                conn.close()


class DatabaseSchemaTool(Tool):
    """
    Inspect and analyze database schema.
    """

    name = "database_schema"
    description = "Get database schema information (tables, columns, indexes)"

    def execute(self, database: str, db_type: str = "sqlite",
                **connection_params) -> Dict[str, Any]:
        """
        Get database schema.

        Args:
            database: Database name or path
            db_type: Database type (sqlite, postgresql, mysql)
            **connection_params: Connection parameters

        Returns:
            Dict with schema information
        """
        try:
            if db_type == "sqlite":
                return self._get_sqlite_schema(database)
            elif db_type == "postgresql":
                return self._get_postgresql_schema(database, connection_params)
            elif db_type == "mysql":
                return self._get_mysql_schema(database, connection_params)
            else:
                return {"error": f"Unsupported database type: {db_type}"}

        except Exception as e:
            return {"error": f"Schema inspection failed: {str(e)}"}

    def _get_sqlite_schema(self, database: str) -> Dict[str, Any]:
        """Get SQLite schema."""
        try:
            conn = sqlite3.connect(database)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            schema = {}
            for table in tables:
                # Get columns
                cursor.execute(f"PRAGMA table_info({table})")
                columns = []
                for col in cursor.fetchall():
                    columns.append({
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'default': col[4],
                        'primary_key': bool(col[5])
                    })

                # Get indexes
                cursor.execute(f"PRAGMA index_list({table})")
                indexes = [{'name': idx[1], 'unique': bool(idx[2])} for idx in cursor.fetchall()]

                schema[table] = {
                    'columns': columns,
                    'indexes': indexes
                }

            return {
                'success': True,
                'database': database,
                'db_type': 'sqlite',
                'tables': list(schema.keys()),
                'schema': schema
            }

        except sqlite3.Error as e:
            return {"error": f"SQLite error: {str(e)}"}
        finally:
            if conn:
                conn.close()

    def _get_postgresql_schema(self, database: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get PostgreSQL schema."""
        try:
            import psycopg2

            conn = psycopg2.connect(
                dbname=database,
                user=params.get('user', 'postgres'),
                password=params.get('password', ''),
                host=params.get('host', 'localhost'),
                port=params.get('port', 5432)
            )

            cursor = conn.cursor()

            # Get all tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            schema = {}
            for table in tables:
                # Get columns
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                """)

                columns = []
                for col in cursor.fetchall():
                    columns.append({
                        'name': col[0],
                        'type': col[1],
                        'nullable': col[2] == 'YES',
                        'default': col[3]
                    })

                schema[table] = {'columns': columns}

            return {
                'success': True,
                'database': database,
                'db_type': 'postgresql',
                'tables': list(schema.keys()),
                'schema': schema
            }

        except ImportError:
            return {"error": "psycopg2 not installed"}
        except Exception as e:
            return {"error": f"PostgreSQL error: {str(e)}"}
        finally:
            if 'conn' in locals():
                conn.close()

    def _get_mysql_schema(self, database: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get MySQL schema."""
        try:
            import mysql.connector

            conn = mysql.connector.connect(
                database=database,
                user=params.get('user', 'root'),
                password=params.get('password', ''),
                host=params.get('host', 'localhost'),
                port=params.get('port', 3306)
            )

            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]

            schema = {}
            for table in tables:
                cursor.execute(f"DESCRIBE {table}")
                columns = []
                for col in cursor.fetchall():
                    columns.append({
                        'name': col[0],
                        'type': col[1],
                        'nullable': col[2] == 'YES',
                        'default': col[4]
                    })

                schema[table] = {'columns': columns}

            return {
                'success': True,
                'database': database,
                'db_type': 'mysql',
                'tables': list(schema.keys()),
                'schema': schema
            }

        except ImportError:
            return {"error": "mysql-connector not installed"}
        except Exception as e:
            return {"error": f"MySQL error: {str(e)}"}
        finally:
            if 'conn' in locals():
                conn.close()


class MigrationTool(Tool):
    """
    Database migration management (Alembic for SQLAlchemy).
    """

    name = "database_migration"
    description = "Create and manage database migrations"

    def execute(self, action: str, migration_dir: str = "migrations",
                message: str = None, **kwargs) -> Dict[str, Any]:
        """
        Manage database migrations.

        Args:
            action: Action (init, create, upgrade, downgrade, current)
            migration_dir: Directory for migrations
            message: Migration message (for create action)
            **kwargs: Additional parameters

        Returns:
            Dict with migration results
        """
        try:
            if action == "init":
                return self._init_migrations(migration_dir)
            elif action == "create":
                if not message:
                    return {"error": "Message required for create action"}
                return self._create_migration(migration_dir, message)
            elif action == "upgrade":
                return self._run_migration(migration_dir, "upgrade")
            elif action == "downgrade":
                return self._run_migration(migration_dir, "downgrade")
            elif action == "current":
                return self._get_current_version(migration_dir)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Migration failed: {str(e)}"}

    def _init_migrations(self, migration_dir: str) -> Dict[str, Any]:
        """Initialize Alembic migrations."""
        try:
            cmd = ["alembic", "init", migration_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': 'Migrations initialized',
                    'directory': migration_dir
                }
            else:
                return {"error": result.stderr}

        except FileNotFoundError:
            return {"error": "Alembic not installed. Install with: pip install alembic"}
        except subprocess.TimeoutExpired:
            return {"error": "Migration initialization timed out"}

    def _create_migration(self, migration_dir: str, message: str) -> Dict[str, Any]:
        """Create new migration."""
        try:
            cmd = ["alembic", "revision", "--autogenerate", "-m", message]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=os.path.dirname(migration_dir) or ".",
                                  timeout=30)

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'Migration created: {message}',
                    'output': result.stdout
                }
            else:
                return {"error": result.stderr}

        except FileNotFoundError:
            return {"error": "Alembic not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "Migration creation timed out"}

    def _run_migration(self, migration_dir: str, direction: str) -> Dict[str, Any]:
        """Run migration upgrade or downgrade."""
        try:
            cmd = ["alembic", direction, "head" if direction == "upgrade" else "-1"]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=os.path.dirname(migration_dir) or ".",
                                  timeout=60)

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'Migration {direction} completed',
                    'output': result.stdout
                }
            else:
                return {"error": result.stderr}

        except FileNotFoundError:
            return {"error": "Alembic not installed"}
        except subprocess.TimeoutExpired:
            return {"error": f"Migration {direction} timed out"}

    def _get_current_version(self, migration_dir: str) -> Dict[str, Any]:
        """Get current migration version."""
        try:
            cmd = ["alembic", "current"]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=os.path.dirname(migration_dir) or ".",
                                  timeout=30)

            if result.returncode == 0:
                return {
                    'success': True,
                    'current_version': result.stdout.strip()
                }
            else:
                return {"error": result.stderr}

        except FileNotFoundError:
            return {"error": "Alembic not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "Version check timed out"}


class ORMModelTool(Tool):
    """
    Generate ORM models from database schema (SQLAlchemy).
    """

    name = "generate_orm_models"
    description = "Generate SQLAlchemy ORM models from database schema"

    def execute(self, database: str, output_file: str = "models.py",
                db_type: str = "sqlite", **connection_params) -> Dict[str, Any]:
        """
        Generate ORM models.

        Args:
            database: Database name or path
            output_file: Output file for models
            db_type: Database type
            **connection_params: Connection parameters

        Returns:
            Dict with generated models
        """
        try:
            # Get schema first
            schema_tool = DatabaseSchemaTool()
            schema_result = schema_tool.execute(database, db_type, **connection_params)

            if 'error' in schema_result:
                return schema_result

            # Generate SQLAlchemy models
            models_code = self._generate_sqlalchemy_models(schema_result['schema'])

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(models_code)

            return {
                'success': True,
                'output_file': output_file,
                'tables_generated': len(schema_result['schema']),
                'preview': models_code[:500] + '...' if len(models_code) > 500 else models_code
            }

        except Exception as e:
            return {"error": f"Model generation failed: {str(e)}"}

    def _generate_sqlalchemy_models(self, schema: Dict[str, Any]) -> str:
        """Generate SQLAlchemy model code."""
        code = "from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text\n"
        code += "from sqlalchemy.ext.declarative import declarative_base\n\n"
        code += "Base = declarative_base()\n\n"

        type_mapping = {
            'INTEGER': 'Integer',
            'TEXT': 'Text',
            'VARCHAR': 'String',
            'FLOAT': 'Float',
            'REAL': 'Float',
            'BOOLEAN': 'Boolean',
            'DATETIME': 'DateTime'
        }

        for table_name, table_info in schema.items():
            class_name = ''.join(word.capitalize() for word in table_name.split('_'))

            code += f"class {class_name}(Base):\n"
            code += f"    __tablename__ = '{table_name}'\n\n"

            for column in table_info['columns']:
                col_type = type_mapping.get(column['type'].upper(), 'String')
                nullable = ', nullable=True' if column['nullable'] else ''
                primary_key = ', primary_key=True' if column.get('primary_key') else ''

                code += f"    {column['name']} = Column({col_type}{primary_key}{nullable})\n"

            code += "\n"

        return code


class NoSQLTool(Tool):
    """
    NoSQL database operations (MongoDB, Redis).
    """

    name = "nosql_operations"
    description = "Perform NoSQL database operations (MongoDB, Redis)"

    def execute(self, db_type: str, operation: str, collection: str = None,
                **params) -> Dict[str, Any]:
        """
        Execute NoSQL operations.

        Args:
            db_type: Database type (mongodb, redis)
            operation: Operation to perform
            collection: Collection/key name
            **params: Operation parameters

        Returns:
            Dict with operation results
        """
        try:
            if db_type == "mongodb":
                return self._mongodb_operation(operation, collection, params)
            elif db_type == "redis":
                return self._redis_operation(operation, collection, params)
            else:
                return {"error": f"Unsupported NoSQL type: {db_type}"}

        except Exception as e:
            return {"error": f"NoSQL operation failed: {str(e)}"}

    def _mongodb_operation(self, operation: str, collection: str,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MongoDB operation."""
        try:
            from pymongo import MongoClient

            client = MongoClient(params.get('host', 'localhost'),
                               params.get('port', 27017))
            db = client[params.get('database', 'test')]
            coll = db[collection]

            if operation == "find":
                query = params.get('query', {})
                results = list(coll.find(query))
                # Convert ObjectId to string
                for doc in results:
                    doc['_id'] = str(doc['_id'])
                return {
                    'success': True,
                    'operation': 'find',
                    'results': results,
                    'count': len(results)
                }

            elif operation == "insert":
                document = params.get('document', {})
                result = coll.insert_one(document)
                return {
                    'success': True,
                    'operation': 'insert',
                    'inserted_id': str(result.inserted_id)
                }

            elif operation == "update":
                query = params.get('query', {})
                update = params.get('update', {})
                result = coll.update_many(query, {'$set': update})
                return {
                    'success': True,
                    'operation': 'update',
                    'modified_count': result.modified_count
                }

            elif operation == "delete":
                query = params.get('query', {})
                result = coll.delete_many(query)
                return {
                    'success': True,
                    'operation': 'delete',
                    'deleted_count': result.deleted_count
                }

            else:
                return {"error": f"Unknown operation: {operation}"}

        except ImportError:
            return {"error": "pymongo not installed. Install with: pip install pymongo"}
        except Exception as e:
            return {"error": f"MongoDB error: {str(e)}"}

    def _redis_operation(self, operation: str, key: str,
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Redis operation."""
        try:
            import redis

            client = redis.Redis(
                host=params.get('host', 'localhost'),
                port=params.get('port', 6379),
                db=params.get('db', 0),
                decode_responses=True
            )

            if operation == "get":
                value = client.get(key)
                return {
                    'success': True,
                    'operation': 'get',
                    'key': key,
                    'value': value
                }

            elif operation == "set":
                value = params.get('value')
                client.set(key, value)
                return {
                    'success': True,
                    'operation': 'set',
                    'key': key
                }

            elif operation == "delete":
                count = client.delete(key)
                return {
                    'success': True,
                    'operation': 'delete',
                    'deleted_count': count
                }

            elif operation == "keys":
                pattern = params.get('pattern', '*')
                keys = client.keys(pattern)
                return {
                    'success': True,
                    'operation': 'keys',
                    'keys': keys,
                    'count': len(keys)
                }

            else:
                return {"error": f"Unknown operation: {operation}"}

        except ImportError:
            return {"error": "redis not installed. Install with: pip install redis"}
        except Exception as e:
            return {"error": f"Redis error: {str(e)}"}
