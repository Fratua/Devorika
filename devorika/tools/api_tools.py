"""
API Development Tools for Devorika
REST, GraphQL, OpenAPI, and API testing capabilities.
"""

import os
import json
import subprocess
from typing import Dict, Any, Optional, List
from .base import Tool


class APIScaffoldTool(Tool):
    """
    Generate API boilerplate code for REST and GraphQL.
    """

    name = "api_scaffold"
    description = "Generate API boilerplate for REST or GraphQL"

    def execute(self, api_type: str = "rest", framework: str = "fastapi",
                **params) -> Dict[str, Any]:
        """
        Generate API scaffold.

        Args:
            api_type: API type (rest, graphql)
            framework: Framework (fastapi, flask, django, express)
            **params: Additional parameters

        Returns:
            Dict with generated files
        """
        try:
            if api_type == "rest":
                return self._generate_rest_api(framework, params)
            elif api_type == "graphql":
                return self._generate_graphql_api(framework, params)
            else:
                return {"error": f"Unknown API type: {api_type}"}

        except Exception as e:
            return {"error": f"API scaffold generation failed: {str(e)}"}

    def _generate_rest_api(self, framework: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate REST API scaffold."""
        output_dir = params.get('output_dir', 'api')
        os.makedirs(output_dir, exist_ok=True)

        if framework == "fastapi":
            main_file = os.path.join(output_dir, 'main.py')
            models_file = os.path.join(output_dir, 'models.py')
            routes_file = os.path.join(output_dir, 'routes.py')

            # Main application
            main_content = '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

app = FastAPI(
    title="Devorika API",
    description="API built by Devorika AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to Devorika API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

            # Models
            models_content = '''from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ItemBase(BaseModel):
    name: str = Field(..., description="Item name")
    description: Optional[str] = Field(None, description="Item description")
    price: float = Field(..., gt=0, description="Item price")

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ItemUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
'''

            # Routes
            routes_content = '''from fastapi import APIRouter, HTTPException, status
from typing import List
from models import Item, ItemCreate, ItemUpdate

router = APIRouter()

# In-memory storage (replace with database)
items_db: List[Item] = []

@router.get("/items", response_model=List[Item])
async def get_items():
    """Get all items"""
    return items_db

@router.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    """Get item by ID"""
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@router.post("/items", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate):
    """Create new item"""
    from datetime import datetime
    new_item = Item(
        id=len(items_db) + 1,
        **item.dict(),
        created_at=datetime.now()
    )
    items_db.append(new_item)
    return new_item

@router.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item_update: ItemUpdate):
    """Update item"""
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            update_data = item_update.dict(exclude_unset=True)
            updated_item = item.copy(update=update_data)
            items_db[idx] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")

@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    """Delete item"""
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            items_db.pop(idx)
            return
    raise HTTPException(status_code=404, detail="Item not found")
'''

            with open(main_file, 'w') as f:
                f.write(main_content)
            with open(models_file, 'w') as f:
                f.write(models_content)
            with open(routes_file, 'w') as f:
                f.write(routes_content)

            return {
                'success': True,
                'framework': 'fastapi',
                'files': [main_file, models_file, routes_file],
                'output_dir': output_dir
            }

        elif framework == "flask":
            main_file = os.path.join(output_dir, 'app.py')

            flask_content = '''from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# In-memory storage
items = []

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to Devorika API'})

@app.route('/api/v1/items', methods=['GET'])
def get_items():
    return jsonify(items)

@app.route('/api/v1/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((i for i in items if i['id'] == item_id), None)
    if item:
        return jsonify(item)
    return jsonify({'error': 'Item not found'}), 404

@app.route('/api/v1/items', methods=['POST'])
def create_item():
    data = request.get_json()
    new_item = {
        'id': len(items) + 1,
        **data
    }
    items.append(new_item)
    return jsonify(new_item), 201

@app.route('/api/v1/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    item = next((i for i in items if i['id'] == item_id), None)
    if item:
        data = request.get_json()
        item.update(data)
        return jsonify(item)
    return jsonify({'error': 'Item not found'}), 404

@app.route('/api/v1/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global items
    items = [i for i in items if i['id'] != item_id]
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
'''

            with open(main_file, 'w') as f:
                f.write(flask_content)

            return {
                'success': True,
                'framework': 'flask',
                'files': [main_file],
                'output_dir': output_dir
            }

        else:
            return {"error": f"Unsupported framework: {framework}"}

    def _generate_graphql_api(self, framework: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate GraphQL API scaffold."""
        output_dir = params.get('output_dir', 'api')
        os.makedirs(output_dir, exist_ok=True)

        schema_file = os.path.join(output_dir, 'schema.py')
        main_file = os.path.join(output_dir, 'main.py')

        # GraphQL schema
        schema_content = '''import strawberry
from typing import List, Optional

@strawberry.type
class Item:
    id: int
    name: str
    description: Optional[str]
    price: float

# In-memory storage
items_db: List[Item] = []

@strawberry.type
class Query:
    @strawberry.field
    def items(self) -> List[Item]:
        return items_db

    @strawberry.field
    def item(self, id: int) -> Optional[Item]:
        return next((item for item in items_db if item.id == id), None)

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_item(self, name: str, price: float, description: Optional[str] = None) -> Item:
        new_item = Item(
            id=len(items_db) + 1,
            name=name,
            description=description,
            price=price
        )
        items_db.append(new_item)
        return new_item

    @strawberry.mutation
    def update_item(self, id: int, name: Optional[str] = None,
                   description: Optional[str] = None, price: Optional[float] = None) -> Optional[Item]:
        item = next((item for item in items_db if item.id == id), None)
        if item:
            if name:
                item.name = name
            if description:
                item.description = description
            if price:
                item.price = price
            return item
        return None

    @strawberry.mutation
    def delete_item(self, id: int) -> bool:
        global items_db
        initial_length = len(items_db)
        items_db = [item for item in items_db if item.id != id]
        return len(items_db) < initial_length

schema = strawberry.Schema(query=Query, mutation=Mutation)
'''

        # Main app
        main_content = '''from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from schema import schema

app = FastAPI()

graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
async def root():
    return {"message": "GraphQL API - Visit /graphql for GraphiQL interface"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

        with open(schema_file, 'w') as f:
            f.write(schema_content)
        with open(main_file, 'w') as f:
            f.write(main_content)

        return {
            'success': True,
            'api_type': 'graphql',
            'files': [schema_file, main_file],
            'output_dir': output_dir
        }


class OpenAPIGeneratorTool(Tool):
    """
    Generate OpenAPI/Swagger documentation.
    """

    name = "openapi_generator"
    description = "Generate OpenAPI specification from API code"

    def execute(self, api_file: str, output_file: str = "openapi.json",
                title: str = "API Documentation") -> Dict[str, Any]:
        """
        Generate OpenAPI spec.

        Args:
            api_file: API source file
            output_file: Output OpenAPI spec file
            title: API title

        Returns:
            Dict with generation results
        """
        try:
            # For FastAPI, it generates OpenAPI automatically
            # This is a simplified version
            spec = {
                "openapi": "3.0.0",
                "info": {
                    "title": title,
                    "version": "1.0.0",
                    "description": "API documentation generated by Devorika"
                },
                "paths": {},
                "components": {
                    "schemas": {}
                }
            }

            with open(output_file, 'w') as f:
                json.dump(spec, f, indent=2)

            return {
                'success': True,
                'output_file': output_file,
                'message': 'OpenAPI spec generated. Customize as needed.'
            }

        except Exception as e:
            return {"error": f"OpenAPI generation failed: {str(e)}"}


class APITestGeneratorTool(Tool):
    """
    Generate API tests automatically.
    """

    name = "api_test_generator"
    description = "Generate automated tests for APIs"

    def execute(self, api_type: str = "rest", base_url: str = "http://localhost:8000",
                endpoints: List[Dict[str, Any]] = None, output_file: str = "test_api.py") -> Dict[str, Any]:
        """
        Generate API tests.

        Args:
            api_type: API type (rest, graphql)
            base_url: Base URL of API
            endpoints: List of endpoints to test
            output_file: Output test file

        Returns:
            Dict with generation results
        """
        try:
            if api_type == "rest":
                test_content = f'''import pytest
import requests

BASE_URL = "{base_url}"

class TestAPI:
    """API integration tests generated by Devorika"""

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{{BASE_URL}}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_get_items(self):
        """Test GET /api/v1/items"""
        response = requests.get(f"{{BASE_URL}}/api/v1/items")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_item(self):
        """Test POST /api/v1/items"""
        item_data = {{
            "name": "Test Item",
            "description": "Test Description",
            "price": 99.99
        }}
        response = requests.post(f"{{BASE_URL}}/api/v1/items", json=item_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == item_data["name"]
        assert "id" in data

    def test_get_item_by_id(self):
        """Test GET /api/v1/items/{{id}}"""
        # First create an item
        item_data = {{"name": "Test", "price": 10.0}}
        create_response = requests.post(f"{{BASE_URL}}/api/v1/items", json=item_data)
        item_id = create_response.json()["id"]

        # Then get it
        response = requests.get(f"{{BASE_URL}}/api/v1/items/{{item_id}}")
        assert response.status_code == 200
        assert response.json()["id"] == item_id

    def test_update_item(self):
        """Test PUT /api/v1/items/{{id}}"""
        # Create item
        item_data = {{"name": "Original", "price": 10.0}}
        create_response = requests.post(f"{{BASE_URL}}/api/v1/items", json=item_data)
        item_id = create_response.json()["id"]

        # Update it
        update_data = {{"name": "Updated"}}
        response = requests.put(f"{{BASE_URL}}/api/v1/items/{{item_id}}", json=update_data)
        assert response.status_code == 200
        assert response.json()["name"] == "Updated"

    def test_delete_item(self):
        """Test DELETE /api/v1/items/{{id}}"""
        # Create item
        item_data = {{"name": "To Delete", "price": 10.0}}
        create_response = requests.post(f"{{BASE_URL}}/api/v1/items", json=item_data)
        item_id = create_response.json()["id"]

        # Delete it
        response = requests.delete(f"{{BASE_URL}}/api/v1/items/{{item_id}}")
        assert response.status_code == 204

        # Verify it's deleted
        get_response = requests.get(f"{{BASE_URL}}/api/v1/items/{{item_id}}")
        assert get_response.status_code == 404

    def test_item_not_found(self):
        """Test 404 for non-existent item"""
        response = requests.get(f"{{BASE_URL}}/api/v1/items/99999")
        assert response.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

            elif api_type == "graphql":
                test_content = f'''import pytest
import requests

GRAPHQL_URL = "{base_url}/graphql"

def graphql_query(query: str, variables: dict = None):
    """Helper function to execute GraphQL queries"""
    payload = {{"query": query}}
    if variables:
        payload["variables"] = variables
    response = requests.post(GRAPHQL_URL, json=payload)
    return response.json()

class TestGraphQLAPI:
    """GraphQL API tests generated by Devorika"""

    def test_query_items(self):
        """Test items query"""
        query = """
        query {{
            items {{
                id
                name
                price
            }}
        }}
        """
        result = graphql_query(query)
        assert "data" in result
        assert "items" in result["data"]

    def test_create_item_mutation(self):
        """Test createItem mutation"""
        mutation = """
        mutation CreateItem($name: String!, $price: Float!) {{
            createItem(name: $name, price: $price) {{
                id
                name
                price
            }}
        }}
        """
        variables = {{"name": "Test Item", "price": 99.99}}
        result = graphql_query(mutation, variables)
        assert "data" in result
        assert result["data"]["createItem"]["name"] == "Test Item"

    def test_query_item_by_id(self):
        """Test item query by ID"""
        # First create an item
        create_mutation = """
        mutation {{
            createItem(name: "Test", price: 10.0) {{
                id
            }}
        }}
        """
        create_result = graphql_query(create_mutation)
        item_id = create_result["data"]["createItem"]["id"]

        # Then query it
        query = f"""
        query {{
            item(id: {item_id}) {{
                id
                name
                price
            }}
        }}
        """
        result = graphql_query(query)
        assert result["data"]["item"]["id"] == item_id

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
            else:
                return {"error": f"Unsupported API type: {api_type}"}

            with open(output_file, 'w') as f:
                f.write(test_content)

            return {
                'success': True,
                'api_type': api_type,
                'output_file': output_file,
                'message': 'API tests generated successfully'
            }

        except Exception as e:
            return {"error": f"Test generation failed: {str(e)}"}


class APIClientGeneratorTool(Tool):
    """
    Generate API client libraries in various languages.
    """

    name = "api_client_generator"
    description = "Generate API client libraries (Python, JavaScript, etc.)"

    def execute(self, openapi_spec: str, language: str = "python",
                output_dir: str = "client") -> Dict[str, Any]:
        """
        Generate API client.

        Args:
            openapi_spec: Path to OpenAPI specification
            language: Target language (python, javascript, typescript)
            output_dir: Output directory

        Returns:
            Dict with generation results
        """
        try:
            if language == "python":
                return self._generate_python_client(openapi_spec, output_dir)
            elif language == "javascript":
                return self._generate_js_client(openapi_spec, output_dir)
            else:
                return {"error": f"Unsupported language: {language}"}

        except Exception as e:
            return {"error": f"Client generation failed: {str(e)}"}

    def _generate_python_client(self, spec: str, output_dir: str) -> Dict[str, Any]:
        """Generate Python API client."""
        os.makedirs(output_dir, exist_ok=True)
        client_file = os.path.join(output_dir, 'client.py')

        client_content = '''import requests
from typing import Dict, Any, Optional, List

class APIClient:
    """Auto-generated API client by Devorika"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})

    def get_items(self) -> List[Dict[str, Any]]:
        """Get all items"""
        response = self.session.get(f"{self.base_url}/api/v1/items")
        response.raise_for_status()
        return response.json()

    def get_item(self, item_id: int) -> Dict[str, Any]:
        """Get item by ID"""
        response = self.session.get(f"{self.base_url}/api/v1/items/{item_id}")
        response.raise_for_status()
        return response.json()

    def create_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item"""
        response = self.session.post(f"{self.base_url}/api/v1/items", json=item_data)
        response.raise_for_status()
        return response.json()

    def update_item(self, item_id: int, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update item"""
        response = self.session.put(f"{self.base_url}/api/v1/items/{item_id}", json=item_data)
        response.raise_for_status()
        return response.json()

    def delete_item(self, item_id: int) -> None:
        """Delete item"""
        response = self.session.delete(f"{self.base_url}/api/v1/items/{item_id}")
        response.raise_for_status()
'''

        with open(client_file, 'w') as f:
            f.write(client_content)

        return {
            'success': True,
            'language': 'python',
            'files': [client_file],
            'output_dir': output_dir
        }

    def _generate_js_client(self, spec: str, output_dir: str) -> Dict[str, Any]:
        """Generate JavaScript API client."""
        os.makedirs(output_dir, exist_ok=True)
        client_file = os.path.join(output_dir, 'client.js')

        client_content = '''/**
 * Auto-generated API client by Devorika
 */

class APIClient {
    constructor(baseURL, apiKey = null) {
        this.baseURL = baseURL.replace(/\\/+$/, '');
        this.apiKey = apiKey;
    }

    async request(endpoint, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        const response = await fetch(`${this.baseURL}${endpoint}`, {
            ...options,
            headers
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return response.json();
    }

    async getItems() {
        return this.request('/api/v1/items');
    }

    async getItem(itemId) {
        return this.request(`/api/v1/items/${itemId}`);
    }

    async createItem(itemData) {
        return this.request('/api/v1/items', {
            method: 'POST',
            body: JSON.stringify(itemData)
        });
    }

    async updateItem(itemId, itemData) {
        return this.request(`/api/v1/items/${itemId}`, {
            method: 'PUT',
            body: JSON.stringify(itemData)
        });
    }

    async deleteItem(itemId) {
        return this.request(`/api/v1/items/${itemId}`, {
            method: 'DELETE'
        });
    }
}

module.exports = APIClient;
'''

        with open(client_file, 'w') as f:
            f.write(client_content)

        return {
            'success': True,
            'language': 'javascript',
            'files': [client_file],
            'output_dir': output_dir
        }
