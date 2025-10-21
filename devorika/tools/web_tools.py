"""
Web browsing and research tools
"""

from typing import Optional
from devorika.tools.base import Tool


class WebSearchTool(Tool):
    """Search the web for information."""

    name = "web_search"
    description = "Search the web using DuckDuckGo"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5)"
            }
        },
        "required": ["query"]
    }

    def execute(self, query: str, num_results: int = 5) -> str:
        """Search the web."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=num_results)):
                    results.append(f"{i+1}. {result['title']}\n   {result['href']}\n   {result['body']}\n")

            return "\n".join(results) if results else "No results found"
        except ImportError:
            return "duckduckgo-search not installed. Run: pip install duckduckgo-search"
        except Exception as e:
            return f"Error searching web: {str(e)}"


class FetchURLTool(Tool):
    """Fetch content from a URL."""

    name = "fetch_url"
    description = "Fetch and extract text content from a URL"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch"
            }
        },
        "required": ["url"]
    }

    def execute(self, url: str) -> str:
        """Fetch URL content."""
        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Limit output
            if len(text) > 5000:
                text = text[:5000] + "\n\n[Content truncated...]"

            return text
        except ImportError:
            return "Required packages not installed. Run: pip install requests beautifulsoup4"
        except Exception as e:
            return f"Error fetching URL: {str(e)}"


class ReadDocumentationTool(Tool):
    """Read and search programming documentation."""

    name = "read_docs"
    description = "Search and read documentation for programming libraries"
    parameters = {
        "type": "object",
        "properties": {
            "library": {
                "type": "string",
                "description": "Library name (e.g., 'python', 'numpy', 'requests')"
            },
            "topic": {
                "type": "string",
                "description": "Specific topic or function to look up"
            }
        },
        "required": ["library", "topic"]
    }

    def execute(self, library: str, topic: str) -> str:
        """Read documentation."""
        # Use web search to find documentation
        search_tool = WebSearchTool()
        query = f"{library} {topic} documentation"
        return search_tool.execute(query, num_results=3)
