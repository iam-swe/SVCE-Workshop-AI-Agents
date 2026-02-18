import os
from typing import List

from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

from langchain_core.tools import tool

@tool
def firecrawl_tool(query: str, num_results: int = 1) -> str:
    """
    Search the web and return cleaned, markdown-formatted academic content for a given topic.

    This tool performs a semantic search using Firecrawl, retrieves the most relevant pages,
    and extracts their main textual content in markdown format along with the source title
    and URL.

    Use this tool when:
    - The question requires up-to-date, real-world, or externally sourced information
    - Additional depth, definitions, examples, applications, or recent developments are needed
    - The topic is not fully covered by core model knowledge
    - Generating detailed, exam-oriented learning material that benefits from authoritative sources

    Do NOT use this tool when:
    - The answer can be generated from standard textbook knowledge
    - The query is simple, conceptual, or does not need external enrichment

    Args:
        query: The academic topic or concept to search for.
        num_results: Number of top relevant sources to retrieve (default: 3).

    Returns:
        A single string containing:
        - Title of each source
        - Source URL
        - Extracted markdown content

        If no relevant content is found, returns:
        "No relevant sources found."
    """
    
    app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    print("Starting firecrawl search with query ",query)

    search_result = app.search(query=query, limit=num_results)
    
    print("Search complete")

    if not search_result.web:
        return "No relevant sources found."

    contents: List[str] = []
    
    print("Starting scrape")

    for item in search_result.web:
        page = app.scrape(item.url)
        

        if not page or not page.markdown:
            continue

        markdown = page.markdown[:4000]

        contents.append(
            f"Title: {item.title}\n"
            f"Source: {item.url}\n"
            f"{markdown}"
        )
    print("Scrape complete")

    return "\n\n".join(contents)

def get_learner_tools():
    return [firecrawl_tool]
