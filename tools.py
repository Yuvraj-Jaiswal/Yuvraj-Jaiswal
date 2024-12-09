from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , SystemMessage
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from langchain_community.tools import TavilySearchResults
import os

load_dotenv()

@tool
def flight_price(query: str):
    """Get Flight price from one place to another given date of travel"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    flight_prompt = f"""
    Generate ( 1 flight ) estimated guess flight price based on the user query:
    Calculate price based on distance between places.
    from: The departure city or airport (e.g., "New York").
    to: The destination city or airport (e.g., "Los Angeles").
    
    Output should only be json nothing else like explanation - the estimated price in a JSON format containing:
    
    from: Departure location.
    to: Destination location.
    price: Estimated flight price in USD.
    
    The JSON structure should look like:
    [{{
      "air line" : "flight company name",
      "from": "start location",
      "to": "destination location",
      "date" : "date of travel",
      "price": "price in rupees add rupees symbol"
    }}]
    
    user query : {query}
    """

    response = llm.invoke([HumanMessage(content=flight_prompt)])
    return response.content.replace("```json","").replace("```","")


@tool
def send_email(sender_email, sender_password, recipient_email):
    """Send email to user given sender email , sender password , reciept email"""
    try:
        # SMTP server configuration for Gmail
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        # Create the email
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = "subject"

        # Establish connection with the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)  # Login to the email account
            server.send_message(message)  # Send the email

        return f"Email send successfully from {sender_email} to {recipient_email}"

    except Exception as e:
        print(f"Error: {e}")


import os
import requests
from typing import List, Dict, Optional


def tavily_search(
        query: str,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = 'basic',
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
) -> Dict:

    api_key = api_key or os.getenv('TAVILY_API_KEY')
    if not api_key:
        raise ValueError("Tavily API key must be provided or set in TAVILY_API_KEY environment variable")

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth
    }

    if include_domains:
        payload["include_domains"] = include_domains

    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        response = requests.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error performing search: {e}")
        return {"error": str(e)}


def clean_tavily_results(search_results: Dict) -> List[Dict]:
    if 'error' in search_results:
        return []

    return [
        {
            'title': result.get('title', 'No Title'),
            'url': result.get('url', ''),
            'content': result.get('content', '')
        }
        for result in search_results.get('results', [])
    ]


@tool
def search(query : str):
    """ Search web tool , provided latest answers from web given search query"""
    results = tavily_search(
        query=query,
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=1,
    )

    clean_results = clean_tavily_results(results)
    search_data = ""
    for result in clean_results:
        search_data += f""" Title: {result['title']}")
                    URL: {result['url']}")
                    Content: {result['content'][:1000]} \n\n"""

    return search_data

