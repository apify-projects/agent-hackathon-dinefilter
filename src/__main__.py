from langchain_core.prompts import ChatPromptTemplate
import operator
from itertools import chain
from datetime import timedelta
import asyncio
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import json
from typing import Dict, List, Optional, TypedDict, Annotated, Union, Literal, Annotated
from dotenv import load_dotenv
from apify import Actor
from pydantic import BaseModel, Field

load_dotenv()

class Settings(BaseModel):
    openai_api_key: Annotated[str, Field(alias="OPENAI_API_KEY")]

settings = Settings.model_validate(os.environ)

# Type definitions
class RestaurantQuery(TypedDict):
    location: str
    restaurant_name: Optional[str]
    date: Optional[str]
    cuisine_type: Optional[str]

class RestaurantData(TypedDict):
    name: str
    address: str
    location: Optional[Dict[str, float]]
    google_rating: Optional[float]
    google_reviews_count: Optional[int]
    google_map_url: str
    web_url: Optional[str]
    web_url_content: Optional[str]
    menu_url: Optional[str]
    menu_url_content: Optional[str]
    top_positive_reviews: List[str]
    top_negative_reviews: List[str]
    price_level: Optional[str]
    cuisine: List[str]

class AgentState(TypedDict):
    query: Annotated[RestaurantQuery, lambda _, new: new]
    restaurants: Annotated[list[RestaurantData], lambda _, new: new]
    top_matches: Annotated[list[RestaurantData], lambda _, new: new]
    current_step: Annotated[str, lambda _, new: new]
    error: Annotated[Optional[str], lambda _, new: new]
    final_response: Annotated[Optional[str], lambda _, new: new]

# APIFY actors
GOOGLE_MAPS_SCRAPER = "compass/crawler-google-places"
WCC = "apify/website-content-crawler"

# LLM setup
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=settings.openai_api_key)

# State transformation functions
async def parse_user_query(state: AgentState) -> AgentState:
    """Parse the user query to extract location, restaurant name, date, etc."""
    print("Parsing user query")
    user_message = state["query"].get("user_input", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Extract the following information from the user query:
        1. Location (city, neighborhood, address) - add country name
        2. Restaurant name (if specified)
        3. Date (if specified)
        4. Cuisine type (if specified)
        
        Format your response as JSON with keys: location, restaurant_name, date, cuisine_type.
        If information is not provided, use null for that field.
        """),
        ("human", "{user_input}")
    ])
    
    formatted_prompt = prompt.invoke({"user_input": user_message})
    response = await llm.ainvoke(formatted_prompt.to_messages())
    parsed_query = json.loads(response.content)

    # Create a new query object that preserves the original user_input
    updated_query = {
        **state["query"],  # Keep all existing fields
        **parsed_query,  # Add the parsed fields
    }

    return {**state, "query": updated_query, "current_step": "query_parsed"}

async def search_google_maps(state: AgentState) -> AgentState:
    """Use APIFY Google Maps Scraper to find restaurants, with caching for debugging."""
    print("Searching Google Maps")
    try:
        location = state["query"]["location"]
        restaurant_name = state["query"].get("restaurant_name")
        cuisine_type = state["query"].get("cuisine_type")

        search_term = restaurant_name or f"{cuisine_type} restaurants" if cuisine_type else "restaurants"

        run_input = {
            "searchStringsArray": [f"{search_term} in {location}"],
            "maxCrawledPlaces": 5,
            "language": "en",
            "maxImages": 0,
            "includeReviews": True,
            "maxReviews": 15,
        }

        run = await Actor.call(GOOGLE_MAPS_SCRAPER, run_input=run_input, timeout=timedelta(seconds=90))
        items = (await Actor.apify_client.dataset(run.default_dataset_id).list_items()).items

        restaurants = []
        for item in items:
            rest_data = {
                "name": item.get("title", ""),
                "address": item.get("address", ""),
                "location": item.get("location"),
                "google_rating": item.get("totalScore"),
                "google_reviews_count": item.get("reviewsCount"),
                "google_map_url": item.get("url", ""),
                "web_url": item.get("website", ""),
                "menu_url": item.get("menu", ""),
                "price_level": item.get("priceLevel"),
                "cuisine": item.get("categories", []),
                "top_positive_reviews": [r.get("text", "") for r in item.get("reviews", []) if r.get("stars", 0) >= 4][:3],
                "top_negative_reviews": [r.get("text", "") for r in item.get("reviews", []) if r.get("stars", 0) <= 2][:3],
            }
            restaurants.append(rest_data)

        return {**state, "restaurants": restaurants, "current_step": "google_maps_completed"}

    except Exception as e:
        return {**state, "error": f"Error in Google Maps search: {str(e)}", "current_step": "error"}

async def crawl_websites(state: AgentState) -> AgentState:
    print("Website crawling")
    """Use website content crawler to fetch information about the restaurant from their website."""
    try:
        if not state.get("top_matches"):
            return {
                **state,
                "error": "Not able to scrape restaurants website",
                "current_step": "error",
            }

        updated_restaurants = []
        start_urls = [*chain(
            ({"url": url} for item in state["top_matches"] if (url := item.get("menu_url"))),
            ({"url": url} for item in state["top_matches"] if (url := item.get("web_url")))
        )]

        if not start_urls:
            return {
                **state,
                "current_step": "content_crawl_completed (no urls)"
            }

        run = await Actor.call(
            WCC, 
            run_input={
                "startUrls": start_urls, 
                "maxCrawlDepth": 0,
            }
        )
        wcc_output = {item["url"]: item for item in (
            await Actor.apify_client.dataset(run.default_dataset_id).list_items()
        ).items}
        
        for item in state["top_matches"]:
            item_copy = {**item}
            updated_restaurants.append(item_copy)

            output_item = wcc_output.get(item["menu_url"])
            if output_item is not None:
                item_copy["menu_url_content"] = output_item["text"]

            output_item = wcc_output.get(item["web_url"])
            if output_item is not None:
                item_copy["web_url_content"] = output_item["text"]

        return {
            **state,
            "top_matches": updated_restaurants,
            "current_step": "content_crawl_completed",
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in WCC crawl: {str(e)}",
            "current_step": "error",
        }

async def analyze_and_summarize(state: AgentState) -> AgentState:
    """Use LLM to analyze and summarize the restaurant data."""
    print("Analysing and summarizing")
    try:
        if not state.get("restaurants"):
            return {
                **state,
                "error": "No restaurants to analyze",
                "current_step": "error",
            }

        analyzed_restaurants = []
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze the restaurant data provided and create a comprehensive summary that includes:
            1. Key themes from positive reviews
            2. Key themes from negative reviews
        
            Focus on being objective and data-driven in your analysis, make it relatively consise. 
            Don't complain about not having much data. Answer straight to the point.
            """),
            ("human", "Restaurant data:\n{restaurant_info}")
        ])

        for restaurant in state["restaurants"]:
            # Prepare the restaurant data for the LLM
            restaurant_info = restaurant.get("top_positive_reviews", [])
            restaurant_info += restaurant.get("top_negative_reviews", [])
            
            # Format the prompt with this restaurant's data
            formatted_prompt = analysis_prompt.invoke({"restaurant_info": restaurant_info})
            response = await llm.ainvoke(formatted_prompt.to_messages())

            # Add the analysis
            restaurant["review_analysis"] = response.content
            analyzed_restaurants.append(restaurant)

        return {
            **state,
            "restaurants": analyzed_restaurants,
            "current_step": "analysis_completed",
        }

    except Exception as e:
        return {
            **state,
            "error": f"Error in analysis: {str(e)}",
            "current_step": "error",
        }

async def pick_top_restaurants(state: AgentState) -> AgentState:
    print("Selecting top restaurants")
    try:
        data = [
            {
                "restaurant_index": index, 
                "review_analysis": item["review_analysis"],
                "google_rating": item["google_rating"],
                "price_level": item["price_level"],
                "address": item["address"],
                "name": item["name"],
                "cuisine": item["cuisine"],
            } for index, item in enumerate(state["restaurants"])
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", """
            Pick up to three restaurants that best match the user criteria. If the criteria are too vague, pick the restaurants with the best quality overall.
            The result should be a JSON list of numbers that contains the `restaurant_index` values of the selected restaurants with no extra characters around it.
            
            User criteria: 
            {user_criteria}
            
            Restaurant data: 
            {restaurant_data}
            """)
        ])
        
        formatted_prompt = prompt.invoke({
            "user_criteria": json.dumps(state["query"], indent=2, ensure_ascii=False),
            "restaurant_data": json.dumps(data, indent=2, ensure_ascii=False)
        })
        
        response = await llm.ainvoke(formatted_prompt.to_messages())
        top_matches = [state["restaurants"][index] for index in json.loads(response.content)]
        
        return {**state, "top_matches": top_matches, "current_step": "pick_top_restaurants_done"}
    except Exception as e:
        return {
            **state,
            "error": f"Error when picking top restaurants: {str(e)}",
            "current_step": "error",
        }

async def format_final_response(state: AgentState) -> AgentState:
    """Format the final response to the user with concise review summaries and menu highlights."""
    print("Formatting final response")
    if state.get("error"):
        return state

    try:
        restaurants = state.get("top_matches", [])
        query = state.get("query", {})

        sorted_restaurants = sorted(
            restaurants, 
            key=lambda x: x.get("google_rating") if x.get("google_rating") is not None else 0,
            reverse=True
        )
        
        # Format restaurant data in a more concise way
        restaurant_summaries = []
        for restaurant in sorted_restaurants:
            summary = {
                "name": restaurant.get("name", ""),
                "address": restaurant.get("address", ""),
                "rating": restaurant.get("google_rating", "N/A"),
                "price_level": restaurant.get("price_level", "N/A"),
                "cuisine": restaurant.get("cuisine", []),
                "map_url": restaurant.get("google_map_url", ""),
                "menu_url": restaurant.get("menu_url", ""),
                "web_url": restaurant.get("web_url", ""),
                "review_analysis": restaurant.get("review_analysis", "No review analysis available"),
                "menu_content": restaurant.get("menu_url_content") or restaurant.get("web_url_content", "")
            }
            restaurant_summaries.append(summary)

        response_prompt = ChatPromptTemplate.from_messages([
            ("human", """
            You are an assistant that generates structured JSON responses.

            Summarize the top restaurant options for {location}.

            Your response must be a JSON object with the following structure:
            {{
                "restaurants": [
                    {{
                        "name": "Restaurant Name",
                        "address": "Restaurant Address",
                        "location": "Restaurant Longtitude and Latitude",
                        "rating": "4.5",
                        "price_level": "$$",
                        "cuisine": ["Cuisine1", "Cuisine2"],
                        "map_url": "Google Maps Link",
                        "menu_url": "Menu Link",
                        "web_url": "Website Link",
                        "review_summary": "Brief review summary",
                        "menu_highlights": "Key menu items if available"
                    }}
                ],
                "notes": "Mention reservation recommendations if a date is provided: {date}. 
                        Prioritize restaurants matching cuisine type: {cuisine_type}."
                        If the cuisine type in the input Restaurant data is too vague, adjust it in the output to fit the name and the reviews.
            }}

            DO NOT return additional text, only a valid JSON response.

            Restaurant data:
            {restaurant_data}
            """)
        ])
        
        formatted_prompt = response_prompt.invoke({
            "location": query.get("location", "the requested location"),
            "date": query.get("date"),
            "cuisine_type": query.get("cuisine_type"),
            "restaurant_data": json.dumps(restaurant_summaries, indent=2, ensure_ascii=False)
        })
        
        response = await llm.ainvoke(formatted_prompt.to_messages())
        final = json.loads(response.content)
        
        return {
            **state,
            "final_response": final,
            "current_step": "completed",
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error formatting response: {str(e)}",
            "current_step": "error",
        }

async def handle_error(state: AgentState) -> Dict:
    """Handle any errors that occurred during processing."""
    error_message = state.get("error", "An unknown error occurred")
    print(f"Handling error: {error_message}")

    prompt = f"""
    The restaurant search encountered an error: {error_message}
    
    Please generate a helpful response to the user explaining what went wrong
    and suggesting alternative actions they could take.
    """

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    return {
        **state,
        "final_response": response.content,
        "current_step": "error_handled",
    }

# Define the graph
def build_restaurant_agent():
    """Build and return the restaurant agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_query", parse_user_query)
    workflow.add_node("search_google_maps", search_google_maps)
    workflow.add_node("analyze", analyze_and_summarize)
    workflow.add_node("pick_top_restaurants", pick_top_restaurants)
    workflow.add_node("crawl_websites", crawl_websites)
    workflow.add_node("format_response", format_final_response)
    workflow.add_node("handle_error", handle_error)

    workflow.add_edge("parse_query", "search_google_maps")
    workflow.add_edge("search_google_maps", "analyze")
    workflow.add_edge("analyze", "pick_top_restaurants")
    workflow.add_edge("pick_top_restaurants", "crawl_websites")
    workflow.add_edge("crawl_websites", "format_response")
    workflow.add_edge("format_response", END)

    # Error handling
    workflow.add_conditional_edges(
        "parse_query",
        lambda state: "handle_error" if state.get("error") else "search_google_maps",
    )
    workflow.add_conditional_edges(
        "search_google_maps",
        lambda state: "handle_error" if state.get("error") else "analyze",
    )
    workflow.add_conditional_edges(
        "analyze",
        lambda state: "handle_error" if state.get("error") else "pick_top_restaurants",
    )
    workflow.add_conditional_edges(
        "pick_top_restaurants",
        lambda state: "handle_error" if state.get("error") else "crawl_websites",
    )
    workflow.add_conditional_edges(
        "crawl_websites",
        lambda state: "handle_error" if state.get("error") else "format_response",
    )
    workflow.add_conditional_edges(
        "format_response",
        lambda state: "handle_error" if state.get("error") else END,
    )

    workflow.add_edge("handle_error", END)

    workflow.set_entry_point("parse_query")

    return workflow.compile()

async def get_restaurant_recommendations(user_query: str):
    """Run the restaurant agent with a user query."""
    agent = build_restaurant_agent()

    initial_state = {
        "query": {"user_input": user_query},
        "restaurants": [],
        "top_matches": [],
        "current_step": "started",
        "error": None,
    }

    result = await agent.ainvoke(initial_state)
    await Actor.set_value("final_state", result)

    if "final_response" in result:
        return result["final_response"]
    elif "error" in result:
        raise RuntimeError(f"There was an error: {result['error']}")
    else:
        raise RuntimeError("Unable to process your request. Please try again with more specific details.")

class Input(BaseModel):
    prompt: Annotated[str, Field()]

async def main() -> None:
    async with Actor:
        input = Input.model_validate(await Actor.get_input())
        try:
            response = await get_restaurant_recommendations(input.prompt)
            for item in response["restaurants"]:
                await Actor.push_data(item)
        except RuntimeError as ex:
            print(str(ex))

if __name__ == "__main__":
    asyncio.run(main())