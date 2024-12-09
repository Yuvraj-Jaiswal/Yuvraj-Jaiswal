from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from tools import flight_price
from draw import show_graph

load_dotenv()

city_selector = ChatOpenAI(model="gpt-4o-mini" , temperature=0.8)
activity_selector = ChatOpenAI(model="gpt-4o-mini" , temperature=0.7)
trip_maker = ChatOpenAI(model="gpt-4o-mini" , temperature=0.2)
trip_maker = trip_maker.bind_tools([flight_price])
flight_price_node = ToolNode([flight_price])

def should_continue(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "flight_price"
    return END


# # Define the function that calls the model
def call_city_selector(state: MessagesState):
    """Research and suggest potential destinations"""
    user_requirements = state['messages'][-1].content

    # Use LLM to generate destination suggestions
    destination_prompt = f"""
        Act as a travel expert. Based on the following user requirements, 
        suggest 3 potential Cities with brief justifications:

        Requirements: {user_requirements}

        For each destination, provide:
        - City
        - Why it matches the requirements
        - Estimated budget range
        - Best time to visit
        
        output format - [
            {{
                "destination name" : {{
                    "City" : "City Name",
                    "Budget Range" : "Estimated budget range",
                    "Best Time" : "Best time to visit",
                    "Explaination" : "Why it matches the requirements"
                }}  
            }}
        ]
        """

    response = city_selector.invoke([
        HumanMessage(content=destination_prompt)
    ])

    return {"messages": [response]}


def call_activity_selector(state: MessagesState):
    """Research and suggest potential activities"""
    city_selector_out = state['messages'][-1].content.replace("```json" , "").replace("```" , "")

    # Use LLM to generate activity suggestions
    activity_prompt = f"""
        Act as a travel activity expert. Based on the following cities provided, 
        suggest 2 potential activities for each given city with brief justifications:

        Give activities for each Cities Name: {city_selector_out}

        For each city, provide:
        - Activity Name
        - Why it matches the requirements
        - Ideal time/season for the activity

        Output format:
        [
            {{
                "City name (array of 3 activities details for city)": [ {{
                    "Activity Name": "Activity Name",
                    "Ideal Time": "Ideal time/season for the activity",
                    "Explanation": "Why it matches the requirements"
                }} ]
            }}
        ]
        """

    response = activity_selector.invoke([
        HumanMessage(content=activity_prompt)
    ])

    return {"messages": [response]}


def call_trip_maker(state: MessagesState):
    """Research and suggest potential activities"""
    if "air line" in state['messages'][-1].content:
        context = "Latest Flight Prices -> "
        for message in state['messages'][::-1]:
            if "air line" in message.content:
                context += message.content
            elif "Activity Name" in message.content:
                context += message.content
                break
    else:
        context = state['messages'][-1].content.replace("```json" , "").replace("```" , "")

    print("############ context : " , context)

    # Use LLM to generate activity suggestions
    trip_maker_prompt = f"""
        You are an expert travel planner. Using the details provided below, create a travel itinerary. The itinerary should be engaging, practical, and tailored for a traveler looking to explore and enjoy these activities in the cities listed.
    
        Details:
        {context}
        
        start date: 24/01/2025 
    
        Instructions:
        1. Plan a trip, allocating appropriate days to each city according to start date.
        2. Ensure activities are spread out evenly and fit well into the schedule for each day.
        3. Also add travel expense and Travel time from from one city to other.
        4. Calculate Latest flight price for path of travel cities ( eg 1 -> 2 -> 3).
        5. Provide the trip summary in a clear, day-by-day format.
    
        Start your trip plan now : 
    """

    response = trip_maker.invoke([
        HumanMessage(content=trip_maker_prompt)
    ])

    return {"messages": [response]}


graph = StateGraph(MessagesState)

# Define the two nodes we will cycle between
graph.add_node("city_sector", call_city_selector)
graph.add_node("activity_sector", call_activity_selector)
graph.add_node("trip_maker", call_trip_maker)
graph.add_node("flight_price", flight_price_node)

graph.add_edge(START, "city_sector")
graph.add_edge("city_sector", "activity_sector")
graph.add_edge("activity_sector" , "trip_maker")
graph.add_conditional_edges("trip_maker" , should_continue , ["flight_price", END])
graph.add_edge("flight_price" , "trip_maker")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

show = False

if not show:
    while True:
        prompt = input("Enter your Prompt : ")
        for chunk in app.stream(
            {"messages": [HumanMessage(content=prompt)]},
            stream_mode="values",
            config={"configurable": {"thread_id": 1}}
        ):
            print(chunk["messages"][-1].content)
else:
    show_graph(app)