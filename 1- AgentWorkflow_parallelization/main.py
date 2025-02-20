import random
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from openai import OpenAI as OpenAI_from_openai
from dummy_data import user_information, product_information, product_reviews
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentWorkflow
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
import json
import time

print("----------------------------------------------------------------------------")
print("User Information: ", user_information)
print("----------------------------------------------------------------------------")
print("Product Information: ", product_information)
print("----------------------------------------------------------------------------")
print("Product Reviews: ", product_reviews)


# Initialize OpenAI LLM
llm = OpenAI(model="gpt-4o", api_key="")

async def retrieve_user_profile(ctx: Context) -> str:
    """Fetches user information."""
    print("\n$$$$$$$   START OF USER_PROFILE   $$$$$$$\n")

    current_state = await ctx.get("state", {})
    users = user_information
    user_id_to_return = 3
    current_state["user_profile"] = users[str(user_id_to_return)]
    await ctx.set("state", current_state)

    print("\n$$$$$$$   END OF USER_PROFILE   $$$$$$$\n")


    return "User profile fetched."

async def retrieve_product_information(ctx: Context) -> str:
    """Fetches product information."""
    print("\n$$$$$$$   START OF PRODUCT_INFORMATION   $$$$$$$\n")

    current_state = await ctx.get("state", {})
    current_state["product_information"] = product_information
    await ctx.set("state", current_state)

    print("\n$$$$$$$   END OF PRODUCT_INFORMATION   $$$$$$$\n")

    return "Product information fetched."

async def retrieve_product_reviews(ctx: Context) -> str:
    """Fetches product reviews."""
    
    print("\n$$$$$$$   START OF PRODUCT_REVIEWS   $$$$$$$\n")

    current_state = await ctx.get("state", {})
    current_state["product_reviews"] = product_reviews
    await ctx.set("state", current_state)

    print("\n$$$$$$$   END OF PRODUCT_REVIEWS   $$$$$$$\n")


    return "Product reviews fetched."


async def features_highlighting(ctx: Context) -> str:
    """Analyzes user profile and product data to highlight relevant features of the product based on the user profile."""
    print("\n$$$$$$$   START OF FEATURES_HIGHLIGHTING   $$$$$$$\n")

    current_state = await ctx.get("state", {})
    user_profile = current_state.get("user_profile", {})
    product_info = current_state.get("product_information", {})

    prompt_template = f"""
    You are given a user profile and product information. Analyze the details of both and suggest which features of the product will be most appealing to the user. As your final output, return only a list of features. Do not put a variable name before the list. Your response should only contain the list and nothing else:
    
    [
        **List of featrures selected**
    ]
    
    Perform the task for the following information:
    User Profile: {user_profile}
    Product Information: {product_info}
    """
    
    client = OpenAI_from_openai(api_key=" ")
    messages = [{"role": "user", "content": prompt_template}]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        top_p=0.1,
        messages=messages
    ).choices[0].message.content.strip()

    # response_in_json_object = json.loads(response)

    current_state["suggested_features"] = response
    await ctx.set("state", current_state)

    print("\n$$$$$$$   END OF FEATURES_HIGHLIGHTING   $$$$$$$\n")


    return "Features suggested."

async def sort_reviews(ctx: Context) -> str:
    """Analyzes user profile and the list of reviews and sort the reviews according to the user profile. The reviews that are most likely to be of interest to the user should be coming first in the list of reviews."""
    print("\n$$$$$$$   START OF SORT REVIEWS   $$$$$$$\n")

    current_state = await ctx.get("state", {})
    user_profile = current_state.get("user_profile", {})
    product_reviews = current_state.get("product_reviews", [])

    prompt_template = f"""
    You are given a user profile and a list of product reviews. Analyze the details of both and sort the reviews based on the user profile information. The reviews that match the users profile the most and are the ones that the user will most likely be interested in should come first. As your final output, return only a list of reviews sorted based on the user profile information. Do not put a variable name before the list. Your response should only contain the list and nothing else:
    
    [
        **List of reviews sorted**
    ]

    User Profile: {user_profile}
    Product Reviews: {product_reviews}
    """
    
    client = OpenAI_from_openai(api_key=" ")
    messages = [{"role": "user", "content": prompt_template}]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        top_p=0.1,
        messages=messages
    ).choices[0].message.content.strip()

    current_state["sorted_reviews"] = response
    await ctx.set("state", current_state)

    print("\n$$$$$$$   END OF SORT_REVIEWS   $$$$$$$\n")


    return "reviews sorted."

async def custom_description(ctx: Context) -> str:
    """Analyzes user profile and product data to create a custom product description that will be most intriguing to the user."""
    print("\n$$$$$$$   START OF CUSTOM_DESCRIPTION   $$$$$$$\n")
    
    current_state = await ctx.get("state", {})
    user_profile = current_state.get("user_profile", {})
    product_info = current_state.get("product_information", {})

    prompt_template = f"""
    You are given a user profile and product information. You have to create a custom description of the product based on the user profile information. The description should be most appealing to the user and should not be longer than 1 sentence.  As your final output, return a json object ONLY, like the follwowing. Do not put a variable name before the json object. Your response should only contain the json object and nothing else:
    {{
        "custom_description": **The custom description that you create**
    }}
    
    User Profile: {user_profile}
    Product Information: {product_info}
    """
    
    client = OpenAI_from_openai(api_key=" ")
    messages = [{"role": "user", "content": prompt_template}]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        top_p=0.1,
        messages=messages
    ).choices[0].message.content.strip()

    current_state["custom_description"] = response
    await ctx.set("state", current_state)

    print("\n$$$$$$$   END OF CUSTOM_DESCRIPTION   $$$$$$$\n")
    return "created custom description."

# Define Agents
from llama_index.core.agent.workflow import FunctionAgent
data_agent = ReActAgent(
    name="DataFetchingAgent",
    description="Used to retrieve user information, product information, product reviews.",
    system_prompt="""You are an agent who has tools to fetch data. You have the following tools: 
    - retrieve_user_profile: this tool is used to fetch user information. Use this when you have to fetch the user details.
    - retrieve_product_information: this tool is used to fetch product information. Use this when you have to fetch the product details.
    - retrieve_product_reviews: this tool is used to fetch product reviews. Use this when you have to fetch the product reviews.
    All of these tools perform tasks independently. You can call and run them in parallel.
    """,
    llm=llm,
    tools=[retrieve_user_profile, retrieve_product_information, retrieve_product_reviews],
    can_handoff_to=["CustomizationAgent"],
    allow_parallel_tool_calls=True,
)

customization_agent = ReActAgent(
    name="CustomizationAgent",
    description="Used to make customizations of the product details based on the user data. Based on the users profile, this agent suggests which features should be highlighted, how the reviews should be sorted, what should be a custom product description.",
    system_prompt="""You are an agent that is used to make customizations based on the user profile. You have the following tools.
    - features_highlighting: this tool is used to suggest which features should be highlighted. Based on the users profile, this tool will suggest which features of the product should be highlighted. Call this tool for features_highlighting.
    - sort_reviews: this tool is used to sort the reviews based on the user profile. Call this tool to sort the reviews based on the user profile.
    - custom_description: this tool is used to create a custom product description based on the user profile. Call this tool to create a custom product description.
    
    These tools are used for customization based on the user profile. Use the 'features_highlighting' tool to suggest which features should be highlighted. Use the 'sort_reviews' tool to sort the reviews based on the user profile. Use the 'custom_description' tool to create a custom product description based on the user profile.

    All of these tools perform tasks independently. You can call and run them in parallel.

    """,
    llm=llm,
    tools=[features_highlighting, sort_reviews, custom_description],
    can_handoff_to=[""],
    allow_parallel_tool_calls=True,
)

# Create Agent Workflow
agent_workflow = AgentWorkflow(
    agents=[data_agent, customization_agent],
    root_agent=data_agent.name,
    initial_state={
        "user_profile": {},
        "product_information": {},
        "product_reviews": [],
    },
)

# from llama_index.utils.workflow import draw_all_possible_flows

# draw_all_possible_flows(agent_workflow, filename="1.html")

# Function to Run Workflow
async def run_workflow():

    handler = agent_workflow.run(
        user_msg="""
    
        Fetch the users detail first. Then fetch the product information. Then fetch the product reviews. Finally, suggest the features after analyzing product information and sort the reviews based on the user profile and also create a custom description of the product based on the user profile.
        
        
        """
    )
    
    # current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            # current_agent = event.current_agent_name
            # print(f"\n{'='*50}")
            # print(f"🤖 Agent: {current_agent}")
            # print(f"{'='*50}\n")
        
        # if isinstance(event, AgentOutput):
        #     if event.response.content:
        #         # print("📤 Output:", event.response.content)
        #     if event.tool_calls:
        #         # print("🛠️  Planning to use tools:", [call.tool_name for call in event.tool_calls])
        # elif isinstance(event, ToolCallResult):
        #     # print(f"🔧 Tool Result ({event.tool_name}):")
        #     # print(f"  Arguments: {event.tool_kwargs}")
        #     # print(f"  Output: {event.tool_output}")
        # elif isinstance(event, ToolCall):
            # print(f"🔨 Calling Tool: {event.tool_name}")
            # print(f"  With arguments: {event.tool_kwargs}")
    
    state = await handler.ctx.get("state")

    print("Keys:  ", state.keys())

    file_path = "formatted_data.json"

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4, ensure_ascii=False)

# Run the workflow (if running as a script)
if __name__ == "__main__":

    import asyncio
    start = time.time()
    asyncio.run(run_workflow())
    end = time.time()
    print("\n\n**** Time taken: ", end - start)
