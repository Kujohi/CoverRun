from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from datetime import datetime
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.types import Command
import random
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# Initialize Supabase client
from supabase import create_client

def get_supabase_client():
    """Create Supabase client with anon key"""
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    return create_client(
        supabase_url,
        supabase_key
    )



def get_user_data(supabase, user_email: str):
    """Get user data by email"""
    try:
        if not user_email:
            raise Exception("User email is required")
        
        user_data = supabase.table('users').select('id, user_description').eq('email', user_email).single().execute()
        if not user_data.data:
            raise Exception(f"User not found in database with email: {user_email}")
        
        return user_data.data
    except Exception as e:
        raise Exception(f"Failed to get user data: {str(e)}")

# FastAPI app
app = FastAPI(title="Jargon AI Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    user_email: str
    image_data: Optional[str] = None
    image_format: Optional[str] = None
    new_thread: Optional[bool] = False  # Flag to indicate if we should create a new thread

class ChatResponse(BaseModel):
    response: str
    success: bool
    function_data: Optional[dict] = None

# LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_email: Optional[str] = None

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Make sure this model supports multimodal inputs
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    convert_system_message_to_human=True
)

search_engine = TavilySearch(
    max_results=5,
    topic="general",
)

# System prompt

system_prompt_template = """
The current email of the user is {user_email}.
Current time: {current_time}
Current date: {current_date}
money currency: VND

User's profile and transaction pattern:
{user_description}
## Guideline for the user's profile and transaction pattern usage:
- This information provide you insightful information about the user's financial behavior and patterns, so you can use it to help the user with their financial management.
- There are some common patterns that you can use to help the user with their financial management, like:
    + The user's income and expenses are not balanced, so you need to help the user to balance their income and expenses.
    + Give them advice to improve their mental and physical health, like: (if they make some coffee transaction but recently he's been drinking too much coffee, you can suggest him to that this is not good for his health)
    + If they ask for how to save money, you can suggest them to save money by reducing their unnecessary expenses, unnecessary purchases depending on the user's profile and transaction pattern.

## 1. Personality
You are a clear, helpful, and respectful assistant focused solely on **booking appointments** for clients.
- **Identity**: You are “CoverRun Financial Assistant”, 
- **Core Traits**: depends on user's profile.
- **Role**: Your job is to help user with their financial management.

## 2. Goal
Your task is to help user with their financial management.

## Here are your abilities:
- Add monthly income to user's jars with proper allocation percentages
- Analyze receipts and products from images to make record for transactions
- Add a specific income or expense transaction to a particular jar. You might need to classify the transaction into a jar category if the user doesn't provide it.
- Set or update the user's savings target amount.
- Predict the user's savings based on their historical savings data.
- Search for transactions based on keywords and filters:
    + Because the keyword from the user might be different to that in the actual database so this is a multistep process:
        - First, you need to understand the schema of the transactions table with get_transaction_schema tool
        - Then, you need to generate a SQL query with multiple possible keywords (OR logic) to search for in the transactions table, use short keywords for example %coffee%, %cafe%, the number of keywords should be less than 5.
        - Then, you need to execute the SQL query with sql_executor tool and get the result
        - Then, you need to read the result answer the user
- You can also search the web for information with search_web tool, to find the price of a product or service, try to extract much information as possible (brandname, weight, etc.) to search the exact product or service.

## 3. Guardrails
- **Do not** discuss anything unrelated to financial management.
- If the user asks for something outside your scope:
    - Say: “I'm only here to help with financial management. For other questions, please contact our support team.”
- NEVER ask the user to classify the transactions into jar categories, just do it automatically.
- NEVER speculate about unavailable data or functions.
- NEVER ask for a date format from the User, like Say date in Day Month and Year format. If you can't understand the user's date, then say Please speak the full date.
- NEVER ask the user to wait for you to excute the tool call, just do it (for example, don't say "Please wait for me to check the availability").
- NOT ALWAYS mention the user's profile and transaction pattern in your response, only mention it when it's relevant to the user's question or when the user asks for it.
"""

system_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("placeholder", "{messages}"),
])

# Tools
@tool
def add_monthly_income(
    monthly_income_amount: float,
    user_email: str,
    month_year: Optional[str] = None,
    necessity_percentage: float = 55,
    play_percentage: float = 10,
    education_percentage: float = 10,
    investment_percentage: float = 10,
    charity_percentage: float = 5,
    savings_percentage: float = 10,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
):
    """Add monthly income to user's jars with proper allocation percentages
    
    Args:
        monthly_income_amount (float): The monthly income amount
        user_email (str): The email of the user
        month_year (str): The month and year for this income in YYYY-MM format (e.g., '2024-01')
        necessity_percentage (float): The percentage for Necessity jar (essential expenses like food, housing, utilities)
        play_percentage (float): The percentage for Play jar (entertainment and leisure activities)
        education_percentage (float): The percentage for Education jar (learning and skill development)
        investment_percentage (float): The percentage for Investment jar (long-term wealth building)
        charity_percentage (float): The percentage for Charity jar (giving back to the community)
        savings_percentage (float): The percentage for Savings jar (emergency fund and future goals)

    returns:
        response (str): The response from the tool
    """
    try:
        if not user_email:
            raise Exception("User email is required")

        supabase = get_supabase_client()
        user_data = get_user_data(supabase, user_email)

        # Validate allocation percentages total to 100%
        total_percentage = (necessity_percentage + play_percentage + education_percentage + 
                          investment_percentage + charity_percentage + savings_percentage)
        
        if abs(total_percentage - 100) > 0.01:
            response = f"Allocation percentages must total 100%. Current total: {total_percentage}%"
            return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

        # Set default month_year to current month if not provided
        if not month_year:
            now = datetime.now()
            month_year = f"{now.year}-{now.month:02d}"

        month_year_date = month_year + '-01'  # Convert to full date
        income_amount_cents = round(float(monthly_income_amount))

        # Create allocation percentages object
        allocation_percentages = {
            'Necessity': necessity_percentage,
            'Play': play_percentage,
            'Education': education_percentage,
            'Investment': investment_percentage,
            'Charity': charity_percentage,
            'Savings': savings_percentage
        }

        # Check if income entry already exists for this month
        existing_entry = supabase.table('monthly_income_entries').select('id').eq('user_id', user_data['id']).eq('month_year', month_year_date).execute()
        
        if existing_entry.data:
            response = f"Income for {month_year} already exists. Please choose a different month."
            return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

        # Insert monthly income entry
        income_entry = supabase.table('monthly_income_entries').insert({
            'user_id': user_data['id'],
            'month_year': month_year_date,
            'total_income_cents': income_amount_cents,
            'allocation_percentages': allocation_percentages
        }).execute()

        if not income_entry.data:
            raise Exception("Failed to create income entry")

        # Get jar categories
        jar_categories = supabase.table('jar_categories').select('id, name').execute()
        
        if not jar_categories.data:
            raise Exception("Failed to fetch jar categories")

        # Create income transactions for each jar
        transactions = []
        for category in jar_categories.data:
            percentage = allocation_percentages.get(category['name'], 0)
            amount = round((income_amount_cents * percentage) / 100)
            
            if amount > 0:
                transactions.append({
                    'user_id': user_data['id'],
                    'jar_category_id': category['id'],
                    'amount_cents': amount,
                    'description': f"Monthly income allocation for {month_year}",
                    'source': 'chatbot',
                    'monthly_income_entry_id': income_entry.data[0]['id']
                })

        # Insert all transactions
        if transactions:
            supabase.table('transactions').insert(transactions).execute()

        # Format success message
        formatted_amount = f"{income_amount_cents:,.0f} VND"
        allocation_details = ', '.join([f"{jar}: {percentage}%" for jar, percentage in allocation_percentages.items()])
        
        response = f"Monthly income of {formatted_amount} for {month_year} added successfully! Allocated to jars: {allocation_details}"
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

    except Exception as error:
        response = f"Error adding monthly income: {str(error)}"
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

@tool
def update_transaction(
    amount: float,
    jar_category_id: int,
    user_email: str,
    description: str = "",
    transaction_type: str = "expense",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
):
    """Add a specific income or expense transaction to a particular jar. You might need to classify the transaction into a jar category if the user doesn't provide it.
    
    Args:
        amount (float): The amount of the transaction
        jar_category_id (int): The ID of the jar category {1: Necessity, 2: Play, 3: Education, 4: Investment, 5: Charity, 6: Savings} 
        user_email (str): The email of the user
        description (str): The description of the transaction
        transaction_type (str): The type of transaction (expense or income)

    returns:
        response (str): The response from the tool
    """
    try:
        if not user_email:
            raise Exception("User email is required")

        supabase = get_supabase_client()
        user_data = get_user_data(supabase, user_email)

        # Get amount and apply sign based on transaction type
        amount_value = round(float(amount))
        final_amount = -amount_value if transaction_type == 'expense' else amount_value

        # Insert transaction
        transaction = supabase.table('transactions').insert({
            'user_id': user_data['id'],
            'jar_category_id': jar_category_id,
            'amount_cents': final_amount,
            'description': description,
            'source': 'chatbot'
        }).execute()

        if not transaction.data:
            raise Exception("Failed to create transaction")

        # Format response
        formatted_amount = f"{amount_value:,.0f} VND"
        response = f"Added {transaction_type} of {formatted_amount} to jar {jar_category_id}"
        if description:
            response += f" for {description}"
        
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

    except Exception as error:
        response = f"Error adding transaction: {str(error)}"
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

@tool
def set_saving_target(
    target_amount: float,
    user_email: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
):
    """Set or update the user's savings target amount.
    
    Args:
        target_amount (float): The target savings amount
        user_email (str): The email of the user

    returns:
        response (str): The response from the tool
    """
    try:
        if not user_email:
            raise Exception("User email is required")

        if target_amount <= 0:
            response = "Target amount must be a positive number in VND."
            return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

        supabase = get_supabase_client()
        user_data = get_user_data(supabase, user_email)

        # Convert VND to cents for storage
        target_cents = round(target_amount * 100)

        # Update the user's saving target
        result = supabase.table('users').update({
            'saving_target_cents': target_cents
        }).eq('id', user_data['id']).execute()

        if not result.data:
            raise Exception("Failed to update saving target")

        # Format the target amount for display
        formatted_target = f"{target_amount:,.0f}"
        response = f"✅ Successfully updated\n\n💰 new target: {formatted_target} VND\n\n. You can now track your savings progress on the savings chart."
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

    except Exception as error:
        response = f"Error setting saving target: {str(error)}"
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})
@tool
def predict_savings(
    target_amount: float,
    target_description: str,
    user_email: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
):
    """Predict when a user can reach a savings goal.
    
    Args:
        target_amount (float): The target savings amount
        target_description (str): The description of the target
        user_email (str): The email of the user

    returns:
        response (str): The response from the tool
    """
    try:
        if not user_email:
            raise Exception("User email is required")

        supabase = get_supabase_client()
        user_data = get_user_data(supabase, user_email)

        # Get historical savings data from transactions
        result = supabase.table('transactions').select(
            'amount_cents, occurred_at, jar_category_id, jar_categories(name)'
        ).eq('user_id', user_data['id']).eq('jar_category_id', 6).order('occurred_at', desc=False).execute()
        
        transactions = result.data or []

        if not transactions:
            response = 'No savings data found. Please start saving money first to get predictions.'
            return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

        # Calculate current savings balance
        current_balance = sum(tx['amount_cents'] for tx in transactions)
        
        # Calculate average monthly savings rate from recent transactions
        recent_transactions = transactions[-12:]  # Last 12 transactions
        if len(recent_transactions) >= 2:
            total_change = sum(tx['amount_cents'] for tx in recent_transactions)
            monthly_rate = total_change / len(recent_transactions) * 30  # Approximate monthly rate
            
            remaining_amount = target_amount - current_balance
            if monthly_rate > 0:
                months_to_target = remaining_amount / monthly_rate
                target_date = datetime.now()
                target_date = target_date.replace(month=target_date.month + int(months_to_target))
                
                response = f"Based on your current savings rate of {monthly_rate:,.0f} VND per month:\n"
                response += f"- Current savings: {current_balance:,.0f} VND\n"
                response += f"- Target amount: {target_amount:,.0f} VND\n"
                response += f"- Remaining amount: {remaining_amount:,.0f} VND\n"
                response += f"- Estimated time to reach goal: {target_date.strftime('%B %Y')}\n\n"
                
                if months_to_target > 24:
                    response += "💡 Tip: Consider increasing your monthly savings or adjusting your target to reach your goal sooner."
            else:
                response = f"Your current savings rate is not positive. You need to save more to reach your goal of {target_amount:,.0f} VND for {target_description}."
        else:
            response = f"Not enough savings data to make a prediction. Current savings: {current_balance:,.0f} VND. Target: {target_amount:,.0f} VND for {target_description}."

        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

    except Exception as error:
        response = f"Error predicting savings: {str(error)}"
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

# Error handler
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    error_msg = str(error) if error else "Unknown error occurred"
    
    return {
        "messages": [
            ToolMessage(
                content=error_msg,
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Tool node with fallback
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


@tool 
def get_transaction_schema(
    user_email: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
):
    """
    Get the schema information of the transactions table

    Args:
        user_email (str): The email of the user (required for security)
        tool_call_id (str): The tool call ID

    returns:
        response (str): The schema information of the transactions table
    """
    try:
        if not user_email:
            raise Exception("User email is required")

        # Query the schema using Postgres information_schema
        query = """
        SELECT 
            column_name, 
            data_type, 
            is_nullable, 
            column_default
        FROM information_schema.columns
        WHERE table_name = 'transactions'
        """
        supabase = get_supabase_client()
        user_data = get_user_data(supabase, user_email)
        
        # Execute the query through RPC
        result = supabase.rpc('run_sql', {'query': query}).execute()
        
        if not result.data:
            return Command(update={"messages": [ToolMessage("No schema information found", tool_call_id=tool_call_id)]})
        
        # Format the schema information
        formatted_schema = ["Transactions Table Schema:"]
        formatted_schema.append("-" * 40)
        
        for column in result.data:
            column_info = [
                f"Column: {column['column_name']}",
                f"Type: {column['data_type']}",
                f"Nullable: {column['is_nullable']}",
                f"Default: {column['column_default'] or 'None'}"
            ]
            formatted_schema.append("\n".join(column_info))
            formatted_schema.append("-" * 40)
        
        response = "\n".join(formatted_schema)
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})

    except Exception as error:
        response = f"Error getting schema information: {str(error)}"
        return Command(update={"messages": [ToolMessage(response, tool_call_id=tool_call_id)]})
    

@tool
def sql_executor(
    sql_query: str,
    user_email: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
):
    """
    Execute a SQL query and return the result. Only SELECT queries are allowed.
    The query must include user_id filter for security.

    Args:
        sql_query (str): The SQL query to execute (SELECT only)
        user_email (str): The email of the user (required for security)
        tool_call_id (str): The tool call ID

    returns:
        response (str): The result of the SQL query
    """
    try:
        if not user_email:
            raise Exception("User email is required")

        # Validate query is SELECT only
        sql_query = sql_query.strip()
        if not sql_query.lower().startswith('select'):
            raise Exception("Only SELECT queries are allowed")

        # Get user data for security check
        supabase = get_supabase_client()
        user_data = get_user_data(supabase, user_email)
        user_id = user_data['id']

        # Execute query
        result = supabase.rpc('run_sql', {
            'query': sql_query,
        }).execute()

        if not result.data:
            return {
                "status": "success",
                "message": "No results found",
                "data": []
            }

        # Format results as JSON
        if isinstance(result.data, list):
            # Process each row to format numbers and dates
            formatted_data = []
            for row in result.data:
                formatted_row = {}
                for key, val in row.items():
                    if isinstance(val, (int, float)):
                        # Format numbers with commas
                        formatted_row[key] = f"{val:,}"
                    elif isinstance(val, datetime):
                        # Format datetime objects
                        formatted_row[key] = val.isoformat()
                    else:
                        formatted_row[key] = str(val)
                formatted_data.append(formatted_row)

            return {
                "status": "success",
                "message": "Query executed successfully",
                "data": formatted_data,
                "total_rows": len(formatted_data)
            }

    except Exception as error:
        return {
            "status": "error",
            "message": f"Error executing query: {str(error)}",
            "data": None
        }

@tool
def search_web(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
):
    """
    Search the web for information

    Args:
        query (str): The query to search the web for
        tool_call_id (str): The tool call ID

    returns:
        response (str): The result of the web search
    """
    try:
        result = search_engine.invoke(query)
        return result
    except Exception as error:
        return {
            "status": "error",
            "message": f"Error searching web: {str(error)}",
            "data": None
        }
# Initialize tools and graph
tools = [
    add_monthly_income,
    update_transaction,
    set_saving_target,
    get_transaction_schema,
    sql_executor,
    predict_savings,
    search_web,
]

def get_assistant_runnable(state):
    # Extract user_email from state
    user_email = state.get("user_email", "unknown@example.com")
    
    # Get current date and time
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%B %d, %Y")
    
    # Get user data including description
    try:
        supabase = get_supabase_client()
        user_data = get_user_data(supabase, user_email)
        user_description = user_data.get('user_description', 'No user description available.')
    except Exception as e:
        user_description = 'Failed to fetch user description.'
    
    # Format the system prompt with the user's email, description and current time
    formatted_prompt = system_prompt.partial(
        user_email=user_email,
        current_time=current_time,
        current_date=current_date,
        user_description=user_description
    )
    
    # Return the runnable with the formatted prompt
    return formatted_prompt | llm.bind_tools(tools)

# Assistant class
class Assistant:
    def __init__(self, runnable_getter):
        self.runnable_getter = runnable_getter

    def __call__(self, state: State, config: RunnableConfig):
        # Get the runnable with the user's email from state
        runnable = self.runnable_getter(state)
        
        while True:
            # Pass the full state including conversation history
            result = runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

builder = StateGraph(State)
builder.add_node("assistant", Assistant(get_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# FastAPI endpoints
@app.get("/")
async def root():
    return {"message": "Jargon AI Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Debug print function
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"

            print(msg_repr)
            
            # Store AI responses in conversation history (only AIMessage, not tool messages)
            if isinstance(message, AIMessage) and message.content:
                print("AI Response:", message.content)

            _printed.add(message.id)

# Streaming event generator
async def generate_chat_stream(request: ChatRequest):
    try:
        if not request.user_email:
            yield f"data: {json.dumps({'type': 'error', 'content': 'User email is required'})}\n\n"
            return
            
        print(f"\n=== Starting streaming request for user: {request.user_email} ===")
        print(f"User message: {request.message}")
        print(f"History length: {len(request.conversation_history)}")
        
        # Create a unique thread ID for this conversation
        if request.new_thread:
            # Generate a completely new thread ID with timestamp to ensure uniqueness
            thread_id = f"user_{hash(request.user_email + str(datetime.now().timestamp())) % 10000}"
            print(f"Created new thread ID: {thread_id} (conversation cleared)")
        else:
            # Use consistent thread ID for continuing conversation
            thread_id = f"user_{hash(request.user_email) % 10000}"
            print(f"Using existing thread ID: {thread_id}")

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Build conversation context
        messages = []
        for msg in request.conversation_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        # Add current message with image if provided
        if request.image_data and request.image_format:
            # Create a multimodal message with text and image
            content = [
                {"type": "text", "text": request.message},
                {"type": "image_url", "image_url": f"data:{request.image_format};base64,{request.image_data}"}
            ]
            messages.append(HumanMessage(content=content))
        else:
            # Text-only message
            messages.append(HumanMessage(content=request.message))
        
        # Create state with user email
        initial_state = {
            "messages": messages,
            "user_email": request.user_email
        }
        
        # Stream the graph execution
        events = graph.stream(initial_state, config, stream_mode="values")
        
        # Track printed events and process stream
        _printed = set()
        step_count = 0
        
        for event in events:
            # Print event for tracking
            _print_event(event, _printed)
            
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                
                # Handle tool calls (thinking steps)
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    step_count += 1
                    for tool_call in last_message.tool_calls:
                        # Create more descriptive thinking content based on tool name
                        tool_name = tool_call['name']
                        tool_args = tool_call.get('args', {})
                        
                        if tool_name == 'add_monthly_income':
                            thinking_content = f"Adding monthly income of {tool_args.get('monthly_income_amount', 'unknown')} VND to user's jars"
                        elif tool_name == 'update_transaction':
                            thinking_content = f"Recording {tool_args.get('transaction_type', 'transaction')} of {tool_args.get('amount', 'unknown')} VND"
                        elif tool_name == 'set_saving_target':
                            thinking_content = f"Setting savings target to {tool_args.get('target_amount', 'unknown')} VND"
                        elif tool_name == 'predict_savings':
                            thinking_content = f"Analyzing savings data to predict when you can reach {tool_args.get('target_amount', 'your goal')}"
                        elif tool_name == 'get_transaction_schema':
                            thinking_content = "Examining transaction database structure to understand your data"
                        elif tool_name == 'sql_executor':
                            thinking_content = "Searching through your transaction history to find relevant information"
                        else:
                            thinking_content = f"Using {tool_name} with parameters: {str(tool_args)[:50]}"
                        
                        # Truncate to first 20 words
                        words = thinking_content.split()[:20]
                        truncated_thinking = " ".join(words) + ("..." if len(thinking_content.split()) > 20 else "")
                        
                        yield f"data: {json.dumps({'type': 'thinking', 'content': truncated_thinking, 'step': step_count})}\n\n"
                        await asyncio.sleep(0.3)  # Slightly longer delay for better readability
                
                # Handle tool responses (intermediate steps)
                elif isinstance(last_message, ToolMessage):
                    step_count += 1
                    # Create more meaningful processing messages
                    content = last_message.content
                    if "successfully" in content.lower():
                        thinking_content = "✅ Task completed successfully, preparing response"
                    elif "error" in content.lower():
                        thinking_content = "⚠️ Encountered an issue, trying alternative approach"
                    elif "found" in content.lower() or "results" in content.lower():
                        thinking_content = "📊 Found relevant data, analyzing results"
                    else:
                        thinking_content = f"Processing: {content[:80]}"
                    
                    # Truncate to first 20 words
                    words = thinking_content.split()[:20]
                    truncated_thinking = " ".join(words) + ("..." if len(thinking_content.split()) > 20 else "")
                    
                    yield f"data: {json.dumps({'type': 'thinking', 'content': truncated_thinking, 'step': step_count})}\n\n"
                    await asyncio.sleep(0.2)
                
                # Handle final AI response
                elif isinstance(last_message, AIMessage) and last_message.content and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                    print(f"\n=== Final response generated ===")
                    print(f"Response length: {len(last_message.content)}")
                    yield f"data: {json.dumps({'type': 'final', 'content': last_message.content})}\n\n"
                    break
        
        print(f"\n=== Stream completed successfully ===")
        print(f"Total thinking steps: {step_count}")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as error:
        print(f"\n=== Streaming error ===")
        print(f"Error details: {error}")
        print(f"Error type: {type(error).__name__}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Failed to process request: {str(error)}'})}\n\n"

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint that shows thinking steps"""
    return StreamingResponse(
        generate_chat_stream(request),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)