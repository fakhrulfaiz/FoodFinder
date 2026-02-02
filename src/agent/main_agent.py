from typing import Annotated, Literal
from typing_extensions import TypedDict
import os
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model

from ..vectorstore.retriever import MultimodalRetriever
from .tools.toolkit import CustomToolkit

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class FoodFinderState(MessagesState):
    """State for the FoodFinder ReAct agent"""
    pass


# Initialize retriever with absolute path
# Get project root (3 levels up from this file: src/agent/main_agent.py)
_project_root = Path(__file__).parent.parent.parent
_index_dir = _project_root / "indexes"

# Create retriever instance with absolute path
retriever = MultimodalRetriever(index_dir=str(_index_dir))

# Flag to track if indices are loaded
_indices_loaded = False


class FoodFinderAgent:
    """ReAct agent for restaurant recommendations using LangGraph
    
    Implements the ReAct (Reasoning + Acting) pattern where the agent:
    1. Reasons about what information it needs
    2. Acts by calling appropriate tools
    3. Observes the results
    4. Repeats until it can answer the user's question
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """Initialize the FoodFinder ReAct agent
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
        """
        self.model = init_chat_model(model_name, temperature=temperature)
        
        # Create toolkit with retriever and LLM
        self.toolkit = CustomToolkit(retriever=retriever, llm=self.model)
        
        # Get all tools from toolkit
        self.tools = self.toolkit.get_tools()
        
        # Bind tools to model for ReAct loop
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        self.graph = None
        
        # System prompt for ReAct reasoning
        self.system_prompt = """You are a helpful restaurant recommendation assistant with access to specialized tools.

You have access to the following tools:
- rag_text_search: Search for restaurants using text queries (cuisine, location, ratings, etc.)
- rag_image_search: Find similar restaurants based on food/restaurant images
- image_qa: Answer questions about specific images (cuisine type, dish description, etc.)

Follow the ReAct (Reasoning + Acting) pattern:
1. REASON about what information you need to answer the user's question
2. ACT by calling the appropriate tool(s)
3. OBSERVE the results from the tools
4. REPEAT if you need more information, or provide a final answer

**IMPORTANT - Tool Combination Strategy:**

When the user uploads an image and asks to find restaurants:
1. First, use `rag_image_search` to find visually similar restaurants
2. OBSERVE the results - they will be in JSON format with restaurant names
3. If the JSON results lack complete information (missing details like full address, hours, reviews, etc.):
   - Extract the restaurant names from the image search results
   - Use `rag_text_search` with those restaurant names to get complete details
   - Combine both results for a comprehensive answer

Example workflow:
- User uploads pizza image and asks "Find similar restaurants"
- Step 1: Call rag_image_search(image_path, k=5) → Get 5 similar restaurants with names
- Step 2: Observe results - if they only have basic info (name, cuisine, rating)
- Step 3: Call rag_text_search("Restaurant Name 1") to get full details
- Step 4: Combine visual similarity scores with detailed information
- Step 5: Provide comprehensive recommendation

When you have enough information, provide a comprehensive and enthusiastic recommendation.
The tool results are in JSON format - parse them to extract relevant details like name, cuisine, rating, location, etc.

Be conversational and helpful!"""
    
    def _should_continue(self, state: MessagesState) -> Literal["tools", "end"]:
        """Determine whether to continue with tool calls or end
        
        This is the ReAct decision point - does the agent need more information?
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes tool calls, continue to tools node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise, end the loop
        return "end"
    
    def _call_model(self, state: MessagesState):
        """Call the model with tools bound (ReAct reasoning step)
        
        This is where the agent REASONS about what to do next
        """
        messages = state["messages"]
        
        # Add system prompt if this is the first call
        if len(messages) == 1 or not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        # Invoke model with tools - it will decide whether to call tools or respond
        response = self.model_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def build_graph(self) -> StateGraph:
        """Build the ReAct agent graph
        
        Graph structure:
        START -> agent (reasoning) -> [tools (acting) -> agent (observe & reason)] -> END
        
        The agent can loop multiple times through the reasoning-acting cycle
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Initialize graph with MessagesState
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        # Agent node: reasoning step (decides what to do)
        workflow.add_node("agent", self._call_model)
        
        # Tools node: acting step (executes tool calls)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Set entry point
        workflow.add_edge(START, "agent")
        
        # Add conditional edge from agent
        # After reasoning, either call tools or end
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",  # Continue to tools if agent wants to act
                "end": END,        # End if agent has final answer
            },
        )
        
        # After tools execute, go back to agent for observation & next reasoning step
        # This creates the ReAct loop: agent -> tools -> agent -> tools -> ... -> END
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        self.graph = workflow.compile()
        return self.graph
    
    def run(self, query: str) -> dict:
        """Run the ReAct agent with a query
        
        Args:
            query: User's restaurant search query
            
        Returns:
            Dictionary containing the response and metadata
        """
        if self.graph is None:
            self.build_graph()
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Run the graph - agent will loop through ReAct cycles
        result = self.graph.invoke(initial_state)
        
        # Extract final answer
        final_message = result["messages"][-1]
        
        return {
            "answer": final_message.content,
            "messages": result["messages"]
        }
    
    def stream(self, query: str):
        """Stream the ReAct agent execution
        
        Args:
            query: User's restaurant search query
            
        Yields:
            Updates from each node in the graph (reasoning and acting steps)
        """
        if self.graph is None:
            self.build_graph()
        
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        for chunk in self.graph.stream(initial_state):
            yield chunk
