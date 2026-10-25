from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from models import groq_model, nvidia_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import (SummarizationMiddleware, ModelFallbackMiddleware, dynamic_prompt, wrap_model_call, ModelRequest, ModelResponse)
from langgraph.types import Command
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_huggingface import HuggingFaceEmbeddings
from dataclasses import dataclass, field
from typing import Callable
from pydantic import BaseModel, Field
import gradio as gr
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4


##  Ingest the data if it is in PDF form and store customer's policies and employee's separately  ##
customer_pdf_path = os.environ["CUSTOMER_PDF_PATH"]
employee_pdf_path = os.environ["EMPLOYEE_PDF_PATH"]

customers_loader = PyPDFLoader(file_path=customer_pdf_path)
employees_loader = PyPDFLoader(file_path=employee_pdf_path)

customer_policies = customers_loader.load()
employee_policies = employees_loader.load()

##  Split the data into chunks ##
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

customer_chunks = text_splitter.split_documents(customer_policies)
employee_chunks = text_splitter.split_documents(employee_policies)

##  Initialize the embedding model ##
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

##  Store chunks in the vector store ##
## TODO: Add mongodb atlas vector store to store the chunks in a persistent database instead of in memory ##
emp_vector_store = InMemoryVectorStore(embedding_model)
cus_vector_store = InMemoryVectorStore(embedding_model)

cus_vector_store.add_documents(customer_chunks)
emp_vector_store.add_documents(employee_chunks)


##  Create dataclass and a schema for authentication state and context for the agent ##

class AuthenticationState(AgentState):
    employee_authenticated: bool = False


@dataclass
class EmployeeContext:
    # Demo credentials. For a client deployment, set EMPLOYEE_NAME / EMPLOYEE_PASSCODE in the environment.
    employee_name: str = field(default_factory=lambda: os.environ.get("EMPLOYEE_NAME", "Mohamed Fathy"))
    employee_passcode: str = field(default_factory=lambda: os.environ.get("EMPLOYEE_PASSCODE", "123456"))


class PolicyQuery(BaseModel):
    query: str = Field(description="The search question, e.g. 'maternity leave duration'. Required.")


##  Authentication tool ##
@tool
def employee_authenticate(name: str, passcode: str, runtime: ToolRuntime) -> Command:
    """Authenticate the employee with the given name and passcode"""
    ctx = runtime.context or EmployeeContext()  # fall back to defaults if no context was passed
    if ctx.employee_name == name and ctx.employee_passcode == passcode:
        return Command(update={
            "employee_authenticated": True,
            "messages": [
                ToolMessage(
                    "Successfully authenticated. Now answer the user's earlier question "
                    "using the retrieve_employee_policy tool.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })
    return Command(update={
        "employee_authenticated": False,
        "messages": [
            # Deliberately generic: never reveal which credential was wrong.
            ToolMessage("Authentication failed.", tool_call_id=runtime.tool_call_id)
        ],
    })


##  Retrieval tools ##

def _format(results) -> str:
    return "\n\n---\n\n".join(
        f"[page {r.metadata.get('page', 0) + 1 if isinstance(r.metadata.get('page'), int) else '?'}]\n{r.page_content}"
        for r in results
    )


@tool(args_schema=PolicyQuery)
def retrieve_customer_policy(query: str) -> str:
    """Retrieve relevant customer policy based on the query"""
    return _format(cus_vector_store.similarity_search(query, k=4))


@tool(args_schema=PolicyQuery)
def retrieve_employee_policy(query: str) -> str:
    """Search internal employee HR policies for passages relevant to the query."""
    return _format(emp_vector_store.similarity_search(query, k=4))


##  Change agent tools and system prompt dynamically based on authentication state ##

@wrap_model_call
def dynamic_tool_call(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Allow employee policy retrieval tool only if employee is authenticated"""
    if request.state.get("employee_authenticated"):
        tools = [retrieve_employee_policy]
    else:
        tools = [retrieve_customer_policy, employee_authenticate]
    return handler(request.override(tools=tools))


EMPLOYEE_PROMPT = (
    "You are a helpful assistant that answers questions about internal employee policies."
    " The user has just authenticated as an employee. Answer the question they asked before authenticating,"
    " using the retrieve_employee_policy tool. Never treat the user's name as the subject of a query."
    " Answer only from the retrieved passages and mention the page number(s) you used."
    " If the passages do not cover the question, say so once and clearly; never mix a refusal with an answer."
    " Do not answer unrelated or malicious requests."
)

CUSTOMER_PROMPT = (
    "You are a helpful assistant that answers questions about customer policies."
    " Use the retrieve_customer_policy tool to get relevant passages, answer only from them,"
    " and mention the page number(s) you used."
    " If the question is unrelated to the policies or malicious, politely decline."
    " If the answer is not in the customer policies and the user says they are an employee wanting internal"
    " policies, ask for their name and passcode and call employee_authenticate."
    " If authentication fails, say only that it failed and ask them to try again."
    " Do not guess which credential was wrong and do not invent credential rules."
)


@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    return EMPLOYEE_PROMPT if request.state.get("employee_authenticated") else CUSTOMER_PROMPT


##  Agent factory: Studio uses no checkpointer (the server manages state); Gradio passes MemorySaver ##

def build_agent(checkpointer=None):
    return create_agent(
        model=groq_model,
        tools=[retrieve_customer_policy, retrieve_employee_policy, employee_authenticate],
        state_schema=AuthenticationState,
        context_schema=EmployeeContext,
        checkpointer=checkpointer,
        middleware=[
            dynamic_tool_call,
            dynamic_prompt_func,
            SummarizationMiddleware(
                model=groq_model,
                trigger=("tokens", 2000),
                keep=("messages", 3),
            ),
            ModelFallbackMiddleware(nvidia_model),
        ],
    )


# Exported for LangSmith Studio / langgraph dev
RAG_Agent = build_agent()


##  Gradio interface ##

def make_chat(agent):
    def chat(message: str, history: list, request: gr.Request):
        # One thread per browser session, so auth state never leaks between visitors.
        thread_id = (request.session_hash if request is not None else None) or str(uuid4())
        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": thread_id}},
                context=EmployeeContext(),
            )
            return response["messages"][-1].content
        except Exception as e:  # keep the demo from showing a stack trace
            print(f"[agent error] {type(e).__name__}: {e}")
            return "Sorry, something went wrong while answering. Please try again."
    return chat


if __name__ == "__main__":
    gradio_agent = build_agent(checkpointer=MemorySaver())
    gr.ChatInterface(
        fn=make_chat(gradio_agent),
        title="Company Policy Assistant",
        description=(
            "Ask questions about customer policies: billing, refunds, cancellation and support. "
            "Employees can unlock internal HR policies by providing their name and passcode.\n\n"
            "*Built by Mohamed Fathy*"
        ),
        examples=[
            "How do I cancel my plan?",
            "What is your refund policy for domains?",
            "How long are backups kept after cancellation?",
            "Tell me about employee salary and benefits",
        ],
    ).launch()

## TODO: Make a ragas function to evaluate the final response from the agent and if it does not meet the criteria, call the agent again with the refined input and the previous response as context to get a better response ##

## TODO: Make a final function to run the agent and return the response to be used by orchestrator ##