from langchain_community.embeddings import HuggingFaceEmbeddings
from operator import itemgetter
from typing import List, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
import qdrant_client

# Load environment variables
load_dotenv()

# Ensure API keys are properly retrieved
qdrant_api_key = os.getenv("QDRANT_API_KEY") or "your_default_qdrant_api_key"
qdrant_url = os.getenv("QDRANT_URL") or "your_default_qdrant_url"
openai_api_key = os.getenv("OPENAI_API_KEY") or "your_default_openai_api_key"
google_api_key=os.getenv("GEMINI_API_KEY_1")

# Initialize Qdrant client
client = qdrant_client.QdrantClient(qdrant_url, api_key=qdrant_api_key)

# Ensure embeddings match the Qdrant vector dimension (384)
embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)



# Connect to Qdrant vector database
vectorstore = Qdrant(
    client=client,
    collection_name="kfueit_edu_pk",
    embeddings=embeddings,
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=google_api_key)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Condense chat history and follow-up questions
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, 
    rephrase the follow-up question to be a standalone question in its original language.

    Chat History:
    {chat_history}

    Follow-Up Input: {question}

    Standalone Question:"""
)

# RAG Answer Synthesis Prompt
template = """You are the official chatbot for Khwaja Fareed University of Engineering & Information Technology, tasked with providing accurate and relevant information exclusively about Khwaja Fareed University of Engineering & Information Technology. 
You must only use the provided context or your own knowledge related to Khwaja Fareed University of Engineering & Information Technology to answer user queries. 
If you are unable to answer a question based on the context provided or your knowledge of Khwaja Fareed University of Engineering & Information Technology, simply state that you do not have the information.

Relevant Context:
{context}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Function to format retrieved documents
def _combine_documents(docs):
    return "\n\n".join(set(d.page_content for d in docs))  # Store page content only

# Function to format chat history
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    if not chat_history:
        return []  # Return empty list to prevent errors

    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# User input schema
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str

# Runnable to check and handle chat history
_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(itemgetter("question")),
)

# Combine input processing
_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
)

# Define the final RAG chain
chain = (
    (_inputs | ANSWER_PROMPT | llm | StrOutputParser())
    .with_types(input_type=ChatHistory)
    .with_fallbacks(
        [
            RunnableLambda(
                lambda prompt: "There was an error while generating your response. Please try again."
            )
        ]
    )
)