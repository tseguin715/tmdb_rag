import streamlit as st
import yaml
import os
import re
from typing import List

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# --- 1. CONFIGURATION ---
RAG_DATABASE = "db/tmdb_faiss_db"
EMBEDDING_MODEL = 'text-embedding-3-large'
CHAT_LLM_OPTIONS = ["gpt-4o", "gpt-4o-mini"]
CREDENTIALS_FILE = "credentials.yml"

# --- 2. PAGE SETUP ---
st.set_page_config(page_title="AI TV & Movie Recommender", page_icon="🎬")
st.title("🎬 AI TV & Movie Recommender")
st.markdown("I'm your AI-powered movie and TV show recommender. Tell me what you're looking for, or try one of the example prompts below to get started!")

# --- SESSION STATE INITIALIZATION ---
# Initialize session state for filters to ensure they are persistent and controllable.
if 'min_rating' not in st.session_state:
    st.session_state.min_rating = 0.0
if 'year_range' not in st.session_state:
    st.session_state.year_range = (1980, 2025)
if 'auto_prompt' not in st.session_state:
    st.session_state.auto_prompt = None

# --- 3. SIDEBAR FILTERS ---
st.sidebar.title("Configuration")

# LLM model selector
if 'LLM' not in st.session_state:
    st.session_state.LLM = CHAT_LLM_OPTIONS[0]
st.session_state.LLM = st.sidebar.selectbox("Choose OpenAI model", CHAT_LLM_OPTIONS, index=CHAT_LLM_OPTIONS.index(st.session_state.LLM))

# Clear Chat History Button
def clear_chat_history():
    """Resets the chat history."""
    st.session_state.langchain_messages = []
    # Optionally reset other states if needed
st.sidebar.button('🧹 Clear Chat History', on_click=clear_chat_history, use_container_width=True)
st.sidebar.divider()

# --- NEW: Quick Filters Section ---
st.sidebar.subheader("⚡ Quick Filters")

DECADES = {
    "All": (1980, 2025), "2020s": (2020, 2025), "2010s": (2010, 2019),
    "2000s": (2000, 2009), "1990s": (1990, 1999), "1980s": (1980, 1989),
}

def handle_decade_change():
    """Callback to update the year range slider when a decade is selected."""
    st.session_state.year_range = DECADES[st.session_state.decade_selector]

st.sidebar.selectbox(
    "Decade Selection", options=list(DECADES.keys()),
    key="decade_selector", on_change=handle_decade_change
)

def handle_rating_checkbox():
    """Callback to update the rating slider when the 8+ checkbox is toggled."""
    if st.session_state.rating_checkbox:
        st.session_state.min_rating = 8.0
    else:
        st.session_state.min_rating = 0.0 # Revert to default

st.sidebar.checkbox("Only 8+ Rated", key="rating_checkbox", on_change=handle_rating_checkbox)
st.sidebar.divider()


st.sidebar.title("Content Filters")

# Content type selector
ALL_TYPES = ["Movie", "TV Show"]
selected_types = st.sidebar.multiselect(
    "Select Content Type",
    options=ALL_TYPES,
    default=ALL_TYPES
)

# Release year slider (now controlled by session state)
selected_year_range = st.sidebar.slider(
    "Select Release Year Range", min_value=1980, max_value=2025,
    key='year_range' # This key binds the slider to st.session_state.year_range
)

# Rating slider (now controlled by session state)
min_rating = st.sidebar.slider(
    "Minimum Average Rating", min_value=0.0, max_value=10.0,
    key='min_rating', # Binds to st.session_state.min_rating
    step=0.1
)

# Genre selector
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 
    'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 
    'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western', 'Action & Adventure',
    'Kids', 'News', 'Reality', 'Sci-Fi & Fantasy', 'Soap', 'Talk', 'War & Politics'
]
selected_genres = st.sidebar.multiselect(
    "Filter by Genres (optional)",
    options=sorted(list(set(ALL_GENRES))),
    default=[]
)

# --- 4. API KEY & SESSION STATE ---
try:
    os.environ['OPENAI_API_KEY'] = st.secrets['openai']
except KeyError:
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets.")
    st.stop()

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! What kind of movie or TV show are you in the mood for?")

# --- 5. RAG CHAIN & FILTERING LOGIC ---
@st.cache_resource(show_spinner="Loading vector database...")
def load_vector_store():
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    try:
        vectorstore = FAISS.load_local(RAG_DATABASE, embedding_function, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading FAISS database: {e}")
        st.stop()

class MetadataFilter:
    """Custom document filter that filters based on metadata criteria."""
    
    def __init__(self, types: List[str], year_range: tuple, min_rating: float, genres: List[str]):
        self.types = types
        self.year_range = year_range
        self.min_rating = min_rating
        self.genres = genres

    def filter_documents(self, documents: List[Document]) -> List[Document]:
        """Filter documents based on metadata criteria."""
        filtered_docs = []
        min_year, max_year = self.year_range
        selected_types_set = set(t.lower() for t in self.types) if self.types else set()
        selected_genres_set = set(g.lower() for g in self.genres) if self.genres else set()

        for doc in documents:
            if not selected_types_set:
                continue
            
            doc_type = doc.metadata.get("type", "").lower()
            if doc_type not in selected_types_set:
                continue
            
            release_year = doc.metadata.get("release_year")
            if release_year:
                try:
                    year_int = int(release_year)
                    if not (min_year <= year_int <= max_year):
                        continue
                except (ValueError, TypeError):
                    continue

            try:
                rating = float(doc.metadata.get("average_rating", 0.0))
                if rating < self.min_rating:
                    continue
            except (ValueError, TypeError):
                continue

            if selected_genres_set:
                doc_genres_raw = doc.metadata.get("genres", [])
                if isinstance(doc_genres_raw, str):
                    doc_genres = set(g.strip().lower() for g in doc_genres_raw.split(',') if g.strip())
                elif isinstance(doc_genres_raw, list):
                    doc_genres = set(g.lower() for g in doc_genres_raw if isinstance(g, str))
                else:
                    doc_genres = set()
                
                if not selected_genres_set.intersection(doc_genres):
                    continue
            
            filtered_docs.append(doc)
        
        return filtered_docs

def create_rag_chain(types: List[str], year_range: tuple, min_rating: float, genres: List[str]):
    """Create RAG chain with metadata filtering."""
    vectorstore = load_vector_store()
    
    class FilteredRetriever:
        def __init__(self, vectorstore, metadata_filter):
            self.vectorstore = vectorstore
            self.metadata_filter = metadata_filter
            
        def get_relevant_documents(self, query: str) -> List[Document]:
            docs = self.vectorstore.similarity_search(query, k=50)
            filtered_docs = self.metadata_filter.filter_documents(docs)
            return filtered_docs[:10]
    
    metadata_filter = MetadataFilter(
        types=types, year_range=year_range, min_rating=min_rating, genres=genres
    )
    filtered_retriever = FilteredRetriever(vectorstore, metadata_filter)

    llm = ChatOpenAI(model=st.session_state.LLM, temperature=0.7)

    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    document_prompt = PromptTemplate.from_template(
        """---
Title: {title}
Type: {type}
Released: {release_year}
Rating: {average_rating}
Genres: {genres}
Image URL: {image_url}
Description: {description}
TMDb URL: {product_url}
---"""
    )

    qa_system_prompt = """You are an expert TV & Movie recommender. Use the following pieces of retrieved context to answer the question. Your primary goal is to be helpful and provide accurate recommendations that match the user's request and filters. If you recommend a title, you MUST use the following format to display it: [Title of Movie or Show: image_url={{image_url}}, product_url={{product_url}}]. If no titles in the context match, say you couldn't find anything matching their criteria. Be conversational and engaging.

    Context:
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    def rag_logic(input_dict):
        query = input_dict["input"]
        chat_history = input_dict.get("chat_history", [])
        
        if chat_history:
            try:
                contextualized_response = llm.invoke(
                    contextualize_q_prompt.format_messages(chat_history=chat_history, input=query)
                )
                contextualized_query = contextualized_response.content
            except Exception:
                contextualized_query = query
        else:
            contextualized_query = query
        
        docs = filtered_retriever.get_relevant_documents(contextualized_query)
        
        formatted_docs = []
        for doc in docs:
            try:
                formatted_doc = document_prompt.format(**doc.metadata)
                formatted_docs.append(formatted_doc)
            except Exception:
                continue
        
        context = "\n\n".join(formatted_docs)
        
        try:
            messages = qa_prompt.format_messages(
                context=context, chat_history=chat_history, input=query
            )
            response = llm.invoke(messages)
            return {"answer": response.content}
        except Exception:
            return {"answer": "I'm sorry, I encountered an issue processing your request. Please try again."}
    
    return RunnableLambda(rag_logic)

# --- 6. RESPONSE RENDERING ---
def display_response_with_images(response_text):
    """Display response text with embedded images for recommended products."""
    # This new pattern optionally matches a title before the colon
    product_pattern = r"\[(?:.*?:)?\s*image_url=(.*?), product_url=(.*?)\]"
    matches = list(re.finditer(product_pattern, response_text))
    
    if not matches:
        st.write(response_text)
        return

    last_end = 0
    for match in matches:
        if last_end < match.start():
            st.write(response_text[last_end:match.start()])
        
        image_url = match.group(1).strip()
        product_url = match.group(2).strip()
        
        if image_url.startswith('http') and product_url.startswith('http'):
            st.markdown(
                f'<a href="{product_url}" target="_blank" style="display:block; text-align:center;"><img src="{image_url}" style="width:50%; max-width: 250px; border-radius: 10px; margin: 10px auto;"></a>',
                unsafe_allow_html=True
            )
        
        last_end = match.end()
    
    if last_end < len(response_text):
        st.write(response_text[last_end:])

# --- 7. CHAT INTERFACE ---
for msg in msgs.messages:
    with st.chat_message(msg.type):
        if msg.type == "ai":
            display_response_with_images(msg.content)
        else:
            st.write(msg.content)

# --- NEW: Example Prompts for new chats ---
if len(msgs.messages) <= 1:
    st.markdown("---")
    st.markdown("#### Or try one of these ideas:")
    
    EXAMPLE_PROMPTS = [
        "Mind-bending sci-fi movies like Inception or The Matrix",
        "A cozy, feel-good TV series to watch on a rainy day",
        "Critically acclaimed thrillers from the 1990s",
        "Hidden gem comedies that flew under the radar"
    ]

    def submit_example_prompt(prompt_text: str):
        """Sets the selected prompt in session state for auto-submission."""
        st.session_state.auto_prompt = prompt_text

    cols = st.columns(2)
    for i, prompt_text in enumerate(EXAMPLE_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt_text, use_container_width=True, key=f"ex_prompt_{i}"):
                submit_example_prompt(prompt_text)
                st.rerun()

# --- NEW: Unified input handling for user text and example clicks ---
prompt_to_process = None

# Check for an auto-submitted prompt first
if st.session_state.auto_prompt:
    prompt_to_process = st.session_state.auto_prompt
    st.session_state.auto_prompt = None # Clear after use

# Then, check for user input from the chat box
user_input = st.chat_input("Suggest a movie or TV show...")
if user_input:
    prompt_to_process = user_input

# If there is a prompt to process, run the RAG chain
if prompt_to_process:
    st.chat_message("human").write(prompt_to_process)

    # Create RAG chain with current filter settings from session state
    rag_chain = create_rag_chain(
        types=selected_types,
        year_range=st.session_state.year_range,
        min_rating=st.session_state.min_rating,
        genres=selected_genres
    )
    
    rag_chain_with_history = RunnableWithMessageHistory(
        rag_chain, lambda session_id: msgs,
        input_messages_key="input", history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain_with_history.invoke(
                    {"input": prompt_to_process},
                    config={"configurable": {"session_id": "any"}}
                )
                display_response_with_images(response["answer"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("I'm sorry, I encountered an issue while processing your request. Please try again.")

# Debug information in sidebar
with st.sidebar.expander("View Message History"):
    st.json(st.session_state.langchain_messages)

with st.sidebar.expander("Current Filter Settings"):
    st.write("**Content Types:**", selected_types)
    st.write("**Year Range:**", st.session_state.year_range)
    st.write("**Minimum Rating:**", st.session_state.min_rating)
    st.write("**Genres:**", selected_genres)