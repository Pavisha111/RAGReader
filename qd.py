import streamlit as st
import os
from pathlib import Path
from llama_index.core import PromptTemplate, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered  # Use DoclingReader here
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
import qdrant_client
from llama_index.core.node_parser import MarkdownNodeParser,MarkdownElementNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define global variables
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
node_parser = MarkdownNodeParser()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index" not in st.session_state:
    st.session_state.index = None
if "file_ingested" not in st.session_state:
    st.session_state.file_ingested = False  # Track if the file has been ingested

def data_ingestion(file_path, base_dir="data_storage"):
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Define the markdown file path
    markdown_filename = Path(base_dir) / "output.md"

    # Check if the markdown file already exists
    if not markdown_filename.exists():
        # If the markdown file does not exist, convert the PDF to Markdown
        path = Path(file_path)
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )

        rendered = converter.convert(file_path)
        text, _, images = text_from_rendered(rendered)

        # Save the converted text to the markdown file
        with open(markdown_filename, 'w') as f:
            f.write(text)
        st.sidebar.success("PDF converted to Markdown and saved.")
    else:
        st.sidebar.info("Markdown file already exists. Skipping PDF conversion.")

    # Load the markdown file as documents
    documents = FlatReader().load_data(markdown_filename)

    # Define the SYSTEM_PROMPT and other configurations
    SYSTEM_PROMPT = """
    # *Instructions:*
    The following markdown document contains information related to **[specific domain, e.g., sensors, circuits, aerospace systems, or cybersecurity]**. When a query is made, please extract *only* the relevant data and information from this document (including both *text* and *tables*) to form an answer. 

    ### *Rules:*
    1. If the query is addressed in the document:
    - Use the provided content (e.g., text, tables, or examples) to form the answer.
    - Reference specific IDs, components, or section headings for clarity.
    2. If the query is *not explicitly covered* in the document:
    - Respond with: *"Not in the document."*
    3. Keep responses **concise**, **accurate**, and **based solely on the document**.

    ---

    ## *Document Overview:*
    - **Text Section**: Contains detailed explanations, descriptions, or problem scenarios.
    - **Tables**: Provides structured data like sensor specifications, circuit configurations, and fault indications.
    - **Problem Section**: Lists possible system issues and troubleshooting steps.

    ---

    ## *Example Queries and Responses:*
    1. **Query**: "What happens if the temperature sensor reads 65°C?"
    **Response**:  
    "The *S002 Temperature Sensor* has a measurement range of *-50°C to 150°C*. A reading of *65°C* is within the normal range. However, if the temperature exceeds the safe operating range for the system, it may indicate an overheating issue, and the circuit may need to be shut down to prevent damage."

    2. **Query**: "What is the voltage sensitivity of the pressure sensor?"
    **Response**:  
    "The *S003 Pressure Sensor* has a voltage sensitivity of *0.5 Pa*."

    3. **Query**: "How can I fix an issue where the voltage exceeds 5V?"
    **Response**:  
    "Not in the document."

    4. **Query**: "Which resistor is used in the circuit with the voltage sensor?"
    **Response**:  
    "The *C001 Resistor* with a value of *1k Ohms* is connected to the *S001 Voltage Sensor*."

    ---

    ## *Expected Response Format:*
    - Responses must reference document content, e.g., IDs, section names, or tables.
    - For out-of-scope queries, always respond with: *"Not in the document."*
    - Avoid guessing or providing extra information outside the document's scope.

    ---

    ## *Text Section Example:*
    This section contains detailed explanations or problem scenarios. For example:

    > If the temperature sensor (S002) reads above 50°C, it can indicate an overheating issue with the system. If the temperature exceeds the safe operating range, the circuit may need to be shut down to prevent damage.

    > A **voltage sensor** detects voltage in a range of 0-5V. If the voltage exceeds this range, the sensor may give incorrect readings.

    ---

    ## *Table Example 1: Sensor Specifications*

    | Sensor ID | Type        | Measurement Range | Voltage Sensitivity | Accuracy (%) |
    |-----------|-------------|-------------------|----------------------|--------------|
    | S001      | Voltage     | 0-5 V             | 0.01 V               | 1.5          |
    | S002      | Temperature | -50°C to 150°C    | 0.1°C                | 0.5          |
    | S003      | Pressure    | 0-1000 Pa         | 0.5 Pa               | 2.0          |

    ---

    ## *Table Example 2: Circuit Configuration*

    | Circuit ID | Component    | Value   | Units  | Notes                     |
    |------------|--------------|---------|--------|---------------------------|
    | C001       | Resistor     | 1k      | Ohms   | Connected to S001 sensor  |
    | C002       | Capacitor    | 100nF   | F      | Parallel with S002 sensor |
    | C003       | Inductor     | 10µH    | H      | Series with S003 sensor   |

    ---

    ## *Problem Section Example:*
    This section describes potential issues and troubleshooting steps. For example:

    > If the temperature sensor reads **65°C**, it may indicate an overheating issue. The circuit might need cooling to prevent damage.

    > If the circuit exceeds **5V** voltage, the **S001 Voltage Sensor** may provide incorrect readings.

    ---

    ## *Query Section:*
    {query_str}
    """

    query_wrapper_prompt = PromptTemplate(
        "[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST]"
    )

    # Define LLM model and embedding
    llm = Ollama(model="mistral", request_timeout=120.0, query_wrapper_prompt=query_wrapper_prompt)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(
        collection_name="paper", client=client, enable_hybrid=True, batch_size=20
    )

    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Create and return the index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        transformations=[node_parser],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
    )

    return index

# Function to handle user queries
def run_query(index, query):
    query_engine = index.as_query_engine()
    chat_engine = index.as_chat_engine(
        query_engine=query_engine,
        chat_mode="context", 
        memory=memory,
        system_prompt=(
            """
            You are an AI assistant that answers questions or solves problems strictly based on the provided source documents. Follow these rules:
            - Generate clear, concise, and human-readable responses without any irrelevant or extraneous text.
            - Provide only the requested output, avoiding additional commentary, explanations, or language before or after the answer.
            - Ensure accuracy by generating responses solely from the information in the given source documents.
            - If the question is not addressed in the documents, respond with: "I don't have any data about [question/query]. Please ask another query I can answer."
            """
        ),
        llm=Ollama(model="mistral", request_timeout=120.0)
    )
    response = chat_engine.chat(query)
    return response

# Streamlit Interface
st.title("QD Bot")

# PDF Ingestion Section
st.sidebar.header("Upload PDF")
if not st.session_state.file_ingested:
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        file_path = f"data_storage/{uploaded_file.name}"
        os.makedirs("data_storage", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.index = data_ingestion(file_path)
        st.session_state.file_ingested = True
        st.sidebar.success("File ingested and indexed successfully!")
else:
    st.sidebar.info("PDF already ingested.")

# Chat Section
st.header("Chat Interface")
chat_container = st.container()

with chat_container:
    if st.session_state.index is None:
        st.warning("No data available. Please ingest a file first.")
    else:
        user_query = st.text_input("Type your question:", key="input_query")
        if user_query:
            with st.spinner("Bot is typing..."):
                bot_response = run_query(st.session_state.index, user_query)
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.markdown(f"**Bot:** {bot_response}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add sidebar with instructions
with st.sidebar:
    st.markdown("""
    ## How to use
    1. Upload your PDF document using the file uploader
    2. Wait for the document to be processed
    3. Start chatting with the assistant about the document content
    4. The assistant will provide relevant answers based on the document
    
    **Note:** The chat history will be cleared when you upload a new document.
    """)