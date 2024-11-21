import streamlit as st
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq # Assuming ChatGroq is available in langchain

groq_api_key = st.secrets["GROQ_API_KEY"] 

# Sales coach prompt template
system_prompt = """
Act as an expert sales coach. Your job is to answer queries from sales agents to help them sell Hyundai Creta only. 
Use the following 7-step framework to guide your responses:

1. Identify the genuine problems or needs the vehicle meets for customers.
2. Acknowledge other competitive vehicles customers may be considering.
3. Describe how the Hyundai Creta ideally improves customers' lives.
4. Present yourself as a trusted advisor aiming to match the right vehicle to each customer.
5. Explain the main benefits and value the Hyundai Creta provides.
6. Provide transparent evidence to back up claims about the vehicle's features.
7. Offer next steps for interested customers to learn more or test drive.

Never recommend any other car other than Hyundai Creta.
If you do not know the answer to a question, simply state so. 
Focus on being helpful, honest, and customer-oriented in crafting sales coaching advice.
"""

# Helper function to process PDF files
def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Helper function to process text files
def process_text(file):
    return file.read().decode("utf-8")  # Decode file to a string

# ChatPromptTemplate configuration
def create_prompt_with_context(raw_text):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt + "\n" + raw_text),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with Sales Coach 🚗", page_icon="🚗")

    # Add custom CSS for chat styling
    st.markdown("""
        <style>
        .user-message {
            background-color: #FFFFFF; /* White background for user messages */
            padding: 8px 12px;
            border-radius: 12px;
            text-align: left;
            margin-left: auto; /* Push the message to the right */
            margin-right: 10px;
            margin-bottom: 10px; /* Add space below user message */
            max-width: 70%;
            color: #000; /* Black text color */
            display: block;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        .assistant-message {
            background-color: #D6EAF8; /* Light Blue background for assistant messages */
            padding: 8px 12px;
            border-radius: 12px;
            text-align: left;
            margin-left: 10px; /* Push the message to the left */
            margin-right: auto;
            margin-bottom: 15px; /* Add space below assistant message */
            max-width: 70%;
            color: #000; /* Black text color */
            display: block;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px; /* Space between messages in the container */
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session states
    if "context_text" not in st.session_state:
        st.session_state.context_text = ""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    st.header("Chat with Sales Coach 🚗")

    # Sidebar for uploading files and chat sessions
    with st.sidebar:
        st.subheader("Upload Your Context")
        uploaded_file = st.file_uploader("Upload your files (PDF or TXT)", type=["pdf", "txt"])
        if st.button("Add File") and uploaded_file:
            try:
                # Check file type and process accordingly
                if uploaded_file.name.endswith(".pdf"):
                    raw_text = process_pdf(uploaded_file)
                elif uploaded_file.name.endswith(".txt"):
                    raw_text = process_text(uploaded_file)
                else:
                    st.error("Unsupported file type. Please upload a PDF or TXT file.")
                    return
                
                # Store the full context in session state
                st.session_state.context_text = raw_text
                
                # Reset conversation with the full context
                prompt = create_prompt_with_context(st.session_state.context_text)
                memory = ConversationBufferMemory(return_messages=True, memory_key="history")
                llm = ChatGroq(api_key=groq_api_key,temperature=0.7, model_name="mixtral-8x7b-32768")
                st.session_state.conversation = ConversationChain(
                    llm=llm,
                    memory=memory,
                    prompt=prompt,
                    verbose=True
                )
                st.success("File processed successfully and context updated!")
            except Exception as e:
                st.error(f"Failed to process file: {str(e)}")

        st.subheader("Chat Sessions")
        # List all existing sessions
        for session_name in list(st.session_state.sessions.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(session_name):  # Load a session when clicked
                    st.session_state.current_session = session_name
                    st.session_state.chat_history = st.session_state.sessions[session_name].copy()
            with col2:
                if st.button("❌", key=f"delete_{session_name}"):  # Delete button
                    del st.session_state.sessions[session_name]
                    if session_name == st.session_state.current_session:
                        st.session_state.current_session = None
                        st.session_state.chat_history = []
                    st.rerun()

        # Button to create a new session
        if st.button("New Chat"):
            new_session_name = f"Chat {len(st.session_state.sessions) + 1}"
            st.session_state.sessions[new_session_name] = st.session_state.chat_history.copy()
            st.session_state.current_session = new_session_name
            st.session_state.chat_history = []

    # Input box for user's question
    col1, col2 = st.columns([9, 1])  # Adjusted column proportions for alignment
    with col1:
        user_input = st.text_input(
            "Ask your question:",
            value=st.session_state.user_input,
            key="dynamic_user_input",
            placeholder="Type your question and click send."
        )
    with col2:
        if st.button("Send"):
            if user_input.strip():
                if not st.session_state.context_text:
                    st.error("Please upload and process a file before asking questions.")
                else:
                    try:
                        response = st.session_state.conversation.run({"input": user_input})
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.session_state.user_input = ""
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a valid question.")

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Clear Chat Button
    if st.button("Clear Chat"):
        if st.session_state.chat_history:
            new_session_name = f"Chat {len(st.session_state.sessions) + 1}"
            st.session_state.sessions[new_session_name] = st.session_state.chat_history.copy()
        st.session_state.chat_history = []
        st.session_state.user_input = ""
        st.rerun()

if __name__ == "__main__":
    main()
