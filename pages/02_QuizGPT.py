import json

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

with st.sidebar:
    user_api_key = st.text_input(
        label="Use your OpenAI API key"
    )


st.title("QuizGPT")

# 함수 정의
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

if user_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=user_api_key
    ).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)
else:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    prompt = PromptTemplate.from_template("Make a quiz about {topic} by using follow documents: {docs}. And set quiz difficulty: {difficulty}")
    chain = prompt | llm
    return json.loads(chain.invoke({ "docs": _docs, "topic": topic, "difficulty": difficulty }).additional_kwargs["function_call"]["arguments"])


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

topic = ""

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "started" not in st.session_state:
    st.session_state.started = False
if "questionsCount" not in st.session_state:
    st.session_state.questionsCount = 0
if "correctCount" not in st.session_state:
    st.session_state.correctCount = 0

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    if st.session_state.submitted:
        st.subheader(f"Quiz Result: {st.session_state.correctCount}/{st.session_state.questionsCount}")

        if st.session_state.questionsCount == st.session_state.correctCount:
            st.balloons()
        
        if st.button("Retry!"):
            st.session_state.questionsCount = 0
            st.session_state.correctCount = 0
            st.session_state.submitted = False
            st.session_state.started = False
    else:
        if not st.session_state.started:
            difficulty = st.selectbox(
                "Choose quiz difficulty.",
                (
                    "Easy",
                    "Hard",
                ),
            )
                
            if st.button("Start Quiz"):
                response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
                st.session_state.started = True
                st.session_state.response = response

        else:
            with st.form("questions_form"):
                for idx, question in enumerate(st.session_state.response["questions"]):

                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        key=f"{idx}_radio",
                        index=None,
                    )
                
                if st.form_submit_button("Submit"):
                    st.session_state.submitted = True
                    st.session_state.questionsCount = len(st.session_state.response["questions"])

                    for idx, question in enumerate(st.session_state.response["questions"]):
                        selected_answer = st.session_state.get(f"{idx}_radio")
                        if selected_answer and {"answer": selected_answer, "correct": True} in question["answers"]:
                            st.session_state.correctCount += 1
    