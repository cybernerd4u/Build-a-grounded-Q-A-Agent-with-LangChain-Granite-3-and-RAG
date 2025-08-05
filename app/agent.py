import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA
from retriever import load_documents, split_documents, get_vectorstore

# Load environment variables
load_dotenv(dotenv_path="../configs/config.env")

API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
MODEL_ID = os.getenv("IBM_FOUNDATION_MODEL_ID")
REGION = os.getenv("IBM_REGION")

# IBM Watsonx config
llm = WatsonxLLM(
    model_id=MODEL_ID,
    apikey=API_KEY,
    url=f"https://{REGION}.ml.cloud.ibm.com",
    project_id=PROJECT_ID
)
def build_agent():
    # Load and split documents
    docs = load_documents("../data/sample_docs.pdf")
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(chunks)
    retriever = vectorstore.as_retriever()

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa

if __name__ == "__main__":
    agent = build_agent()
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() in ["quit", "exit"]:
            break
        answer = agent.run(question)
        print(f"\nAnswer: {answer}")
