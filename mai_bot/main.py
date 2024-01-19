import os
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores import FAISS


def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # keep embeddings model in sync with create_db.py!
    
    print(f"Loading path, {os.path.join(os.path.dirname(__file__),'db')}")
    vectorstore = FAISS.load_local(os.path.join(os.path.dirname(__file__),"db"), embeddings)
    retriever = vectorstore.as_retriever(k=8)

    return retriever

def get_prompt():
    template = """Answer the question using only information from the following, related previous answers or context:
    # CONTEXT:
    {context}
    # INSTRUCTIONS:
    - Replace any mentions of "provider", "supplier" or similar with "Metamaze"
    - Replace any mentions of "AXA", "AG Insurance", "KBC", or other potential client names with "Client"
    # QUESTION: 
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def get_pipeline(prompt, db, model):
    rag_chain_from_docs = (
    {
        "context": lambda input: input["documents"],
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)
    rag_chain_with_source = RunnableMap(
    {"documents": db, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: input["documents"],
    "answer": rag_chain_from_docs,
}
    
    return rag_chain_with_source

def init_pipeline():
    db = load_db()
    model = ChatOpenAI(model_name="gpt-4")
    prompt = get_prompt()
    rag_chain_with_source = get_pipeline(prompt, db, model)
    return rag_chain_with_source


if __name__ == "__main__":
    pipeline = init_pipeline()

    s = """Does the provider perform regular vulnerability assessments / penetration tests to determine security gaps?"""
    result = pipeline.invoke(s)
    
    print(f"## Reference data")
    
    for doc in result['documents']:
        print(f"({doc.metadata})")
        print(doc.page_content)
        print("-------")
    
    print(f"\n ## Answer:\n\n{result['answer']}")







