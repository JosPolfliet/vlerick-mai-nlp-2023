
"""
Streamlit application for the bot
To run, use the command: streamlit run app.py
"""

import pandas as pd
import streamlit as st
from main import init_pipeline

pipeline = init_pipeline()

def process_question(question):
    # Add your processing logic here
    result =  pipeline.invoke(question)
    return result

question = st.text_input("Enter your question")

if st.button('Submit'):
    response = process_question(question)
    st.write(response["answer"])
    st.markdown("<H1>Sources consulted</H1>", unsafe_allow_html=True)
    docs = [{
        "source": doc.metadata.get("source") or doc.metadata.get("title"),
        "text": doc.page_content}
        for doc in response['documents']
    ]

    df = pd.DataFrame(response["documents"]) 

    st.write(df.to_html(), unsafe_allow_html=True) # unsafe_allow_html: don't run this in production without cleaning source html