
import streamlit as st
import pandas as pd
import openai

# Load the API key from Streamlit secrets
openai.api_key = st.secrets['openai_key']

# Load the CSV file
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to get a response from OpenAI API
def get_openai_response(context):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": context}],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message["content"].strip()

# Load data
data = load_data("baruch.csv")

st.title("Retrieval-Augmented Generation (RAG) App")

# Input for user query
query = st.text_input("Enter your query:")

if query:
    # Search for the most relevant context
    results = data[data['context'].str.contains(query, case=False, na=False)]
    
    if not results.empty:
        # Display the top matching context
        context = results.iloc[0]['context']
        st.subheader("Top Matching Context")
        st.write(context)
        
        # Generate a response using the context
        response = get_openai_response(context)
        st.subheader("Generated Response")
        st.write(response)
    else:
        st.write("No matching context found.")
