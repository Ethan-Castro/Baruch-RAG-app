import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_key"])


# Load the specific CSV file
@st.cache_data
def load_data():
    # Replace 'your_file.csv' with the path to your actual CSV file
    df = pd.read_csv('your_file.csv')
    return df

# TF-IDF Vectorizer
@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(stop_words='english')

# Semantic search function using TF-IDF
def semantic_search(query, df, vectorizer, top_k=5):
    # Convert text column to string and handle NaN values
    df['text'] = df['text'].fillna('').astype(str)
    
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results

# Generate response using OpenAI
def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

# Streamlit app
st.set_page_config(page_title="RAG App", layout="wide")

st.title("🧠 Retrieval Augmented Generation App")

# Load the data
df = load_data()
vectorizer = get_vectorizer()

# Display information about the dataset
st.subheader("Dataset Information")
st.write(f"Number of records: {len(df)}")
st.write(f"Columns: {', '.join(df.columns)}")

# Settings in sidebar
st.sidebar.header("Settings")
text_column = st.sidebar.selectbox("Select the column containing the text data", df.columns)
top_k = st.sidebar.slider("Number of relevant documents", 1, 10, 5)

# Main interface
st.subheader("Ask a question about the data")
query = st.text_input("Enter your question:")

if query:
    try:
        # Use the selected text column
        df['text'] = df[text_column]
        
        with st.spinner("Searching for relevant information..."):
            results = semantic_search(query, df, vectorizer, top_k)

        st.subheader("Relevant Information")
        for _, row in results.iterrows():
            st.markdown(f"**Similarity: {row['similarity']:.2f}**")
            st.info(row['text'])

        with st.spinner("Generating answer..."):
            context = " ".join(results['text'].tolist())
            answer = generate_response(query, context)

        st.subheader("Generated Answer")
        st.success(answer)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please ensure that the selected column contains valid text data.")

# Styling
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stTitle {
    color: #2c3e50;
    font-family: 'Helvetica Neue', sans-serif;
}
.stSubheader {
    color: #34495e;
    font-family: 'Helvetica Neue', sans-serif;
}
.stMarkdown {
    font-family: 'Helvetica Neue', sans-serif;
}
</style>
""", unsafe_allow_html=True)
