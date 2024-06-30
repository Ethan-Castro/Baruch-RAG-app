import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_key"])

# Load and preprocess the data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# TF-IDF Vectorizer
@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(stop_words='english')

# Semantic search function using TF-IDF
def semantic_search(query, df, vectorizer, top_k=5):
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

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    vectorizer = get_vectorizer()

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of relevant documents", 1, 10, 5)

    # Display the first few rows of the CSV
    st.subheader("Preview of uploaded CSV")
    st.write(df.head())

    # Let user select the text column
    text_column = st.selectbox("Select the column containing the text data", df.columns)

    st.subheader("Ask a question")
    query = st.text_input("Enter your question:")

    if query and text_column:
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

else:
    st.info("Please upload a CSV file to get started.")

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
