import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# 1. Load the NLP Model (SentenceTransformer)
# -------------------------------
@st.cache_resource
def load_model():
    # Load the pre-trained model (you can change the model as needed)
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------------
# 2. PDF Text Extraction
# -------------------------------
@st.cache_data
def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file using PyPDF2.
    Caches the result for a given PDF input.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -------------------------------
# 3. Splitting Text into Chunks
# -------------------------------
@st.cache_data
def split_text(text, max_chunk_length=500):
    """
    Split the extracted text into smaller chunks for better embedding quality.
    This function caches the result because it's a pure function.
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_length:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += " " + para
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

# -------------------------------
# 4. Generate Embeddings for Each Text Chunk
# -------------------------------
def get_embeddings(text_chunks):
    """
    Generate embeddings for each text chunk using the NLP model.
    """
    embeddings = model.encode(text_chunks)
    return np.array(embeddings).astype("float32")

# -------------------------------
# 5. Create and Cache a FAISS Index for Similarity Search
# -------------------------------
@st.cache_resource
def create_faiss_index(embeddings):
    """
    Create a FAISS index from the provided embeddings using L2 (Euclidean) distance.
    This is a resource-heavy operation, so we cache it as a resource.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# -------------------------------
# 6. Main Application with Streamlit
# -------------------------------
def main():
    st.title("AI-Powered PDF Question Answering")
    st.write("Upload a PDF, and then ask questions based on its content.")

    # PDF Upload
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    if pdf_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(pdf_file)
        
        # Display the extracted text
        st.subheader("Extracted Text")
        st.text_area("PDF Content", text, height=200)

        # Split text into manageable chunks
        text_chunks = split_text(text)
        st.write(f"Number of text chunks: {len(text_chunks)}")

        # Generate embeddings for each chunk
        with st.spinner("Generating embeddings..."):
            embeddings = get_embeddings(text_chunks)

        # Create a FAISS index for similarity search
        index = create_faiss_index(embeddings)

        # Query Section
        st.subheader("Ask a Question")
        query = st.text_input("Enter your question:")
        if query:
            # Generate embedding for the query
            query_embedding = model.encode([query]).astype("float32")
            k = 3  # Retrieve the top 3 most similar chunks
            distances, indices = index.search(query_embedding, k)

            st.write("### Results:")
            for i, idx in enumerate(indices[0]):
                st.write(f"**Result {i+1}:**")
                st.write(text_chunks[idx])
                st.write("---")

if __name__ == '__main__':
    main()
