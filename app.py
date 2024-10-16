import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Step 1: Encode Document and Store Embeddings with FAISS
def encode_document_with_faiss(document_text):
    # Load pre-trained sentence transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings from the document (split text into paragraphs or sections)
    document_embeddings = embedding_model.encode(document_text)

    # Initialize FAISS index
    dimension = document_embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric (Euclidean)

    # Add document embeddings to the FAISS index
    index.add(np.array(document_embeddings))

    return index, embedding_model, document_embeddings

# Step 2: Query FAISS for Relevant Document Sections
def retrieve_relevant_document_sections(query, index, embedding_model, document_text):
    # Encode the query to find similar sections in the document
    query_embedding = embedding_model.encode([query])

    # Search for similar documents
    D, I = index.search(np.array(query_embedding), k=3)  # Top 3 matching sections
    relevant_sections = [document_text[i] for i in I[0]]

    return relevant_sections

# Step 3: Use GPT-2 Model to Generate New Features
def generate_features_with_gpt2(relevant_context):
    # Load GPT-2 model and tokenizer (this can be replaced with any similar model)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set the pad_token to eos_token to avoid the padding issue
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the relevant context
    inputs = tokenizer(relevant_context, return_tensors="pt", padding=True, truncation=True).input_ids

    # Generate features using GPT-2
    generated_output = model.generate(inputs, max_length=500, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return generated_text

# Step 4: Main Function to Execute the Full Pipeline
def main():
    # Example document with multiple sections
    document_text = ['Walking for 30 minutes a day or more on most days of the week is a great way to improve or maintain your overall health.',
        'If you can’t manage 30 minutes a day, remember even short walks more frequently can be beneficial.',
        'Walking with others can turn exercise into an enjoyable social occasion.',
        'See your doctor for a medical check-up before embarking on a higher-intensity new fitness program, particularly if you are aged over 40 years, are overweight or haven’t exercised in a long time.'
    ]

    # New business rules for feature generation
    query = "Should I walk?"

    # Step 1: Encode document and store embeddings in FAISS
    faiss_index, embedding_model, document_embeddings = encode_document_with_faiss(document_text)

    # Step 2: Retrieve relevant sections from the document based on the query
    relevant_context = retrieve_relevant_document_sections(query, faiss_index, embedding_model, document_text)

    # Step 3: Generate new features using GPT-2 with the retrieved context
    generated_features = generate_features_with_gpt2(relevant_context)

    # Output the generated features
    print(f"Generated Features:\n{generated_features}")

if __name__ == "__main__":
    main()
