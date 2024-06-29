from utils import load_documents, split_texts, create_embeddings, create_vectorstore

def main():
    directory = 'readme_files'
    documents = load_documents(directory)
    texts = [doc['text'] for doc in documents]
    chunks = split_texts(texts)
    embeddings = create_embeddings(chunks)
    vectorstore = create_vectorstore(embeddings, chunks)
    faiss.write_index(vectorstore.index, 'readme_index.faiss')
    with open('chunks.npy', 'wb') as f:
        np.save(f, chunks)

if __name__ == "__main__":
    main()
