import pickle
import os
import functionz as fz
# Loading data from the .pkl file
with open('doc_objects/doc_table_chunks.pkl', 'rb') as file:
    doc_table_chunks = pickle.load(file)
with open('doc_objects/doc_texts_chunks.pkl', 'rb') as file:
    doc_text_chunks = pickle.load(file)

# print("TABLES:", doc_table_chunks)
# print("TEXTS: ", doc_text_chunks)


outputpath = "doc_objects/vectordb"
result = fz.into_vector_db(doc_text_chunks, outputpath )
result = fz.into_vector_db(doc_table_chunks, outputpath )