import jieba
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import torch
from langchain.embeddings import HuggingFaceEmbeddings
import custom_llm as md
import copy
import os
from custom_llm import ZhiPuLLm
from langchain.vectorstores import FAISS


def jaccard_similarity(str1, str2):
    words1 = set(jieba.cut(str1))
    words2 = set(jieba.cut(str2))
    return len(words1.intersection(words2)) / len(words1.union(words2))


def doc_loader(file, type, elements):
    if elements and type in ['pdf', 'docx']:
        loader = UnstructuredWordDocumentLoader(file, mode="elements")
        data = loader.load()
    elif elements == 0 and type in ['pdf', 'docx']:
        loader = UnstructuredWordDocumentLoader(file)
        data = loader.load()
    return data
def get_tables(doc_data):
    text_data = []
    table_data = []
    for v in doc_data:
        if v.metadata['category'] == "Table":
            # print("table")
            table_data.append(v)
        else:
            # print("doc")
            text_data.append(v)
    return text_data, table_data
def get_headers(doc_data):
    page_content_list = []
    data_title_ind_original = []
    for i, v in enumerate(doc_data):
        cat = v.metadata['category']
        if cat == 'Title' and\
                (v.page_content[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']):
            #get page content for input into further prompt
            page_content_list.append(v.page_content)
            #get indices for the title documents
            data_title_ind_original.append(i)
    #use the page content to input to prompt
    page_content_string = "\n".join(page_content_list)
    prompt = """帮我整理一下目录，不需要小标题：\n {}""".format(page_content_string)
    headers_result = md.zhipu_turbo(prompt)
    print(headers_result)
    headers_list = headers_result.split('\n')
    print(headers_list)
    original_data_index = []
    score = 0.4
    for ind, content in enumerate(page_content_list):
        for header in headers_list:
            sim = jaccard_similarity(content, header)
            if sim >= score:
                #get original index
                orig_ind = data_title_ind_original[ind]
                original_data_index.append(orig_ind)
    return headers_result, headers_list, original_data_index


def doc_split(doc, ind_list, chunk):

    metadataz = doc[0].metadata

    if len(ind_list)>0 and chunk[0] == 0 :
        ind_list2 = [0] + ind_list
        index_ranges = [(ind_list2[i - 1], ind_list2[i]) for i in range(1, len(ind_list2))]
        page_content = [v.page_content for i, v in enumerate(doc) ]
        page_content_list = [''.join(page_content[start:end]) for start, end in index_ranges]
        doc_chunks = [Document(page_content = content, metadata = metadataz) for
                      content in page_content_list]
        return doc_chunks
    elif len(ind_list)>0 and chunk[0]>0:
        ind_list2 = [0] + ind_list
        index_ranges = [(ind_list2[i - 1], ind_list2[i]) for i in range(1, len(ind_list2))]
        page_content = [v.page_content for i, v in enumerate(doc)]
        page_content_list = [''.join(page_content[start:end]) for start, end in index_ranges]
        doc_chunks0 = [Document(page_content=content, metadata=metadataz) for
                      content in page_content_list]
        doc_chunks1 = []
        for docz in doc_chunks0:
            docz2 = copy.deepcopy(docz)
            docz2.metadata['original_text'] = docz2.page_content
            # print(docz2)
            doc_chunks1.append(docz2)
        print(doc_chunks1)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk[0], chunk_overlap=chunk[1])
        doc_chunks2 = text_splitter.split_documents(doc_chunks1)
        return doc_chunks2

    elif chunk[0]> 0:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk[0], chunk_overlap=chunk[1])
        doc_chunks = text_splitter.split_documents(doc)
        return doc_chunks


def into_vector_db(docs, outputpath):
    embedding_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    emb_mod_path = "C:/Users/40467/Desktop/新建文件夹 (2)/langchainz/text2vec-base-chinese"
    local_embeddings = HuggingFaceEmbeddings(model_name=emb_mod_path,
                                             model_kwargs={'device': embedding_device})
    if os.path.exists(outputpath):
        print("The file '{}' exists.".format(outputpath))
        newdocs = FAISS.from_documents(docs, local_embeddings)
        docsearch = FAISS.load_local(folder_path=outputpath, embeddings=local_embeddings)
        docsearch.merge_from(newdocs)
        docsearch.save_local(outputpath)
    else:
        print("The file '{}' does not exist.".format(outputpath))
        docsearch = FAISS.from_documents(docs, local_embeddings)
        docsearch.save_local(outputpath)
    return 1


def search_vector_database(vectordb_path,main_query,  filtermetadata, topk):
    embedding_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    emb_mod_path = "C:/Users/40467/Desktop/新建文件夹 (2)/langchainz/text2vec-base-chinese"
    local_embeddings = HuggingFaceEmbeddings(model_name=emb_mod_path,
                                             model_kwargs={'device': embedding_device})
    docsearch = FAISS.load_local(folder_path=vectordb_path, embeddings=local_embeddings)
    # get particular document
    y = docsearch.similarity_search(query=main_query, filter=filtermetadata, k=topk)
    return y
def get_vector_index(vectordb_path):
    embedding_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    emb_mod_path = "C:/Users/40467/Desktop/新建文件夹 (2)/langchainz/text2vec-base-chinese"
    local_embeddings = HuggingFaceEmbeddings(model_name=emb_mod_path,
                                             model_kwargs={'device': embedding_device})
    docsearch = FAISS.load_local(folder_path=vectordb_path, embeddings=local_embeddings)
    return docsearch

def search_vector_index(index,query,filter, topk):
    result = index.similarity_search(query=query, filter=filter, k=topk)
    return result

def build_chain():
    # 1. embedding model
    embedding_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    local_embeddings = HuggingFaceEmbeddings(model_name="./text2vec-base-chinese",
                                             model_kwargs={'device': embedding_device})
    # 2. load vectordb
    vectorindex = FAISS.load_local(folder_path='vectordb', embeddings=local_embeddings)

    # 3. filter conditions
    filtermetadata = {'source': "docs/protocol.docx", 'category': 'Header'}

    # 4. load llm
    llm = ZhiPuLLm()

    # 5. build chain
    chain_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorindex.as_retriever(
            search_type="similarity",  # Also test "similarity_score_threshold", "mmr"
            search_kwargs={"k": 8},
            filter=filtermetadata
            # query="身高值的单位是什么"
        ),
        return_source_documents=True
    )
    return chain_qa
