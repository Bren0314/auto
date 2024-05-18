
#pip install "unstructured[all-docs]"  -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install --upgrade langchain
#source:pip install langchain  -i https://pypi.tuna.tsinghua.edu.cn/simple
#punkt: nltk.download('punkt')
#if not able to download you will have to download from https://www.nltk.org/nltk_data/
#and then put it in your pycharm project under C:\Users\junka\PycharmProjects\untitled1\venv\nltk_data\tokenizers\
#add punkt and other packages, for other package you have to look at the location it mentions such as \nltk_data\taggers

import nltk
import functionz as fz
import pickle



doc_path = "docs/2022 AHA-ACC-HFSA Guideline for the Management of Heart Failure.docx"
type = 'docx'
elements = 1
doc_data = fz.doc_loader(doc_path, type, elements)
text_data, table_data = fz.get_tables(doc_data)

prompt_result, headers_list, original_data_index = fz.get_headers(text_data)
# # title_doc_data = [v for i, v in enumerate(doc_data) if i in original_data_index]
# # print(title_doc_data)

#need to cut the document into usable pieces, then use langchain to search for the most relevant section
#may need to run once for paragraphs and once for tables
doc_text_split = fz.doc_split(text_data, original_data_index, (50,20))
doc_table_split = fz.doc_split(table_data, [], (50, 20))
# print([v.page_content for v in doc_table_split])
# print(len(doc_table_split))
#take out 'text_as_html' as source_data for table_data,
# and 'original_text' as source data for text_data
# print("text")
# print(doc_text_split)
# print("table")
# print(doc_table_split[0])


# Saving data to a .pkl file
with open('doc_objects/doc_texts_chunks.pkl', 'wb') as file:
    pickle.dump(doc_text_split, file)

with open('doc_objects/doc_table_chunks.pkl', 'wb') as file:
    pickle.dump(doc_table_split, file)



