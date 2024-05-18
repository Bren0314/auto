# -*- coding: utf-8 -*-
import pandas as pd
import functionz as fz
import custom_llm as md


#question list
df = pd.read_excel("docs/protocolquestions.xlsx",sheet_name="main")
# print(df)
df_out = []

#initialize index
# chain_qa = fz.build_chain()
vectordb_path = 'doc_objects/vectordb'
index = fz.get_vector_index(vectordb_path)
#filter conditions
filterTable = {'source': "docs/protocol.docx", 'category': 'Table'}
filterText = {'source': "docs/protocol.docx", 'category': 'Header'}

#loop questions
for i, v in df.iterrows():
    # if i != 4:
    #     continue
    if pd.notnull(v['group_query_search']):
        query_search = v["group_query_search"]
        print(query_search)
        query_question = v["group_query_question"]
        format = v['group_format']
        print(query_question)
        topk = 4
        relevantTable = fz.search_vector_index(index, query_search, filterTable, topk)
        relevantTable = [i.metadata['text_as_html'] for i in relevantTable]
        relevantText = fz.search_vector_index(index, query_search, filterText, topk)
        relevantText = [i.metadata['original_text'] for i in relevantText]
        relevant = relevantTable + relevantText

        # print(relevant)
        #deduplicate similar search returns
        context = list(set(relevant))
        print(context)
        prompt = '''context: \n {} \n  question: {} \n result_format: {} \n answer: '''.format('\n'.join(context), query_question,format)
        print(prompt)
        question_result = md.zhipu_turbo(prompt)
        print(question_result)

    # 将结果保存到DataFrame中
        df.at[i, 'group_output'] = question_result

    # 将修改后的DataFrame保存回Excel文件
    df.to_excel("docs/protocolquestions.xlsx", sheet_name="main", index=False)




























#
#
#     chain_qa.combine_documents_chain.llm_chain.llm.filter = filtermetadata
#     if query_question == query_search:
#         chain_qa.combine_documents_chain.llm_chain.llm.query_answer = query_search
#     else:
#         chain_qa.combine_documents_chain.llm_chain.llm.query_answer = query_question
#     # chain_qa.run(query, return_only_outputs=True)
#     response = chain_qa({"query": query_search}, return_only_outputs=True)
#     # print(response)
#     result = response['result']
#     source = response['source_documents']
#     # file_name = response['metadata']
#     json_result = {"file": filtermetadata['source'],
#                    "search": query_search,
#                    "question":query_question,
#                    "result":result,
#                    "source":source }
#     df_out.append(json_result)
#
# df_out_excel= pd.DataFrame(df_out)
# df_out_excel.to_excel('output.xlsx', index=False, engine='openpyxl')
