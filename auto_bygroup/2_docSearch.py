import functionz as fz
vectordb_path = "doc_objects/vectordb"
query = "主要研究者：
名称 卢洪洲教授
联系地址 上海市金山区漕廊公路2901号
联系电话 021-37990333

临床研究牵头单位：
名称 上海市公共卫生临床中心
单位地址 上海市金山区漕廊公路2901号
联系电话 021-37990333
邮编 201508

项目数据管理和统计分析单位：
名称 北京大学第一医院医学统计室
单位地址 北京市西城区西什库大街8号
联系电话 010-66115216
邮编 100034

申办者：
名称 江西青峰药业有限公司
单位地址 北京市朝阳区新源里16号琨莎中心3座9层
联系电话 010-84682600
邮编 100027

版本号/日期：
2.0版/2020年02月20日

保密：
本方案为江西青峰药业有限公司版权所有。未经允准，不得擅自使用、泄露、公布及出版。上面的话语中，机构的名称都有哪些"
# query = '研究编号'
# filtermetadata = {'source': "docs/protocol.docx", 'category': 'Table'} # for text use 'Header'
filtermetadata = {'source': "docs/protocol.docx", 'category': 'Header'}
topk = 3
result = fz.search_vector_database(vectordb_path,query,  filtermetadata, topk)
print(result)
# print(result[0].metadata['text_as_html'])