from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# query_set: query
# support_set: faq候选列表，一般实际应用场景中通过检索得到
pipeline = pipeline(Tasks.faq_question_answering, 'damo/nlp_structbert_faq-question-answering_chinese-base')
outputs = pipeline({"query_set": ['如何使用优惠券', '在哪里领券', '购物评级怎么看'],
                   "support_set": [{'text': '卖品代金券怎么用', 'label': '2931733'}, 
                                   {'text': '怎么使用优惠券', 'label': '2931733'}, 
                                   {'text': '这个可以一起领吗', 'label': '3626004'}, 
                                   {'text': '付款时送的优惠券哪里领', 'label': '3626004'}, 
                                   {'text': '购物等级怎么长', 'label': '6344909'}, 
                                   {'text': '购物等级二心', 'label': '6344909'}]})
# 如果输入数据中每个label只有一个句子，则可以做句子相似度计算任务

# outputs
# 输出每一个类的分值，并进行排序
# {'output': [[{'label': '6527856', 'score': 0.9982811212539673}, {'label': '1000012000', 'score': 0.0280130784958601}, {'label': '13421097', 'score': 8.978261757874861e-05}], 
#            [{'label': '1000012000', 'score': 0.8750997185707092}, {'label': '6527856', 'score': 0.0031510782428085804}, {'label': '13421097', 'score': 0.0007711253711022437}], 
#            [{'label': '13421097', 'score': 0.6274582743644714}, {'label': '1000012000', 'score': 0.0035026895347982645}, {'label': '6527856', 'score': 0.001228355336934328}]]}