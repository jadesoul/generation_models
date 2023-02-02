import os, json
from transformers import BertTokenizer
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.nlp.space_T_cn.fields.database import Database
from modelscope.utils.constant import ModelFile, Tasks

model_id = 'damo/nlp_convai_text2sql_pretrain_cn'
test_case = {
    'utterance':
    [['长江流域的小型水库的库容总量是多少？', 'reservoir'], ['那平均值是多少？', 'reservoir'], ['那水库的名称呢？', 'reservoir'], ['换成中型的呢？', 'reservoir']]
}

model = Model.from_pretrained(model_id)
tokenizer = BertTokenizer(
    os.path.join(model.model_dir, ModelFile.VOCAB_FILE))
db = Database(
    tokenizer=tokenizer,
    table_file_path=os.path.join(model.model_dir, 'table.json'),
    syn_dict_file_path=os.path.join(model.model_dir, 'synonym.txt'),
    is_use_sqlite=True)
preprocessor = TableQuestionAnsweringPreprocessor(
    model_dir=model.model_dir, db=db)
pipelines = [
    pipeline(
        Tasks.table_question_answering,
        model=model,
        preprocessor=preprocessor,
        db=db)
]

for pipeline in pipelines:
    historical_queries = None
    for question, table_id in test_case['utterance']:
        output_dict = pipeline({
            'question': question,
            'table_id': table_id,
            'history_sql': historical_queries
        })[OutputKeys.OUTPUT]
        print('question', question)
        print('sql text:', output_dict[OutputKeys.SQL_STRING])
        print('sql query:', output_dict[OutputKeys.SQL_QUERY])
        print()
        historical_queries = output_dict[OutputKeys.HISTORY]