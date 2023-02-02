from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import ConversationalTextToSqlPreprocessor
from modelscope.utils.constant import Tasks

model_id = 'damo/nlp_star_conversational-text-to-sql'
test_case = {
    "database_id": 'employee_hire_evaluation',
    'local_db_path':None,
    "utterance":[
    "I'd like to see Shop names.",
    "Which of these are hiring?",
    "Which shop is hiring the highest number of employees? | do you want the name of the shop ? | Yes"]
}

model = Model.from_pretrained(model_id)
preprocessor = ConversationalTextToSqlPreprocessor(model_dir=model.model_dir)
pipeline = pipeline(
                    task=Tasks.table_question_answering,
                    model=model,
                    preprocessor=preprocessor)
last_sql, history = '', []
for item in test_case['utterance']:
    case = {"utterance": item,
            "history": history,
            "last_sql": last_sql,
            "database_id": test_case['database_id'],
            'local_db_path': test_case['local_db_path']}
    results = pipeline(case)
    print(results)
    history.append(item)