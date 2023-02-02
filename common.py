from modelscope.pipelines import pipeline
from conf import models

def get_predictor(key):
    assert key in models
    print('\n-- getting predictor ...')
    name = models[key]['name']
    print('Name:', name)
    task = models[key]['task']
    print('Task:', task)
    model = models[key]['model']
    print('Model:', model)
    inputs = models[key]['inputs']
    print('Inputs:', inputs)
    print('--\n')
    predictor = pipeline(task, model=model)

    print('-- got predictor:', predictor)

    return predictor, inputs

