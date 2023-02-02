import yaml

models=None
with open('conf.yml') as fin:
    models = yaml.full_load(fin)

for name, model in models.items():
    assert 'name' in model

    if 'model' not in models[name]:
        model_id=f'damo/{name}'
        models[name]['model']=model_id
    else:
        model_id=models[name]['model']

    if 'task' not in models[name]:
        models[name]['task']='text-generation'
    else:
        models[name]['task']=models[name]['task'].replace('_', '-')

    if 'url' not in models[name]:
        models[name]['url']=f'https://modelscope.cn/models/{model_id}/summary'


# name='春联生成模型-中文-base'
# name='GPT-3预训练生成模型-中文-large'
# task=models[name]['task']
# model=models[name]['model']
# inputs=models[name]['inputs']

if __name__=='__main__':
    # from modelscope.utils.constant import Tasks

    # models = {
    #     '春联生成模型-中文-base': {
    #         'task': Tasks.text_generation
    #         , 'model': 'damo/spring_couplet_generation'
    #         , 'url': ''
    #     }
    #     , '春联生成模型-中文-base': {
    #         'task': Tasks.text_generation
    #         , 'model': 'damo/spring_couplet_generation'
    #         , 'url': 'https://modelscope.cn/models/damo/spring_couplet_generation/summary'
    #     }   
    # }

    # print(yaml.dump(models))

    import json
    print(json.dumps(models, indent=1, ensure_ascii=0))