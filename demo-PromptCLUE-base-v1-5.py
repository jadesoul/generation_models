from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.nlp import T5ForConditionalGeneration
from modelscope.preprocessors import TextGenerationTransformersPreprocessor


import sys 
model=sys.argv[1]

if __name__ == "__main__":

    model = T5ForConditionalGeneration.from_pretrained('ClueAI/PromptCLUE-base-v1-5', revision='v0.1')
    preprocessor = TextGenerationTransformersPreprocessor(model.model_dir)
    pipeline_t2t = pipeline(task=Tasks.text2text_generation, model=model, preprocessor=preprocessor)

    print(pipeline_t2t('生成与下列文字相同意思的句子：\n白云遍地无人扫\n答案：', do_sample=True, top_p=0.8))
    # {'text': '白云散去无踪，没人扫。'}
    
    print(pipeline_t2t('改写下面的文字，确保意思相同：\n一个如此藐视本国人民民主权利的人，怎么可能捍卫外国人的民权？\n答案：', do_sample=True, top_p=0.8))
    # {'text': '对一个如此藐视本国人民民主权利的人，怎么能捍卫外国人的民权？'}

    print(pipeline_t2t('根据问题给出答案：\n问题：手指发麻的主要可能病因是：\n答案'))
    # {'text': '神经损伤，颈椎病，贫血，高血压'}

    print(pipeline_t2t('问答：\n问题：黄果悬钩子的目是：\n答案：'))
    # {'text': '蔷薇目'}

    print(pipeline_t2t('情感分析：\n这个看上去还可以，但其实我不喜欢\n选项：积极，消极'))
    # {'text': '消极'}

    print(pipeline_t2t("下面句子是否表示了相同的语义：\n文本1：糖尿病腿麻木怎么办？\n文本2：糖尿病怎样控制生活方式\n选项：相似，不相似\n答案："))
    # {'text': '不相似'}

    print(pipeline_t2t('这是关于哪方面的新闻：\n如果日本沉没，中国会接收日本难民吗？\n选项：故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏'))
    # {'text': '国际'}
    
    print(pipeline_t2t("阅读文本抽取关键信息：\n张玄武1990年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。\n问题：机构，人名，职位，籍贯，专业，国籍，学历，种族\n答案："))
    # {'text': '机构：杭州线锁科技技术_人名：张玄武_职位：博士学历'}

    print(pipeline_t2t("翻译成英文：\n杀不死我的只会让我更强大\n答案："))
    # {'text': 'To kill my life only let me stronger'}

    print(pipeline_t2t('为下面的文章生成摘要：\n北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡'))
    # {'text': '四川甘孜发生6.8级地震'}
    
    print(pipeline_t2t("推理关系判断：\n前提：小明今天在北京\n假设：小明在深圳旅游\n选项：矛盾，蕴含，中立\n答案："))
    # {'text': '蕴涵'}

    print(pipeline_t2t('阅读以下对话并回答问题。\n男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。\n问题：女的怎么样？\n选项：正在工作，感冒了，在打电话，要出差。'))
    # {'text': '感冒了'}

    print(pipeline_t2t("文本纠错：\n告诉二营长，叫他彻回来，我李云龙从不打没有准备的杖\n答案："))
    #{'text'：'告诉二营长，叫他下来，我李云龙从不打没有准备的仗'}

    print(pipeline_t2t("问答：\n问题：小米的创始人是谁？\n答案："))
    # {'text': '小米创始人：雷军'}


    # for text in inputs:
    #     result = predictor(text)

    #     print('输入:' + text + '\n')
    #     print('输出:' + result[OutputKeys.TEXT])

    # while True:
    #     text = input('\nprompt:')
    #     if text=='/exit': break
    #     result = predictor(text)
    #     print(result[OutputKeys.TEXT])
