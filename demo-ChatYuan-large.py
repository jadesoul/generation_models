from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.nlp import T5ForConditionalGeneration
from modelscope.preprocessors import TextGenerationT5Preprocessor

import sys 
model=sys.argv[1]

if __name__ == "__main__":
    
    model = T5ForConditionalGeneration.from_pretrained('ClueAI/ChatYuan-large', revision='v1.0.0')
    preprocessor = TextGenerationT5Preprocessor(model.model_dir)
    pipeline_t2t = pipeline(task=Tasks.text2text_generation, model=model, preprocessor=preprocessor)

    def preprocess(text):
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        return text

    def postprocess(text):
        return text.replace("\\n", "\n").replace("\\t", "\t")

    def answer(text, sample=True, top_p=1, temperature=0.7):
        '''sample：是否抽样。生成任务，可以设置为True;
        top_p：0-1之间，生成的内容越多样'''
        text = preprocess(text)

        if not sample:
            out_text = pipeline_t2t(text, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
        else:
            out_text = pipeline_t2t(text, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)

        return postprocess(out_text["text"])
    
    input_text0 = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
    input_text1 = "你能干什么"
    input_text2 = "用英文写一封道歉的邮件，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失"
    input_text3 = "写一个文章，题目是未来城市"
    input_text4 = "写一个诗歌，关于冬天"
    input_text5 = "从南京到上海的路线"
    input_text6 = "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字"
    input_text7 = "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。"
    input_text8 = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
    
    input_list = [input_text0, input_text1, input_text2, input_text3, input_text4, input_text5, input_text6, input_text7, input_text8]

    for i, input_text in enumerate(input_list):
        input_text = "用户：" + input_text + "\n小元："
        print(f"示例{i}".center(50, "="))
        output_text = answer(input_text)
        print(f"{input_text}{output_text}")

    # for text in inputs:
    #     result = predictor(text)

    #     print('输入:' + text + '\n')
    #     print('输出:' + result[OutputKeys.TEXT])

    # while True:
    #     text = input('\nprompt:')
    #     if text=='/exit': break
    #     result = predictor(text)
    #     print(result[OutputKeys.TEXT])
