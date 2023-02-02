from modelscope.outputs import OutputKeys
from common import get_predictor
import sys 
model=sys.argv[1]
predictor, inputs = get_predictor(model)

if __name__ == "__main__":
    for text in inputs:
        print(text)
        result = predictor(text)

        # print('输出:' + result[OutputKeys.TEXT])
        print(result)

    while True:
        text = input('\nprompt:')
        if text=='/exit': break
        result = predictor(text)
        # print('输出:' + result[OutputKeys.TEXT])
        print(result)
