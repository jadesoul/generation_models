from modelscope.outputs import OutputKeys
from common import get_predictor
import sys 
model=sys.argv[1]
predictor, inputs = get_predictor(model)

if __name__ == "__main__":
    while True:
        text = input('\nprompt:')
        if not text: continue
        if text=='/exit': break

        text = f'以关键词“{text}”写一副春联'
        result = predictor(text)
        print(result[OutputKeys.TEXT])