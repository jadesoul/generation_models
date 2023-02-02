from modelscope.outputs import OutputKeys
from common import get_predictor
import sys 
model=sys.argv[1]
predictor, inputs = get_predictor(model)

if __name__ == "__main__":
    for i in range(100):
        result = predictor(inputs[0])
        print(result[OutputKeys.TEXT], flush=True)
