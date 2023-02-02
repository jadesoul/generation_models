model=$1
workspace=$2

pip install -r requirements.txt
# pip install megatron_util -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

# bash

main=demo-${model}.py 
if [ ! -f $main ]; then
    main=demo.py
fi

python $main $model $workspace

# python demo.py $model
# python demo-prompt.py $model
# python demo-prompt-open.py $model

# python demo-prompt-clue.py $model
# python demo-chat-yuan.py $model
# python demo-zero-shot-cls.py $model

# python demo-batch.py $model | tee $workspace/${model}.out
# python server.py $model

# bash
