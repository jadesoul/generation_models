set -x
cd -P $(dirname $0)

MODEL=$1

NAME=$MODEL
# IMAGE=registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.1.2 #cpu
IMAGE=registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.2.0
#IMAGE=registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.1.2 #gpu
CACHE=modelscope

docker pull $IMAGE
( docker ps | grep ${NAME}$ ) && ( docker stop $NAME ; docker rm $NAME ) || echo container not exists

WORK=/mnt/workspace
MOUNT=$HOME/.cache/docker/${CACHE}${WORK}
mkdir -p $MOUNT

docker run \
	--rm \
	--name=$NAME \
	--cpuset-cpus=0-8 \
	-it \
	-v $MOUNT:$WORK \
	-v $PWD:/root/$(basename $PWD) \
	-w /root/$(basename $PWD) \
	$IMAGE \
	/bin/bash ./entry.sh $MODEL $WORK

	# -p 8003:8003 \


cd $MOUNT/.cache/modelscope
echo Models:
du -sh */*

echo Total:
du -sh .

