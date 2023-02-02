import sys 
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id=sys.argv[1]
workspace=sys.argv[2]

img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                       model='damo/cv_unet_person-image-cartoon-3d_compound-models')
# 图像本地路径
img_path = workspace+'/input.png'
# 图像url链接
# img_path = 'https://invi-label.oss-cn-shanghai.aliyuncs.com/label/cartoon/image_cartoon.png'
result = img_cartoon(img_path)

out_path = workspace+'/result.png'
cv2.imwrite(out_path, result[OutputKeys.OUTPUT_IMG])
print('finished!')

