import torch
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.text_to_image_synthesis
model_id = 'damo/multi-modal_chinese_stable_diffusion_v1.0'
# 基础调用
pipe = pipeline(task=task, model=model_id)
output = pipe({'text': '中国山水画'})
cv2.imwrite('result.png', output['output_imgs'][0])
# 输出为opencv numpy格式，转为PIL.Image
# from PIL import Image
# img = output['output_imgs'][0]
# img = Image.fromarray(img[:,:,::-1])
# img.save('result.png')

# 更多参数
pipe = pipeline(task=task, model=model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
output = pipe({'text': '中国山水画', 'num_inference_steps': 50, 'guidance_scale': 7.5, 'negative_prompt':'模糊的'})
cv2.imwrite('result.png', output['output_imgs'][0])

# 采用DPMSolver
from diffusers.schedulers import DPMSolverMultistepScheduler
pipe = pipeline(task=task, model=model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
pipe.pipeline.scheduler.config)
output = pipe({'text': '中国山水画', 'num_inference_steps': 25})
cv2.imwrite('result.png', output['output_imgs'][0])


