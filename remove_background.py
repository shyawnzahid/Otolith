import os
import torch
import numpy as np

from PIL import Image
from absl import app, flags
from dotenv import load_dotenv, find_dotenv
from sam2.sam2_image_predictor import SAM2ImagePredictor

flags.DEFINE_string('device', 'cpu', 'Whether or not to use GPU (cuda)')
flags.DEFINE_string('output_directory', 'background_masks', 'Output directory for the SAM2 background mask predictions')
flags.DEFINE_string('img_path', 'data/OA/1.png', 'Which image to predict a mask for')
flags.DEFINE_list('prompt_locations', ['240-480-1'], 'Pixel location(s) in h-w-v format of the prompt to SAM2, where (h,w) is the pixel and v is the binary valued label (0 - background or 1 - foreground) assigned to the prompt. For multiple prompts, use comma separated h-w-v values.')

FLAGS = flags.FLAGS
load_dotenv(find_dotenv())

def get_bg_mask(predictor, img, points, labels):
	with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
		predictor.set_image(img)
		masks, scores, _ = predictor.predict(
		    point_coords=points,
		    point_labels=labels,
		    multimask_output=True,
		)

	sorted_ind = np.argsort(scores)[::-1]
	masks = masks[sorted_ind]

	mask = torch.from_numpy(masks[0]).unsqueeze(0)
	mask = torch.cat([mask]*3, dim=0)

	return mask

def main(argv):
	base_directory = os.environ['BASE_DIRECTORY']
	output_directory = f'{base_directory}/{FLAGS.output_directory}'
	try:
		os.makedirs(output_directory)
	except:
		pass

	img = Image.open(f'{base_directory}/{FLAGS.img_path}')
	img = np.array(img.convert('RGB'))
	height, width, _ = img.shape
	
	prompts = []
	labels = []

	for prompt in FLAGS.prompt_locations:
		try:
			location = prompt.split('-')
			assert len(location) == 3

			h = int(location[0])
			assert 0 <= h and h < height

			w = int(location[1])
			assert 0 <= w and w < width

			v = int(location[2])
			assert v == 0 or v == 1

			prompts.append((h, w))
			labels.append(v)
		except Exception as e:
			print('Make sure prompt_locations flag is a comma separated list of h-w-v values, where (h,w) is the pixel location of the prompt and v is the binary valued label (0 or 1) assigned to the prompt: ' + repr(e))
	
	sam2 = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-large', device=FLAGS.device)
	
	bg_mask = get_bg_mask(
		predictor=sam2,
		img=img,
		points=np.array(prompts),
		labels=np.array(labels)
	)

	bg_mask = bg_mask.permute(1, 2, 0).numpy()
	bg_mask = (bg_mask * 255).astype(np.uint8)

	output_path = f'{output_directory}/{FLAGS.img_path.split('/')[-1]}'
	Image.fromarray(bg_mask).save(output_path)

if __name__ == '__main__':
	app.run(main)