import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
#from models.MixDehazeNet_FiLM6 import *
from models.MixDehazeNet_FiLM_all6_FPN_TA_CSAM_add_mul import *
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
#from models import *
import os
gpu_ids = [0, 1]
gpu_str = ','.join(str(x) for x in gpu_ids)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-s', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--data_dir', default='/public/home/shaojx8/Restormer-main/Defocus_Deblurring/Datasets', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/real_FiLM6_FPN_TA_CSAM_add_mul/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/real_FiLM6_FPN_TA_CSAM_add_mul/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


import time
import torch.cuda

def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    TIME = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()
        TM = batch['source'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            torch.cuda.synchronize()  # 确保GPU操作完成
            start_time = time.time()
            
            output = network(input, TM).clamp_(-1, 1)
            
            torch.cuda.synchronize()  # 确保GPU操作完成
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            TIME.update(elapsed_time)

            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\tTime: {time.val:.4f} ({time.avg:.4f})\tPSNR: {psnr.val:.02f} ({psnr.avg:.02f})\tSSIM: {ssim.val:.03f} ({ssim.avg:.03f})'.format(
            idx, time=TIME, psnr=PSNR, ssim=SSIM))
        f_result.write('%s,%.02f,%.03f,%.4f\n' % (filename, psnr_val, ssim_val, elapsed_time))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()
    os.rename(os.path.join(result_dir, 'results.csv'), 
              os.path.join(result_dir, '%.02f | %.04f | %.4f.csv' % (PSNR.avg, SSIM.avg, TIME.avg)))

    print('Average running time: {:.4f} seconds'.format(TIME.avg))
    print('Total running time for {} images: {:.4f} seconds'.format(len(test_loader), TIME.sum))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, 'MixDehazeNet-sreal_FiLM6_FPN_TA_CSAM_add_mul.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	#macs, params = get_model_complexity_info(network, [(3, 256, 256),(3, 256, 256)], as_strings=True,
	# 										 print_per_layer_stat=True, verbose=True)
	#print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	#print('{:<30}  {:<8}'.format('Number of parameters: ', params))

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir)



