import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader_UIEB import PairLoader
from models.MixDehazeNet_FiLM_all6_FPN_TA_CSAM_add_mul import *
from utils.CR import ContrastLoss
from utils.CR_res import ContrastLoss_res
import torch_optimizer as optim
import os
gpu_ids = [0, 1]
gpu_str = ','.join(str(x) for x in gpu_ids)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-s', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/real_FiLM6_FPN_TA_CSAM_add_mul_UIEB/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/public/home/shaojx8/palette-mask2/datasets/UIEB_dataset/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		TM_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			output = network(source_img,TM_img)
			loss = criterion[0](output, target_img)#+criterion[1](output, target_img, source_img)*0.1
			# ablation-base
# 			loss = criterion[0](output, target_img)


		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()
		TM_img = batch['source'].cuda()

		with torch.no_grad():							# torch.no_grad() may cause warning
			output = network(source_img,TM_img).clamp_(-1, 1)		

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

    #   Start training model from checkpoint
	checkpoint=torch.load('/public/home/shaojx8/MixDehaceNet_new_6_new/saved_models/real_FiLM6_FPN_TA_CSAM_add_mul_UIEB/indoor/MixDehazeNet-sreal_FiLM6_FPN_TA_CSAM_add_mul_UIEB.pth')

	#   Start training model from NULL
	#checkpoint=None
	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()
	if checkpoint is not  None:
		network.load_state_dict(checkpoint['state_dict'])

	criterion = []
	criterion.append(nn.L1Loss())
	criterion.append(ContrastLoss_res(ablation=False).cuda())

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
		#optimizer = optim.Lookahead(optimizer, alpha=1, k=6)
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
		#optimizer = optim.Lookahead(optimizer, alpha=1, k=6)
	else:
		raise Exception("ERROR: unsupported optimizer")

	# CosineAnnealingWarmRestarts
	# CosineAnnealingLR
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,  eta_min=setting['lr'] * 1e-1)
	scaler = GradScaler()

	if checkpoint is not None:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['lr_scheduler'])
		scaler.load_state_dict(checkpoint['scaler'])
		best_psnr = checkpoint['best_psnr']
		start_epoch = checkpoint['epoch'] + 1
	else:
		best_psnr = 0
		start_epoch = 0

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	print(dataset_dir)
	train_dataset = PairLoader(dataset_dir, 'train_val', 'train', 
								setting['patch_size'],
							    setting['edge_decay'],
							    setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'TEST', setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		print('==> Start training, current model name: ' + args.model)
		print(network)

	writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

	train_ls, test_ls, idx = [], [], []
	#compiled_model = torch.compile(network,mode="max-autotune")

	for epoch in tqdm(range(start_epoch,setting['epochs'] + 1)):
		loss = train(train_loader, network, criterion, optimizer, scaler)

		train_ls.append(loss)
		idx.append(epoch)

		# writer.add_scalar('train_loss', loss, epoch)

		scheduler.step()


		if epoch % setting['eval_freq'] == 0:
			avg_psnr = valid(val_loader, network)

			# writer.add_scalar('valid_psnr', avg_psnr, epoch)

			if avg_psnr > best_psnr:
				best_psnr = avg_psnr
				print(avg_psnr)

				torch.save({'state_dict': network.state_dict(),
							'optimizer':optimizer.state_dict(),
							'lr_scheduler':scheduler.state_dict(),
							'scaler':scaler.state_dict(),
							'epoch':epoch,
							'best_psnr':best_psnr
							},
						   os.path.join(save_dir, args.model+'real_FiLM6_FPN_TA_CSAM_add_mul_UIEB.pth'))

			writer.add_scalar('best_psnr', best_psnr, epoch)

