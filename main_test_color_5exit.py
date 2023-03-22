import os.path
import logging
import numpy as np
from datetime import datetime
from collections import OrderedDict
import torch
import cv2
from utils import utils_logger
from utils import utils_image as util
import requests
# ----------------------------------------
# load model
# ----------------------------------------
from CEESDB_arch import CEESDBNet as net

def main():

    quality_factor_list = [10, 50, 80]
    testset_name = 'urban100'
    n_channels = 3            # set 1 for grayscale image, set 3 for color image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'CEESDB_rbqe'
    model_path = './pretrain_model/CEESDBNet_rbqe.pth'
    model = net(in_nc=3, out_nc=3, nf=64, cond_dim=1, ca_type='CE', order=6)
    # show_img = False                 # default: False

    testsets = '/data/dataset'
    results = '../results'

    result_name = testset_name + '_' + model_name
    util.mkdir(os.path.join(results, result_name))
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(results, result_name, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # model.load_state_dict(torch.load(model_path)['network']['net'], strict=True)
    load_net = torch.load(model_path)  # ['network']['net']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=True)
    
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    
    logger.info('Model params: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))))    

    for quality_factor in quality_factor_list:

        H_path = os.path.join(testsets, testset_name)
        E_path = os.path.join(results, result_name, str(quality_factor))   # E_path, for Estimated images
        util.mkdir(E_path)

        logger.info('--------------- quality factor: {:d} ---------------'.format(quality_factor))
        border = 0

        test_results = OrderedDict()
        test_results['psnrlq'] = []
        test_results['ssimlq'] = []
        test_results['psnrblq'] = []
        test_results['psnr1'] = []
        test_results['ssim1'] = []
        test_results['psnrb1'] = []

        test_results['psnr2'] = []
        test_results['ssim2'] = []
        test_results['psnrb2'] = []

        test_results['psnr3'] = []
        test_results['ssim3'] = []
        test_results['psnrb3'] = []

        test_results['psnr4'] = []
        test_results['ssim4'] = []
        test_results['psnrb4'] = []

        test_results['psnr5'] = []
        test_results['ssim5'] = []
        test_results['psnrb5'] = []

        H_paths = util.get_image_paths(H_path)
        for idx, img in enumerate(H_paths):

            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))

            img_L = util.imread_uint(img, n_channels=n_channels)
            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
            _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 1)
            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

            img_L = util.uint2tensor4(img_L)
            img_L = img_L.to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------
            img_E1, img_E2, img_E3, img_E4, img_E5, cond = model(img_L)
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels).squeeze()
            QF = (1 - cond)
            logger.info('predicted quality factor: {:d}'.format(round(float(QF*100))))
            img_L = util.tensor2single(img_L)
            img_L = util.single2uint(img_L)
            psnr = util.calculate_psnr(img_L, img_H, border=border)
            ssim = util.calculate_ssim(img_L, img_H, border=border)
            psnrb = util.calculate_psnrb(img_H, img_L, border=border)
            test_results['psnrlq'].append(psnr)
            test_results['ssimlq'].append(ssim)
            test_results['psnrblq'].append(psnrb)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name + ext, psnr, ssim, psnrb))
            for i in range(1, 6):
                img_E_name = 'img_E' + str(i)
                img_E = locals()[img_E_name]
                img_E = util.tensor2single(img_E)
                img_E = util.single2uint(img_E)
                # --------------------------------
                # PSNR and SSIM, PSNRB
                # --------------------------------
                psnr = util.calculate_psnr(img_E, img_H, border=border)
                ssim = util.calculate_ssim(img_E, img_H, border=border)
                psnrb = util.calculate_psnrb(img_H, img_E, border=border)
                test_results['psnr'+str(i)].append(psnr)
                test_results['ssim'+str(i)].append(ssim)
                test_results['psnrb'+str(i)].append(psnrb)
                logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name+ext, psnr, ssim, psnrb))
                util.imsave(img_E, os.path.join(E_path, img_name+'ex' + str(i) + '.png'))
        ave_psnr = sum(test_results['psnrlq']) / len(test_results['psnrlq'])
        ave_ssim = sum(test_results['ssimlq']) / len(test_results['ssimlq'])
        ave_psnrb = sum(test_results['psnrblq']) / len(test_results['psnrblq'])
        logger.info('Lq images' + 'Average PSNR/SSIM/PSNRB-{} -: {:.2f}$\\vert${:.4f}$\\vert${:.2f}.'
                    .format(result_name+'_'+str(quality_factor), ave_psnr, ave_ssim, ave_psnrb))
        for i in range(1, 6):
            ave_psnr = sum(test_results['psnr' + str(i)]) / len(test_results['psnr' + str(i)])
            ave_ssim = sum(test_results['ssim' + str(i)]) / len(test_results['ssim' + str(i)])
            ave_psnrb = sum(test_results['psnrb' + str(i)]) / len(test_results['psnrb' + str(i)])
            logger.info('Exit for stage' + str(i) + 'Average PSNR/SSIM/PSNRB-{} -: {:.2f}$\\vert${:.4f}$\\vert${:.2f}.'
                        .format(result_name+'_'+str(quality_factor), ave_psnr, ave_ssim, ave_psnrb))


if __name__ == '__main__':
    main()
