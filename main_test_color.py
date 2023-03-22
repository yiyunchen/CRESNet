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
# from CEESDB_arch import CEESDBNet2 as net

def main():

    quality_factor_list = [50, 70]
    testset_name = 'urban100'
    n_channels = 3  # set 1 for grayscale image, set 3 for color image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'CEESDB_rbqe'
    model_path = './pretrain_model/CEESDBNet_rbqe.pth'
    model = net(in_nc=3, out_nc=3, nf=64, cond_dim=1, ca_type='CE', order=6)

    # model_name = 'CEESDB_fbcnn'
    # model_path = './pretrain_model/CEESDB_fbcnn.pth'
    # model = net(order=5)

    show_img = False                 # default: False
    testsets = '/data/dataset'
    results = '../results'

    result_name = testset_name + '_' + model_name
    util.mkdir(os.path.join(results, result_name))
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(results, result_name, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    model.load_state_dict(torch.load(model_path), strict=True)
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
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnrb'] = []
        test_results['psnrlq'] = []
        test_results['ssimlq'] = []
        test_results['psnrblq'] = []
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

            img_E, QF = model(img_L, mode='val')
            QF = 1 - QF
            img_E = util.tensor2single(img_E)
            img_E = util.single2uint(img_E)
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels).squeeze()
            # --------------------------------
            # PSNR and SSIM, PSNRB
            # --------------------------------
            img_L = util.tensor2single(img_L)
            img_L = util.single2uint(img_L)
            psnr = util.calculate_psnr(img_L, img_H, border=border)
            ssim = util.calculate_ssim(img_L, img_H, border=border)
            psnrb = util.calculate_psnrb(img_H, img_L, border=border)
            test_results['psnrlq'].append(psnr)
            test_results['ssimlq'].append(ssim)
            test_results['psnrblq'].append(psnrb)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name + ext, psnr, ssim, psnrb))
            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            psnrb = util.calculate_psnrb(img_H, img_E, border=border)

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnrb'].append(psnrb)

            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name+ext, psnr, ssim, psnrb))
            logger.info('predicted quality factor: {:d}'.format(round(float(QF*100))))

            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None
            util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])

        logger.info(
             'Average PSNR/SSIM/PSNRB - {} -: {:.2f}$\\vert${:.4f}$\\vert${:.2f}.'.format(result_name+'_'+str(quality_factor), ave_psnr, ave_ssim, ave_psnrb))


if __name__ == '__main__':
    main()
