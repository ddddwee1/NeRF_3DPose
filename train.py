import yaml
from tqdm import tqdm
from easydict import EasyDict
from lib import dataloader, trainer

if __name__=='__main__':
    cfg = EasyDict(yaml.load(open('config.yaml'), Loader=yaml.FullLoader))

    loader, pose_metadata = dataloader.get_dataloader(cfg)

    nerf_trainer = trainer.Trainer(cfg, pose_metadata)
    nerf_trainer.initialize()

    bar = tqdm(loader)
    for i,batch in enumerate(bar):
        loss_coarse, loss_fine, loss_weight_coarse, loss_weight_fine, loss_vgg_coarse, loss_vgg_fine = nerf_trainer.train_batch(batch, global_step=i)
        out_str = 'ls_c: %.4e  ls_f: %.4e  ls_wc: %.4e  ls_wf: %.4e  ls_vgc:%.4e  lsvgf:%.4e' % \
                        (loss_coarse.detach().cpu().numpy(), loss_fine.detach().cpu().numpy(), \
                        loss_weight_coarse.detach().cpu().numpy(), loss_weight_fine.detach().cpu().numpy(),\
                        loss_vgg_coarse.detach().cpu().numpy(), loss_vgg_fine.detach().cpu().numpy())
        bar.set_description(out_str)

        if (i+1)%cfg.TRAIN.save_interval==0:
            nerf_trainer.save(i+1)
