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
        loss_coarse, loss_fine, loss_weight_coarse, loss_weight_fine = nerf_trainer.train_batch(batch, global_step=i)
        out_str = 'loss_c: %.4e  loss_f: %.4e  loss_wc: %.4e  loss_wf: %.4e' % \
                        (loss_coarse.detach().cpu().numpy(), loss_fine.detach().cpu().numpy(), loss_weight_coarse.detach().cpu().numpy(), loss_weight_fine.detach().cpu().numpy())
        bar.set_description(out_str)

        if (i+1)%cfg.TRAIN.save_interval==0:
            nerf_trainer.save(i+1)
