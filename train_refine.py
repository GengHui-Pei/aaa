import argparse
from common.config import cfg
import torch,cv2,sys
from common.base import Refine_Trainer
import torch.backends.cudnn as cudnn
from net.misc import NestedTensor
from common.utils.preprocessing import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=str(0), type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--annot_subset', type=str, dest='annot_subset')
    parser.add_argument('--dataset', type=str, dest='dataset')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    trainer = Refine_Trainer()
    trainer._make_batch_generator()
    trainer._make_test_batch_generator()
    if args.dataset == 'interhand':
        trainer._make_model(pre_train=cfg.model_dir + '/SkeletonNet.trained.on.AMB/inter_refine_snapshot_49.pth.tar')
    else:
        trainer._make_model(pre_train=cfg.model_dir + '/SkeletonNet.trained.on.AMB/refine_snapshot_6.pth.tar')

    # train

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}
        # for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
        #     trainer.read_timer.toc()
        #     trainer.gpu_timer.tic()
        #     # forward
        #     trainer.optimizer.zero_grad()
        #
        #     img_mask = inputs['img'].cuda()
        #
        #     joint_sp = targets['joint_coord'][:, :21]
        #     joint_km_sp = targets['joint_coord_kmean'][:, :21]
        #     pre_joint = joint_km_sp.clone().cuda()
        #     gt_joint = joint_sp.clone().cuda()
        #
        #     refine_heatmap, loss, refine_joint = trainer.model(img_mask, pre_joint, gt_joint, 'train')
        #
        #     # joint_coord = np.zeros([refine_joint.shape[0], 42, 3])
        #     # joint_coord[:, :21] = refine_joint.detach().cpu().numpy()
        #
        #     # preds['joint_coord'].append(joint_coord) #targets['joint_coord_kmean_space'].cpu().numpy())
        #     # preds['rel_root_depth'].append(targets['rel_root_depth'].cpu().numpy())
        #     # preds['hand_type'].append(targets['hand_type'].cpu().numpy())
        #     # preds['inv_trans'].append(meta_info['inv_trans'].cpu().numpy())
        #
        #     loss = {k: loss[k].mean() for k in loss}
        #     sum(loss[k] for k in loss).backward()
        #     trainer.optimizer.step()
        #     trainer.gpu_timer.toc()
        #
        #     screen = [
        #         'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
        #         'lr: %g' % (trainer.get_lr()),
        #         'speed: %.2f(%.2fs r%.2f)s/itr' % (
        #             trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
        #         '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
        #     ]
        #     screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss.items()]
        #
        #     if itr % 100 == 0:
        #         trainer.logger.info(' '.join(screen))
        #
        #     trainer.tot_timer.toc()
        #     trainer.tot_timer.tic()
        #     trainer.read_timer.tic()
        # # save model
        # trainer.save_model({
        #     'epoch': epoch,
        #     'network': trainer.model.state_dict(),
        #     'optimizer': trainer.optimizer.state_dict(),
        # }, epoch)

        # calc mpjpe
        preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}
        for itr, (inputs, targets, meta_info) in enumerate(trainer.test_batch_generator):
            with torch.no_grad():
                joint_sp = targets['joint_coord'][:, :21]
                joint_km_sp = targets['joint_coord_kmean'][:, :21]
                pre_joint = joint_km_sp.clone().cuda()
                gt_joint = joint_sp.clone().cuda()

                img_mask = inputs['img'].cuda()
                refine_heatmap, loss, refine_joint = trainer.model(img_mask, pre_joint, gt_joint, 'train')

                joint_coord = np.zeros([refine_joint.shape[0], 42, 3])
                joint_coord[:, :21] = refine_joint.detach().cpu().numpy()

                preds['joint_coord'].append(joint_coord)  # targets['joint_coord_kmean_space'].cpu().numpy())
                preds['rel_root_depth'].append(targets['rel_root_depth'].cpu().numpy())
                preds['hand_type'].append(targets['hand_type'].cpu().numpy())
                preds['inv_trans'].append(meta_info['inv_trans'].cpu().numpy())

        preds = {k: np.concatenate(v) for k, v in preds.items()}
        trainer.testset_loader.evaluate(preds)



if __name__ == "__main__":
    main()