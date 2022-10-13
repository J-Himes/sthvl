import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

logger = logging.getLogger()


def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_bboxes = model.predict(seq_torch)

            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)

            # Adding in our cps code
#            num_cps = min(n_frames, 30)
#            sample_rate = np.floor(n_frames / num_cps)
            sample_rate = 15
            picks_1 = np.arange(0, seq_len) * sample_rate
            cps_1 = []
            nfps_1 = []
            i = 0

            # for i in range(num_cps):
            #     start = i * sample_rate
            #     end = start + sample_rate
            #     if (i + 1 == num_cps):
            #         end = n_frames - 1
            #     cps_1.append([int(start), int(end)])
            #     nfps_1.append(end - start)

            while i < seq_len:
                start = i * sample_rate
                i += 1
                end = i * sample_rate
                i += 1
                if i == seq_len:
                    end = min(n_frames - 1, (i + 1) * sample_rate)
                    i += 1

                cps_1.append([start, end])
                nfps_1.append((end - start) + 1)

            cps_1 = np.asarray(cps_1)
            nfps_1 = np.asarray(nfps_1)
            cps = cps_1
            nfps = nfps_1
            print('Test')

            # End of our code

            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)

            pred_summ = vsumm_helper.downsample_summ(pred_summ)
            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            stats.update(fscore=fscore, diversity=diversity)


    return stats.fscore, stats.diversity


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            fscore, diversity = evaluate(model, val_loader, args.nms_thresh, args.device)
            stats.update(fscore=fscore, diversity=diversity)

            logger.info(f'{split_path.stem} split {split_idx}: diversity: '
                        f'{diversity:.4f}, F-score: {fscore:.4f}')

        logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                    f'F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()
