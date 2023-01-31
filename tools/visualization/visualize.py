import argparse
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('idx', type=int,
        help='index of data to visualize')
    parser.add_argument('--result', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument('--thr', 
        type=float,
        default=0,
        help='score threshold to filter predictions')
    parser.add_argument(
        '--out-dir', 
        default='demo',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def import_plugin(cfg):
    '''
        import modules from plguin/xx, registry will be update
    '''

    import sys
    sys.path.append(os.path.abspath('.'))    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            
            def import_path(plugin_dir):
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            plugin_dirs = cfg.plugin_dir
            if not isinstance(plugin_dirs, list):
                plugin_dirs = [plugin_dirs,]
            for plugin_dir in plugin_dirs:
                import_path(plugin_dir)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)

    # build the dataset
    dataset = build_dataset(cfg.data.val)
    idx = args.idx
    out_dir = os.path.join(args.out_dir, str(idx))
    gt_dir = os.path.join(out_dir, 'gt')
    pred_dir = os.path.join(out_dir, 'pred')

    if args.result is not None:
        os.makedirs(pred_dir, exist_ok=True)
        results = mmcv.load(args.result)
        dataset.show_result(
                submission=results, 
                idx=idx, 
                score_thr=args.thr, 
                out_dir=pred_dir
            )
        
    os.makedirs(gt_dir, exist_ok=True)
    dataset.show_gt(idx, gt_dir)


if __name__ == '__main__':
    main()