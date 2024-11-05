import argparse
import os
import sys
from pathlib import Path
import yaml
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(ROOT)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(ROOT)

from utils.check import check_img_size, increment_path, colorstr, init_seeds, check_suffix
from utils.torch_utils import select_device
from models.detect_yolo import YOLOModel

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train():
    pass


def before_train(opt):
    """检查文件"""
    # 1.读取yaml配置
    # if len(opt.config):
    #     with open(opt.config, "w") as f:
    #         config_yaml = yaml.safe_load(f)


def run(opt):
    # before_train(opt)
    """  参数解析  """
    if len(opt.config) != 0:
        with open(opt.config, "r") as f:
            config_yaml = yaml.safe_load(f)
        cfg = config_yaml["cfg"]
        weights = config_yaml["weights"]
        epochs = config_yaml["epochs"]
        batch_size = config_yaml["batch_size"]
        img_size = check_img_size(config_yaml["img_size"], 32, floor=32 * 2)
        device = select_device(config_yaml["device"], batch_size)
        hyp = config_yaml["hyp"]
        with open(hyp, "r") as f:
            hyp = yaml.safe_load(f)
    else:
        weights = opt.weights
        cfg = opt.cfg
        epochs = opt.epochs
        batch_size = opt.batch_size
        img_size = check_img_size(opt.img_size, 32, floor=32 * 2)
        device = select_device(opt.device, batch_size)
        hyp = opt.hyp
        with open(hyp, "r") as f:
            hyp = yaml.safe_load(f)
    save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    """  参数解析  """

    """ train """
    # Directories
    # 创建 weights / last.pt  , weights / last.pt
    w = Path(save_dir) / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    # cpu cuda设置
    cuda = device.type != 'cpu'

    # 随机数
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    # 检查数据集
    # with torch_distributed_zero_first(LOCAL_RANK):
    #     data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = cfg['train'], cfg['val']
    nc = int(cfg['nc'])
    names = {0: 'item'} if nc == 1 and len(cfg['names']) != 1 else cfg['names']
    # 检查模型权重
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    # 如果有预训练权重
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')
        model = YOLOModel(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    else:
        model = YOLOModel(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    """ train """


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--config', type=str, default='', help='config')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    opt0 = parse_opt()
    run(opt0)
