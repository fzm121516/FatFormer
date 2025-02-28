import torch
import random
import numpy as np
import argparse
import torch.distributed as dist
from myutils.misc import get_rank, init_distributed_mode
from myutils.dataset import Dataset_Creator
import myutils.misc as utils
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from models import build_model
from sklearn.metrics import average_precision_score, accuracy_score
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE  # 导入t-SNE算法
import matplotlib.pyplot as plt  # 导入Matplotlib绘图库

def plot_tsne(features, labels, title, save_path=None, dpi=1200):
    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=123)
    tsne_results = tsne.fit_transform(features)

    # 绘制散点图
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='coolwarm', s=10)
    plt.title(title)
    plt.xticks([])  # 隐藏 X 轴刻度
    plt.yticks([])  # 隐藏 Y 轴刻度
    # 隐藏坐标轴标签
    plt.xlabel("")  # X 轴标签设置为空
    plt.ylabel("")  # Y 轴标签设置为空

    # 隐藏刻度
    plt.xticks([])  # 隐藏 X 轴刻度
    plt.yticks([])  # 隐藏 Y 轴刻度

    # 添加颜色条
    # plt.colorbar(scatter, label='Label')

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, format='svg', dpi=dpi)  # 修改为保存 SVG 格式
    else:
        plt.show()

    plt.close()

def evaluate_and_plot_tsne(model, data_loader, file_name, save_path):
    features = []  # 存储特征
    labels = []  # 存储标签

    with torch.no_grad():  # 禁用梯度计算
        for img1, label, path in tqdm(data_loader, desc=f"Processing {file_name}", leave=False):
            feature = model.encode_image(img1.cuda()).detach().cpu().numpy()  # 提取特征
            features.extend(feature)  # 存储特征
            labels.extend(label.flatten().tolist())  # 存储标签

    features = np.array(features)  # 转换为NumPy数组
    labels = np.array(labels)  # 转换为NumPy数组

    # 绘制并保存 t-SNE 图
    plot_tsne(features, labels, title=f't-SNE plot for {file_name}', save_path=save_path)



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # dataset parameters
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--img_resolution', type=int, default=256)
    parser.add_argument('--crop_resolution', type=int, default=224)
    parser.add_argument('--test_selected_subsets', nargs='+', required=True)
    parser.add_argument('--batchsize', type=int, default=32)

    # model
    parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_vit_adapter', type=int, default=8)

    # text encoder
    parser.add_argument('--num_context_embedding', type=int, default=8)
    parser.add_argument('--init_context_embedding', type=str, default="")

    # frequency
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--clip_vision_width', type=int, default=1024)
    parser.add_argument('--frequency_encoder_layer', type=int, default=2)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)

    # output
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    return parser


@torch.no_grad()
def gather_together(data):
    world_size = dist.get_world_size()
    if world_size < 2:
        return data
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


@torch.no_grad()
def evaluate(model, data_loaders, device, args=None, test=False):
    model.eval()
    # if test:
    #     images = data_loaders.to(device)
    #     outputs = model(images)
    #     print(outputs.softmax(dim=1)[:, 1].flatten().tolist())
    #     return

    for data_name, data_loader in data_loaders.items():
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        print_freq = args.print_freq

        y_true, y_pred = [], []
        logits_list = []
        logits0_list = []
        # for samples in metric_logger.log_every(data_loader, print_freq, header):
        #     images, labels = [sample.to(device) for sample in samples]
        #     outputs = model(images)
        #     logits_list.extend(outputs[:, 1].flatten().tolist())
        #     logits0_list.extend(outputs[:, 0].flatten().tolist())

        #     y_pred.extend(outputs.softmax(dim=1)[:, 1].flatten().tolist())

        #     y_true.extend(labels.flatten().tolist())
        

        features = []  # 存储特征
        labels = []  # 存储标签


        for samples in metric_logger.log_every(data_loader, print_freq, header):
            images, label = [sample.to(device) for sample in samples]

            feature = model.module.encode_img(images)  # 通过 .module 访问原始模型的方法
            # print("Feature shape1:", feature.shape)  # 打印 features 的形状
            features.extend(feature)  # 存储特征
            labels.extend(label.flatten().tolist())  # 存储标签
        if any(f.is_cuda for f in features):  # 检查列表中是否有张量位于 GPU 上
            features = [f.cpu() for f in features]  # 将列表中的每个张量移动到 CPU
        features = np.array(features)  # 转换为 NumPy 数组

        # features = np.array(features)  # 转换为NumPy数组
        labels = np.array(labels)  # 转换为NumPy数组
        tsne_dir = 'tsne_0227fat'
        if not os.path.exists(tsne_dir):
            os.makedirs(tsne_dir)
        tsne_save_path = os.path.join(tsne_dir, f'tsne_{data_name}.svg')  # 保存路径设置为当前路径下的 tsne 文件夹
        # 绘制并保存 t-SNE 图
        plot_tsne(features, labels, title=f't-SNE plot for {data_name}', save_path=tsne_save_path)



def main(args):
    init_distributed_mode(args)

    # set device
    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    # infer dataset
    dataset_creator = Dataset_Creator(dataset_path=args.dataset_path, batch_size=args.batchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)
    dataset_vals, selected_subsets = dataset_creator.build_dataset("test", selected_subsets=args.test_selected_subsets)
    data_loader_vals = {}
    for dataset_val, selected_subset in zip(dataset_vals, selected_subsets):
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = SequentialSampler(dataset_val)
        data_loader_vals[selected_subset] = DataLoader(dataset_val, args.batchsize, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)
    
    # model
    model = build_model(args)
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    if args.eval:
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        evaluate(model, data_loader_vals, device,args=args)




if __name__ == "__main__":    
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
