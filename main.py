import os
import argparse
from train import train_model
from torch.backends import cudnn
#image_dir_train, image_dir_test, lr = 0.002, load_epoch = 0,
#batch_size = 16, epoch_size = 20, datasets = 'CKplus', net = None ,
#use_gpu = True, resize = True, category = 7, num_work=0, just_real_test = False, part_num = 10

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    datasets = config.datasets
    models = config.nets
    cudnn.benchmark = True
    for dataset in datasets:
        for model in models:
            if 'OuluCasIA' in dataset or dataset == 'MMI' or dataset == 'Jaffe6' or dataset == 'MMI_CKplus':
                category = 6
                total_heatmap = [[0 for i in range(6)] for i in range(6)]
                temp = config.start_part
            else:
                category = 7
                total_heatmap = [[0 for i in range(7)] for i in range(7)]
                temp = config.start_part
            for part_number in range(temp, config.part_sum):

                image_dir_train = os.path.join(config.image_dir_train, str(part_number), 'train')
                image_dir_test = os.path.join(config.image_dir_test, str(part_number), 'test')
                heatmap = train_model(image_dir_train=image_dir_train, image_dir_test=image_dir_test, lr = config.lr, batch_size=config.batch_size,
                                            datasets=dataset,use_gpu=config.use_gpu, num_work=config.num_work,epoch_size=config.epoch_size,
                                            resize=False, net=model, part_num=part_number, category=category).training()
                #     #image_dir_train, image_dir_test, lr = 0.002, load_epoch = 0,  batch_size = 16, epoch_size = 20, datasets = 'CKplus', net = None ,
                # #use_gpu = True, resize = True, category = 7, num_work=0, just_real_test = False, part_num = 10
                # else:
                #     # identity_num = 117
                #     heatmap = train_model(image_dir_train=image_dir_train, image_dir_test=image_dir_test,
                #                                 datasets=dataset, resize=False, net=model,
                #                                 part_num=part_number).training()
                total_heatmap = heatmap + total_heatmap
            if 'MSE' in dataset:
                accuracy = open(os.path.join('../DML-results(MSE)', dataset, 'test_heatmap.txt'), 'w')
            else:
                accuracy = open(os.path.join('../DML-results(WMSE)', dataset, 'test_heatmap.txt'), 'w')
            for loss in total_heatmap:
                accuracy.write(str(loss))
                accuracy.write('\n')
            accuracy.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir_train', type = str)
    parser.add_argument('--image_dir_test', type=str)
    parser.add_argument('--lr', type = float, default = 0.01)
    parser.add_argument('--load_epoch', type = int, default='0')
    parser.add_argument('--batch_size', type = int, default='16')
    parser.add_argument('--epoch_size', type = int, default='20')
    parser.add_argument('--datasets', nargs = '+')
    parser.add_argument('--nets', nargs = '+', default = ['IAFER_model'])
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--start_part', type = int, default=0)
    parser.add_argument('--num_work', type = int, default=0)
    parser.add_argument('--part_sum', type = int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)
