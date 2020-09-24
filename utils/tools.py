import numpy as np
# from sklearn.model_selection import train_test_split
import torch
from matplotlib import pyplot as plt
import os
import seaborn as sns

def draw_line(all_epoch, train, test, strings, save_dir):
    # fig = plt.figure(figsize=(10,6))
    plt.plot(all_epoch, train, color = "red", label = 'train')
    plt.plot(all_epoch, test, color = "blue", label = 'test')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel(strings)
    plt.title(strings)
    plt.savefig(os.path.join(save_dir, strings+ '3_real' + ".png"))
    # plt.show()

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

def select_X(Xs, labels):
    Xp = []
    for j, label in enumerate(labels):
        indices = torch.cuda.LongTensor([j])
        temp = torch.index_select(Xs[label], 0, indices)
        Xp.append(temp)
    Y = torch.cat(Xp, 0)
    return Y

def select_Y(labels, category, batch_size):
    temp = []
    for i in range(category):
        temp.append(torch.full((batch_size, 1), -1).cuda())

    for j, label in enumerate(labels):
        x = int(label.cpu().numpy())
        y = torch.LongTensor([j]).cuda()
        (temp[x]).index_fill_(0, y, 1)
    return temp

def C_MSE(X, Y, batch_size):
    results = []
    for i in range(batch_size):
        results.append(torch.mean(torch.pow((X[i] - Y[i]), 2)).view(1, 1))

    return torch.sum(torch.cat(results, 0))

def TwoCN_loss(o_params, y_params):
    a = torch.mul(o_params, y_params)
    b = torch.Tensor([1]).cuda() - a
    temp = torch.clamp(b, min=0)
    return torch.sum(temp)

def save_2_file(dataset, net, part_num, train_loss, test_loss, train_accuracy, test_accuracy, model_dict, epoch_size, save):
    save_dir = os.path.join(save, dataset, net, part_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    savefile_train_loss = open(os.path.join(save_dir, 'train_loss.txt'), 'w')
    savefile_test_loss = open(os.path.join(save_dir, 'test_loss.txt'), 'w')
    savefile_train_accuracy = open(os.path.join(save_dir, 'train_accuracy.txt'), 'w')
    savefile_test_accuracy = open(os.path.join(save_dir, 'test_accuracy.txt'), 'w')
    for loss in train_loss:
        savefile_train_loss.write(str(loss))
        savefile_train_loss.write('\n')
    savefile_train_loss.close()
    for loss in test_loss:
        savefile_test_loss.write(str(loss))
        savefile_test_loss.write('\n')
    savefile_test_loss.close()
    for accuracy in train_accuracy:
        savefile_train_accuracy.write(str(accuracy))
        savefile_train_accuracy.write('\n')
    savefile_train_accuracy.close()
    for accuracy in test_accuracy:
        savefile_test_accuracy.write(str(accuracy))
        savefile_test_accuracy.write('\n')
    savefile_test_accuracy.close()
    checkpoint_path = os.path.join(save_dir, '{dataset}-{net}-{epoch}.pth')
    torch.save(model_dict, checkpoint_path.format(dataset = dataset, net=net, epoch=epoch_size))
    print('saving success!')

def Prepare_heatmap(Graph, box, X, Y):
    for i in range(len(X)):
        Xi = int((X.cpu().numpy())[i])
        Yi = int((Y.cpu().numpy())[i])
        Graph[Xi][Yi] = Graph[Xi][Yi] + 1
        box[Yi] = box[Yi] + 1
    return Graph, box

def Draw_heatmap(Graph, box, dataset, part_num, save_dir):
    sns.set()
    # print(box)
    heatmap_data = [[0 for i in range(len(box))] for i in range(len(box))]
    for i, row in enumerate(Graph):
        for j, X in enumerate(row):
            heatmap_data[j][i] = round(X / float(box[j]),4)
    print(heatmap_data)
    if dataset == 'CKplus':
        sns.heatmap(heatmap_data, annot=True, cmap='Blues_r', linewidths=.5,
                    xticklabels=['An', 'Co', 'Di', 'Fe', 'Ha', 'Sa', 'Su'],
                    yticklabels=['An', 'Co', 'Di', 'Fe', 'Ha', 'Sa', 'Su'])
    elif 'OuluCasIA' in dataset or dataset == 'MMI_CKplus' or 'Oulu_CasIA_Strong':
        sns.heatmap(heatmap_data, annot=True, cmap='Blues_r', linewidths=.5, xticklabels=['An', 'Di', 'Fe', 'Ha', 'Sa', 'Su'],
                yticklabels=['An', 'Di', 'Fe', 'Ha', 'Sa', 'Su'])
    else:
        sns.heatmap(heatmap_data, annot=True, cmap='Blues_r', linewidths=.5,
                    xticklabels=['An', 'Di', 'Fe', 'Ne', 'Ha', 'Sa', 'Su'],
                    yticklabels=['An', 'Di', 'Fe', 'Ne', 'Ha', 'Sa', 'Su'])
    if not os.path.exists(os.path.join(save_dir, dataset, part_num)):
        os.makedirs(os.path.join(save_dir, dataset, part_num))
    print(os.path.join(save_dir, dataset, part_num, 'heatmap.png'))
    plt.savefig(os.path.join(save_dir, dataset, part_num, 'heatmap.png'), format='png')
    plt.close()
    return heatmap_data