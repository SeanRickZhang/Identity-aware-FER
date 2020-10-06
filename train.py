import torch
from torch.autograd import Variable
from data_loader import data_loader
from utils.tools import save_2_file, print_network
import numpy as np
from utils.tools import draw_line
from utils.tools import select_Y, C_MSE, TwoCN_loss, Prepare_heatmap, select_X, Draw_heatmap

import os
import time
## MSE loss + hinge loss
class train_model(object):

    def __init__(self, image_dir_train, image_dir_test, lr = 0.01, load_epoch = 0,  batch_size = 16, epoch_size = 20, datasets = 'CKplus', net = None ,
                use_gpu = True, resize = True, category = 7, num_work=0, just_real_test = False, part_num = 10):

        self.datasets = datasets
        self.image_dir_train = image_dir_train
        self.image_dir_test = image_dir_test
        self.batch_size = batch_size
        self.net = net
        self.epoch_size = epoch_size
        self.use_gpu = use_gpu
        self.load_epoch = load_epoch
        self.resize = resize
        self.num_work = num_work
        self.category = category
        self.part_num=str(part_num)
        self.lr = lr
        self.just_real_test = just_real_test
        self.Graph = [[0 for col in range(self.category)] for row in range(self.category)]
        self.box = [0 for col in range(self.category)]
        # self.training()
        # self.box = {'An':0, 'Di':0, 'Fe':0, 'Ha':0, 'Sa':0, 'Su':0}

    def training(self):
        # margin  = self.triplet_margin * float(self.category-1/self.category)

        train_dataloader = data_loader.get_loader(self.image_dir_train, batch_size=self.batch_size,
                                                         dataset=self.datasets, mode='train', num_workers=self.num_work, resize=self.resize)
        test_dataloader = data_loader.get_loader(self.image_dir_test, batch_size=self.batch_size,
                                                        dataset=self.datasets, mode='test', num_workers=self.num_work, resize=self.resize, just_real_test=self.just_real_test)
        if self.net == 'IAFER_model':
            from models.IAFER_model import create_IAFER_model
            model = create_IAFER_model()

        print_network(model, self.net)
        optimizer_model = torch.optim.SGD(model.parameters(), self.lr, weight_decay = 0.005)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model = model.cuda()
        # time_open = time.time()

        total_loss_train = []
        total_loss_test = []

        CNN_train_accuracy = []
        CNN_test_accuracy = []

        all_epoch = []
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_model, milestones=[10, 15], gamma=0.1)
        for epoch in range(self.load_epoch, self.epoch_size):

            lr = str(optimizer_model.param_groups[0]['lr'])
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train(True)
                    dataloader = train_dataloader
                else:
                    model.train(False)
                    dataloader = test_dataloader
                correct = 0.0
                running_loss = 0.0
                print(len(dataloader.dataset))
                start1 = time.perf_counter()
                for i, data in enumerate(dataloader):
                    if self.datasets == 'MMI_CKplus' or 'OuluCasIA' or self.datasets == 'ISAFE' in self.datasets:
                        img0, img1, img2, img3, img4, img5, imgreal, label = data
                        if self.use_gpu:
                            #
                            X0, X1, X2, X3, X4, X5, Xreal, Y = Variable(img0.cuda()), Variable(
                                img1.cuda()), Variable(img2.cuda()), Variable(img3.cuda()), \
                                                                   Variable(img4.cuda()), Variable(
                                img5.cuda()), Variable(imgreal.cuda()), Variable(label.cuda())
                        else:
                            #
                            X0, X1, X2, X3, X4, X5, Xreal, Y = Variable(img0), Variable(img1), Variable(
                                img2), Variable(img3), Variable(img4), Variable(img5), \
                                                                   Variable(imgreal), Variable(label)
                        X_images = [X0, X1, X2, X3, X4, X5]

                    elif self.datasets == 'CKplus':
                        img0, img1, img2, img3, img4, img5, img6, imgreal, label = data
                        if self.use_gpu:
                            #
                            X0, X1, X2, X3, X4, X5, X6, Xreal, Y = Variable(img0.cuda()), Variable(img1.cuda()),Variable(img2.cuda()),Variable(img3.cuda()),\
                                                                   Variable(img4.cuda()),Variable(img5.cuda()),Variable(img6.cuda()),Variable(imgreal.cuda()),\
                                                                      Variable(label.cuda())
                        else:
                            #
                            X0, X1, X2, X3, X4, X5, X6, Xreal, Y = Variable(img0), Variable(img1), Variable(img2), Variable(img3), Variable(img4), Variable(img5),\
                                                                   Variable(img6), Variable(imgreal), Variable(label)
                        X_images = [X0, X1, X2, X3, X4, X5, X6]
                    elif self.datasets == 'ISED':
                        img0, img1, img2, img3, imgreal, label = data
                        if self.use_gpu:
                            #
                            X0, X1, X2, X3, Xreal, Y = Variable(img0.cuda()), Variable(img1.cuda()),Variable(img2.cuda()),Variable(img3.cuda()),\
                                                                   Variable(imgreal.cuda()), Variable(label.cuda())
                        else:
                            #
                            X0, X1, X2, X3, Xreal, Y = Variable(img0), Variable(img1), Variable(img2), Variable(img3), Variable(imgreal), Variable(label)
                        X_images = [X0, X1, X2, X3]
                    Y_blocks = select_Y(Y, self.category, (X0.shape)[0])
                    outputs, outputx, outputr = model(X_images, Xreal)
                    Xs = select_X(outputx, label)

                    loss = 0
                    for j, Xi in enumerate(outputs):
                        Yi = Y_blocks[j]
                        TCN_loss = TwoCN_loss(Xi, Yi)
                        loss = loss + TCN_loss
                    if 'MSE' in self.datasets:
                        MSE_loss = C_MSE(Xs, outputr, (X0.shape)[0])
                        loss = loss + MSE_loss * 100
                    temp = torch.cat(outputs, 1)
                    _, index = torch.max(temp, 1)
                    correct = correct + index.view(-1, 1).eq(Y).sum()
                    if epoch == self.epoch_size-1 and phase == 'test':
                        self.Graph, self.box = Prepare_heatmap(self.Graph, self.box, index.view(-1, 1), Y)
                    optimizer_model.zero_grad()

                    if phase == 'train':
                        loss.backward()
                        optimizer_model.step()
                    running_loss += loss.item()
                print(correct)
                end1 = time.perf_counter()
                print("final is in : %s Seconds " % (end1 - start1))
                # print(identity_correct)
                if phase == "train":
                    print("".format())
                    print("-" * 10)
                    print('Epoch {}/{},Train set:, Average loss {:.6f},FER Accuracy: {:.6f}, lr: {}'.format(epoch, self.epoch_size - 1,
                                                                                                                 running_loss / len(dataloader.dataset),
                                                                                                         float(correct) / len(dataloader.dataset),
                        lr
                    ))
                    total_loss_train.append(running_loss / len(dataloader.dataset))
                    CNN_train_accuracy.append(float(correct) / len(dataloader.dataset))
                else:
                    print('Test set: Average loss {:.6f},FER Accuracy: {:.6f}'.format(
                        running_loss / len(dataloader.dataset),
                        float(correct) / len(dataloader.dataset)
                    ))
                    total_loss_test.append(running_loss / len(dataloader.dataset))
                    CNN_test_accuracy.append(float(correct) / len(dataloader.dataset))
            all_epoch.append(epoch)
            train_scheduler.step()

        total_loss_train = np.array(total_loss_train)
        total_loss_test = np.array(total_loss_test)

        CNN_train_accuracy = np.array(CNN_train_accuracy)
        CNN_test_accuracy = np.array(CNN_test_accuracy)
        if 'MSE' in self.datasets:
            save_file = '../DML-resuts(MSE)'
        else:
            save_file = '../DML-resuts(WMSE)'
        save_2_file(self.datasets, self.net, self.part_num, total_loss_train, total_loss_test, CNN_train_accuracy,
                    CNN_test_accuracy, model.state_dict(), self.epoch_size,save_file)
        save_dir = os.path.join(save_file, self.datasets, self.part_num)
        heatmap_data = Draw_heatmap(self.Graph, self.box, self.datasets, self.part_num, save_file)
        draw_line(all_epoch, total_loss_train, total_loss_test,strings = 'loss', save_dir =save_dir)
        draw_line(all_epoch,CNN_train_accuracy, CNN_test_accuracy,'accuracy', save_dir = save_dir)
        return heatmap_data