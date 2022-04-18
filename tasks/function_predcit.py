import os

import click as ck
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append('../')
from deepfold.data.uniprot_dataset import AnnotatedSequences
from deepfold.loss.metrics import compute_roc
from deepfold.model.deepgoplus import DeepGOPlusModel


@ck.command()
@ck.option('--data-path', '-dp', default='data', help='Path to store data')
@ck.option('--model-path',
           '-mp',
           default='models/s',
           help='Path to save model')
@ck.option('--summary-path',
           '-sp',
           default='logs/deepgoplus',
           help='Path to save summary')
def main(data_path, model_path, summary_path):
    # check data_path
    if not os.path.exists(data_path):
        print('Unable to find data path %s.' % data_path)

    # check model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print('Create %s to save model.' % model_path)
    # filename format to save or load model
    model_prefix = r'deepgoplus_parameters_'
    model_suffix = r'.pth'
    model_file = os.path.join(model_path, model_prefix + r'%d' + model_suffix)

    # check summary_path
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
        print('Create %s to save summary.' % summary_path)

    # Gene Ontology file in OBO Format
    go_file = os.path.join(data_path, u'go.obo')
    # Result file with a list of terms for prediction task
    out_terms_file = os.path.join(data_path, u'terms.pkl')
    # Result file with a DataFrame for training
    train_data_file = os.path.join(data_path, u'train_data.pkl')
    # Result file with a DataFrame for testing
    test_data_file = os.path.join(data_path, u'test_data.pkl')
    # Result file with a DataFrame of prediction annotations
    predictions_file = os.path.join(data_path, 'predictions.pkl')

    data_path_dict = {}
    data_path_dict['go'] = go_file
    data_path_dict['terms'] = out_terms_file
    data_path_dict['train_data'] = train_data_file
    data_path_dict['test_data'] = test_data_file
    data_path_dict['predictions'] = predictions_file

    train_df = pd.read_pickle(data_path_dict['test_data'])
    test_df = pd.read_pickle(data_path_dict['test_data'])
    terms_df = pd.read_pickle(data_path_dict['terms'])

    # Hyper parameters of model and training
    params = {
        'nb_filters': 128,
        'max_kernel': 129,
        'fc_depth': 0,
        'learning_rate': 3e-4,  # 学习率
        'loss': 'binary_crossentropy',  #
        'initializer': 'glorot_normal',  #
        'epochs': 10,  # 迭代次数
        'model_save_interval': 1,  # save model after interval epoches
        'train_batch_size': 32,  # 训练集一个batch大小
        'valid_batch_size': 64,  # 验证集一个batch大小
        'test_batch_size': 64,  # 测试集一个batch大小
        'threshold': 0.5,
    }

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Dataset and DataLoader
    train_dataset = AnnotatedSequences(train_df, terms_df)
    test_dataset = AnnotatedSequences(test_df, terms_df)

    train_valid_split = 0.9
    train_dataset_size = int(len(train_dataset) * train_valid_split)
    valid_dataset_size = len(train_dataset) - train_dataset_size
    train_dataset, valid_dataset = random_split(
        train_dataset, [train_dataset_size, valid_dataset_size])

    print('Size of train dataset:', len(train_dataset))
    print('Size of valid dataset:', len(valid_dataset))
    print('Size of test dataset:', len(test_dataset))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params['train_batch_size'],
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=params['valid_batch_size'],
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=params['test_batch_size'],
                                 shuffle=False)

    # Terms of annotations
    terms = terms_df['terms'].values.flatten()
    nb_classes = len(terms)

    print('Number of classes: ', nb_classes, '\n')

    # model
    model = DeepGOPlusModel(nb_classes, params)
    model.to(device)

    start_epoch = 0
    optimizer = optim.Adam(model.parameters(),
                           lr=params['learning_rate'])  # 使用Adam优化器
    crition = torch.nn.BCELoss()
    # record data for tensorboard
    writer = SummaryWriter(summary_path)

    print('Training starts:')
    for epoch in range(1, params['epochs'] + 1):
        print('--------Epoch %02d--------' % (epoch + start_epoch))
        train_loss, train_auc, valid_loss, valid_auc = train(
            model,
            optimizer,
            crition,
            train_dataloader,
            valid_dataloader,
            train_len=len(train_dataset),
            valid_len=len(valid_dataset),
            nb_classes=nb_classes,
            device=device)
        y_trues, y_preds, test_loss, test_auc = test(
            model,
            crition,
            test_dataloader,
            test_len=len(test_dataset),
            nb_classes=nb_classes,
            device=device)

        writer.add_scalar('train_loss', train_loss, epoch + start_epoch)
        writer.add_scalar('train_auc', train_auc, epoch + start_epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch + start_epoch)
        writer.add_scalar('valid_auc', valid_auc, epoch + start_epoch)
        writer.add_scalar('test_loss', test_loss, epoch + start_epoch)
        writer.add_scalar('test_auc', test_auc, epoch + start_epoch)

        # save model
        if (epoch + start_epoch) % params['model_save_interval'] == 0:
            torch.save(model.state_dict(), model_file % (epoch + start_epoch))
            print('Model parameters are saved!')

    writer.close()
    print('Training is done!')

    test_labels, preds, test_loss, test_auc = test(model,
                                                   crition,
                                                   test_dataloader,
                                                   test_len=len(test_dataset),
                                                   nb_classes=nb_classes,
                                                   device=device)

    # 保存df文件
    test_df = pd.read_pickle(data_path_dict['test_data'])
    test_df['labels'] = list(test_labels)
    test_df['preds'] = list(preds)
    test_df.to_pickle(data_path_dict['predictions'])


# Training function
def train(model,
          optimizer,
          crition,
          train_loader,
          valid_loader,
          train_len,
          valid_len,
          nb_classes,
          device='cpu'):
    model.train()  # 设置为训练模式
    train_loss = 0  # 初始化训练损失为0
    train_auc = 0  # 初始化预测正确个数为0

    train_preds = np.zeros((train_len, nb_classes))
    train_trues = np.zeros((valid_len, nb_classes))

    # 训练过程
    for index, (data, target) in enumerate(train_loader):
        train_batch = data.shape[0]
        data = data.to(device)
        target = target.to(device)

        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        loss = crition(output, target.float())  # 交叉熵损失函数
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新参数

        pred_numpy = output.detach().cpu().numpy()
        true_numpy = target.detach().cpu().numpy()

        train_preds[index * train_batch:(index + 1) *
                    train_batch, :] = pred_numpy  # 获取预测值
        train_trues[index * train_batch:(index + 1) *
                    train_batch, :] = true_numpy

    train_auc = compute_roc(train_trues, train_preds)  # 训练集准确率

    model.eval()  # 设置为test模式
    valid_loss = 0  # 初始化测试损失值为0
    valid_auc = 0  # 初始化预测正确的数据个数为0

    valid_preds = np.zeros((len(valid_len), nb_classes))
    valid_trues = np.zeros((len(valid_len), nb_classes))

    # 验证效果
    for index, (data, target) in enumerate(valid_loader):
        valid_batch = data.shape[0]
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(
            target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度

        with torch.no_grad():
            output = model(data)

        loss = crition(output, target.float())
        valid_loss += loss.item() * valid_batch

        pred_numpy = output.detach().cpu().numpy()
        true_numpy = target.detach().cpu().numpy()

        valid_preds[index * valid_batch:(index + 1) *
                    valid_batch, :] = pred_numpy  # 获取预测值
        valid_trues[index * valid_batch:(index + 1) *
                    valid_batch, :] = true_numpy

    valid_auc = compute_roc(valid_trues, valid_preds)  # 训练集准确率

    # total loss should be averaged over dataset
    train_loss /= len(train_len)
    valid_loss /= len(valid_len)

    print('Train set: Average loss: {:.4f}, ROC AUC: {:.4f}'.format(
        train_loss, train_auc))
    print('Valid set: Average loss: {:.4f}, ROC AUC: {:.4f}'.format(
        valid_loss, valid_auc))

    return train_loss, train_auc, valid_loss, valid_auc


# Testing function
def test(model, crition, test_loader, test_len, nb_classes, device):
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    test_auc = 0  # 初始化预测正确的数据个数为0

    y_preds = np.zeros((len(test_len), nb_classes))
    y_trues = np.zeros((len(test_len), nb_classes))

    # 测试结果
    for index, (data, target) in enumerate(test_loader):
        test_batch = data.shape[0]
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(
            target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度

        with torch.no_grad():
            output = model(data)

        loss = crition(output, target.float())
        test_loss += loss.item() * test_batch

        pred_numpy = output.detach().cpu().numpy()
        true_numpy = target.detach().cpu().numpy()

        y_preds[index * test_batch:(index + 1) *
                test_batch, :] = pred_numpy  # 获取预测值
        y_trues[index * test_batch:(index + 1) * test_batch, :] = true_numpy

    test_auc = compute_roc(y_trues, y_preds)
    test_loss /= len(test_loader.dataset)
    # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('Test set: Average loss: {:.4f}, ROC AUC: {:.2f}'.format(
        test_loss, test_auc))

    return y_trues, y_preds, test_loss, test_auc


if __name__ == '__main__':
    main()