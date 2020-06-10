import torch
import os
import data_loader
from TransferNet import Transfer_Net
from config import CFG
import utils
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch DDC_DeepCoral')

parser.add_argument('--backbone', type=str, default='resnet50', help='backbone of the model')
parser.add_argument('--epochs', type=int, default=100, help='num of epoch')

args = parser.parse_args()

torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(model, target_test_loader):
    
    print()
    print('Testing')
    print()
    
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target_label in target_test_loader:
            data, target_label = data.to(DEVICE), target_label.to(DEVICE)
            
            s_output = model.predict(data)
            
            loss = criterion(s_output, target_label)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]

            correct += torch.sum(pred == target_label)
    
    print('{} {} --> {}: correct: {}, accuracy{: .2f}%\n'.format(
        CFG['backbone'], source_name, target_name, correct, 100. * correct / len_target_dataset))
    
    return 1. * correct / len_target_dataset


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    '''training phase'''
    
    max_acc = 0.0
    optimal_model = None
    
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    
    for e in range(args.epochs):
        
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        for i in range(n_batch):
            
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CFG['lambda'] * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CFG['log_interval'] == 0:
                print(
                    'Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                        e + 1,
                        args.epochs,
                        int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        
        # Test
        acc = test(model, target_test_loader)
        if acc > max_acc:
            max_acc = acc
            optimal_model = model
            model_path = os.path.join('./models_office_caltech/', '{0}_{1}_{2}.model.pt'.format(
                         CFG['backbone'], CFG['source_name'], CFG['target_name']))
            torch.save(optimal_model.state_dict(), model_path)

        print('{} {} --> {}: max accuracy{: .2f}%\n'.format(CFG['backbone'], source_name, target_name, 100 * max_acc))

    return max_acc, optimal_model, model_path


def load_data(src, tar, root_dir):
    '''Train data and test data initialization'''


    folder_src = os.path.join(root_dir, src)
    folder_tar = os.path.join(root_dir, tar)

    source_loader = data_loader.load_data(folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(folder_tar, CFG['batch_size'], True, CFG['kwargs'])
    target_test_loader = data_loader.load_data(folder_tar, CFG['batch_size'], False, CFG['kwargs'])

    print('source len: {0}, target train len: {1}, target test len: {2}'.format(
        len(source_loader.dataset), len(target_train_loader.dataset), len(target_test_loader.dataset)))

    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':

    
    CFG['backbone'] = args.backbone

    source_name = os.path.join(CFG['source_name'], 'train')
    target_name = os.path.join(CFG['target_name'], 'test')

    print('Src: %s, Tar: %s' % (source_name, target_name))
    source_loader, target_train_loader, target_test_loader = load_data(source_name, target_name, CFG['data_path'])

    model = Transfer_Net(CFG['n_class'], transfer_loss='mmd', base_net=CFG['backbone']).to(DEVICE)
    
    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])
    
    max_acc, optimal_model, model_path = train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG)
    
    print("max accuracy in target domain: {0}".format(max_acc))
    
    model = Transfer_Net(CFG['n_class'], transfer_loss='mmd', base_net=CFG['backbone']).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    print(model)
    acc = test(model, target_test_loader)

