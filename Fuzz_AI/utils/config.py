CFG = {
    'data_path': '/home/leohoo/work/code/tlfuzz/datasets/office_caltech/',
    'kwargs': {'num_workers': 1},
    'batch_size': 32,
    'epoch': 100,
    'lr': 1e-3,
    'momentum': .9,
    'log_interval': 10,
    'l2_decay': 0,
    'lambda': 10,
    'backbone': 'resnet50',  # alexnet, resnet18, resnet34, resnet50, resnet101, resnet152
    'n_class': 10,
    'source_name': "amazon",  # Office31: amazon, webcam, dslr; OfficeHome: Art, Clipart, Product, RealWorld
    'target_name': "caltech",
}

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

