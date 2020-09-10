def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    ids = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
            ids[name.strip('\n')] = ID
    return names, ids


# common
debug = False
classes_file = '/media/data_it/thiennt/centernet/datasets/id.names'
id2class, class2id = read_class_names(classes_file)

n_classes = len(id2class)

input_image_h = 256
input_image_w = 256
down_ratio = 4

output_h = input_image_h // down_ratio
output_w = input_image_w // down_ratio

n_kps = 4


max_objs = 10


# model
backbone = {
    'type': 'resnet18'
}

heads = {
    'hm': n_classes,
    'offset': 2,
    'wh': 2,

    'kp_hm': n_kps,
    'kp_offset': 2,
    'kps': n_kps * 2
}

weight_loss = {
    "hm": 1,
    "wh": 1,
    "offset": 1,
    "kp_hm": 1,
    "kp_offset": 1,
    "kps": 1
}

dataset_name = "id"

# train
train_data_file = "/media/data_it/thiennt/centernet/datasets/test/id_train.txt"
use_aug = False
batch_size = 8
epochs = 3

# learning rate
lr_type = "piecewise"  # "exponential","piecewise","CosineAnnealing"
lr = 1e-3  # exponential
lr_decay_steps = 5000  # exponential
lr_decay_rate = 0.95  # exponential
lr_boundaries = [40000, 60000]  # piecewise
lr_piecewise = [0.0001, 0.00001, 0.000001]  # piecewise
warm_up_epochs = 2  # CosineAnnealing
init_lr = 1e-4  # CosineAnnealing
end_lr = 1e-6  # CosineAnnealing
pre_train = True


# test
test_data_file = '/media/data_it/thiennt/centernet/datasets/test/id_test.txt'
score_threshold = 0.3
use_nms = True
nms_thresh = 0.4