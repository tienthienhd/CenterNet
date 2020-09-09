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
debug = True
classes_file = '/media/data_it/thiennt/cv_end_to_end/training/object_detection/centernet/datasets/id.names'
id2class, class2id = read_class_names(classes_file)

n_classes = len(id2class)

input_image_h = 512
input_image_w = 512
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

    'hm_kp': n_kps,
    'kp_offset': 2,
    'kps': n_kps * 2
}

weight_loss = {
    "hm": 1,
    "wh": 1,
    "offset": 1,
    "hm_kp": 1,
    "kp_offset": 1,
    "kps": 1
}

# train
train_data_file = "/media/data_it/thiennt/cv_end_to_end/training/object_detection/centernet/datasets/id_train.txt"
batch_size = 16
epochs = 80

