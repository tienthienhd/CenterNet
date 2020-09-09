

backbone = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
}


def get_backbone(input, type):
    network = backbone[type]
    return network(include_top=False, weights=None, input_tensor=input)
