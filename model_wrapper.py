from __future__ import absolute_import, division, print_function


def get_model(name):
    if name == 'vgg_16':
        from nets.mobilenet import model, forward
    if name == 'MobilenetV1':
        from nets.mobilenet import model, forward
    else:
        raise AttributeError

    return model, forward
