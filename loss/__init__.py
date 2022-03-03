from loss import losses


def create_loss(name, model=None, weight_decay=None, p=2):
    if name == 'Regularization':
        return losses.__dict__[name](model, weight_decay, p)
    return losses.__dict__[name]()