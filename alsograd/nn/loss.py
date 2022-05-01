from alsograd.core import Parameter


def MSE(y_true: Parameter, y_pred: Parameter) -> Parameter:
    return ((y_true - y_pred)**2).mean()
