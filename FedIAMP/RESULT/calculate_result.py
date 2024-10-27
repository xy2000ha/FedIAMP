from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score


def count_error(pred, label):
    print('pred:', pred)
    print('label:', label)
    print('mae:', mean_absolute_error(label, pred),
          '\nmse:', mean_squared_error(label, pred),
          '\nrmse', sqrt(mean_squared_error(label, pred)),
          '\nmape', mean_absolute_percentage_error(label, pred),
          '\nr2:', r2_score(label, pred))
