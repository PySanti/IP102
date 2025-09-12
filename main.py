from utils.load_set import load_set
from utils.normalization_metrics_calc import normalization_metrics_calc

"""
means = [0.5058171864589405, 0.5286897495022889, 0.37927134012176345]
stds = [0.26876003294998324, 0.25613632070425424, 0.285797550958417]

"""

if __name__ == "__main__":
    train_X_paths, train_Y = load_set("./archive/classification","train" )
    val_X_paths, val_Y = load_set("./archive/classification","val" )
    test_X_paths, test_Y = load_set("./archive/classification","test" )
    print(len(train_X_paths))
    print(len(val_X_paths))
    print(len(test_X_paths))
    means, stds = normalization_metrics_calc(train_X_paths)

