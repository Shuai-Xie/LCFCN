import matplotlib

matplotlib.use('Agg')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import applyOnImage
import experiments, train, test, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default="trancos")
    parser.add_argument('-m', '--mode', default=None)
    parser.add_argument('--image_path', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('-r', '--reset', action="store_const", const=True, default=False,
                        help="If set, a new model will be created, overwriting any previous version.")

    args = parser.parse_args()

    dataset_name, model_name, metric_name = experiments.get_experiment(args.exp_name)

    # Paths
    name = "{}_{}".format(dataset_name, model_name)  # pascal_ResFCN

    path_model = "checkpoints/model_{}.pth".format(name)
    path_opt = "checkpoints/opt_{}.pth".format(name)

    # best model and history
    path_best_model = "checkpoints/best_model_{}.pth".format(name)
    path_history = "checkpoints/history_{}.json".format(name)

    # demo
    if args.image_path is not None:
        applyOnImage.apply(args.image_path, args.model_name, args.model_path)

    elif args.mode == "train":
        train.train(dataset_name, model_name, metric_name, path_history, path_model, path_opt, path_best_model, args.reset)

    elif args.mode == "test":
        test.test(dataset_name, model_name, metric_name, path_history, path_best_model)

    elif args.mode == "summary":
        summary.summary(dataset_name, model_name, path_history)


if __name__ == "__main__":
    main()
