import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ray import tune 
from ray.air import session, RunConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray

import sys, getopt
import math

from scores import accuracy, precision, recall, f1_score, kappa_score
from datasets import Malware14, get_dataloader
from models import VGG, ResNet
from train_model import test_model_performance

NUM_EPOCHS = 23

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

@ray.remote 
class GlobalBestConfig:
    def __init__(self):
        self.best_config = {}
    def set_config(self, config):
        self.best_config = config
    def get_config(self):
        return self.best_config

@ray.remote
class GlobalBestLoss:
    def __init__(self):
        self.best_loss = math.inf
    def set_loss(self, loss):
        self.best_loss = loss
    def get_loss(self):
        return self.best_loss

def optimize_model(hyperparameter_config: dict, train_dataset: Dataset, val_dataset: Dataset, num_classes: int, model_type, over_sampling, save_dir_best_result, global_best_loss, global_best_config):
    torch.cuda.empty_cache()
    batch_size = hyperparameter_config["batch_size"]

    val_size = len(val_dataset)
    train_data_loader = get_dataloader(dataset=train_dataset, batch_size=batch_size, over_sampling=over_sampling)
    val_data_loader = get_dataloader(dataset=val_dataset, batch_size=batch_size)

    output_channels = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    pool_positions = [1, 3, 7, 11, 15]

    if hyperparameter_config["last_pooling"] == "avg_pool":
        last_pooling = nn.AdaptiveAvgPool2d
    else:
        last_pooling = nn.AdaptiveMaxPool2d

    if model_type == "notop":
        model = VGG(
            num_classes=num_classes,
            output_channels=output_channels,
            pool_positions=pool_positions,
            last_pooling=last_pooling
        )
    elif model_type == "fc":
        model = VGG(
            num_classes=num_classes,
            output_channels=output_channels,
            pool_positions=pool_positions,
            use_fc_layers=True,
            dropout=hyperparameter_config["dropout"],
            last_pooling=last_pooling
        )
    elif m_type == "resnet":
        model = ResNet(num_classes=num_classes, last_pooling=last_pooling)
    elif m_type == "next":
        model = ResNet(num_classes=num_classes, last_pooling=last_pooling, resnext=True)

    model = nn.DataParallel(model)
    model = model.to(device)

    loss_criterion = nn.CrossEntropyLoss()
    if hyperparameter_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter_config["lr"], betas=hyperparameter_config["betas"], weight_decay=hyperparameter_config["weight_decay"])
    elif hyperparameter_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameter_config["lr"], weight_decay=hyperparameter_config["weight_decay"], momentum=0.9)

    # Training
    batch_print_interval = int(len(train_data_loader) / 4)
    adapted_learning_rate = False
    for i in range(NUM_EPOCHS): 
        model.train()
        train_losses = []
        running_loss = 0.
        running_n_instances = 0
        for j, batch in enumerate(train_data_loader):
            X = batch[0].to(device)
            Y = batch[1].to(device)
            output = model(X)
            loss = loss_criterion(output, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*X.size(0)
            running_n_instances += X.size(0)

            if j == len(train_data_loader)-1 or (j % batch_print_interval == 0 and j!= 0):
                print(f"epoch {i} batch {j} current running training loss: {running_loss / running_n_instances}")
                train_losses.append(running_loss / running_n_instances)
                running_loss = 0.
                running_n_instances = 0
        
        # Evaluation on validation set
        total_loss, total_accuracy, macro_recall, macro_precision, macro_f1, kappa, *_ = test_model_performance(model, val_data_loader, val_size, num_classes, batch_size, loss_criterion)

        if not adapted_learning_rate and total_accuracy > 0.93:
            adapted_learning_rate = True
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 10

        if save_dir_best_result and (total_loss < ray.get(global_best_loss.get_loss.remote())):
            ray.get(global_best_loss.set_loss.remote(total_loss))
            ray.get(global_best_config.set_config.remote(hyperparameter_config))
            torch.save(model.state_dict(), save_dir + "best-model.pt")
            torch.save(optimizer.state_dict(), save_dir + "best-optimizer.pt")

        session.report({"loss": total_loss, "accuracy": total_accuracy, "precision": macro_precision, "recall": macro_recall, "macro-f1": macro_f1, "kappa": kappa, "training-losses": train_losses})
                    
if __name__ == "__main__":

    argv = sys.argv[1:]
    over_sampling=False
    save_dir = None

    try:
      opts, args = getopt.getopt(argv,"m:p:s:o",["model_type=","preprocessing_type=","save_dir_best_result="])
    except getopt.GetoptError:
      print('hyperparameter_optimization.py -m <model_type> -p <preprocessing_type> -s <save_dir_best_result>')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("-m", "--model_type"):
         m_type = arg
      elif opt in ("-p", "--preprocessing_type"):
        preprocessing_type = arg
      elif opt in ("-s", "--save_dir_best_result"):
        save_dir = arg
      elif opt == "-o":
        over_sampling=True

    train_dataset = Malware14(preprocessing_type=preprocessing_type)
    val_dataset = Malware14(set_type="Val", preprocessing_type=preprocessing_type)
    test_dataset = Malware14(set_type="Test", preprocessing_type=preprocessing_type)
    test_size = len(test_dataset)

    print(f"Done initializing Dataset with {len(train_dataset.classes)} classes")

    print(f"Optimizing for model type {m_type}")
    print(f"Train oversampling: {over_sampling}")
    print(f"Best resulting model will be saved in {save_dir}")

    global_best_loss = GlobalBestLoss.remote()
    global_best_config = GlobalBestConfig.remote()

    if m_type == "notop":
        hyperparameter_config = {
        "lr": tune.choice([0.01, 0.001, 0.0001]),
        "batch_size": tune.choice([32, 64]),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "betas": tune.choice([(0.9, 0.999), (0.5, 0.999), (0.4, 0.9)]),
        "weight_decay": tune.choice([0.0, 0.01, 0.001, 0.0001]),
        "last_pooling": tune.choice(["avg_pool", "max_pool"])
        }
    elif m_type == "fc":
        hyperparameter_config = {
        "lr": tune.choice([0.01, 0.001, 0.0001]),
        "batch_size": tune.choice([32, 64]),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "betas": tune.choice([(0.9, 0.999), (0.5, 0.999), (0.4, 0.9)]),
        "weight_decay": tune.choice([0.0, 0.01, 0.001, 0.0001]),
        "dropout": tune.choice([0.0, 0.35, 0.5, 0.7]),
        "last_pooling": tune.choice(["avg_pool", "max_pool"])
        }
    else:
        # Resnet or ResneXt
        hyperparameter_config = {
        "lr": tune.choice([0.01, 0.001, 0.0001]),
        "batch_size": tune.choice([32, 64]),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "betas": tune.choice([(0.9, 0.999), (0.5, 0.999), (0.4, 0.9)]),
        "weight_decay": tune.choice([0.0, 0.01, 0.0001]),
        "last_pooling": tune.choice(["avg_pool", "max_pool"])
        }

    scheduler = ASHAScheduler(
        max_t=23,
        grace_period=4,
        reduction_factor=2
    )
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "macro-f1", "kappa", "training-losses", "training_iteration"],
        max_report_frequency=180
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(optimize_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset, 
            num_classes=len(train_dataset.classes),
            model_type = m_type,
            over_sampling = over_sampling,
            save_dir_best_result=save_dir,
            global_best_loss=global_best_loss,
            global_best_config=global_best_config
            ),
            resources={"cpu": 1, "gpu": 4},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=25,
        ),
        run_config=RunConfig(
            progress_reporter=reporter,
            ),
        param_space=hyperparameter_config,
    )

    result = tuner.fit()

    best_result = result.get_best_result("loss", "min")
    best_config = ray.get(global_best_config.get_config.remote())
    print(f"Model reported as best model at any epoch had config: {best_config}")
    print(f"Model reported as best model at any epoch had loss: {ray.get(global_best_loss.get_loss.remote())}")
    print("Best trial config at the end of all epochs: {}".format(best_result.config))
    print("Best trial final validation loss at the end of all epochs: {}".format(best_result.metrics["loss"]))
    print("Best trial final validation accuracy at the end of all epochs: {}".format(best_result.metrics["accuracy"]))
    print("Best trial final validation macro-f1 at the end of all epochs: {}".format(best_result.metrics["macro-f1"]))
    print("Best trial final validation kappa at the end of all epochs: {}".format(best_result.metrics["kappa"]))
    
    print("Testing performance of best model on testset")
    test_data_loader = get_dataloader(dataset=test_dataset, batch_size=best_config["batch_size"])

    if best_config["last_pooling"] == "avg_pool":
        last_pooling = nn.AdaptiveAvgPool2d
    else:
        last_pooling = nn.AdaptiveMaxPool2d
    
    if m_type == "fc":
        model = VGG(
            num_classes=len(train_dataset.classes),
            output_channels=[64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
            pool_positions=[1, 3, 7, 11, 15],
            use_fc_layers=True,
            dropout=best_config["dropout"],
            last_pooling=last_pooling
        )
    elif m_type == "notop":
        model = VGG(
            num_classes=len(train_dataset.classes),
            output_channels=[64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
            pool_positions=[1, 3, 7, 11, 15],
            use_fc_layers=False,
            last_pooling=last_pooling
        )
    elif m_type == "resnet":
        model = ResNet(num_classes=len(train_dataset.classes), last_pooling=last_pooling)
    elif m_type == "next":
        model = ResNet(num_classes=len(train_dataset.classes), resnext=True, last_pooling=last_pooling)
    
    model = nn.DataParallel(model)
    model = model.to(device)

    model.load_state_dict(torch.load("best-model.pt"))
    total_loss, total_accuracy, macro_recall, macro_precision, macro_f1, kappa, wrong_predictions, *_ = test_model_performance(model, test_data_loader, test_size, len(train_dataset.classes), best_config["batch_size"], nn.CrossEntropyLoss(), return_stats=False)
    print(f"Loss on the test set: {total_loss}")
    print(f"Accuracy on the test set: {total_accuracy}")
    print(f"Macro recall on the test set: {macro_recall}")
    print(f"Macro precision on the test set: {macro_precision}")
    print(f"Macro F1-Score on the test set: {macro_f1}")
    print(f"Kappa-Score on the test set: {kappa}")
    print()

    




    



