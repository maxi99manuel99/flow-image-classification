import torch
import torch.nn as nn
import sys, getopt
import numpy as np
import pandas as pd
import json

from torchmetrics.classification import BinaryROC, MulticlassROC
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix

from scores import accuracy, TP_FP_TN_FN, precision, recall, f1_score, kappa_score
from datasets import Malware14, MalwareBinary, get_dataloader
from models import VGG, ResNet

class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.1):
        self.patience = patience 
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
    
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter == self.patience:
                return True
        return False        

#Hyperparameters

OPTIMAL_HYPEPARAMETERS = {
    "notop": {
        "optimizer": "SGD",
        "batch_size": 32,
        "lr": 0.001,
        "last_pooling": nn.AdaptiveMaxPool2d,
        "weight_decay": 0.0001
    },
    "fc": {
        "optimizer": "SGD",
        "batch_size": 32,
        "lr": 0.001,
        "dropout": 0.5,
        "weight_decay": 0.01,
        "last_pooling": nn.AdaptiveMaxPool2d
    },
    "resnet": {
        "optimizer": "SGD",
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.01,
        "last_pooling": nn.AdaptiveMaxPool2d
    },
    "next": {
        "optimizer": "Adam",
        "batch_size": 32,
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 0.0001,
        "last_pooling": nn.AdaptiveMaxPool2d
    }

}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

def test_model_performance(model, test_data_loader, test_size, num_classes, batch_size, loss_criterion, return_stats=False):
    with torch.no_grad():
        wrong_predictions = []
        if num_classes > 2:
            model_outputs = torch.zeros((test_size, num_classes))
        else:
            model_outputs = torch.zeros((test_size, 1))
        confusion_matrix = []
        roc = []
        label_prediction_dist = {}
        model.eval()
        predicted_labels = torch.zeros(test_size, dtype=torch.int)
        true_labels = torch.zeros(test_size, dtype=torch.int)
        test_loss = 0.
        test_steps = 0
        for z, (test_X, test_Y) in enumerate(test_data_loader):
            test_X = test_X.to(device)
            test_Y = test_Y.to(device)
            test_output = model(test_X)
            model_outputs[z*batch_size:z*batch_size+len(test_X)] = test_output
            
            if num_classes == 2:
                # BCELossWithLogits needs targets to be float
                test_Y = test_Y.float().unsqueeze(1)
            
            loss = loss_criterion(test_output, test_Y)
            if num_classes > 2:
                test_prediction = nn.functional.softmax(test_output, dim=1)
                predicted_labels[z*batch_size:z*batch_size+len(test_X)] = torch.argmax(test_prediction, dim=1)
            else:
                predicted_labels[z*batch_size:z*batch_size+len(test_X)] = torch.Tensor([1 if logit >= 0.0 else 0 for logit in test_output.view(test_output.size(0))])

            true_labels[z*batch_size:z*batch_size+len(test_X)] = test_Y.view(test_Y.size(0))
            
            # stats that we only want at the end of training
            if return_stats:
                wrong_indices = predicted_labels[z*batch_size:z*batch_size+len(test_X)] != true_labels[z*batch_size:z*batch_size+len(test_X)]
                for idx, wrong in enumerate(wrong_indices):
                    if wrong:
                        wrong_predictions.append((true_labels[z*batch_size:z*batch_size+len(test_X)][idx].item(), predicted_labels[z*batch_size:z*batch_size+len(test_X)][idx].item()))

            test_loss += loss.item()
            test_steps += 1

            del test_X, test_Y
        
        # stats that we only want at the end of training
        if return_stats:
            # ROC
            if num_classes > 2:
                roc_metric = MulticlassROC(num_classes=num_classes)
            else:
                roc_metric = BinaryROC()
                model_outputs = model_outputs.view(model_outputs.size(0))
            roc = roc_metric(preds=model_outputs, target=true_labels)
            
            # label_prediction_distribution
            if num_classes > 2:
                all_output_activations = nn.functional.softmax(model_outputs, dim=1)
            else:
                all_output_activations = torch.sigmoid(model_outputs) 
            for t, label in enumerate(true_labels):
                label = label.item()
                if label not in label_prediction_dist.keys():
                    label_prediction_dist[label] = [all_output_activations[t].tolist()]
                else:
                    label_prediction_dist[label].append(all_output_activations[t].tolist())
            
            # confusion matrix
            if num_classes > 2:
                conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
            else:
                conf_matrix = BinaryConfusionMatrix()

            confusion_matrix = conf_matrix(preds=model_outputs, target=true_labels)
        
        macro_recall = 0
        macro_precision = 0
        macro_f1 = 0 
        if num_classes == 2:
            # if we are dealing with binary classification
            # we need no macro average
            all_labels = [1]
        else:
            all_labels = true_labels.unique()
        n_labels = len(all_labels)
        f1_per_label = []
        recall_per_label = []
        precision_per_label = []
        for label in all_labels:
            tp, fp, tn, fn = TP_FP_TN_FN(predicted_labels, true_labels, label)
            label_recall = recall(tp, fn)
            label_precision = precision(tp, fp)
            label_f1 = f1_score(label_precision, label_recall)
            if return_stats:
                recall_per_label.append(label_recall)
                precision_per_label.append(label_precision)
                f1_per_label.append(label_f1)
            macro_recall += label_recall
            macro_precision += label_precision
            macro_f1 += label_f1
        
        kappa = kappa_score(predicted_labels, true_labels)
        macro_recall = macro_recall / n_labels
        macro_precision = macro_precision / n_labels
        macro_f1 = macro_f1 / n_labels
        total_loss = test_loss / test_steps
        total_accuracy = accuracy(predicted_labels, true_labels)

        return total_loss, total_accuracy, macro_recall, macro_precision, macro_f1, kappa, wrong_predictions, roc, label_prediction_dist, confusion_matrix, recall_per_label, precision_per_label, f1_per_label

if __name__ == "__main__":

    argv = sys.argv[1:]
    over_sampling=False
    model_path = None
    optimizer_path = None
    learning_rate = None
    starting_epoch = 1
    adaptive_learning_rate = None
    NUM_EPOCHS = 35


    try:
      opts, args = getopt.getopt(argv,"d:m:p:s:t:l:e:n:oa",["dataset_type=","model_type=","preprocessing_type=","saved_model=","training_optimizer=","learning_rate=","epoch=","num-epochs="])
    except getopt.GetoptError:
      print('train_model.py -d <dataset_type> -m <model_type> -p <preprocessing_type>')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("-d", "--dataset_type"):
         d_type = arg
      elif opt in ("-m", "--model_type"):
         m_type = arg
      elif opt in ("-p", "--preprocessing_type"):
        preprocessing_type = arg
      elif opt in ("-s", "--saved_model"):
        model_path = arg
      elif opt in ("-t", "--training_optimizer"):
        optimizer_path = arg
      elif opt in ("-l", "--learning_rate"):
        learning_rate = arg
      elif opt in ("-e", "--starting-epoch"):
        starting_epoch = int(arg)
      elif opt in ("-n", "--num-epochs"):
        NUM_EPOCHS = int(arg)
      elif opt == "-o":
        over_sampling=True
      elif opt == "-a":
        adaptive_learning_rate = True

    torch.cuda.empty_cache()

    if d_type == "binary":
        print("Initializing binary Dataset with preprocessing type: " + preprocessing_type)
        train_dataset = MalwareBinary(preprocessing_type=preprocessing_type)
        val_dataset = MalwareBinary(set_type="Val", preprocessing_type=preprocessing_type)
        test_dataset = MalwareBinary(set_type="Test", preprocessing_type=preprocessing_type)
    elif d_type == "multiclass":
        print("Initialzing multiclass Dataset with preprocessing type: " + preprocessing_type)
        train_dataset = Malware14(preprocessing_type=preprocessing_type)
        val_dataset = Malware14(set_type="Val", preprocessing_type=preprocessing_type)
        test_dataset = Malware14(set_type="Test", preprocessing_type=preprocessing_type)

    num_classes = len(train_dataset.classes)
    print(f"Done initializing Dataset with classes: {train_dataset.classes}")

    # VGG output channels and max pool positions
    output_channels = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    pool_positions = [1, 3, 7, 11, 15]

    print(f"Initialising model with type {m_type}")

    last_pooling = OPTIMAL_HYPEPARAMETERS[m_type]["last_pooling"]
    if m_type == "fc":
        dropout = OPTIMAL_HYPEPARAMETERS["fc"]["dropout"]
        model = VGG(
            num_classes=num_classes,
            output_channels=output_channels,
            pool_positions=pool_positions,
            use_fc_layers=True,
            dropout=dropout,
            last_pooling=last_pooling
        )
    elif m_type == "notop":
        model = VGG(
            num_classes=num_classes,
            output_channels=output_channels,
            pool_positions=pool_positions,
            use_fc_layers=False,
            last_pooling=last_pooling
        )
    elif m_type == "resnet":
        model = ResNet(num_classes=num_classes, last_pooling=last_pooling)
    elif m_type == "next":
        model = ResNet(num_classes=num_classes, resnext=True, last_pooling=last_pooling)

    model = nn.DataParallel(model)
    model = model.to(device)
    print("Done initializing model")

    early_stopper = EarlyStopper(patience = 3, min_delta=0.01)
    
    optimal_hyperparameters = OPTIMAL_HYPEPARAMETERS[m_type]
    optimizer_name = optimal_hyperparameters["optimizer"]
    print(f"Using optimizer: {optimizer_name}")
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimal_hyperparameters["lr"], betas=optimal_hyperparameters["betas"], weight_decay=optimal_hyperparameters["weight_decay"])
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=optimal_hyperparameters["lr"], weight_decay=optimal_hyperparameters["weight_decay"], momentum=0.9)

    BATCH_SIZE = optimal_hyperparameters["batch_size"]
    
    if num_classes == 2:
        loss_criterion = nn.BCEWithLogitsLoss()
    else:
        loss_criterion = nn.CrossEntropyLoss()

    print(f"Initializing Dataloader (train oversampling={over_sampling})")
    train_data_loader = get_dataloader(dataset=train_dataset, batch_size=BATCH_SIZE, over_sampling=over_sampling)
    val_data_loader = get_dataloader(dataset=val_dataset, batch_size=BATCH_SIZE)
    test_data_loader = get_dataloader(dataset=test_dataset, batch_size=BATCH_SIZE)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    print("Done Initializing Dataloader")


    # If we are using an already trained model
    if model_path:
        model.load_state_dict(torch.load(model_path))
        # if we do not get a better model during training save the first model
        # as best-model
        torch.save(model.state_dict(), "best-model.pt")
        print(f"Using already trained model loaded from: {model_path}")
        print(f"The model was trained {starting_epoch-1} epochs already")
        print("Testing performance at the current state:")
        total_loss, total_accuracy, macro_recall, macro_precision, macro_f1, kappa, *_ = test_model_performance(model, val_data_loader, val_size, num_classes, BATCH_SIZE, loss_criterion)
        print(f"Loss on the validation set: {total_loss}")
        print(f"Accuracy on the validation set: {total_accuracy}")
        print(f"Macro recall on the validation: {macro_recall}")
        print(f"Macro precision on the validation set: {macro_precision}")
        print(f"Macro F1-Score on the validation set: {macro_f1}")
        print(f"Kappa-Score on the validation set: {kappa}")
        print()
        early_stopper.min_validation_loss = total_loss

    # If we are using an saved optimizer to resume training
    if optimizer_path:
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("Using the old optimizer to continue training")

    #if we want to adjust the learning rate and not use the one that was optimal after 10 epochs
    if learning_rate:
        for g in optimizer.param_groups:
            g['lr'] = float(learning_rate) 
    print(f"Learning rate that will be used: {optimizer.param_groups[0]['lr']}")
    if adaptive_learning_rate:
        print("If validation accuracy reaches 93% lr will be deivded by 10")
    
    # Training
    batch_print_interval = int(len(train_data_loader) / 4)
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    all_val_recalls = []
    all_val_precisions = []
    all_val_f1_scores = []
    all_val_kappa_scores = []
    print(f"Starting training. Training will last at maximum {NUM_EPOCHS} epochs")
    adapted_lr = False
    for i in range(starting_epoch, starting_epoch+NUM_EPOCHS):
        model.train() 
        train_losses = []
        running_loss = 0.
        steps = 0
        for j, batch in enumerate(train_data_loader):
            X = batch[0].to(device)
            Y = batch[1].to(device)
            
            if num_classes == 2:
                # BCELossWithLogits needs targets to be float
                # also target needs to be of shape batch x 1
                Y = Y.float().unsqueeze(1)
            
            output = model(X)
            loss = loss_criterion(output, Y)
            del X, Y
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if j == len(train_data_loader)-1 or (j % batch_print_interval == 0 and j!= 0):
                print(f"epoch {i} batch {j} current running training loss: {running_loss / steps}")
                train_losses.append(running_loss / steps)
                running_loss = 0.
                steps = 0

        print("Testing performance on validation set")
        
        total_loss, total_accuracy, macro_recall, macro_precision, macro_f1, kappa, *_ = test_model_performance(model, val_data_loader, val_size, num_classes, BATCH_SIZE, loss_criterion)
        all_train_losses.append(train_losses)
        all_val_losses.append(total_loss)
        all_val_accuracies.append(total_accuracy)
        all_val_recalls.append(macro_recall)
        all_val_precisions.append(macro_precision)
        all_val_f1_scores.append(macro_f1)
        all_val_kappa_scores.append(kappa)
        print(f"Loss on the validation set after epoch {i} batch {j}: {total_loss}")
        print(f"Accuracy on the validation set after epoch {i} batch {j}: {total_accuracy}")
        print(f"Macro recall on the validation set after epoch {i} batch {j}: {macro_recall}")
        print(f"Macro precision on the validation set after epoch {i} batch {j}: {macro_precision}")
        print(f"Macro F1-Score on the validation set after epoch {i} batch {j}: {macro_f1}")
        print(f"Kappa-Score on the validation set after epoch {i} batch {j}: {kappa}")
        print()
        torch.cuda.empty_cache()
        if total_loss < early_stopper.min_validation_loss:
            torch.save(model.state_dict(), "best-model.pt")
            torch.save(optimizer.state_dict(), "best-optimizer.pt")
        if early_stopper.early_stop(total_loss):
            print("Stopping training because the validation loss is not improving")
            break
        if adaptive_learning_rate and not adapted_lr and total_accuracy > 0.93:
            adapted_lr = True
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 10

    scores_dict = {"loss": all_val_losses, "accuracy": all_val_accuracies, "recall": all_val_recalls, "precision": all_val_precisions, "macro-f1": all_val_f1_scores, "kappa": all_val_kappa_scores, "training-losses": all_train_losses}
    df = pd.DataFrame(data = scores_dict)
    df.to_csv('scores' + str(starting_epoch) + '.csv', index=None)

    bonus_test_scores = {
        "Wrong_prediction_classes": None,
        "Label_prediction_distribution": None,
        "Recall_per_label": None,
        "Precision_per_label": None,
        "F1_per_label": None,
    }
    print("Measuring performance of the current model at the end of trainign")
    total_loss, total_accuracy, macro_recall, macro_precision, macro_f1, kappa, *_ = test_model_performance(model, test_data_loader, test_size, num_classes, BATCH_SIZE, loss_criterion, return_stats=True)
    print(f"Loss on the test set: {total_loss}")
    print(f"Accuracy on the test set: {total_accuracy}")
    print(f"Macro recall on the test set: {macro_recall}")
    print(f"Macro precision on the test set: {macro_precision}")
    print(f"Macro F1-Score on the test-set: {macro_f1}")
    print(f"Kappa-Score on the test set: {kappa}")
    print()

    print("Measuring performance of the best model on the testset after training")
    model.load_state_dict(torch.load("best-model.pt"))
    total_loss, total_accuracy, macro_recall, macro_precision, macro_f1, kappa, wrong_predictions, roc, label_prediction_dist, confusion_matrix, recall_per_label, precision_per_label, f1_per_label  = test_model_performance(model, test_data_loader, test_size, num_classes, BATCH_SIZE, loss_criterion, return_stats=True)
    print(f"Loss on the test set: {total_loss}")
    print(f"Accuracy on the test set: {total_accuracy}")
    print(f"Macro recall on the test set: {macro_recall}")
    print(f"Macro precision on the test set: {macro_precision}")
    print(f"Macro F1-Score on the test set: {macro_f1}")
    print(f"Kappa-Score on the test set: {kappa}")
    print()
    bonus_test_scores["Wrong_prediction_classes"] = wrong_predictions
    bonus_test_scores["Label_prediction_distribution"] = label_prediction_dist
    bonus_test_scores["Recall_per_label"] = recall_per_label
    bonus_test_scores["Precision_per_label"] = precision_per_label
    bonus_test_scores["F1_per_label"] = f1_per_label
    with open("bonus_scores.json", "w") as fp:
        json.dump(bonus_test_scores, fp)
    torch.save(roc, "roc.pt")
    torch.save(confusion_matrix, "confusion_matrix.pt")



    




    



