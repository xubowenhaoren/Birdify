import matplotlib.pyplot as plt


log_folder_location = "logs/0.9_per_epoch_decay/"
file_name = "efficient_net_cv_log_no_pretrain.txt"


def parse_logs():
    with open(log_folder_location + file_name) as log_file:
        lines = log_file.read().splitlines()
    # print(lines)

    train_acc_logs = []
    validation_acc_logs = []
    train_loss_logs = []
    validation_loss_logs = []
    max_epoch = 0
    for line in lines:
        info_arr = line.split(" ")
        epoch_num = int(info_arr[1])
        epoch_type = info_arr[2]
        accuracy = float(info_arr[4])
        loss = float(info_arr[-1])
        # print("epoch", epoch_num, "epoch type", epoch_type, "acc", accuracy, "loss", loss)
        max_epoch = max(epoch_num, max_epoch)
        if epoch_type == "train":
            train_acc_logs.append((epoch_num, epoch_type, accuracy))
            train_loss_logs.append((epoch_num, epoch_type, loss))
        else:
            validation_acc_logs.append((epoch_num, epoch_type, accuracy))
            validation_loss_logs.append((epoch_num, epoch_type, loss))
    # print(len(train_logs), train_logs)
    # print(len(validation_logs), validation_logs)
    return train_acc_logs, validation_acc_logs, train_loss_logs, validation_loss_logs, max_epoch


def plot_helper(data_arr, max_epoch, title, y_label):
    plt.figure(figsize=(10, 7))
    for data in data_arr:
        log_arr, style, m_label = data
        if len(log_arr) > 0:
            plt.plot([log[2] for log in log_arr], style, label=m_label)
    plt.title(title)
    plt.xticks(range(0, max_epoch + 1, 1))
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig("graphs/" + title + ".png")
    plt.show()


if __name__ == '__main__':
    train_acc_logs, validation_acc_logs, train_loss_logs, validation_loss_logs, max_epoch = parse_logs()
    data_arr = [(train_acc_logs, "-o", "train"), (validation_acc_logs, "--o", "validation")]
    plot_helper(data_arr, max_epoch, "EfficientNet with no pre-trained weights: Accuracy", "Accuracy")

    data_arr = [(train_loss_logs, "-o", "train"), (validation_loss_logs, "--o", "validation")]
    plot_helper(data_arr, max_epoch, "EfficientNet with no pre-trained weights: Loss", "Loss")
