from config.train_config import parse_train_configs
import matplotlib.pyplot as plt


def plot(training, validation):
    # config = parse_train_configs()
    # epochs = [i for i in range(1, config.num_epochs)]
    epochs = [i for i in range(1, 189)]

    # Plot training and validation errors
    plt.plot(epochs, training, 'b', label='Training Loss')
    plt.plot(epochs, validation, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


class LossComputation:

    def __init__(self, training_path, validation_path):
        self.training_loss = training_path
        self.validation_loss = validation_path

    def load_file(self):
        training_data = []
        validation_data = []

        # Reading training errors from file
        with open(self.training_loss, 'r') as file:
            for line in file:
                training_data.append(float(line.strip()))

        # Reading validation errors from file
        with open(self.validation_loss, 'r') as file:
            for line in file:
                validation_data.append(float(line.strip()))
        # print(training_data, validation_data)
        return training_data, validation_data


if __name__ == '__main__':
    training_errors = 'D:/downloads_d/Jnaved/plot/training_loss.txt'
    validation_errors = 'D:/downloads_d/Jnaved/plot/validation_loss.txt'
    # Writing errors to a file
    loss = LossComputation(training_errors, validation_errors)
    training_file, validation_file = loss.load_file()
    plot(training_file, validation_file)
