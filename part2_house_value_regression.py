import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from part1_nn_lib import Trainer, MultiLayerNetwork


class Regressor():

    def __init__(self,
                 x,
                 nb_epoch=20,
                 neurons=[16, 64, 4, 1],
                 activations=["relu", "relu", "relu", "relu"],
                 batch_size=400,
                 learning_rate=0.1,
                 loss_function="mse"):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.output_size = 1
        self.nb_epoch = nb_epoch

        #  pre processing variables:
        self.lb = LabelBinarizer()
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.a = 0
        self.b = 1
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]

        # Set up the trainer and the Multilayer
        self.neurons = neurons
        self.activations = activations
        self.network = MultiLayerNetwork(self.input_size, self.neurons, self.activations)
        self.trainer = Trainer(self.network, batch_size, nb_epoch, learning_rate, loss_function, shuffle_flag=False)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        if isinstance(x, pd.DataFrame):
            # set up variables
            if training:
                self.lb.fit(x['ocean_proximity'])
                # Identify numeric columns (exclude 'ocean_proximity' which will be one-hot encoded)
                numeric_cols = x.columns.drop('ocean_proximity')
                self.x_min = x[numeric_cols].min()
                self.x_max = x[numeric_cols].max()
                if y is not None:
                    self.y_min = y.min()
                    self.y_max = y.max()

            ocean_proximity_encoded = self.lb.transform(x['ocean_proximity'])
            if len(self.lb.classes_) == 2:
                # For binary classification, LabelBinarizer returns a single column array
                encoded_df = pd.DataFrame(ocean_proximity_encoded, columns=[self.lb.classes_[1]])
            else:
                encoded_df = pd.DataFrame(ocean_proximity_encoded, columns=self.lb.classes_)
            # fill out empty values
            x_numeric = x[x.columns.drop('ocean_proximity')]
            x_numeric = x_numeric.fillna(0)
            if y is not None:
                y = y.fillna(0)
            x_numeric = (x_numeric - self.x_min) / (self.x_max - self.x_min)
            x_numeric = x_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero

            # combine normalized data with encoded non numerical data
            x_numeric = x_numeric.reset_index(drop=True)
            encoded_df = encoded_df.reset_index(drop=True)
            x = pd.concat([x_numeric, encoded_df], axis=1)
            x = x.to_numpy()

            if y is not None:
                y = (y - self.y_min) / (self.y_max - self.y_min)
                y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
                y = y.to_numpy()

                # print("Pre: ", y)
                # print("Revert: ", self.revert_prediction(y))

            return x, y
        return None, None
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert_prediction(self, data):
        return np.array(
            [(y_normal - self.a) / (self.b - self.a) * (
                    self.y_max.to_numpy() - self.y_min.to_numpy()) + self.y_min.to_numpy() for y_normal in data])[:,
               0]

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        self.trainer.train(X, Y)
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        preds = self.trainer.network(X)
        reverted = self.revert_prediction(preds)
        return reverted

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, conf_intv=0.2, plot=False):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        y_pred = self.predict(x)
        y_true = y["median_house_value"].to_numpy()
        plt.title("Prediction and true values, " + str(conf_intv * 100) + "% error tolerance")
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.scatter(y_pred, y_true, marker="1")
        ref = max(y_pred)
        plt.plot([0, ref], [0, ref], c="k")
        plt.plot([0, ref], [0, ref * (1 - conf_intv)], c="m")
        plt.plot([0, ref / (1 + conf_intv)], [0, ref], c="c")
        plt.savefig("prediction.png")

        naive_Y = np.full((1, len(y_true)), np.mean(y))[0]
        # Mean Squared Error metric: mean average of squared differences between predicted and expected
        # Squared magnifies errors
        # Unit: squared dollar
        mse = mean_squared_error(y_true, y_pred)
        naive_mse = mean_squared_error(y_true, naive_Y)
        # Root Mean Squared Error metric
        # Unit: dollar
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        naive_rmse = mean_squared_error(y_true, naive_Y, squared=False)
        # Mean Absolute Error metric: score increasing linearly with errors (same weight for all types of errors)
        # Unit: dollar
        mae = mean_absolute_error(y_true, y_pred)
        naive_mae = mean_absolute_error(y_true, naive_Y)

        accuracy = 0
        naive_acc = 0
        for i in range(len(y_true)):
            diff = abs(y_pred[i] - y_true[i]) / y_true[i]
            naive_diff = abs(naive_Y[i] - y_true[i]) / y_true[i]
            if diff <= conf_intv:
                accuracy += 1
            if naive_diff <= conf_intv:
                naive_acc += 1
        accuracy /= len(y_true)
        naive_acc /= len(y_true)

        return (mse, naive_mse), (rmse, naive_rmse), (mae, naive_mae), (accuracy * 100, naive_acc * 100)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model, fn='part2_model.pickle'):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open(fn, 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in", fn + "\n")


def load_regressor(fn='part2_model.pickle'):
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open(fn, 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in", fn + "\n")
    return trained_model


def RegressorHyperParameterSearch(x_set, y_set, regressor_model: Regressor, param_grid: dict,
                                  epoch_freq, scoring_metric="neg_mean_squared_error", cv=5):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        x_train: Input data for training
        y_train:Expected data for training
        regressor_model:Our regressor
        param_grid: A dictionary defining the parameter grid for tuning
        scoring_metric:How are we gonna evaluate our metrics
        cv:Number of test we're gonna run        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    #
    # #Explanation:We need to test our network cv times changing our hyperparameters(Parameters that are given) #In
    # our case our hyperparameters are number of neurons,epochs,learning rate...(Arguments passed to the regressor
    # before training) #We check which one performed the best, and we return its parameters. #We can use external
    # library's for this last step #Should work with our regressor however,if not,we would either need to use an sck
    # regressor or our own function #Documentation:https://scikit-learn.org/stable/modules/generated/sklearn
    # .model_selection.GridSearchCV.html
    #
    # grid_search=GridSearchCV(regressor_model,param_grid,scoring=scoring_metric,cv=cv)
    # grid_search.fit(x_train,y_train)#Fitting our created
    #
    # return grid_search.best_params_  # Return the chosen hyper parameters
    def combine(prefix, choices):
        if not choices:
            return prefix
        new_pref = []
        if not prefix:
            new_pref = [[c] for c in choices[0]]
        else:
            for p in prefix:
                for c in choices[0]:
                    new_pref.append(p + [c])
        return combine(new_pref, choices[1:])

    combinations = combine([], list(param_grid.values()))
    ks = param_grid.keys()

    x_val = None
    y_val = None

    def validate():
        preds = regressor_model.trainer.network(x_val)
        y_pred = regressor_model.revert_prediction(preds)
        return mean_squared_error(y_val, y_pred)

    epoch_intv = int(regressor_model.nb_epoch / epoch_freq)
    epoch_checkpoints = [epoch_intv * i for i in range(1, epoch_freq + 1)]

    best_params = None
    best_epoch = None
    best_score = None
    x_folds = np.array_split(x_set, cv)
    y_folds = np.array_split(y_set, cv)
    for i in range(len(x_folds)):
        # Take 1 fold out as testing set
        x_val = x_folds[i]
        x_remaining = x_folds[:i] + x_folds[i + 1:]
        x_train = x_remaining.pop(0)
        y_val = y_folds[i]
        y_remaining = y_folds[:i] + y_folds[i + 1:]
        y_train = y_remaining.pop(0)
        while len(x_remaining) > 0:
            x_train = np.concatenate((x_train, x_remaining.pop(0)))
            y_train = np.concatenate((y_train, y_remaining.pop(0)))

        for comb in combinations:
            params = {}
            c_num = 0
            for k in ks:
                params[k] = comb[c_num]
                c_num += 1
            prev_net = regressor_model.trainer.network
            if prev_net.neurons != params["neurons"] or prev_net.activations != params["activations"]:
                regressor_model.trainer.network = MultiLayerNetwork(prev_net.input_dim, params["neurons"],
                                                                    params["activations"])
            regressor_model.trainer.batch_size = params["batch_size"]
            regressor_model.trainer.learning_rate = params["lr"]
            decay = params["decay"]
            score, epoch = regressor_model.trainer.train(x_train, y_train, epoch_checkpoints, validate)
            print(params, score, epoch)
            if best_score is None or score < best_score:
                best_score = score
                best_epoch = epoch
                best_params = params
                save_regressor(regressor_model, fn="evaluation.pickle")
    return best_params, best_epoch

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main(training):
    output_label = "median_house_value"

    # HyperParams dictionary:
    # params = {'nb_epoch': [1000, 2000, 3000, 3500, 4000],
    #           'neurons': [[56], [128], [256], [512], [1024]]
    #           }
    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    regressor = None
    if training:
        # Training
        # This example trains on the whole available dataset. 
        # You probably want to separate some held-out data 
        # to make sure the model isn't overfitting
        regressor = Regressor(x_train)
        regressor.fit(x_train, y_train)

        save_regressor(regressor)
    else:
        regressor = load_regressor()

    prediction = regressor.predict(x_train)
    print("Predicted y:", prediction, "\nLength: ", len(prediction))

    # Error
    conf_intv = 0.2
    (mse, naive_mse), (rmse, naive_rmse), (mae, naive_mae), (accuracy, naive_acc) = regressor.score(x_train, y_train,
                                                                                                    conf_intv)
    print("""\nEvaluation Metrics:\n
          - Mean Squared Error: {}$^2 compared to naive model: {}$^2;\n
          - Root Mean Squared Error: {}$ compared to naive model: {}$;\n
          - Mean Absolute Error: {}$ compared to naive model: {}$.\n
          - Accuracy with {}%$ confident interval: {}%$ compared to naive model: {}%$.
          """.format(mse, naive_mse, rmse, naive_rmse, mae, naive_mae, conf_intv * 100, accuracy, naive_acc))
    # best_params= RegressorHyperParameterSearch(x_train,y_train,regressor,params)
    # print("Best parameters after hyperparametertuning:",best_params)


def evaluation(training=True):
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    param_grid = {'neurons': [[16, 1], [32, 1]],
                  'activations': [["relu", "relu"], ["sigmoid", "relu"]],
                  'batch_size': [500], 'lr': [0.05],
                  'decay': [1]
                  }

    length = len(data)
    test_len = int(length / 10)
    shuffled_df = data.sample(frac=1)
    test_data = shuffled_df[:test_len]
    train_val_data = shuffled_df[test_len:]
    print(len(test_data))

    x_test_data = test_data.loc[:, data.columns != output_label]
    y_test_data = test_data.loc[:, [output_label]]
    x_train_val_data = train_val_data.loc[:, data.columns != output_label]
    y_train_val_data = train_val_data.loc[:, [output_label]]
    regressor = Regressor(x_train_val_data)
    x_set, y_set = regressor._preprocessor(x_train_val_data, y_train_val_data)
    params, epoch = RegressorHyperParameterSearch(
        x_set=x_set, y_set=y_set, regressor_model=regressor, param_grid=param_grid, epoch_freq=4)
    best_regressor = Regressor(x=x_train_val_data, neurons=params["neurons"], activations=params["activations"],
                               batch_size=params["batch_size"], learning_rate=params["lr"], nb_epoch=epoch)
    best_regressor.fit(x_train_val_data, y_train_val_data)

    save_regressor(best_regressor)

    (mse, naive_mse), (rmse, naive_rmse), (mae, naive_mae), (acc, naive_acc) = best_regressor.score(x_test_data, y_test_data)
    print("""\nEvaluation Metrics:
                - Mean Squared Error: {}^2 compared to naive model: {}^2;
                - Root Mean Squared Error: {} compared to naive model: {};
                - Mean Absolute Error: {} compared to naive model: {}."""
          .format(mse, naive_mse, rmse, naive_rmse, mae, naive_mae))


def map_houses():
    data = pd.read_csv("housing.csv")
    colors = "kbcgyr"
    sdata = data.sort_values(by="median_house_value").to_numpy()
    size = int(len(sdata) / len(colors))
    lons = [[] for i in range(len(colors))]
    lats = [[] for i in range(len(colors))]
    for c in range(len(colors)):
        for i in range(c * size, (c + 1) * size):
            lons[c].append(sdata[i][0])
            lats[c].append(sdata[i][1])
    plt.title("Geo-informatics house price map")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    for i in range(len(colors)):
        plt.scatter(lons[i], lats[i], c=colors[i], marker="1")
    plt.annotate("San Francisco", (-122.4, 37.77), xytext=(-124, 35), arrowprops={"width": 0.5, "headwidth": 5})
    plt.annotate("Los Angeles", (-118.26, 34.05), xytext=(-124, 33), arrowprops={"width": 0.5, "headwidth": 5})
    plt.annotate("Sacramento", (-121.48, 38.57), xytext=(-124, 35.5), arrowprops={"width": 0.5, "headwidth": 5})
    plt.annotate("San Diego", (-117.16, 32.72), xytext=(-124, 32.5), arrowprops={"width": 0.5, "headwidth": 5})
    plt.savefig("house map.png")


if __name__ == "__main__":
    # example_main(training=False)
    evaluation()
    # map_houses()
