import os, sys



class Learner(object):
    def __init__(self, huc2 = None):

        self.X = [] # m * n
        self.y = [] # m * 1
        pass


    def generate_dataset(self):
        """
        1 -
        2-
        :return:
        """

        # todo: report this

        self.X = 1
        self.y = 1



    def remove_outliers(self):
        #
        # filtering -->
        X, y
        pass

    def augment_data(self):
        #  X where new_m > m
        #  X  increase n


        pass

    def train(self, method = 'keras'):
        # X, y
        pass

    def tune(self):
        pass

    def evaluate(self):
        pass

    def generate_model(self):

        # 1) assemble raw dataset in pandas dataframe (generate_dataset)

        # 2) apply the QA/QC (remove_outliers) [X,y]

        # 3) augment the dataset by adding features that helps extrapolate [min and max bounds]

        # 4) split dataset training/validation. Maybe kfold partitioning

        # training

        # 5) Hyper-parameter tuning

        pass

    def post_training(self):
        # features importance, model performance, others
        pass

    def report(self):
        """
        Generate a pdf (or HTML) report that document (with plots) all decision made to produce the model.
        Report model training progress and performance
        Plot data for visual inspection
        Plot validation results.
        .
        .

        :return:
        """
        pass

    def load_model(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":

    Learner(huc2=2, data_ws = "")
    md = Learner.generate_model()
    Y_hat = Learner.predict()





################3

#date, pop, climate1, wu_daily, montly, annual
