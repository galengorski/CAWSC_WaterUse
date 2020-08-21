from wuse import data_collector, Learner

data_collector.get_training_data(huc2=13)
data_collector.prediction_data(huc2 = 13)

Learner.generate_model(huc2 = 13)

Learner.predict(huc2 = 13)

