from initializing import Initialize
from make_model import MakeModel
from train_model import Training
from inference import Inference

#INITIALIZING.PY SCRIPT
#used to make dataset
training_data = Initialize('aclImdb_v1.tar.gz')
training_data.download_dataset('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')

#MAKE_MODEL.PY SCRIPT
#used to make the model structure
makemodel = MakeModel()
bert_model = makemodel.select_model()
makemodel.preview_untrained_model(bert_model, ['This is a good movie'])
classifier_model = makemodel.build_model
makemodel.test_model(classifier_model, ['This is a good movie'])
makemodel.see_model_structure(classifier_model)

#TRAIN THE MODEL
#train the model for the first time. Prefereably using GPUs or TPUs
training_model = Training(5,3e-5)

training_model.compile('adamw')
training_model.fit()
training_model.predict_model()

#INFERENCE.PY SCRIPT
inference = Inference()
inference.run_batch_inference("/home/harsh/Documents/new_data/train", num_files=3)
inference.run_live_inference()

#Execute this in the monitoring.py file
#train_ds, val_ds, test_ds = training_data.make_dataset('aclImdb/train', '/home/harsh/Documents/combined_dataset/train')
#run_drift_monitoring(train_ds, test_ds)

