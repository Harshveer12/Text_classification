I have the following python files that help implement the pipeline
*initializating.py* with class *Initialize*
*make_model.py* with class *MakeModel*
*train_model.py* with class Training*
*inference.py* with class *Inference*
*monitoring.py* with class *Monitoring*
*retraining.py* with class *Retrain*

## THIS ZIP FILE ALSO CONTAINS THE MODEL I TRAINED IN THE model DIRECTORY

Basic workflow is *initializing.py* makes training testing and validation sets for model. *MakeModel* is used to initialize the model with all the hyperparameters 

Next *Training* class makes a *MakeModel* object, initilises it and compiles it with an optimizer. It then fits the model on the data from an object of *Initialize* class and saves it for future use

To use this model, we can make an object of the Inference class which has methods for live inference for any sentence/review from the viewer and has a batch inference method.

The Monitoring class monitors the drift in live data from that of the training data and if that goes above a certain threshold calls the retrain class to train the model on the updated dataset. 

*Retrain* class is used by the *Monitoring* class and when called it combined the new dataset with the old one and calls the *Training* class to train the model again on the combined dataset 




**Workplan:**
There is a imdb movie review dataset. The inference script has an *Inference* class that has both live and batch inference allowed. 
Live inference allows for user to give a review as an input and gives the score. Score closer to 1 means review is positive and closer to 0 means negative review. If the user provides the sentiment as well (0 or 1) it stores the review to form a new dataset
with pos and neg directories with each review in a diffrent text file
For batch inference, we select a few files at random from both the folders (pos and neg) and perform prediction on those.
We also have a monitoring script which has a *Monitoring* class which is used to calculate data drift which again uses this new dataset since that would be relatively fresh/latest review.
If the drift is more than a certain threshold retrain script is triggered which traiggers combines the old dataset with the new dataset and triggers the train_model script to train the model again on this combined dataset

**Use The Scripts**
1. Basic setup - 
See the requirements.txt and execute it to install the dependencies
or build the docker image using the dockerfile provided

Using the Initialisation class, run the download_dataset method with file name and url to install the training data
I have used the following.
    training_data = Initialize('aclImdb_v1.tar.gz')
    training_data.download_dataset('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')

2. Training The model -
To train the model for the first time 2 scripts are bieng used, make_model and train_model
You can create an object for the MakeModel class and use preview_untrained_model, test model and see_model_structure to preview the untrained model, perform inference on the untrained model and see visually, the structure of the model
I have used the following. 
    makemodel = MakeModel()
    bert_model = makemodel.select_model()
    makemodel.preview_untrained_model(bert_model, ['Input String'])
    classifier_model = makemodel.build_model
    makemodel.test_model(classifier_model, ['Input text'])
    makemodel.see_model_structure(classifier_model)

To train the model use the train_model.py script. Follow the steps below
    training = Training()
    training.compile() {can pass the optimiser as an argument. Default is set to Adam}
    training.fit_model()

training object is initialise with the following arguments
 epochs = 5,
 init_lr = 3e-5,
 dataset_path = 'aclImdb_v1.tar.gz',
 model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8',
 name_to_handle='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
 model_to_handle = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', 
 url=''

All of the above argument are fed to initialize the training object as parameters. The above are the values I have used and set as default
If the url is provided, the download_dataset method will be called from Initialization class to download the training data from the given url

3. Use the model -
To see the sentiment of a sentence, like a movie review, use the infernce.py script
Inference class has both methods for live and batch inference
For live inference run the following 
    inference = Inference(model_path)
    inference.run_live_inference()

this will ask you to input a sentence or a paragraph and if you chose to, what the sentiment is
If the sentiment is also provided, the sentence is recorded in a new dataset which is used for model monitoring and training.
The model the gives you a score between 0 and 1 based on what it thinks is the review. Clode to 1 is positive and closer to 0 is negative

For batch inference run the following
    inference = Inference(model_path)
    inference.run_batch_inference(path to dataset, num_files)

num files specifies how many files from each of pos and neg directory of the dataset shall be taken for batch inference
It will output the review and the score for each

4. Monitoring - 
Monitor the model statistics and performance and if it drifts too far, retrain it
Run the following to run monitoring
    train_ds, val_ds, test_ds = training_data.make_dataset(original_training_data, latest_data) 
    run_drift_monitoring(train_ds, test_ds, threshold)

use the make_dataset method of Initialize class to get the train test split. train_ds and test_ds would be taken from original_training_data and latest_test_data for calculating the drift.
If the drift is above a threshold it triggers retraining.

5. Retraining Logic -
Retraining is triggered by monitoring.py script when the drift is unacceptable. 
Retrain class takes the latest data and combines it with the older data and retrains the model on this combined data using the fit method of Training class and saves the newly retrained model if it is accurate enough



**{due to time constraint, could  not deploy model for monitoring to work in the background}**




# How to build the docker image
docker build -t bert-image .
docker run -it bert-image

# Install Requirements
pip install -r requirements.txt


