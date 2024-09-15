import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization 
from initializing import Initialize
from make_model import MakeModel
tf.get_logger().setLevel('ERROR')



# url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
class Training:
    def __init__(self, epochs = 5,init_lr = 3e-5,dataset_path = 'aclImdb_v1.tar.gz', model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8',name_to_handle='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',model_to_handle = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', url=''):    
        
        training_data = Initialize(dataset_path)
        if url != '':
            training_data.download_dataset(url)

        self.train_ds, self.val_ds, self.test_ds = training_data.make_dataset('aclImdb/train','aclImdb/test')

        made_model = MakeModel(model_name,
                                name_to_handle,
                                model_to_handle)
        
        self.bert_preprocess_model = made_model.select_model()
        self.classifier_model = made_model.build_model()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.init_lr = init_lr
        self.epochs = epochs


    def compile(self,optimiser='adamw'):
        
        metrics = [tf.metrics.BinaryAccuracy(), tf.metrics.Accuracy(), tf.metrics.F1Score(), tf.metrics.Recall(), tf.metrics.Precision()]
        epochs = self.epochs

        # tf.data.experimental.cardinality(train_ds) returns the number of batches in the dataset, and .numpy() converts this to a Python integer.
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        optimizer = optimization.create_optimizer(init_lr=self.init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type=optimiser)
        # For the learning rate (init_lr), you will use the same schedule as BERT pre-training: linear decay of a notional initial learning rate,
        # prefixed with a linear warm-up phase over the first 10% of training steps (num_warmup_steps). In line with the BERT paper, the initial learning rate is smaller for fine-tuning (best of 5e-5, 3e-5, 2e-5).
        self.classifier_model.compile(optimizer=optimizer,      #**#
                             loss=self.loss,
                             metrics=metrics)


    def fit_model(self):
        print(f'Training model')
        history = self.classifier_model.fit(x=self.train_ds,
                                   validation_data=self.val_ds,
                                   epochs=self.epochs)
        return history

    def predict_model(self):
        loss, accuracy = self.classifier_model.evaluate(self.test_ds)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

    
    def save_model(self,dataset_name='my_model'):
        saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

        self.classifier_model.save(saved_model_path, include_optimizer=False)
        print(f'Model saved to: {saved_model_path}')

# training = Training()
# training.compile()
# training.fit_model()