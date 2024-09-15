import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization 
import matplotlib.pyplot as plt
import os
import shutil
tf.get_logger().setLevel('ERROR')
from tensorflow import keras


# url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
# file_name = 'aclImdb_v1.tar.gz'

class Initialize:

    def __init__(self, file_name):
        self.file_name = file_name

    def download_dataset(self,url, name='aclImdb'):
        dataset = tf.keras.utils.get_file(self.file_name, url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

        dataset_dir = os.path.join(os.path.dirname(dataset), name) 
        train_dir = os.path.join(dataset_dir, 'train')

        # remove unused folders to make it easier to load the data
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)


    #train_dir = 'aclImdb/train'
    #test_dir = 'aclImdb/test'
    def make_dataset(self,train_dir, test_dir,val_split = 0.2, batch_size = 32):

        AUTOTUNE = tf.data.AUTOTUNE

        # TRAINING DATA
        # 80%
        # will read subdirectory train and read the subdirectory names (pos and neg) in this case and label them as class names
        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            train_dir,
            batch_size=batch_size,
            validation_split=val_split,
            subset='training',
            seed=42)

        self.class_names = raw_train_ds.class_names

        # On the first epoch (or pass) through the dataset, TensorFlow will read the data from the
        # disk or other source and store it in memory. On subsequent epochs, it will use the cached
        # data from memory instead of reloading it from the source

        # prefetch will overlap the training with preparing the dataset. It will prepare the number of batches = buffer size
        # while the training is already in progress, reducing the idle time

        train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # VALIDATION DATA
        # 20%
        val_ds = tf.keras.utils.text_dataset_from_directory(
            train_dir,
            batch_size=batch_size,
            validation_split=val_split,
            subset='validation',
            seed=42)

        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # TEST DATA
        test_ds = tf.keras.utils.text_dataset_from_directory(
            test_dir,
            batch_size=batch_size)

        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds,val_ds,test_ds


    def preview(self,train_ds):
        for text_batch, label_batch in train_ds.take(1):
            for i in range(3):
                print("Batch",text_batch.numpy()[i])
                label = label_batch.numpy()[i]
                print(f'Label : {label} ({self.class_names[label]})')


# training_data = Initialize('aclImdb_v1.tar.gz')
# training_data.download_dataset('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
# training_data.make_dataset('aclImdb/train', 'aclImdb/test')