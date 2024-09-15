from train_model import Training
from initializing import Initialize
import os
import shutil

class Retrain:
    def __init__(self,old_dataset_path, new_dataset_path, combined_dataset_path,epochs=5,init_lr = 3e-5, 
                 dataset_path = 'aclImdb_v1.tar.gz', model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8',name_to_handle='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
                 model_to_handle = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3' ):
        print("2")
        self.combine_datasets(old_dataset_path, new_dataset_path, combined_dataset_path)
        training_data = Initialize('combined_dataset')
        new_train_ds, new_val_ds, new_test_ds = training_data.make_dataset('combined_dataset/train', 'combined_dataset/test')

        self.training = Training(epochs=epochs, init_lr=init_lr, model_name=model_name,
                                 name_to_handle=name_to_handle, model_to_handle=model_to_handle,
                                 dataset_path=combined_dataset_path)
        self.training.train_ds = new_train_ds
        self.training.val_ds = new_val_ds
        self.training.test_ds = new_test_ds
        # Compile model (with old data for initial setup)
        self.training.compile()
        
    

    def combine_datasets(self,old_dataset_dir, new_dataset_dir, combined_dataset_dir):
        """
        Combine old and new datasets into a single dataset directory.
        Args:
            old_dataset_dir (str): Path to the old dataset directory.
            new_dataset_dir (str): Path to the new dataset directory.
            combined_dataset_dir (str): Path to the directory where the combined dataset will be saved.
        """
        if not os.path.exists(combined_dataset_dir):
            os.makedirs(combined_dataset_dir)
        
        # Define dataset splits
        splits = ['train', 'test']
        labels = ['pos', 'neg']
        
        for split in splits:
            for label in labels:
                old_path = os.path.join(old_dataset_dir, split, label)
                new_path = os.path.join(new_dataset_dir, split, label)
                combined_path = os.path.join(combined_dataset_dir, split, label)

                if not os.path.exists(combined_path):
                    os.makedirs(combined_path)

                # Copy old dataset files
                for file_name in os.listdir(old_path):
                    full_file_name = os.path.join(old_path, file_name)
                    if os.path.isfile(full_file_name):
                        shutil.copy(full_file_name, combined_path)
                
                # Copy new dataset files
                for file_name in os.listdir(new_path):
                    full_file_name = os.path.join(new_path, file_name)
                    if os.path.isfile(full_file_name):
                        shutil.copy(full_file_name, combined_path)

        print(f"Combined dataset saved at: {combined_dataset_dir}")

    def retrain_model(self):
        history = self.training.fit_model()
        return history        

    def save_retrained_model(self, new_model_name='retrained_model'):
        loss, accuracy = self.training.predict_model()
        if accuracy > 0.8:
            self.training.save_model(new_model_name)

    # Define paths
# old_dataset_path = 'aclImdb'
# new_dataset_path = '/home/harsh/Documents/new_data'
# combined_dataset_path = 'combined_dataset'

# retrain = Retrain(old_dataset_path,new_dataset_path,combined_dataset_path)
# retrain.retrain_model()