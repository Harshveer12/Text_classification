import tensorflow as tf
import tensorflow_text as text
import tensorflow as tf
import os
import uuid
import random

class Inference:

    def __init__(self, model_path='/home/harsh/Documents/Bert_classification/model/zipfolder'):

        print(f'Loading model from {model_path}')
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()


    def save_to_directory(self, input_text, sentiment, save_dir='/home/harsh/Documents/new_data/train/'):
        # Define the directories
        pos_dir = os.path.join(save_dir,'pos')
        neg_dir = os.path.join(save_dir,'neg')

        # Create directories if they don't exist
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # Generate a unique file name
        file_name = str(uuid.uuid4()) + ".txt"

        # Set the path based on sentiment
        if sentiment == 1:
            file_path = os.path.join(pos_dir, file_name)
        else:
            file_path = os.path.join(neg_dir, file_name)

        # Write the input text to the corresponding file
        with open(file_path, 'w') as file:
            file.write(input_text[0])

        print(f'Saved to: {file_path}\n')


    def predict_live_result(self, text,sentiment):
        # Get the model's prediction (logits)
        reloaded_results = tf.sigmoid(self.model(tf.constant(text)))
        self.save_to_directory(text,sentiment)
        result_for_printing = \
        [f'input: {text[0]:<30} : score: {reloaded_results[0][0]:.6f}']
        print(*result_for_printing, sep='\n')
        print()


    def run_batch_inference(self, dir_path,num_files=3):        
        # Get the model's prediction (logits)
        example = []
        text = self.load_texts_from_files(dir_path,'pos', num_files)
        example.extend(text)
        print(example)
        text = self.load_texts_from_files(dir_path,'neg', num_files)
        example.extend(text)
        print(example)
        reloaded_results = tf.sigmoid(self.model(tf.constant(text)))
        result_for_printing = \
        [f'input: {text[i]:<30} : score: {reloaded_results[i][0]:.6f}'
            for i in range(len(text))]
        print(*result_for_printing, sep='\n')
        print()


    def load_texts_from_files(self, dir_path, sentiment, num_files=20):
        # Set the path to the pos/neg directory
        sentiment_dir = os.path.join(dir_path, sentiment)
        
        # List all files in the sentiment directory
        file_list = os.listdir(sentiment_dir)
        
        # Randomly sample 20 files (or fewer if there are less than 20)
        selected_files = random.sample(file_list, min(len(file_list), num_files))
        
        text_list = []
        
        for file_name in selected_files:
            file_path = os.path.join(sentiment_dir, file_name)
            with open(file_path, 'r') as file:
                text_list.append(file.read().strip())  # Read the content and strip whitespace
                
        return text_list

    def run_live_inference(self):
        print("\n\nLIVE INFERENCE\n")
        while True:
            text=[]
            user_input = input("Enter text for live inference (or type 'exit' to quit): ")
            text.append(user_input)
            if user_input.lower() == 'exit':
                break
            sentiment_input = input("Enter Yes or No")
            if sentiment_input == "Yes":
                while True:
                    try:
                        sentiment = int(input("Enter sentiment (1 for positive, 0 for negative): "))
                        if sentiment not in [0, 1]:
                            raise ValueError("Sentiment must be 0 or 1.")
                        break
                    except ValueError as e:
                        print(e)
                self.predict_live_result(text,sentiment)        


inference = Inference()


# examples = [
#     'this is such an amazing movie!', 
#     'The movie was great!',
#     'The movie was meh.',
#     'The movie was okish.',
#     'The movie was terrible...'
# ]


print("BATCH INFERENCE\n")
inference.run_batch_inference("/home/harsh/Documents/new_data/train", num_files=3)
inference.run_live_inference()