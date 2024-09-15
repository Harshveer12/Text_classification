import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel('ERROR')



class MakeModel:

    def __init__(self,bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8',
                 map_name_to_handle ='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
                 map_model_to_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'):
        
        self.bert_model_name = bert_model_name
        self.tfhub_handle_encoder = map_name_to_handle
        self.tfhub_handle_preprocess  = map_model_to_preprocess


    def select_model(self):
        print(f'BERT model selected           : {self.tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {self.tfhub_handle_preprocess}')

        bert_preprocess_model = hub.KerasLayer(self.tfhub_handle_preprocess)

        return bert_preprocess_model

    # Sanity check
    # BERT model would use (input_words_id, input_mask and input_type_ids).
    # ##READ THIS ONCE##

    
    def preview_untrained_model(self,bert_preprocess_model, text_test = ["This is really good"]):

        text_preprocessed = bert_preprocess_model(text_test)

        print(f'Keys       : {list(text_preprocessed.keys())}')
        print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
        print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
        print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
        print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

        # Before putting BERT into your own model, let's take a look at its outputs
        bert_model = hub.KerasLayer(self.tfhub_handle_encoder)

        bert_results = bert_model(text_preprocessed)

        print(f'Loaded BERT: {self.tfhub_handle_encoder}')
        print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
        print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
        print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
        print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')



        # The type_ids only have one value (0) because this is a single sentence input. 
        # For a multiple sentence input, it would have one number for each input.


    def build_classifier_model_structure(self,dropout=0.1):
      
      text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
      preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
      encoder_inputs = preprocessing_layer(text_input)
      encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
      outputs = encoder(encoder_inputs)
      net = outputs['pooled_output']
      net = tf.keras.layers.Dropout(dropout)(net)
      net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
      return tf.keras.Model(text_input, net)


    def build_model(self):
        classifier_model = self.build_classifier_model_structure()
        return classifier_model

    def test_model(self, classifier_model,text_test):
        bert_raw_result = classifier_model(tf.constant(text_test))
        print(tf.sigmoid(bert_raw_result))

    def see_model_structure(self,classifier_model,to_file = "model_structure.png"):
        tf.keras.utils.plot_model(classifier_model,to_file=to_file, show_shapes=True)

