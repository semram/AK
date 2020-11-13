import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam

def getUnreadableCount(df):
    return (df['IDENTITY'] == 'UNREADABLE').sum();

def getLowercaseCount(df):
    return (df['IDENTITY'].str.islower()).sum();

def getDigitCount(df):
    return (df['IDENTITY'].str.isdigit()).sum();
  
def removeUnreadableEntries(df):
    is_unreadable = df['IDENTITY'] != 'UNREADABLE';
    df = df[is_unreadable];
    return df;

def removeDigitEntries(df):
    is_digit = df['IDENTITY'].str.isdigit();
    df = df[is_digit];
    return df;

def cleanDataSet(df):
    empty_count = df.isnull().sum().sum();      # 565 for train, 78 for validation
    if(empty_count):
        df.dropna(inplace=True);
    unreadable_count = getUnreadableCount(df);  # 102 for train, 12 for validation
    if(unreadable_count):
        df = removeUnreadableEntries(df);
    digit_count = getDigitCount(df);            # 0 for train, 0 for validation
    if(digit_count):
        df = removeDigitEntries(df);
    lowercase_count = getLowercaseCount(df);    # 13 for train, 2 for validation
    if(lowercase_count):                        # Pictures are all uppercase, we have to make our data uppercase
        df.loc[:, 'IDENTITY'] = df['IDENTITY'].apply(lambda x: x.upper());
    notpicture_count = df[~df["FILENAME"].str.endswith('.jpg')].sum().sum().astype(int);
    if(notpicture_count):
         df = df[df["FILENAME"].str.contains('.jpg')];
    return df;

def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

# the ctc loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

if __name__ == "__main__":
    alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
    max_str_len = 24 # max length of input labels
    num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
    num_of_timestamps = 64 # max length of predicted labels

    # Load the written data into dataframes with the first row as header
    training_written_df = pd.read_csv('written_name_train_v2.csv')
    validation_written_df = pd.read_csv('written_name_validation_v2.csv')
    
    # DATA CLEANING + PREPARATION (train + validation)
    training_written_df = cleanDataSet(training_written_df);
    validation_written_df = cleanDataSet(validation_written_df);
    
     # To make sure our indices are one behind the other
    training_written_df.reset_index(inplace = True, drop=True)
    validation_written_df.reset_index(inplace = True, drop=True) 
    
    train_size = 30000
    valid_size= 3000
    
    valid_x = []
    #validation_indices = validation_written_df.index.values[:valid_size]
    for i in range(valid_size):
        img_dir = 'validation_v2/validation/'+validation_written_df.loc[i, 'FILENAME']   
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        image = image/255.
        valid_x.append(image)
    
    train_x = []
    #train_indices = training_written_df.index.values[:train_size]

    for j in range(train_size):
        img_dir = 'train_v2/train/'+training_written_df.loc[j, 'FILENAME']
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        image = image/255.
        train_x.append(image)
    
    valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)
    train_x = np.array(train_x).reshape(-1, 256, 64, 1)

    # Preparing the labels for CTC Loss
    
    #name = 'JEBASTIN'
    #print(name, '\n',label_to_num(name))
    
    train_y = np.ones([train_size, max_str_len]) * -1
    train_label_len = np.zeros([train_size, 1])
    train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
    train_output = np.zeros([train_size])
    
    for i in range(train_size):
        train_label_len[i] = len(training_written_df.loc[i, 'IDENTITY'])
        train_y[i, 0:len(training_written_df.loc[i, 'IDENTITY'])]= label_to_num(training_written_df.loc[i, 'IDENTITY']) 

    valid_y = np.ones([valid_size, max_str_len]) * -1
    valid_label_len = np.zeros([valid_size, 1])
    valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
    valid_output = np.zeros([valid_size])

    for i in range(valid_size):
        valid_label_len[i] = len(validation_written_df.loc[i, 'IDENTITY'])
        valid_y[i, 0:len(validation_written_df.loc[i, 'IDENTITY'])]= label_to_num(validation_written_df.loc[i, 'IDENTITY'])
    

    print('True label : ',training_written_df.loc[100, 'IDENTITY'] , '\ntrain_y : ',train_y[100],'\ntrain_label_len : ',train_label_len[100], 
      '\ntrain_input_len : ', train_input_len[100])
    
    #Building our model
    input_data = Input(shape=(256, 64, 1), name='input')

    inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)
    
    inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
    inner = Dropout(0.3)(inner)
    
    inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
    inner = Dropout(0.3)(inner)
    
    # CNN to RNN
    inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)
    
    ## RNN
    inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
    inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)
    
    ## OUTPUT
    inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
    y_pred = Activation('softmax', name='softmax')(inner)
    
    model = Model(inputs=input_data, outputs=y_pred)
    model.summary()
    
    
    #The output shape of the predictions is (64, 30).
    #The model predicts words of 64 characters and each character contains the probability of
    #the 30 alphabets which we defined earlier
    
    labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)
    
    #Train our model
    
    # the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
    model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = 0.0001))

    model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                epochs=60, batch_size=128)
    
    
    #Check model performance on validation set 
    preds = model.predict(valid_x)
    decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                       greedy=True)[0][0])
    
    prediction = []
    for i in range(valid_size):
        prediction.append(num_to_label(decoded[i]))
        
    y_true = validation_written_df.loc[0:valid_size, 'IDENTITY']
    correct_char = 0
    total_char = 0
    correct = 0
    
    for i in range(valid_size):
        pr = prediction[i]
        tr = y_true[i]
        total_char += len(tr)
        
        for j in range(min(len(tr), len(pr))):
            if tr[j] == pr[j]:
                correct_char += 1
                
        if pr == tr :
            correct += 1 
        
    print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
    print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))     
    
    #Some predictions on test set    
    test = pd.read_csv('written_name_test_v2.csv')
    plt.figure(figsize=(15, 10))
    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        img_dir = 'test_v2/test/'+test.loc[i, 'FILENAME']
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        plt.imshow(image, cmap='gray')
        
        image = preprocess(image)
        image = image/255.
        pred = model.predict(image.reshape(1, 256, 64, 1))
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                           greedy=True)[0][0])
        plt.title(num_to_label(decoded[0]), fontsize=12)
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            