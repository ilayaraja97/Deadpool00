import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def cnn_arch(x_train, y_train, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(7, 7), data_format="channels_last", activation="relu",
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(4048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('Training....')
    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, shuffle=True, verbose=1)
    train_val_accuracy = hist.history
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print('          Done!')
    print('     Train acc: ', train_acc[-1])
    print('Validation acc: ', val_acc[-1])
    print(' Overfit ratio: ', val_acc[-1] / train_acc[-1])
    save_model(model)
    return model


def save_model(model):
    model_json = model.to_json()
    with open("../data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../data/model.h5")
    print("Saved model to disk")
