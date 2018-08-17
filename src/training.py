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
    # 11 layers
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


def deep_cnn_arch(x_train, y_train, epochs, batch_size, validation_split):
    # 8s per epoch
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), data_format="channels_last", activation="relu",
                     input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(filters=128, kernel_size=(4, 4), activation="relu"))
    model.add(Flatten())
    model.add(Dense(3072, activation="softmax"))
    model.add(Dense(6, activation="softmax"))
    # 8 layers
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('Training....')
    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, shuffle=True, verbose=2)
    train_val_accuracy = hist.history
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print('          Done!')
    print('     Train acc: ', train_acc[-1])
    print('Validation acc: ', val_acc[-1])
    print(' Overfit ratio: ', val_acc[-1] / train_acc[-1])
    save_model(model, index="deepcnn")
    return model


def deep2_cnn_arch(x_train, y_train, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    conv_arch = [(32, 3), (64, 3), (128, 3)]
    dense = [64, 2]
    if (conv_arch[0][1] - 1) != 0:
        for i in range(conv_arch[0][1] - 1):
            model.add(Conv2D(conv_arch[0][0], kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[1][1] != 0:
        for i in range(conv_arch[1][1]):
            model.add(Conv2D(conv_arch[1][0], kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[2][1] != 0:
        for i in range(conv_arch[2][1]):
            model.add(Conv2D(conv_arch[2][0], kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
    if dense[1] != 0:
        for i in range(dense[1]):
            model.add(Dense(dense[0], activation='relu'))
            model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    # 16 layers
    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Training....')
    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=validation_split, shuffle=True, verbose=2)
    train_val_accuracy = hist.history
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print('          Done!')
    print('     Train acc: ', train_acc[-1])
    print('Validation acc: ', val_acc[-1])
    print(' Overfit ratio: ', val_acc[-1] / train_acc[-1])
    save_model(model, index="deep2")


def save_model(model, index=""):
    model_json = model.to_json()
    with open("../data/model" + index + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../data/model" + index + ".h5")
    print("Saved model to disk")
