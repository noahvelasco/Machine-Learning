batch_size = 5000
epochs = 20

history = model.fit(
    x_train.astype(np.float32), y_train.astype(np.float32),
    initial_epoch = len(loss),
    epochs =  epochs + len(loss), 
    validation_data=(x_test.astype(np.float32), y_test.astype(np.float32)),
    batch_size=batch_size, 
    verbose = 1,
)

loss = loss + history.history['loss']
val_loss = val_loss + history.history['val_loss']
accuracy = accuracy + history.history['accuracy']
val_accuracy = val_accuracy + history.history['val_accuracy']

print('Lowest training loss = {:9.6f}, in epoch {} '.format(np.amin(loss),np.argmin(loss)+1))
print('Lowest validation loss = {:9.6f}, in epoch {} '.format(np.amin(val_loss),np.argmin(val_loss)+1))
print('Highest training accuracy = {:9.6f}, in epoch {} '.format(np.amax(accuracy),np.argmax(accuracy)+1))
print('Highest validation accuracy = {:9.6f}, in epoch {} '.format(np.amax(val_accuracy),np.argmax(val_accuracy)+1))

model.save_weights('last_model.h5', overwrite=True)


