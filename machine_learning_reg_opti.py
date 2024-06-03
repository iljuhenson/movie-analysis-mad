from machine_learning import *

# Training the Model
for i in range(8, 80, 2):
    for j in range(1, 60):
        history_reg = model_reg.fit(
            X_train, y_train, epochs=i, batch_size=j, validation_split=0.2, verbose=1
        )
        # Evaluating the Model
        y_pred = model_reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        with open(r"output/machine_learning_reg_opti.txt", "a") as file:
            file.write(
                f"Number of epochs: {i}, batch_size: {j},  Mean Absolute Error for Average Rating: {mae:.5f}\n"
            )
