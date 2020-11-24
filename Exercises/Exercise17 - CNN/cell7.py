with strategy.scope():
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

