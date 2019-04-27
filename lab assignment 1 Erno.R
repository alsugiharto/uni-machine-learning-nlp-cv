library(keras)

#Loading the dataset
mnist <- dataset_mnist()

#Preparing the arrays
x_test <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), 784))
x_train <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), 784))

#Setting the values between 0 and 1
x_test <- x_test/255
x_train <- x_train/255

#Turning the labels into categories
y_test <- to_categorical(mnist$test$y)
y_train <- to_categorical(mnist$train$y)

#Building the model
model <- keras_model_sequential()
model %>% 
  layer_dense(units=256, input_shape=c(784)) %>%
  layer_dense(units=10, activation = 'softmax')

#Compiling
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#Training, saving steps into history
history <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)

#Saving evaluation
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)

