library(keras)

cifar10 <- dataset_cifar10()

x_train <- cifar10$train$x
x_test <- cifar10$test$x

#rescale by devide by 255
x_train = x_train/255
x_test = x_test/255

#to categorical
y_train = to_categorical(cifar10$train$y)
y_test = to_categorical(cifar10$test$y)

#MODEL DEFINITION
#watch that 2 lines of dropout layes are added

model_deep_dropout_cifar <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(32, 32, 3), padding = 'same') %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', padding = 'same') %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

model_deep_dropout_cifar %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = c('accuracy')
)

#TRAINING AND EVALUATION
# plot history result
history_deep_dropout_cifar <- model_deep_dropout_cifar %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 20,
  verbose = 1,
  validation_data = list(x_test, y_test),
  shuffle = TRUE
)

plot(history_deep_dropout_cifar)