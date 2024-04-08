# DL_assignment2

**Libraries:**

     -> Required liabaries torch, DataLoader, transforms and optim are imported.

**Dataset Preparation:**

      -> I loaded the iNaturalist dataset, comprising both training and testing data. To evaluate the model's performance and prevent overfitting, I partitioned             the training data into an 80% training set and a 20% validation set. This partition allowed me to monitor whether the model was learning patterns or                   simply memorizing images.

**GPU Integration:**

      -> To accelerate model training, I leveraged GPU parallelism using the torch.device function. This enabled the model to execute computations in parallel.

**Data Preprocessing:**

      Each image in the iNaturalist dataset had varying dimensions, necessitating preprocessing. I resized all images to a standardized size of 128x128x3 pixels 
      and applied tensor normalization. Additionally, I grouped images into batches of 32 to facilitate efficient training.

**Small CNN Architecture:**

      I implemented a SmallCNN class, customizable with parameters such as the number of filters, activation function, data augmentation, batch normalization,            number of dense neurons, and dropout. This CNN architecture comprised five convolutional layers, each followed by max-pooling, and five batch normalization 
      layers. It also included a dense layer and an output layer for classification.

**Forward Propagation:**

      The forward() function in the SmallCNN class applied the specified activation function, performed max-pooling, and optionally applied dropout to prevent 
      overfitting. Additionally, it allowed the choice of applying batch normalization based on the specified parameter.

**Model Training:**

      For training, I employed the cross-entropy loss function and Adam optimizer. In each epoch, the model traversed each batch of the training set, calculated          the loss, and updated the model parameters using backpropagation. After training, I evaluated the model's performance on the validation set, monitoring both        accuracy and loss to assess whether the model effectively learned patterns from the dataset or resorted to memorization.

**wandb:** I have integrated wandb to visualize the results

