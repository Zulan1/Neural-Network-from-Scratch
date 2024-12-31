# Neural-Network-from-Scratch
Learning Project - Coding Neural Network from scratch using NumPy.

Requirements:
NumPy
Matplotlib
Click
SciPy
tqdm
Optuna (optional)

To optimize a neural network, run:

python . 

example usage:

python . \
    --dataset 'SwissRoll' \
    --net_shape '2x5' \
    --activation 'relu' \
    --resnet \
    --loss crossentropy \
    --optim sgd \
    --batch_size 512 \
    --epochs 100 \
    --lr' 1e-2 \
    --momentum' 0.0 \