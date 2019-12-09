NOTES for testing

12062019 - Jigsaw run successfully but did not learn. See yaml in extra scripts.
12072019

    Hyperparameter search for learning rate with SGD and ADAM. All tests are for 10 epochs.

    unsupervised_alexnet_jigsaw_stl10_1.yaml: ADAM, lr = 0.01
    unsupervised_alexnet_jigsaw_stl10_2.yaml: ADAM, lr = 0.001
    unsupervised_alexnet_jigsaw_stl10_3.yaml: ADAM, lr = 0.0001
    unsupervised_alexnet_jigsaw_stl10_4.yaml: SGD, lr = 0.001
    unsupervised_alexnet_jigsaw_stl10_5.yaml: SGD, lr = 0.0001

    Since no global normalization is done in image preprocessing, We test adding BN layers to the conv and fc

    unsupervised_alexnet_jigsaw_bn_stl10_1.yaml: ADAM, lr = 0.1, 1000permutation
    unsupervised_alexnet_jigsaw_bn_stl10_2.yaml: ADAM, lr = 0.01, 1000permutation
    unsupervised_alexnet_jigsaw_bn_stl10_3.yaml: ADAM, lr = 0.001, 1000permutation
    unsupervised_alexnet_jigsaw_bn_stl10_4.yaml: ADAM, lr = 0.1, 100permutation
    unsupervised_alexnet_jigsaw_bn_stl10_5.yaml: ADAM, lr = 0.01, 100permutation
    unsupervised_alexnet_jigsaw_bn_stl10_6.yaml: ADAM, lr = 0.001, 100permutation