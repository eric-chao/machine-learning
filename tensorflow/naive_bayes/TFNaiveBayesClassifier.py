#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:03:10 2018

Naive Bayes Classifier on TensorFlow.

For more info: 
    http://nicolovaligi.com/naive-bayes-tensorflow.html

@author: zxj

"""

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.utils.fixes import logsumexp

class TFNaiveBayesClassifier:
    dist = None

    def fit(self, X, y):
        # Separate training points by class (nb_classes * nb_samples * nb_features)
        unique_y = np.unique(y)
        points_by_class = np.array([
            [x for x, t in zip(X, y) if t == c]
            for c in unique_y])

        # Estimate mean and variance for each class / feature
        # shape: nb_classes * nb_features
        mean, var = tf.nn.moments(tf.constant(points_by_class), axes=[1])

        # Create a 3x2 univariate normal distribution with the 
        # known mean and variance
        self.dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))


    def predict(self, X, priors_prob):
        assert self.dist is not None
        nb_classes, nb_features = map(int, self.dist.scale.shape)

        # Conditional probabilities log P(x|c) with shape
        # (nb_samples, nb_classes)
        cond_probs = tf.reduce_sum(
            self.dist.log_prob(
                tf.reshape(
                    tf.tile(X, [1, nb_classes]), [-1, nb_classes, nb_features])),
            axis=2)

        # uniform priors
        # priors = np.log(np.array([1. / nb_classes] * nb_classes))
        priors = np.log(np.array(priors_prob))

        # posterior log probability, log P(c) + log P(x|c)
        joint_likelihood = tf.add(priors, cond_probs)

        # normalize to get (log)-probabilities
        norm_factor = tf.reduce_logsumexp(joint_likelihood, axis=1, keepdims=True)
        log_prob = joint_likelihood - norm_factor

        # exp to get the actual probabilities
        return tf.exp(log_prob)