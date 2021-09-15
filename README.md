# Change Detection in High-dimensional data streams

This repository contains code of the change detection framework Stream Autoencoder Windowing (SAW) for the detection of concept drift in high dimensional data streams: We train an autoencoder on the incoming data stream and monitor its reconstruction error with a sliding window of adaptive size to detect ''when'' and ''where'' a drift occurs.

## Abstract 

The data collected in many real-world scenarios such as environmental analysis, manufacturing, and e-commerce are high dimensional and come as a stream, i.e., data properties evolve over time – a phenomenon known as "concept drift". This brings numerous challenges: data-driven models become outdated, and one is typically interested in detecting specific events, e.g., the critical wear and tear of industrial machines. Hence, it is crucial to detect change, i.e., concept drift, to design a reliable and adaptive predictive system for streaming data. However, existing techniques can only detect the “change point”, i.e. “when” a drift occurs. As drifts may occur only in certain dimensions, a change detector should be able to identify the drifting dimensions, i.e. “where” a change occurs. This is particularly challenging when data streams are high dimensional because of the so-called “curse of dimensionality”: neighborhood becomes meaningless, and a concept drift might be only visible in sub spaces. 

We introduce Stream Autoencoder Windowing (SAW), an unsupervised change detection framework based on the online training of an autoencoder, while monitoring its reconstruction error via a sliding window of adaptive size. Our approach allows to effciently and effectively detect “when” and “where” drift occurs in high dimensional data streams. Unsupervised methods do not require ground truth or labels, which is an advantage in the data streaming environment, where obtaining labels can be very expensive or even impossible. We will evaluate the performance of our method against synthetic data, in which the characteristics of drifts are known. We then show how our method improves the accuracy of existing classiers for predictive systems on real data streams. We evaluate our framework SAW against state-of-the-art methods.
