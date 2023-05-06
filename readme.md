# Exploring Outlier Exposure Methods for Deep Margin-Based Anomaly Detection

In standard training regimes, one assumes that the classes presented to a model are representative of the classes that the model will encounter when it is deployed. In real deployment scenarios, however, a model can sometimes encounter situations or objects that it has never seen. When these scenarios are safety-critical, a model's response to an out-of-distribution (OOD) input can be the difference between success and catastrophic failure. In some cases, one has access to some OOD examples and can exploit these during training to make the model more robust at deployment time. This technique is known as outlier exposure (OE) and has been shown in the literature to improve novelty detection performance. In this project, two OE strategies, one established and one speculative, are explored in two different deployment scenarios: one where OE data are representative of the OOD examples that will be seen during test time and one where they are not. Lastly, an attempt is made to better understand these OE techniques when used in tandem with deep margin-based classifiers, an approach that has not yet appeared in the literature.

<hr>

You can read my Master's paper [here](https://ir.library.oregonstate.edu/concern/graduate_projects/ks65hn11h).
