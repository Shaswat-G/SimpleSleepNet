1. Standardize the final layer of all encoder backbones to output normalized (why? CRL using SupCon Loss and NT-Xent Loss expects normalized embeddings and the Latent space evaluator expects normalized embeddings) 128 D (latent-space dimension - configurable parameter). Can use code from the projection head - flattening, adaptiveavgpooling and normalizing.
2. Ensure that the one-fold trainer works as expected, no test data is left out or no validation data is left out.
3. Modify the training pipeline Use the pretrainer to just train the backbone, remove the other training paradigms without the projection heead.
4. Add an MLp classifier to the classifiers.py / or not required and just modify the traiing pipeline to generate embeddings, then train an MLP classifier.
5. Divide the experiments into GPU and CPU with logging.





Training pipeline

1. Pre-train the encoder from the main model on MP
2. Generate Embeddings - save encoder checkpoint (optionally)
3. Train Classifier (the same standard) - save checkpoint separately (optional)
4. evaluate





1. CNN backbone, we were able to get the forward pass to generate [B (10) x Embedding (128) x 1 (spatial dimension)] and we used a 1-d convolutional layer. In the train_mp, our latent space is learning 128x6 objects rather than 128x1 objects. We are choosing to arbitrarily squash (pool) the 128x6 mathematical object into 128x1 vector embedding (hoping ) it preserves the information richness, feature capturing anyway.
2. We modifed the encoder architecture to generate 128x6 objects.
3. The else in the trainning mode now now generates a normalized 128x1 mebeddings (wehter pretrian or anything else.)