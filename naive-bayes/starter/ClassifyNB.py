def classify(features_train, labels_train):   
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(features_train, labels_train)
    return classifier
