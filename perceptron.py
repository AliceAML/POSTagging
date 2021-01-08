# VERSION mise à jour partielle de a : taux d'erreurs vers 16%
from random import shuffle

def perceptron_train(training_set, MAX_EPOCH=3):
    
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    # initialisation de a (poids totaux)
    a = {}
    for tag in tags:
        a[tag] = {}

    # initialisation des vecteurs de poids
    w = {}
    for tag in tags:
        w[tag] = {}

    n_update = 0 # nombre de mots sur lequel l'entraînement a été effectué

    last_update = {} # dictionnaire de dictionnaire suivant la même structure que les vecteurs de poids a et w
    # et qui stocke la valeur de n lors de la dernière modification d'un poid
    for tag in tags:
        last_update[tag] = {}
    
    for i in range(0, MAX_EPOCH):
        shuffled_set = training_set.copy() # copie du training set
        shuffle(shuffled_set) # mélange du training set

        for word in shuffled_set:
            n_update += 1 # on compte le nb de mots déjà vus

            vec, gold = word
            prediction = predict(vec, w) # trouve étiquette plus probable avec les poids w

            if not prediction == gold: # si le gold_label n'est pas égal à celui prédit
                # on ignore les mots dont le gold_label est "_" ("au" et "du") car ils sont ensuite analysés comme "à le" et "de le"

                for feat in vec: # pour chaque feature du mot

                    # on met à jour a : ajout de l'ancienne valeur dans w * le nombre de fois où elle n'a pas été modifiée
                    a[gold][feat] = a[gold].get(feat, 0) + w[gold].get(feat,0)*(n_update-last_update[gold].get(feat,0))
                    a[prediction][feat] = a[prediction].get(feat, 0) + w[prediction].get(feat,0)*(n_update-last_update[prediction].get(feat, 0))
                     # on modifie le dernier update de l'élément du vecteur
                    last_update[gold][feat] = n_update
                    last_update[prediction][feat] = n_update

                    # on modifie les poids de w pour les 2 étiquettes concernées
                    w[gold][feat] = w[gold].get(feat,0) + vec.get(feat) #  on ajoute x_i à chaque poids de l'étiquette correcte
                    w[prediction][feat] = w[prediction].get(feat,0) - vec[feat] #  on retire x_i à chaque poids de l'étiquette mal prédite

                   

    # mise à jour finale tous les poids qui n'ont pas été modifiés lors de la dernière update
    for tag in a:
        for feat in a[tag].keys():
            if last_update[tag][feat] < n_update: 
                a[tag][feat] = a[tag].get(feat, 0) + w[tag][feat]*(n_update - last_update[tag].get(feat,0))
                last_update[tag][feat] = n_update
        
    return a



    ## VERSION 2
    # msie à jour de A en entier
    # beaucoup trop long à faire tourner

from random import shuffle    

def perceptron_train(training_set, MAX_EPOCH=3):
    
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    # initialisation de a (poids totaux)
    a = {}
    for tag in tags:
        a[tag] = {}

    # initialisation des vecteurs de poids
    w = {} # TODO defaultdictionary
    for tag in tags:
        w[tag] = {}

    n_update = 0 # nombre de mots sur lequel l'entraînement a été effectué

    last_update = 0 # dictionnaire de dictionnaire suivant la même structure que les vecteurs de poids a et w
    # et qui stocke la valeur de n lors de la dernière modification d'un poid

    
    for i in range(0, MAX_EPOCH):
        shuffled_set = training_set.copy() # copie du training set
        shuffle(shuffled_set) # mélange du training set

        for word in shuffled_set    :
            n_update += 1 # on compte le nb de mots déjà vus

            vec, gold = word
            prediction = predict(vec, w) # trouve étiquette plus probable avec les poids w

            if not prediction == gold: # si le gold_label n'est pas égal à celui prédit
                # on ignore les mots dont le gold_label est "_" ("au" et "du") car ils sont ensuite analysés comme "à le" et "de le"

                # mise à jour de a : on ajoute chaque poids * le nombre d'updates sans modifications
                for tag in tags:
                    for feat in w[tag]:
                        if not w[tag].get(feat,0) == 0:
                            a[tag][feat] = a[tag].get(feat,0) + w[tag].get(feat,0)*(n_update-last_update)

                # modification de w
                for feat in vec: # pour chaque feature du mot
                    # on modifie les poids de w pour les 2 étiquettes concernées
                    w[gold][feat] = w[gold].get(feat,0) + 1 #  on ajoute x(i) à chaque poids de l'étiquette correcte
                    w[prediction][feat] = w[prediction].get(feat,0) - 1 #  on retire x(i) à chaque poids de l'étiquette mal prédite

                # modification de last_update
                last_update = n_update

    # mise à jour finale de a 
    for tag in w:
        for feat in w[tag]:
            a[tag][feat] = a[tag].get(feat, 0) + w[tag][feat]*(n_update - last_update)
    

    return a
