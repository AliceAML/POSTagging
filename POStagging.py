from random import shuffle
from collections import defaultdict 
import pandas as pd
from time import time
import matplotlib.pyplot as plt

taux = []


def load_corpus(file):
    with open(file, "r", encoding = "utf8") as f: 
        content = f.read() # chargement du corpus
    content = content.split("\n\n") # séparation en phrases
    corpus = []
    for phrase in content: # pour chaque phrase
        phrase_dico = {"mots" : [], "gold_labels" : []} # liste qui contiendra 1 dictionnaire par mot de la phrase
        for line in phrase.splitlines():
            if not line.startswith("#"): # on ignore les lignes qui commencent par #
                features = line.split("\t")
                phrase_dico["mots"].append(features[1])
                # phrase_dico["lemme"].append(features[2])
                phrase_dico["gold_labels"].append(features[3])
        corpus.append(phrase_dico)
    return corpus

def feature_extraction(corpus, feat_mots=True, feat_maj=True, feat_non_alpha=True, feat_long=True, feat_suff=True):

    corpus_features = []

    list_vb = ["iser","ifier", "oyer","ailler", "asser","eler", "eter","iller", "iner","nicher", "ocher","onner",
    "otter","oter", "ouiller"]
    list_adj = ["ain", "aine","ains", "aines","aire", "aires","é", "ée","ées", "és","iel", "iels","uel", "uels", 
    "lle", "lles","els", "el" "al", "ales", "al", "ial", "aux","iaux", "er","ers", "ère","ères", "ier", "iers", 
    "esque","esques", "eur","eurs", "euse","euses", "ieux","ueux", "if", "ifs","ive", "ives","in", "ins","ine", 
    "ines","iques", "ique","atoire", "u","ue", "us","ues", "issime","issimes","able","ible", "ibles","ables", 
    "uble","ubles", "ième","ièmes", "uple"]
    list_noun = ["ade", "ades", "age", "ages","aille", "ailles", "aison", "ison", "isons","oison", "ation", 
    "itions", "ition", "ssion", "sion","xion", "isation","ment", "ement","erie", "eries","ure","ures","ature", 
    "atures","at", "ance","ence", "escence","ité", "eté","té", "ie","erie", "esse", "ise", "eur","isme", "iste",
    "istes","eurs", "seur","seurs", "isseur","isseurs", "isateur","euse", "euses","isseuse", "isseuses", "atrice",
    "atrices","ier", "iers","ière", "ières","aire","aires","ien", "iens","ienne", "iennes","iste", "istes","er", 
    "ers","eron", "erons","eronne","trice","oir", "oire","oires", "oirs","ier", "iers","ière","ières","erie",
    "eries","anderie","aire", "aires","ain", "aines", "ée","ées","aille", "ard","asse", "asses", "assier","âtre",
    "aut","eau", "eaux","ceau", "ereau","eteau", "elle","elles", "et","elet","ets","ette","elette","ettes", 
    "elettes","in", "ins","otin", "ine","ines", "illon","on","ons","ille", "erole","eroles", "ole","oles", "iche"]

    for phrase in corpus: # ajout des features additionnelles
        for prev, word, suiv in zip(["START"] + phrase["mots"][:-1], phrase["mots"], phrase["mots"][1:] + ["END"]):
            # création de triplets (mot précédent, mot, mot suivant)
            # avec "START" en prev pour le 1er mot
            # et "END" en suiv pour le dernier

            # dictionnaire de features du mot
            if feat_mots :
                features_mot = { 
                    # on récupère le gold_label correspondant
                    f"mot - {word.lower()}" : 1,
                    f"prec - {prev.lower()}" : 1,
                    f"mot_suiv - {suiv.lower()}" : 1,
                    }
            else:
                features_mot = {}
            
            if feat_maj:
                if word.istitle(): features_mot["maj"] = 1 
                if word.isupper(): features_mot["all_caps"] = 1

            if feat_non_alpha:
                if any(char.isdigit() for char in word): features_mot["num"] = 1 # mieux que isnumeric(), car renvoie false si espace (40 000) ou virgule (50,6) par ex
                if not word.isalnum(): features_mot["nonAlphanum"] = 1

            if feat_long:
                if len(word) <= 3: features_mot["court"] = 1 
                if len(word) > 3: features_mot["long"] = 1
                if len(word) == 1: features_mot["un_car"] = 1
            
            if feat_suff:
                if word.endswith("ment"): features_mot["suff_adv"] = 1
                if any(word.endswith(elem) and len(word) != len(elem) for elem in list_noun): features_mot["suff_noun"] =1 
                if any(word.endswith(elem) and len(word) != len(elem) for elem in list_adj): features_mot["suff_adj"] = 1
                if any(word.endswith(elem) for elem in list_vb): features_mot["suff_vb"] = 1
                # on vérifie la longueur du mot pour être sûr que ce soit un suffixe car on peut avoir le mot                      age avec le suffixe age par exemple ou bien aux
            
            # ajout au corpus
            corpus_features.append(features_mot)

    return corpus_features # renvoie les features transformés en vecteurs one-hot

def add_gold(features, corpus, addMot=False):
    '''Ajoute les gold labels pour créer un corpus d'entraînement / de test'''
    i = 0
    gold_corpus = []
    for phrase in corpus:
        for word, pos_gold in zip(phrase["mots"],phrase["gold_labels"]):
            if not pos_gold == "_": # on ignore les mots sans gold_labels
                if addMot:
                    gold_corpus.append((features[i], pos_gold, word))
                else:
                    gold_corpus.append((features[i], pos_gold))
            i += 1

    return gold_corpus

def predict(word_features, weights):
    """Renvoie l'étiquette avec le plus gros score (argmax)"""
    scores = {}
    for tag, w in weights.items():
        scores[tag] =  sum(word_features[feat]*w[feat] for feat in word_features)
    return max(scores, key=scores.get)

def perceptron_train(training_set, MAX_EPOCH=3):
    
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    # initialisation de a (poids totaux)
    a = defaultdict(lambda: defaultdict(int)) # source : https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict

    # initialisation des vecteurs de poids
    w = {}
    for tag in tags:
        w[tag] = defaultdict(int)

    n_update = 0 # nombre de mots sur lequel l'entraînement a été effectué

    last_update = defaultdict(lambda: defaultdict(int))  # dictionnaire de dictionnaire suivant la même structure que les vecteurs de poids a et w
    # et qui stocke la valeur de n lors de la dernière modification d'un poid
    
    for i in range(0, MAX_EPOCH):
        shuffled_set = training_set.copy() # copie du training set
        shuffle(shuffled_set) # mélange du training set

        for x in shuffled_set:
            n_update += 1 # on compte le nb de mots déjà vus

            vec = x[0]
            gold = x[1]

            prediction = predict(vec, w) # trouve étiquette plus probable avec les poids w

            if not prediction == gold: # si le gold_label n'est pas égal à celui prédit

                for feat in vec: # pour chaque feature du mot

                    # on met à jour a : ajout de l'ancienne valeur dans w * le nombre de fois où elle n'a pas été modifiée
                    a[gold][feat] += w[gold][feat]*(n_update-last_update[gold][feat])
                    a[prediction][feat] += w[prediction][feat]*(n_update-last_update[prediction][feat])
                     # on modifie le dernier update de l'élément du vecteur
                    last_update[gold][feat] = n_update
                    last_update[prediction][feat] = n_update

                    # on modifie les poids de w pour les 2 étiquettes concernées
                    w[gold][feat] += 1 #  on ajoute x_i à chaque poids de l'étiquette correcte
                    w[prediction][feat] -= 1 #  on retire x_i à chaque poids de l'étiquette mal prédite
                   

    # mise à jour finale tous les poids qui n'ont pas été modifiés lors de la dernière update
    for tag in a:
        for feat in a[tag]:
            a[tag][feat] += w[tag][feat]*(n_update - last_update[tag][feat])

    return a

def test(corpus, poids):
    """Prédit les étiquettes et renvoie un taux d'erreur"""
    nb_erreurs = 0
    for word in corpus:
        vec = word[0]
        gold = word[1]
        prediction = predict(vec, poids)
        if not gold == "_" and not prediction == gold:
            nb_erreurs +=1

    return nb_erreurs/len(corpus)

def test_features(corpus, poids, test_mots=False, test_maj=False, test_non_alpha=False, test_long=False, test_suff=False):
    '''Effectue les 3 étapes nécessaires pour effectuer le test'''
    features = feature_extraction(corpus, feat_mots=test_mots, feat_maj=test_maj, feat_non_alpha=test_non_alpha, feat_long=test_long, feat_suff=test_suff)
    features_gold = add_gold(features, corpus)
    tx_erreur = test(features_gold, poids)
    
    return tx_erreur

def matrice_confusion(corpus_feat, poids):
    predictions = []
    gold = []

    for word in corpus_feat:
        pred = predict(word[0], poids)
        predictions.append(pred)
        gold.append(word[1])
        # if pred != 'PROPN' and word[1] == 'PROPN':
        #     with open('filename.txt', 'a', encoding="utf-8") as f:
        #         print(f"{pred} au lieu de PROPN : {word[0]} | la PROPN: {poids_gsd_train['PROPN']['prec - la']} |maj PROPN: {poids_gsd_train['PROPN']['maj']} | le PROPN: {poids_gsd_train['PROPN']['prec - le']}",file = f)
              
                
    preds = pd.Series((item for item in predictions), name = "Prédictions")
    refs = pd.Series((item for item in gold), name = "Références")
    matrice_confusion = pd.crosstab(refs, preds, margins=True)
    print(matrice_confusion)

def getVoc(corpus):
    '''Renvoie le vocabulaire (set)'''
    voc = set()
    for phrase in corpus:
        for mot in phrase["mots"]:
            voc.add(mot)
    return voc

def test_hors_voc(features_gold_mot, poids, voc):
    """Prédit les étiquettes des mots hors vocabulaire et renvoie un taux d'erreur"""
    nb_erreurs_hors_voc = 0
    nb_erreurs_in_voc = 0

    mots_hors_voc=0
    mots_in_voc = 0

    for vec, gold, mot in features_gold_mot:
        if mot in voc:
            mots_in_voc +=1
            prediction = predict(vec, poids)
            if not gold == "_" and not prediction == gold:
                nb_erreurs_in_voc +=1
        else:
            mots_hors_voc +=1
            prediction = predict(vec, poids)
            if not gold == "_" and not prediction == gold:
                # print(mot, vec, prediction, gold, sep="\t", end="\n")
                nb_erreurs_hors_voc +=1


    return (nb_erreurs_hors_voc/mots_hors_voc, nb_erreurs_in_voc/mots_in_voc)

def temps():

    for i in range(1,6):
        t0 = time()
        poids_gsd_train = perceptron_train(gsd_train_features_gold, MAX_EPOCH=i)
        tx_erreur = test(gsd_dev_features_gold, poids_gsd_train)
        taux.append(tx_erreur)
        print(f"{i} epochs : {tx_erreur:.2%} d'erreurs - temps training + test : {time()-t0:.2f}s")

def graph():

    x_range = range(1,6)
    plt.figure()
    plt.plot(x_range,taux)
    plt.ylabel("Taux d'erreur")
    plt.xlabel("MAX_EPOCH")
    plt.title("Taux d'erreur selon le nombre d'itérations pendant l'apprentissage")
    plt.show()



if __name__ == "__main__":

    # on charge nos différents corpus in-domain
    gsd_train = load_corpus("corpus-in-domain/fr_gsd-ud-train.conllu")
    gsd_test = load_corpus("corpus-in-domain/fr_gsd-ud-test.conllu")
    gsd_dev = load_corpus("corpus-in-domain/fr_gsd-ud-dev.conllu")
    # corpus hors domaine
    oral_dev = load_corpus("corpus-hors-domaine/spoken/fr_spoken-ud-dev.conllu")
    old_dev = load_corpus("corpus-hors-domaine/old/fro_srcmf-ud-dev.conllu")

    # extraction des caractéristiques des différents corpus et ajout des étiquettes in-domain
    gsd_train_features = feature_extraction(gsd_train)
    gsd_train_features_gold = add_gold(gsd_train_features, gsd_train)
    gsd_dev_features = feature_extraction(gsd_dev)
    gsd_dev_features_gold = add_gold(gsd_dev_features, gsd_dev)
    gsd_test_features = feature_extraction(gsd_test)
    gsd_test_features_gold = add_gold(gsd_test_features, gsd_test)
    # extraction hors-domaine
    oral_dev_features = feature_extraction(oral_dev)
    oral_dev_features_gold = add_gold(oral_dev_features, oral_dev)
    old_dev_features = feature_extraction(old_dev)
    old_dev_features_gold = add_gold(old_dev_features, old_dev)


    # corpus d'entrainement passé dans le perceptron
    poids_gsd_train = perceptron_train(gsd_train_features_gold)

    # affichage des tests
    print(f"\nTaux d'erreur sur corpus d'évaluation in-domain : {test(gsd_test_features_gold, poids_gsd_train):.3%}")
    print("Affichage de la matrice de confusion pour train : \n")
    matrice_confusion(gsd_dev_features_gold, poids_gsd_train)

    print(f"\nTaux d'erreur sur corpus d'évaluation hors domaine (oral) : {test(oral_dev_features_gold, poids_gsd_train):.3%}")
    print("Affichage de la matrice de confusion pour le corpus hors domaine (oral) : \n")
    matrice_confusion(oral_dev_features_gold, poids_gsd_train)

    print(f"\nTaux d'erreur sur corpus d'évaluation hors domaine (ancien francais) : {test(old_dev_features_gold, poids_gsd_train):.3%}")
    print("Affichage de la matrice de confusion pour le corpus hors domaine (ancien français) : \n")
    matrice_confusion(old_dev_features_gold, poids_gsd_train)
    print("\n")

    # temps() # sert à afficher les temps en fonction des différentes valeurs d'EPOCH
    # graph() # affiche le graphique qui en résulte 
 
    # nouvelle extraction de features mais avec les mots       #
    # pour pouvoir ensuite voir s'ils sont dans le vocabulaire #
    # gsd_dev_features_gold_mot = add_gold(gsd_dev_features, gsd_dev, addMot=True)
    # resultats_test_voc = test_hors_voc(gsd_dev_features_gold_mot, poids_gsd_train, getVoc(gsd_train))
    # print(f"Taux d'erreur hors voc : {resultats_test_voc[0]:.3%}")
    # print(f"Taux d'erreur dans voc : {resultats_test_voc[1]:.3%}")

