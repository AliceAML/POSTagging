def make_all_one_hots(corpus_features):
    '''Renvoie un dictionnaire qui associe à chaque mot du corpus un vecteur one hot'''
    
    # création liste de mots
    all_words = set()
    for item in corpus_features:
        all_words.add(item[mot].lower()) # on ne prend pas en compte les majuscules pour les vecteurs one hot

    all_words.update({"start", "end"}) # ajout de START et END, qui ne sont pas dans les mots

    all_words = list(all_words) # transformation du set en liste pour conserver l'ordre

    all_one_hots = {}
    i = 0
    for word in all_words:
        vec = [0]*len(all_words)
        vec[i] = 1
        i+= 1
        all_one_hots[word] = vec

    return all_one_hots


def make_one_hot_features(corpus_features, all_one_hots):
    '''Transforme dictionnaire de features en vecteur one hot'''

    vecteurs = []

    count = 0

    for token in corpus_features:
        count += 1
        print(count)

        vec_item = []

        for key, value in token.items(): # pour chaque feature de notre mot
            if isinstance(value, str) and not key == gold: # si la valeur est une chaîne de caractère > encodage one hot   
                # cette méthode est affreusement lente....

                #for word in all_words:
                #    if word == item[mot]:
                #        vec_item.append(1)
                #    else:
                #        vec_item.append(0)

                # deuxième méthode, peut-être plus rapide
                #one_hot = [0]*len(all_words) # on initialise une liste de zéros de longueur du vocabulaire
                
                #try :
                #    one_hot[all_words.index(value.lower())] = 1 # 1 à la place correspondant au mot
                #except:
                #    pass

                #vec_item += one_hot # on concatène la liste one hot


                # troisième méthode...
                try:
                    vec_item += all_one_hots.get(value)
                except:
                    pass

            elif isinstance(value, bool): # si c'est un booléen, append après l'avoir converti en entier
                vec_item.append(int(value))
            elif isinstance(value, int): # si c'est un entier, append
                vec_item.append(value)
        
        vecteurs.append(vec_item)
    
    return vecteurs
    

#gsd_train_vec = make_one_hot_features(gsd_train_features)
gsd_train_all_one_hots = make_all_one_hots(gsd_train_features)