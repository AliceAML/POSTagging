# POSTagging with perceptron from scratch
This project was carried out in January 2020 in a two-person team, at the end of the first semester of a *Linguistics and Computer Science Master's Degree* at the University of Paris.

The projet is a POS tagger based on a perceptron implemented from scratch using Python dictionaries rather than np arrays (we have since moved on to using np arrays and sklearn). It runs on CoNLL-style files.




## Detailed instructions (in French :

### Objectif
L'objectif du projet est de développer classifieur statistique capable de prédire les PoS d'une phrase.

### Travail à effectuer
Certains points seront précisés au fur et à mesure des dernières séances de TP. Les points pouvant déjà être traités sont précédés d'une astérisque

- [x] Développement d'un classifieur multiclasse (régression logistique, perceptron ou SVM). Votre classifieur devra (à minima) implémenter une méthode fit capable d'estimer les paramètres du modèle à partir d'un ensemble d'apprentissage et une méthode predict qui retournera l'étiquette d'une observation
- [x] Extraction des caractéristiques pour le PoS tagging. Les caractéristiques généralement utilisées sont le mot, le mot précédent, le mot suivant, la présence de certains préfixes, la présence de lettre en majuscule ; vous êtes libres de définir toutes les caractéristiques vous semblant pertinentes.
- [x] Évaluation des performances du classifieur et des erreurs fréquentes. On distinguera notamment les performances obtenus sur les mots vue en apprentissage et sur les mots n'apparaissant qu'en test (mot hors vocabulaire).
- [ ] Évaluation de votre PoS tagger sur des données hors-domaine
- [ ] Amélioration des performances sur les données hors-domaines. On pourra considérer les deux méthodes suivantes :
- [ ] Sélection des exemples d'apprentissage en fonction du domaine cible. Pour cela : on apprends un modèle de langue sur le domaine cible (par exemple avec kenLM) et on sélectionne les phrases du corpus d'apprentissage ayant la plus forte probabilité d'avoir été générée par ce modèle de langue
- [ ] Définition de nouvelles caractéristiques robuste au changement de domaine (p. ex. après avoir identifié les erreurs fréquentes)

La séance du 2020-12-10 a été consacrée (en partie) à une description plus détaillée des différentes parties du projet).


### Jeux de données

Les développements se feront sur le corpus [French-GSD](https://universaldependencies.org/treebanks/fr_gsd/index.html) disponible sur le site du projet Universal Dependencies (qui sera considéré comme le corpus in-domain).
Pour les évaluations hors-domaine, on considérera les corpus de test issues des deux treebanks :
- [Old_French-SRCMF](https://universaldependencies.org/treebanks/fro_srcmf/index.html), du projet Universal Dependencies
- [French Spoken](https://universaldependencies.org/treebanks/fr_spoken/index.html), du projet Universal Dependencies


### Informations pratiques

- à réaliser en binome
- date de rendu 4 janvier 2021
