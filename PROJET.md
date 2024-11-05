---
title: Spam Classification Model
subject: Natural Language Processing
short_title: Spam Detection
banner: ./assets/banner.png
license: CC-BY-4.0
keywords: classifier, NLP, spam
---

+++ {"part": "abstract"}
Dans ce projet recherche nous avons choisi de ...
+++

## Introduction 

Aujourd'hui, les gens utilisent des services de messagerie électronique tels que Gmail, Outlook, etc. pour communiquer entre eux le plus rapidement possible afin d'envoyer des informations et des lettres officielles. Le courrier indésirable ou le spam est un défi majeur pour ce type de communication, souvent envoyé par des botnets dans le but de faire de la publicité, de nuire et de voler des informations en masse à différentes personnes. Recevoir quotidiennement des courriels indésirables remplit la boîte de réception. Par conséquent, la détection du spam est un défi fondamental, jusqu'à présent, de nombreux travaux ont été réalisés pour détecter le spam en utilisant des méthodes de regroupement et de catégorisation de texte. Dans ce projet, nous utilisons les bibliothèques de traitement du langage naturel spaCy et NLTK ainsi que 3 algorithmes d'apprentissage automatique (algorithme de Bayes naïf, la régression logistique et SVM) avec Python afin d'entraîner un classifieur binaire à détecter les courriels indésirables sur un ensemble de données disponible sur Kaggle.

Le terme **ham** est utilisé pour décrire les emails qui sont authentiques et qui ne sont pas considérés comme du spam. Les emails de type "ham" sont ceux que les utilisateurs veulent recevoir, comme des correspondances personnelles, des newsletters auxquelles ils se sont abonnés, etc.

Le terme **spam** désigne quant à lui les emails indésirables, souvent envoyés en masse à une grande quantité de destinataires sans leur consentement. Le spam peut inclure des publicités non sollicitées, des offres frauduleuses, ou même des emails contenant des logiciels malveillants.

## Source des données

Les données sont issues d'un jeu de données disponible sur [Kaggle](https://www.kaggle.com/datasets/rajnathpatel/multilingual-spam-data/).

Décrire les données disponibles

:::{table} Extrait du jeu de donnée original
:label: table_original_data_head1
:align: center
![](#table_original_data_head)
:::

Le jeu de données est filtré afin de ne considérer que les mails écrit en Français. 

:::{table} Extrait du jeu de donnée d'intérêt
:label: table_data_head1
:align: center
![](#table_data_head)
:::

## Bases théoriques

### Classification binaire

Le modèle prédit {0, 1} (ici spam ou ham).
Les valeurs réelles sont dans {0, 1}.

La matrice de confusion est une matrice 2x2 qui permet de visualiser les performances d'un algorithme de classification. Elle contient quatre éléments : les vrais positifs (TP), les faux positifs (FP), les vrais négatifs (TN) et les faux négatifs (FN). Ces éléments sont utilisés pour calculer des métriques telles que la précision, le rappel et le score F1.

Seuil de décision : C'est un chiffre compris entre 0 et 1, mais en général nous utilisons un seuil 0.5, de manière à ajuster le risque de première et seconde espèce à celui désiré.

Expliquer l'ajustement possible du seuil de décision afin d'arbitrer entre les taux de faux positifs et vrai négatif. Expliquer pourquoi cela peut être important (par exemple, dans le cas de la détection de maladies, il est préférable de privilégier un taux de faux positifs élevé pour éviter de passer à côté de cas positifs).

Expliquer les problèmatiques liées aux classes déséquilibrées :
- moins de données d'apprentissage pour certaines classe -> - le modèle n'apprend pas suffisamment à distinguer la classe minoritaire
- la précision est biaisée car la classe majoritaire (ham) est sureprésentée
Mention des techniques de rééquilibrage des classes (sous-échantillonnage, sur-échantillonnage, SMOTE)

### Prétraitement des données

#### Nettoyage des données

- unescape HTML
- retrait des mots d'arrêts

#### Tokenisation 

Processus de transformations des textes en token (unité linguistique)

#### Vectorisation des textes

Description des bases du CountVectorizer

Le CountVectorizer est une méthode de représentation du texte. Elle fait partie des techniques de transformation en sac de mots (Bag of Words), où chaque texte est représenté par les fréquences d’apparition de ses mots, indépendamment de leur ordre.

L'objectif de cette méthode est de convertir des données textuelles en données numériques en utilisant une matrice de comptage des mots. (Partie à retravailler nottament sur le fonctionnement)

#### Stemming / Lemmatisation

Description des bases du stemming et de la lemmatisation

Le stemming et la lemmatisation sont deux techniques utilisés pour réduire des mots à leur forme de base. Cela aide à simplifier les texte.

Le stemming est une technique de traitement des mots qui consiste à supprimer les suffixes (et parfois les préfixes) pour ne conserver que la racine du mot. Cette méthode n’analyse pas le sens des mots, ce qui peut entraîner des erreurs de syntaxe. Son objectif est de simplifier les mots en réduisant les variations morphologiques, mais sans distinction de sens contextuel.


La lemmatisation vise à ramener les mots à leur forme de base, appelée lemme, telle qu’elle apparaît dans le dictionnaire. Contrairement au stemming, la lemmatisation prend en compte la grammaire et le contexte d’usage, permettant d’identifier la forme correcte d’un mot en fonction de son rôle dans la phrase. Elle nécessite donc une compréhension linguistique plus approfondie.

On peut dire que le stemming est plus rapide à utiliser que la lemmatisation mais par contre au niveau de la structure des mots le stemming est beaucoup moins précis.

#### Utilisation de la fréquences des mots pour normaliser les données

Description du TdifTransformer

:::{warning}
À contrario des étapes précedentes, cette étape nécessite un corpus de données de manière à calculer les fréquences des mots. Cette étape est donc réalisée après la séparation des données en jeu d'entrainement et de test afin d'éviter les fuites de données entre les deux jeux de données.
:::

## Les modèles

### Méthode de Bayes naïve

Le classificateur de Naïve Bayes est un algorithme de machine learning supervisé. Il est utilisé comme méthode de 
classification basée sur le théorème de Bayes. Il repose sur une hypothèse simplificatrice selon laquelle chaque 
caractéristique fonctionne indépendamment des autres, ce qui rend les calculs plus rapides.

Comment fonctionne la méthode de Bayes :

Pour classifier des textes, comme dans les applications de détection de spam ou de non-spam pour les e-mails, ou pour la 
catégorisation automatique d'articles, Naïve Bayes modélise la probabilité qu'un texte appartienne à une catégorie spécifique en se 
basant sur les mots qu'il contient. Chaque mot (ou groupe de mots) est considéré comme une caractéristique du texte, et l'algorithme 
estime comment ces mots se distribuent en fonction des différentes catégories à classifier.

Dans Naïve Bayes, l’hypothèse fondamentale est que chaque mot est indépendant des autres. Cette simplification 
réduit considérablement la complexité des calculs. Bien que cette hypothèse soit souvent irréaliste 
(les mots dans une phrase ne sont généralement pas totalement indépendants), elle se révèle efficace dans 
la plupart des cas de classification de texte. En effet, le modèle évalue les probabilités individuelles de chaque mot, 
puis combine ces informations pour déterminer la probabilité globale que le texte appartienne à une catégorie.

Avantages :
Rapide et simple à mettre en œuvre.

Inconvénient :
Difficile à utiliser sur des données complexes.

### Régression logistique

La régression logistique est une méthode de machine learning. Elle s'appuie sur un modèle statistique permettant de prédire la probabilité qu'un événement se produise. Elle est également largement utilisée comme méthode de classification pour déterminer si une observation appartient à une classe ou à une autre.

Fonctionnement de la méthode :

La régression logistique utilise une fonction logistique pour transformer une combinaison linéaire de variables prédictives en une probabilité comprise entre 0 et 1.

Modèle : Le modèle évalue une probabilité en se basant sur plusieurs variables d'entrée. Par exemple, pour déterminer si un e-mail est un spam, il peut prendre en compte des caractéristiques telles que le nombre de mots-clés, la longueur de l'e-mail et d'autres éléments pertinents.

Avantage :

Méthode simple et facile à utiliser. Elle est efficace pour des ensembles de données de taille modérée et est souvent utilisée comme méthode de référence pour les problèmes de classification.

Inconvénient :
Peut être moins performante que d'autres algorithmes sur des données complexes.

### Support-vector classification

Le Support Vector Classification (SVC) est une méthode de classification largement utilisée en machine learning. Elle permet de regrouper des données en trouvant la meilleure séparation possible entre différentes catégories.


Principe de l'algorithme :
Les SVM (Support Vector Machines) identifient la séparation optimale entre les différentes catégories de données. Cette séparation est représentée par un hyperplan qui maximise la distance entre les points de données les plus proches de chaque côté. Ces points, appelés vecteurs de support, sont essentiels pour définir cette marge. En augmentant cette marge, le modèle est mieux équipé pour classer correctement de nouvelles données, améliorant ainsi sa capacité de généralisation.

Fonctionnement de la méthode :
L'algorithme SVM suit trois étapes principales 

Déterminer l'hyperplan optimal en maximisant la marge entre les classes.
Utiliser les vecteurs de support, qui sont les points d'entraînement définissant la marge et influençant l'hyperplan.
Classer les nouvelles observations en fonction de leur position par rapport à cet hyperplan.

Avantage :
Efficace avec un petit nombre de vecteurs de support, ce qui le rend économe en mémoire.

Inconvénient :
Très lent pour des ensembles de données complexes.

### Vue du modèle complet


```{mermaid}
flowchart LR
    A[(Labeled data)] --> G[Unescape HTML]
    G --> B[CountVectorizer]
    B --> C[TfidfTransformer]
    
    subgraph Models
        D([Naive Bayes])
        E([SVC])
        F([Logistic Regression])
    end
    
    C --> D
    C --> E
    C --> F

    style A fill:#ffecb3,stroke:#f39c12,stroke-width:2px
    style G fill:#f7cac9,stroke:#c0392b,stroke-width:2px
    style B fill:#d1c4e9,stroke:#8e44ad,stroke-width:2px
    style C fill:#b2dfdb,stroke:#16a085,stroke-width:2px
```

## Les résultats des modèles

La précision (accuracy) répond à la question : de toutes les prédictions positives, combien sont correctement classifiées ?
Le rappel (recall) répond à la question : de tous les cas positifs, combien sont actuellement correctement classifiés ?   

### Definition des termes : 

Précision : est une mesure utilisée pour indiquer la proportion de prédictions positives qui sont correctes. Une haute précision signifie que lorsque le modèle prédit un élément comme positif, il y a de grandes chances qu'il ait raison.

Recall (rappel) : est une mesure d'évaluation employée dans les modèles de classification pour évaluer la capacité d'un modèle à détecter correctement tous les cas positifs. Autrement dit, elle mesure la proportion des éléments réellement positifs qui ont été identifiés correctement par le modèle.

F1-score : est une mesure d'évaluation qui combine à la fois la précision et le rappel pour donner une indication unique de la performance d'un modèle de classification. Plus il est proche de 1, meilleure est la précision et le rappel.

Precision-Recall curve : est une courbe qui permet d'analyser la relation entre la précision et le rappel. Elle est particulièrement utilisée pour évaluer des modèles de classification binaire. Nous utilisons différents seuils pour évaluer la précision et le rappel. Un modèle idéal afficherait une courbe qui reste élevée et proche de 1 pour la précision et le rappel sur toute la plage des valeurs de rappel, indiquant qu'il maintient une précision élevée tout en capturant presque tous les vrais positifs.

Courbe ROC (Receiver Operating Characteristic) : La courbe ROC permet d'évaluer la performance d'un classificateur binaire, c’est-à-dire un système conçu pour diviser des éléments en deux catégories distinctes en fonction de certaines caractéristiques. Cette mesure est généralement illustrée par une courbe qui affiche le taux de vrais positifs (proportion des cas positifs correctement détectés) en fonction du taux de faux positifs (proportion des cas négatifs incorrectement identifiés comme positifs). Elle s'interprète avec l'aire sous la courbe (AUC) : plus la valeur est proche de 1, plus le modèle est performant pour déterminer les classes positives et négatives. On peut interpréter le modèle par des points ; par exemple, si le point est en (0,0), il n'y a ni vrais positifs ni faux positifs. Le point en (1,1) signifie que le modèle détecte tous les positifs ainsi que tous les négatifs comme positifs, et enfin, un autre point important en (0,1) signifie que tous les positifs ont été détectés et qu'il n'y a aucune erreur.

### Naive Bayes

Possibilité de faire référence à une [table](#table_report_bayes1) ou une figure : [](#figure_pr_bayes1)

:::{table} Classification report of Naive Bayes model
:label: table_report_bayes1
![](#table_report_bayes)
:::

Analyse du tableau.
Précision : Pour la classe "spam", la précision est de 98,07 %. Cela signifie que sur toutes les prédictions du modèle classées comme spam, 98,07 % étaient effectivement des messages "spam". De même, pour le "ham", où la précision est de 98,57 %, cela indique que la quasi-totalité des messages "ham" analysés étaient corrects.

Recall : Le recall pour la classe "spam" est de 90,62 %, ce qui signifie que sur tous les messages réellement étiquetés comme "spam", le modèle a correctement identifié 90,62 % d'entre eux. En revanche, pour les messages catégorisés comme "ham", le recall est de 99,72 %, ce qui démontre une très bonne capacité à détecter les messages "ham".

F1-score : Pour le "spam", il est de 94,10 %, tandis que pour le "ham", il est de 99,14 %. Cela montre que le modèle est plus efficace pour détecter les messages étiquetés comme "ham".

:::{figure} #figure_pr_bayes
:label: figure_pr_bayes1
Precision-Recall Curve of Naive Bayes model
:::

On peut observer sur ce graphique que la courbe bleue (la précision) reste toujours à un niveau de score quasiment proche de 1, tandis que pour la courbe verte (le recall), on constate qu'à mesure que le seuil augmente, le score de recall diminue. Cela signifie que plus le score est élevé, plus la probabilité de détecter un message comme spam diminue, même si la précision demeure satisfaisante.

:::{figure} #figure_roc_bayes
:label: figure_roc_bayes1
ROC Curve of Naive Bayes model
:::

On constate que l'aire sous la courbe est de 0,99, ce qui est très proche de 1, ce qui indique que le modèle est excellent pour identifier les messages "spam" par rapport aux messages "ham". La courbe montre également une forte sensibilité et une faible probabilité de faux positifs, car elle se trouve dès le départ dans l'extrême coin supérieur gauche.

Pour conclure, nous pouvons constater que le modèle de Naive Bayes a montré de très bonnes performances pour classer les messages en "spam" et "ham", avec un rappel et une précision élevés, surtout pour la classe "ham". Cela peut également s'expliquer par le fait qu'il y a moins de messages à classer.

### Logistic regression

:::{table} Classification report of Logistic Regression model
:label: table_report_LR1
![](#table_report_LR)
:::

Précision : La précision pour la classe "ham" de ce modèle est de 97,05 %, ce qui signifie que parmi tous les messages prédits comme "ham", 97,05 % étaient bien classés, ce qui montre qu'il y a très peu de mauvaises classifications. Parmi les messages classés comme "spam", 98,90 % sont correctement identifiés.

Recall : Pour la classe "spam", le recall est de 80,86 %, ce qui signifie que parmi tous les messages réellement étiquetés comme "spam", le modèle en a identifié 80,86 %. Ce chiffre est bien inférieur à celui de la catégorie "ham", où le recall atteint 99,86 %. Cela montre que le modèle est plus performant pour détecter les messages étiquetés comme "ham".

F1-score : Pour le "spam", l'F1-score est de 88,67 %, tandis que pour le "ham", il est de 98,43 %. Cela montre que le modèle est plus efficace pour détecter les messages étiquetés comme "ham".

:::{figure} #figure_pr_LR
:label: figure_pr_LR1
Precision-Recall Curve of Logistic Regression model
:::

On remarque sur le graphique de Precision-Recall que, plus on augmente le seuil, plus la courbe bleue (précision) s'élève, atteignant un score de 1 à partir d'un seuil de 0,4 et se maintenant à ce niveau jusqu'au seuil maximum. En revanche, pour la courbe verte (recall), on constate que plus le seuil augmente, plus le score diminue de façon marquée : dès un seuil de 0,2, le recall commence à chuter, passant sous 0,5 à un seuil de 0,7. Cela suggère que le modèle conserve une bonne précision même à des seuils plus élevés, mais perd de sa capacité à identifier les "spams" au fur et à mesure que le seuil augmente.

:::{figure} #figure_roc_LR
:label: figure_roc_LR1
ROC Curve of Logistic Regression model
:::

On constate que l'aire sous la courbe est de 0.99, très proche de 1, ce qui indique que le modèle est excellent pour distinguer les messages "spam" des messages "ham". La courbe montre également une forte sensibilité et une faible probabilité de faux positifs, car elle se situe dès le départ dans l'extrême coin supérieur gauche.

Le modèle de régression logistique est globalement performant, avec une AUC élevée et une bonne précision pour la classe "spam". Cependant, le rappel pour le "spam" est un peu plus faible que celui pour le "ham", ce qui pourrait indiquer une certaine difficulté à identifier tous les spams. On observe également, grâce au graphique de précision-rappel, que le modèle aura davantage de mal à classer correctement un message comme "spam" ou "ham" lorsque le seuil est élevé.

### Support-vector clustering

:::{table} Classification report of Support-vector clustering model
:label: table_report_SVC1
:align: center
![](#table_report_SVC)
:::

Précision : Pour les messages classés comme "spam", la précision est de 100 %, ce qui signifie que tous les messages prédits comme "spam" sont effectivement des spams. Pour la classe "ham", la précision est de 98,37 %, ce qui indique que parmi tous les messages prédits comme "ham", 98,37 % étaient bien classés.

Recall : Pour la classe "spam", le recall est de 89,29 %, ce qui signifie que parmi tous les messages réellement étiquetés comme "spam", le modèle en a identifié 89,29 %. Ce chiffre est inférieur à celui de la classe "ham", où le recall est de 100 %, signifiant que tous les messages "ham" ont été correctement identifiés.

F1-score : Pour le "spam", le F1-score est de 94,34 %, tandis que pour le "ham", il est de 99,18 %. Cela montre que le modèle est plus efficace pour détecter les messages étiquetés comme "ham".

:::{figure} #figure_pr_SVC
:label: figure_pr_SVC1
Precision-Recall Curve of Support-vector clustering model
:::

On peut observer sur ce graphique que la courbe bleue (la précision) reste toujours à un niveau de score quasiment proche de 1, tandis que pour la courbe verte (le recall), on constate qu'à mesure que le seuil augmente, le score de recall diminue légérement à partir d'un seuil de 0.4. Cela signifie que plus le score est élevé, plus la probabilité de détecter un message comme spam diminue, même si la précision demeure satisfaisante.

:::{figure} #figure_roc_SVC
:label: figure_roc_SVC1
ROC Curve of Support-vector clustering model
:::

On constate que l'aire sous la courbe est de 0,99, très proche de 1, ce qui indique que le modèle est excellent pour distinguer les messages "spam" des messages "ham". La courbe montre également une forte sensibilité et une faible probabilité de faux positifs, car elle se situe dès le départ dans l'extrême coin supérieur gauche.

Le modèle de classification par SVM présente une excellente précision, notamment pour la classe "spam". Toutefois, le rappel pour le "spam" est un peu faible comparé à celui pour la classe "ham", ce qui suggère une légère difficulté à identifier tous les spams.

## Exemple des possibilités du Markdown 

:::{code} python
:filename: 00 Data extraction.ipynb
def some_code(text='Hello World):
    print(text)
:::

:::{caution}
Une p-valeur faible dans le cadre de la corrélation de Pearson ne signifie pas nécessairement qu'il existe une relation de cause à effet entre les deux variables observées. Elle indique simplement que la corrélation observée est statistiquement significative. Ainsi, les hypothèses explicatives fournies sont simplement des hypothèses plausibles et non des conclusions définitives.
:::

Réutilisation d'une sortie d'un Notebook en tant que table :

Some latex : $\frac{p}{n} = \Delta$

1. Some list
    - sub list 1
    - sub list 2
2. List 2
    - sub list 1

## Conclusion

+++ {"part": "data_availability"}
L'ensemble des fichiers et données relatif à ce travail sont disponible en accès libre sur le [dépot GitHub](https://github.com/mathisdrn/head_coach_dismissal) sous licence MIT.
+++