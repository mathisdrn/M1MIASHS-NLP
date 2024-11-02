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

Aujourd'hui, les gens utilisent des services de messagerie électronique tels que Gmail, Outlook, AOL Mail, etc. pour communiquer entre eux le plus rapidement possible afin d'envoyer des informations et des lettres officielles. Le courrier indésirable ou le spam est un défi majeur pour ce type de communication, souvent envoyé par des botnets dans le but de faire de la publicité, de nuire et de voler des informations en masse à différentes personnes. Recevoir quotidiennement des courriels indésirables remplit la boîte de réception. Par conséquent, la détection du spam est un défi fondamental, jusqu'à présent, de nombreux travaux ont été réalisés pour détecter le spam en utilisant des méthodes de regroupement et de catégorisation de texte. Dans ce projet, nous utilisons les bibliothèques de traitement du langage naturel spaCy et NLTK ainsi que 3 algorithmes d'apprentissage automatique : le Naive Bayes (NB) en langage de programmation Python pour détecter les courriels indésirables collectés à partir d'un ensemble de données Kaggle. Les observations montrent un taux de précision de 96% pour l'algorithme Naive Bayes dans la détection du spam.

Ham : C'est un terme utilisé pour décrire les emails qui sont authentiques et qui ne sont pas considérés comme du spam. Les emails de type "ham" sont ceux que les utilisateurs veulent recevoir, comme des correspondances personnelles, des newsletters auxquelles ils se sont abonnés, etc.

Spam : Ce terme est utilisé pour décrire les emails indésirables, souvent envoyés en masse à une grande quantité de destinataires sans leur consentement. Le spam peut inclure des publicités non sollicitées, des offres frauduleuses, ou même des emails contenant des logiciels malveillants.

## Source des données

Les données sont issues d'un jeu de données disponible sur [Kaggle](https://www.kaggle.com/datasets/rajnathpatel/multilingual-spam-data/).

:::{table} Extrait du jeu de donnée original
:label: table_original_data_head1
:align: center
![](#table_original_data_head)
:::

:::{table} Extrait du jeu de donnée d'intérêt
:label: table_data_head1
:align: center
![](#table_data_head)
:::

## Bases théoriques
### Différence de proportion des classes 
Description des bases du CountVectorizer et du TdifTransformer

Définir risque première espèce et risque seconde espèce
Définition de risque premier 

Présenter le trade-off entre les deux

Expliquer les risques lorsque les classes du jeu d'entraînement sont trop disproportionné :
- le modèle n'apprend pas suffisamment à distinguer la classe minoritaire
- l'évaluation d'un modèle de classification binaire est biaisé car la classification de la classe minoritaire est sous représenté, etc.

Présenter les techniques permettant d'améliorer l'entraînement d'un modèle avec des classes disproportionés : technique de resampling (SMOTE), paramètre de poids des classes dans la regression logistique et SVC (class weight = 'balanced' (voir doc)), ajustement du seuil en sortie de modèle

Seuil de décision : C'est un chiffre compris entre 0 et 1, mais en général nous utilisons un seuil 0.5, de manière à ajuster le risque de première et seconde espèce à celui désiré.

## Les modèles

### Preprocessing 

Les étapes de Preprocessing sont :
- Le nettoyage de données : conversion des majuscules en minuscules pour harmoniser le texte. Il peut y avoir aussi par exemple la suppresion des ponctuations.
- Etapes normalisation du texte
    - Tokenisation : division du texte en unités linguistiques, appelés tokens. Cela permet de transformer le texte en sous-parties ou en séquence de mots
    - Stops words : signifie suprimmer des mots très fréquents dans le texte qui ne sont souvent pas pertinents pour l'analyse
    - Stemming : Tronque les mots à leur racine, sans prendre en compte leur contexte
- La dernière étape consiste à transformer les textes en valeurs numérique en utilisant par exemple la commande TdifTransformer.

### 

```{mermaid}

flowchart LR
    A[(Labeled data)] --> B[CountVectorizer]
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
    style B fill:#d1c4e9,stroke:#8e44ad,stroke-width:2px
    style C fill:#b2dfdb,stroke:#16a085,stroke-width:2px
```

### text feature extraction 

### description des modèles

#### Methode de Naïve Bayes
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

#### Méthode de Logreg

La régression logistique est une méthode de machine learning. Elle s'appuie sur un modèle statistique permettant de prédire la probabilité qu'un événement se produise. Elle est également largement utilisée comme méthode de classification pour déterminer si une observation appartient à une classe ou à une autre.

Fonctionnement de la méthode :

La régression logistique utilise une fonction logistique pour transformer une combinaison linéaire de variables prédictives en une probabilité comprise entre 0 et 1.

Modèle : Le modèle évalue une probabilité en se basant sur plusieurs variables d'entrée. Par exemple, pour déterminer si un e-mail est un spam, il peut prendre en compte des caractéristiques telles que le nombre de mots-clés, la longueur de l'e-mail et d'autres éléments pertinents.

Avantage :

Méthode simple et facile à utiliser. Elle est efficace pour des ensembles de données de taille modérée et est souvent utilisée comme méthode de référence pour les problèmes de classification.


Inconvénient :
Peut être moins performante que d'autres algorithmes sur des données complexes.

#### Support vector Clustering

Le Support Vector Clustering est une méthode de classification largement utilisée en machine learning. Elle permet de regrouper des données en trouvant la meilleure séparation possible entre différentes catégories.


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

:::{figure} #figure_pr_LR
:label: figure_pr_LR1
Precision-Recall Curve of Logistic Regression model
:::

:::{figure} #figure_roc_LR
:label: figure_roc_LR1
ROC Curve of Logistic Regression model
:::

### Support-vector clustering

:::{table} Classification report of Support-vector clustering model
:label: table_report_SVC1
:align: center
![](#table_report_SVC)
:::

:::{figure} #figure_pr_SVC
:label: figure_pr_SVC1
Precision-Recall Curve of Support-vector clustering model
:::

:::{figure} #figure_roc_SVC
:label: figure_roc_SVC1
ROC Curve of Support-vector clustering model
:::

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