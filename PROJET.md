---
title: Spam Classification Model
subject: Natural Language Processing
short_title: Spam Detection
banner: ./assets/banner.png
license: CC-BY-4.0
---

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

Le TF-IDF Transformer (Transformateur de Fréquence de Terme-Fréquence Inverse de Document) transforme les matrices de comptage en représentations pondérées, mettant en avant les termes les plus informatifs d'un document tout en réduisant l'impact des mots courants qui apportent peu de distinction au sein du corpus.

L’objectif principal du TF-IDF Transformer est de transformer des données textuelles en données numériques, de manière à mettre en avant les mots les plus importants pour chaque document. Ce procédé est crucial car il rend les données exploitables par les modèles de machine learning, tout en augmentant leur capacité de discrimination entre les documents.

:::{warning}
À contrario des étapes précedentes, cette étape nécessite un corpus de données de manière à calculer les fréquences des mots. Cette étape est donc réalisée après la séparation des données en jeu d'entrainement et de test afin d'éviter les fuites de données entre les deux jeux de données.
:::

## Les modèles

### Méthode de Bayes naïve

Le classificateur de Naïve Bayes est un algorithme de classification supervisé largement utilisé en machine learning. Sa méthode repose sur le théorème de Bayes, appliqué à des problèmes de classification où il est particulièrement efficace, notamment dans le traitement des textes et la détection de spams.

La méthode Bayes naïve est un algorithme d'apprentissage supervisé utilisé principalement pour des problèmes de classification binaire (ou multiclasse). Elle repose sur le théorème de Bayes, qui permet de calculer la probabilité qu'une observation appartienne à une classe donnée, en tenant compte de ses caractéristiques. Le terme "naïve" provient de l'hypothèse simplificatrice selon laquelle toutes les caractéristiques sont conditionnellement indépendantes les unes des autres, ce qui n'est généralement pas le cas dans la réalité. Par exemple, les mots dans un texte sont souvent liés par leur contexte, mais l'algorithme de Bayes naïf traite chaque mot comme une caractéristique indépendante. Ainsi, le modèle calcule la probabilité de chaque mot en fonction de la catégorie, puis les combine pour estimer la probabilité globale que le texte appartienne à une catégorie donnée.

Avantages :
- Très rapide et efficace même avec de grands ensembles de données.
- Fonctionne bien avec des variables catégorielles et pour des problèmes de filtrage de texte, comme la détection de spam.
- Facile à interpréter et à mettre en œuvre.

Inconvénients :
- L'hypothèse d'indépendance entre les caractéristiques est souvent irréaliste, ce qui peut réduire les performances.
- Peut ne pas être adapté aux situations où les relations entre les caractéristiques sont importantes.
- Sensible à des échantillons de données avec de faibles fréquences de certaines catégories, nécessitant parfois un lissage pour améliorer la robustesse.

### Régression logistique

La régression logistique est une méthode d'apprentissage supervisé largement utilisée pour la classification binaire. Elle modélise la probabilité qu'une observation appartienne à une des deux classes en utilisant une fonction logistique (ou sigmoïde) pour transformer une combinaison linéaire des caractéristiques en une probabilité. La fonction sigmoïde prend une valeur comprise entre 0 et 1, ce qui permet de prédire l'appartenance à une classe en appliquant un seuil, généralement 0,5. L'objectif de l'algorithme est d'ajuster les coefficients de la combinaison linéaire en maximisant la vraisemblance des observations données.

Avantages :
- Facile à interpréter grâce aux coefficients qui indiquent l'influence des variables explicatives.
- Efficace pour des problèmes de classification linéairement séparables.
- Performant même lorsque les classes sont partiellement chevauchées, tant que la relation est linéaire.

Inconvénients :
- Ne fonctionne pas bien lorsque la relation entre les variables explicatives et la probabilité d'appartenance à une classe est non linéaire.
- Peut être sensible aux valeurs aberrantes, nécessitant un nettoyage préalable des données.
- Peut nécessiter une régularisation (comme la pénalisation L1 ou L2) pour éviter le surajustement dans le cas de nombreuses variables explicatives.

### Support-vector classification

L'algorithme Support Vector Classification (SVC) est une méthode d'apprentissage supervisé utilisée pour résoudre des problèmes de classification binaire. Il repose sur le concept des marges maximales, c'est-à-dire qu'il cherche à séparer deux classes dans l'espace des caractéristiques en traçant un hyperplan qui maximise la distance (ou la marge) entre les points de chaque classe les plus proches de cette hyperplan, appelés vecteurs de support. Lorsque les classes ne sont pas linéairement séparables, l'algorithme utilise des noyaux pour projeter les données dans un espace de dimension supérieure où elles peuvent être séparées.

Avantages :
- Efficace dans des espaces de grande dimension.
- Peut être modifié pour des cas non linéaires grâce aux fonctions noyau.
- Utilise seulement les vecteurs de support, ce qui le rend plus économe en mémoire.
Inconvénients :
- Peut être sensible aux choix des paramètres (comme C et le type de noyau).
- Moins performant avec de très grands ensembles de données ou quand les classes sont fortement chevauchées.

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

### Definitions 

Dans le cadre d'une classification binaire, on peut définir les termes suivants :

Précision
: La précision (ou en anglais accuracy) désigne la proportion de prédictions correctes parmi toutes les prédictions effectuées par le modèle. Elle permet d'évaluer la qualité des prédictions positives du modèle.

Rappel
: Le rappel (ou en anglais recall) mesure la proportion de vrais positifs correctement identifiés parmi tous les éléments réellement positifs. Il permet d'évaluer la capacité du modèle à détecter tous les cas positifs.

F1-score
: Le F1-score est la moyenne harmonique de la précision et du rappel, permettant d'évaluer la performance globale d'un modèle en équilibrant ces deux métriques. Un score proche de 1 indique une excellente performance.

Weighted Average F1-score
: Le F1-score moyen pondéré est une mesure utilisée pour évaluer les performances d'un modèle de classification binaire. Il prend en compte le déséquilibre des classes en calculant une moyenne pondérée des F1-scores de chaque classe, où les poids sont proportionnels au nombre d'instances de chaque classe. Cela permet d'obtenir une évaluation plus représentative des performances globales du modèle, en particulier lorsque les classes sont déséquilibrées.

Les modèles de classifications binaires utilisées produisent en sortie un chiffre entre 0 et 1. Pour transformer ces chiffres en classes, on utilise un seuil de décision. Par défaut, ce seuil est de 0.5. Si le chiffre est supérieur à 0.5, la prédiction est de 1, sinon elle est de 0. Certains graphiques permettent de visualiser les performances des modèles selon ce seuil :

Courbe de précision-rappel
: La courbe de précision-rappel affiche la précision et le rappel en fonction du seuil de décision. Elle permet d'évaluer la performance du modèle en fonction de ces deux métriques. Plus la courbe est proche du coin supérieur droit, meilleure est la performance du modèle.

Courbe ROC (Receiver Operating Characteristic)
: La courbe ROC permet d'évaluer la performance d'un classificateur binaire, c’est-à-dire un système conçu pour diviser des éléments en deux catégories distinctes en fonction de certaines caractéristiques. Cette mesure est illustrée par une courbe qui affiche le taux de vrais positifs en fonction du taux de faux positifs. Elle permet d'observer la capacité du modèle à correctement distinguer les classes positives et négatives et de visualier l'arbitrage réalisé entre les taux de faux positifs et de vrais négatifs. De plus l'aire sous la courbe (AUC) permet de quantifier la performance du modèle : plus la valeur est proche de 1, plus le modèle est performant pour déterminer les classes positives et négatives.

### Bayes naïf

La [](#table_report_bayes1) montre les résultats de la classification par le modèle de Bayes naïf. On observe que :
- $90,62 \%$ des *spams* sont correctement identifiés
- $99,72 \%$ des *hams* sont correctement identifiés
- $98,07 \%$ des observations classifiées en tant que *spam* sont des *spams*
- $98,57 \%$ des observations classifiées en tant que *ham* sont des *hams*
- Le score F1 moyen pondéré est de $98,48 \%$ 

Ces chiffres montrent que le modèle de Bayes naïf est très performant pour classer les messages en *spam* et *ham*.

:::{table} Classification report of Naive Bayes model
:label: table_report_bayes1
![](#table_report_bayes)
:::

La [](#figure_pr_bayes1) présente la courbe précision-rappel pour le modèle de Bayes naïf. Cette figure permet d'observer les compromis possible entre la précision et le rappel en fonction du seuil de décision.

:::{figure} #figure_pr_bayes
:label: figure_pr_bayes1
Precision-Recall Curve of Naive Bayes model
:::

La [](#figure_roc_bayes1) présente la courbe ROC pour le modèle de Bayes naïf. On constate que l'aire sous la courbe est de $0,99$. Ce chiffre une excellente performance du modèle.

:::{figure} #figure_roc_bayes
:label: figure_roc_bayes1
ROC Curve of Naive Bayes model
:::

Pour conclure, le modèle de Bayes Naïf présente d'excellentes performances. Il est capable de distinguer les messages *spam* des messages *ham* avec une grande précision et un bon rappel. La courbe ROC montre une forte sensibilité et une faible probabilité de faux positifs, ce qui indique que le modèle est très performant pour distinguer les deux classes.

### La régression logistique

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