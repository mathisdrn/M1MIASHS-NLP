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

Rappeler l'importance de la détection de spams (sécurité, arnaque, etc.). Rappeler que les hébergeurs des principales boîtes mail assurent ce travail de filtre au quotidien.

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

Description des bases du CountVectorizer et du TdifTransformer

```{mermaid}
---
title: "Spam Classifier Model"
---
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

## Les modèles utilisés

Description des modèles 

## Les résultats des modèles

La précision (accuracy) répond à la question : de toutes les prédictions positives, combien sont correctement classifiées ?
Le rappel (recall) répond à la question : de tous les cas positifs, combien sont actuellement correctement classifiés ?   

### Naive Bayes

:::{table} Classification report of Naive Bayes model
:label: table_report_bayes1
:align: center
![](#table_report_bayes)
:::

:::{figure} #figure_pr_bayes
:name: figure_pr_bayes1
:align: center
:height: 2em
:width: 2em
Precision-Recall Curve of Naive Bayes model
:::

:::{figure} #figure_roc_bayes
:name: figure_roc_bayes1
:align: center
:height: 2em
:width: 2em
ROC Curve of Naive Bayes model
:::

### Logistic regression

:::{table} Classification report of Logistic Regression model
:label: table_report_LR1
:align: center
![](#table_report_LR)
:::

:::{figure} #figure_pr_LR
:name: figure_pr_LR1
:align: center
:height: 2em
:width: 2em
Precision-Recall Curve of Logistic Regression model
:::

:::{figure} #figure_roc_LR
:name: figure_roc_LR1
:align: center
:height: 2em
:width: 2em
ROC Curve of Logistic Regression model
:::

### Support-vector clustering

:::{table} Classification report of Support-vector clustering model
:label: table_report_SVC1
:align: center
![](#table_report_SVC)
:::

:::{figure} #figure_pr_SVC
:name: figure_pr_SVC1
:align: center
:height: 2em
:width: 2em
Precision-Recall Curve of Support-vector clustering model
:::

:::{figure} #figure_roc_SVC
:name: figure_roc_SVC1
:align: center
:height: 2em
:width: 2em
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

Possibilité de faire référence à une [table](#table_report_LR1) ou une figure [](#figure_roc_SVC1)

## Conclusion

+++ {"part": "data_availability"}
L'ensemble des fichiers et données relatif à ce travail sont disponible en accès libre sur le [dépot GitHub](https://github.com/mathisdrn/head_coach_dismissal) sous licence MIT.
+++