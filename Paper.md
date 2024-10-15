---
title: Projet NLP - M1 MIASHS
subject: Projet NLP - M1 MIASHS
short_title: Projet NLP
banner: ./misc/banner.jpg
license: CC-BY-4.0
keywords: 
date: 2024-04-14
---

+++ {"part": "abstract"}
Dans ce projet recherche nous avons choisi de ...
+++

## Introduction 

Cette analyse vise à mieux comprendre la relation entre la performance d'une équipe de football et l'entraîneur sportif. Nous allons nous intéresser à trois aspects différents de cette relation : l'effet de l'ancienneté de l'entraîneur sur la performance de l'équipe, l'effet du renouvellement régulier de l'entraîneur par le club sur la performance de l'équipe, et l'effet du changement de clubs sur la performance de l'entraîneur.

### Source des données

Les données utilisées au cours de cette analyse sont extraites de deux sites spécialisés dans les statistiques de football : [Fbref](https://fbref.com/) et [Transfermakt](https://www.transfermarkt.com/). 

### Références et outils utilisés

L'intégralité du travail de récupération, de pré-traitement, d'analyse et visualisation des données a été réalisé au sein de Notebook Jupyter. 

La récupération des données footballistique a été effectuée à l'aide du package R [WorldFootBallR](https://github.com/JaseZiv/worldfootballR/). Ce package est régulièrement mis à jour et implémente des outils de web scraping afin d'extraire les données des principaux sites footballistiques.

:::{code} r
:filename: 00 Data extraction.ipynb
country <- c("ENG", "ESP", "ITA", "GER", "FRA")
year <- c(2015:2022)
:::

Enfin, la lecture de Fundamentals of Data Visualization [@DataViz] a permis d'améliorer la qualité des graphiques et de la présentation des données en les rendants plus clairs et informatifs.

:::{caution}
Une p-valeur faible dans le cadre de la corrélation de Pearson ne signifie pas nécessairement qu'il existe une relation de cause à effet entre les deux variables observées. Elle indique simplement que la corrélation observée est statistiquement significative. Ainsi, les hypothèses explicatives fournies sont simplement des hypothèses plausibles et non des conclusions définitives.
:::

Une corrélation négative modérée ($r = −0.38$) statistiquement significative ($p = 0.00$) entre la durée du mandat de l'entraîneur et le pourcentage de défaites.

Les hypothèses explicatives du lien entre la durée du mandat de l'entraîneur et la performance de l'équipe sont similaires aux hypothèses fournies pour expliquer le lien entre la fréquence du renouvellement des entraîneurs par les clubs et la performance du club :

1. La durée du mandat comme facteur explicatif de la performance de l'équipe :
    - Les entraîneurs qui restent plus longtemps à la tête de l'équipe ont tendance à mieux connaître les joueurs et à mieux comprendre les forces et les faiblesses de l'équipe, ce qui peut contribuer à améliorer les performances de l'équipe.
    - Les entraîneurs qui restent moins longtemps à la tête de l'équipe n'ont pas eu le temps de mettre en place leur stratégie de jeu et de s'adapter à leur nouvel environnement, ce qui peut affecter négativement les performances de l'équipe.
2. La performance de l'équipe comme facteur explicatif de la durée du mandat de l'entraîneur :
    - Les équipes qui obtiennent de bons résultats ont tendance à garder leur entraîneur plus longtemps, car il est une partie intégrante de la stabilité et de la performance actuelle de l'équipe.
    - Les équipe qui obtiennent de mauvais résultats ont tendance à renouveler leurs entraîneurs plus fréquemment pour tenter d'améliorer leurs performances.
3. Existence de facteurs tiers qui affecte à la fois la durée du mandat de l'entraîneur et la performance de l'équipe.

## Conclusion

+++ {"part": "data_availability"}
L'ensemble des fichiers et données relatif à ce travail sont disponible en accès libre sur le [dépot GitHub](https://github.com/mathisdrn/head_coach_dismissal) sous licence MIT.
+++