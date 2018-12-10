# SHI-FU-mIA
_Pierre feuille ciseaux avec reconnaissance des signes grâce à un reseau de neurones convolutionel._


## Installation

1. Installer Python (2.x ou 3.x)

Recommandé mais pas obligatoire : Installer un environnement virtuel pour python
avec **venv** ou **conda**

2. Installer les dépendances avec `pip`

- `pip install numpy`
- `pip install tensorflow`
- `pip install keras`
- `pip install python-opencv`
- `pip install scikit-image`
- `pip install matplotlib`

## Utilisation

###Entrainement du reseau
Dans cette classe python, on peut choisir entre utiliser un nouveau modèle et
entrainer un modèle déjà existant.

```python
#Sur cette ligne, il faut choisir le bon titre pour le modèle
#Pour charger un modèle existant, mettre 
# e nom du modèle (model_x.json => model_x)
#
CNN = Train_Model("model_x",500,0.001,2)
CNN.LoadData()
#
####### Choisir entre "CreateModel" ou "LoadModel" (ici on charge un modèle,
# CreateModel() est commenté donc désactivé
#
#CNN.CreateModel()
CNN.LoadModel()
#

```


Lancer le programme Traincnn.py

`python Traincnn.py`

###Test du modèle
Lancer le programme Test_model.py

`python Test_model.py`

###Créer des données avec la webcam

**Modifications préliminaires**

Dans la fonction `show_webcam()`, définir i sur le valeur de la dernière photo du dossier voulu (par exemple 
s'il y a déjà 1000 photos dans le dossier, écrire `i=1000`)

Toujours dans la même fonction, ligne 25 :

`io.imsave("images_paper/"+str(i)+".png", img)`

Changer "images_paper/" par le chemin du dossier voulu.

**Utilisation**

Lancer `python GetDataFromWebcam.py`.
Seules les images dans le cadre vert seront enregistrées.

**Il ne faut pas utiliser d'autres dossiers que les trois images_xxxx/ existants, a moins de 
réecrire les chemins dans les autres classes python.

