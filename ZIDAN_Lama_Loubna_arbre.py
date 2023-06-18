
#charger les libraries:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import numpy as np 




#Etape 1: Choix de métrique d'impureté
#Nous utiliserons l'indice de Gini


#Etape 2: fonction pour calculer l'impureté d'un dataset 
def gini_impurete(y):
    unique, compter = np.unique(y, return_counts=True)
    proba = compter / len(y)
    gini = 1 - np.sum(np.square(proba))
    return gini



#Etape 3: Implémentation de la fonction gain d'information
def information_gain(y, y_gauche, y_droit):
    gauche_poids = len(y_gauche) / len(y)
    droit_poids = len(y_droit) / len(y)
    
    gini_gauche = gini_impurete(y_gauche)
    gini_droit = gini_impurete(y_droit)
    
    impurete_apres = gauche_poids * gini_gauche + droit_poids * gini_droit
    impurete_avant = gini_impurete(y)
    
    return impurete_avant - impurete_apres



#Etape 4: Trouver meilleur point de division

def meilleur_division(X, y):
    meilleur_gain = 0
    feature_index = None
    seuil = None

    #pour chaque feature:
    for i in range(X.shape[1]):
        #pour chaque valeur unique du feature:
        for value in np.unique(X[:, i]):
            #division feuille gauche et droit:
            y_gauche=y[X[:, i] <= value]
            y_droit=y[X[:, i] > value]
            
            #calculer le gain d'information pour cette division
            gain = information_gain(y,y_gauche ,y_droit )
            #si le gain est meilleur que le meilleur gain actuelle
            if gain > meilleur_gain:
                #mettre à jour le meilleur gain et l'index du feature et la seuil
                meilleur_gain= gain
                feature_index= i 
                seuil = value
                
    return feature_index, seuil




#Etapes 5 :Créer l'arbre de décision:

class Tree:
    def __init__(self, X, y, profondeur, max_profondeur):
        self.gauche = None
        self.droit = None
        #stoker les infos(seuile et feature) des divisions
        self.feature_index, self.seuil = meilleur_division(X, y)
        #stocker les labels uniques:
        self.label = np.argmax(np.bincount(y))
        # stocker l'indice de Gini:
        self.gini = gini_impurete(y) 
        #stocker les labels dans les feuilles:
        self.data = y
        
        #effectuer la division si la profondeur max n'est pas atteint et si le feature existe:
        if profondeur < max_profondeur and self.feature_index is not None:
            
            #feuille gauche: chercher les labels pour les valeurs <= le seuil 
            X_gauche= X[X[:, self.feature_index] <= self.seuil]
            y_gauche = y[X[:, self.feature_index] <= self.seuil]
            
            #feuille droit: chercher les labels pour les valeurs > le seuil 
            X_droit= X[X[:, self.feature_index] > self.seuil]
            y_droit = y[X[:, self.feature_index] > self.seuil]
            
            #affecter les infos de chaque coté: les valeurs, les labels de ces valeurs, la profondeur et la profondeur max:
            self.gauche = Tree(X_gauche, y_gauche, profondeur + 1, max_profondeur)
            self.droit = Tree(X_droit, y_droit, profondeur + 1, max_profondeur)

    
    def predict(self, x):
        if self.feature_index is None or x[self.feature_index] <= self.seuil:
            if self.gauche:
                return self.gauche.predict(x)
            else:
                return self.label
        else:
            if self.droit:
                return self.droit.predict(x)
            else:
                return self.label
            


#Etape 6: Implementer une classe pour le classificateur basé sur l'arbre de décision.

class classificateur:
    #constructeur pour la classe DecisionTreeClassifier:
    def __init__(self, max_profondeur=None):
        self.max_profondeur = max_profondeur
        
    #fonction d'apprentissage accepte les entrées (X et y):
    #elle crée le noeud racine de l'arbre en utilisant la classe Tree
    def fit(self, X, y):
        self.root = Tree(X, y, 0, self.max_profondeur)
        
    #fonction de prédiction prends les entrées (X) et renvoie les prédictions
    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])
        
   


#Etape 7: Chargement des données Iris, entrainement et evaluation

iris = load_iris()
X= iris.data
y = iris.target

# Diviser l'ensemble de données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle et faire des prédictions
model = classificateur(max_profondeur=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Calculer l'accuracy du modèle
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)




# Calculer le rappel (Recall)
recall = recall_score(y_test, y_pred, average='macro')
print('Recall:', recall)



# Calculer le F1-Score
f1 = f1_score(y_test, y_pred, average='macro')
print('F1-Score:' ,f1)


'''
Le paramètre average='macro' est utilisé pour spécifier le type de calcul à effectuer.
Lorsqu'il est défini sur 'macro', le F1-score et recall_score spnt calculés en prenant la moyenne non pondérée de chaque classe individuelle
indépendamment de sa taille ou de son déséquilibre par rapport aux autres classes.
'''

# Afficher la matrice de confusion
matrice_confusion = confusion_matrix(y_test, y_pred)
print('Matrice de confusion: \n ', matrice_confusion)



print("-"*50)
print(""*50)



#8.Representer l'arbre
def represent_arbre(node, profondeur=0, espace=""):
    if node is None:
        return
    #pour respecter les endentation selon le niveau de division:
    endent = "  " * profondeur
    #afficher seuil et feature de division de ce noeud (si applicable)
    if node.feature_index is not None:
        print(endent,espace,"Feature:",node.feature_index,"<=",node.seuil)
        represent_arbre(node.gauche, profondeur + 1, "Left: ")
        represent_arbre(node.droit, profondeur + 1, "Right: ")
    else:
        #comme c'est une feuille, afficher le label:
        print(endent,espace,"Label:",node.label)
        #afficher l'indice de Gini:
        print(endent,espace,"Gini:",node.gini)
        #afficher les données dans le feuille:
        print(endent,espace,"Data:",node.data)
represent_arbre(model.root)