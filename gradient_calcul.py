import torch


"""
Pour rapel, le gradient est la dérivé parielle d'une fonction par rapport à ses variables d'entrée.
Il est utilisé dans l'algorithme de descente de gradient pour mettre à jour les paramètres d'un modèle
En mathématiques, calculer la dérivée d'une fonction permet d'en déterminer la pente à un point donné.
Le gradient est alors utilisé pour minimiser une fonction convexe (qui fonctionne en un seul point), c'est à dire
trouver le minimum de cette fonction donc le point où la pente est nulle sois avec l'erreur la plus faible possible.
"""
x = torch.tensor([2.0], requires_grad=True)  # Création d'un tenseur avec suivi des gradients
y = x ** 2  # Opération qui génère un gradient

y.backward()  # Calcul du gradient de y par rapport à x
print(x.grad)  # Donne 4.0 (car dy/dx = 2x et x=2 donc 2*2=4)

# Désactivation du calcul des gradients
with torch.no_grad():
    z = x ** 2  # Aucun gradient ne sera stocké

print(z.requires_grad)  # False, aucun suivi des gradients ici
