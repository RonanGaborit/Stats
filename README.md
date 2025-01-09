# Stats

# information git
## Cloner un dépôt GitHub
git clone https://github.com/utilisateur/repo.git
## Créer une nouvelle branche
git checkout -b nouvelle-branche
## Ajouter des modifications et les valider
git add .

git commit -m "Message du commit"

## Envoyer les modifications sur GitHub
git push origin nouvelle-branche

## Ouvrir une Pull Request
Sur GitHub, allez dans votre dépôt, sélectionnez votre branche et cliquez sur "New Pull Request" .

## Gérer les versions
### Voir l'historique
git log 
git revert <commit-id>  
git revert <commit-id>

git revert
### Annuler un commit spécifique
git reset --hard HEAD~1  
git reset --hard HEAD~1
### Supprimer le dernier commit
git reset --
