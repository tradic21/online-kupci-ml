# Online kupci – ML analiza (Online Shoppers Intention)

Projekt analizira ponašanje posjetitelja e-trgovine i radi prediktivne modele:
- **Klasifikacija:** predikcija kupnje (*Revenue*)
- **Regresija:** predikcija vrijednosti stranice (*PageValues*)
  
## Što je napravljeno
- Početna analiza i priprema podataka (osnovne statistike, grafovi, korelacije, provjera neuravnoteženih klasa)
- Čišćenje podataka
- Podjela na skup za učenje i testiranje uz stratifikaciju
- Usporedba više modela i podešavanje hiperparametara (GridSearchCV)
- Evaluacija klasifikacije: precision/recall/F1 + ROC AUC i PR AUC
- Evaluacija regresije: standardne regresijske metrike

## Sprječavanje target leakage-a
- Kod predikcije **Revenue** izbačen je atribut **PageValues**
- Kod predikcije **PageValues** izbačen je atribut **Revenue**

## Datoteke
- `projekt_faza1.ipynb` – analiza i predikcija (faza 1)
- `projekt_faza_2.ipynb` – dodatni modeli/analiza (faza 2)
- `analiza_online_shoppers_faza1.py` – skripta za početnu analizu podataka (izvještaji i grafovi)
- `SUIS_projekt.pdf` – opis i rezultati projekta
