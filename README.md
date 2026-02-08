# Online kupci - ML analiza (Online Shoppers Intention)

Cilj projekta je izraditi modele koji pomažu procijeniti namjeru kupnje i vrijednost posjete kako bi se bolje razumjelo ponašanje korisnika e-trgovine.

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

## Struktura repozitorija
- `notebook/` – Jupyter notebook datoteke (glavni dio projekta)
  - `notebook/projekt_faza1.ipynb` – analiza i predikcija (faza 1)
  - `notebook/projekt_faza_2.ipynb` – dodatni modeli/analiza (faza 2)
- `skripte/` – pomoćne skripte
  - `skripte/analiza_online_shoppers_faza1.py` – skripta za početnu analizu podataka (izvještaji i grafovi)
- `podaci/` – dataset
  - `podaci/online_shoppers_intention.csv`
- `dokumentacija/` – projektna dokumentacija
  - `dokumentacija/SUIS_projekt.pdf`

## Podaci (dataset)
Korišten je **Online Shoppers Purchasing Intention Dataset** (UCI Machine Learning Repository), licenca **CC BY 4.0**.
DOI: https://doi.org/10.24432/C5F88Q

## Kako pokrenuti
1. Otvoriti `notebook/projekt_faza1.ipynb` ili `notebook/projekt_faza_2.ipynb` u Jupyteru/VS Code-u.
2. Dataset se nalazi u `podaci/online_shoppers_intention.csv`.
3. Bilježnice je moguće pokrenuti i u Google Colab-u (upload notebooka i dataseta).
