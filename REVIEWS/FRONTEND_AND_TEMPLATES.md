# Frontend & Templates — Revue détaillée

Fichiers analysés: `app/templates/diagnosis_console.html`, `app/templates/auth.html`, `app/templates/landing_page.html`, `app/static/*`.

Objectif: vérifier cohérence UI-server, risques UX, performance, sécurité client.

---

1) Architecture client
- La console diagnostic (`diagnosis_console.html`) contient logique riche :
  - validation côté client, preview asynchrone via `/api/predict` (fetch),
  - génération PDF via `html2pdf`, debounce et history view.

2) Cohérence champs & serveurs
- Le client envoie un grand nombre de champs (age, sex, weight_kg, ...)
  dont certains ne font pas partie des 10 features du modèle. Le serveur
  `_build_input_row` se base sur `_feature_cols` et `_defaults`; il accepte
  champs additionnels et les ignore, mais il faut :
  - documenter la liste exacte des champs attendus (`feature_cols`) côté client,
  - éviter divergences entre labels HTML et noms de colonnes (cas-sensibles).

3) Performance & SHAP image
- Le client reçoit `shap_b64` (base64 PNG). Cela peut être volumineux.
  - Pour mobiles/bas débit, considérer réduction DPI ou thumbnails via
    serveur.
  - Côté serveur, limitez la taille de l'image encodée (p.ex. 200–400KB).

4) Export PDF
- Le rapport PDF intègre l'image SHAP base64; vérifier anonymisation
  (nom patient) et consentement avant export. Offre un risque de fuite
  si l'utilisateur exporte et partage un fichier contenant données PHI.

5) Sécurité client
- Protéger contre Cross-Site Scripting (XSS) : les templates Jinja doivent
  échapper les variables. Dans les fichiers présents l'usage est standard
  mais valider qu'aucune variable utilisateur n'est rendue non-escaped.

6) Dépendances externes
- Les templates utilisent CDN pour Bootstrap, FontAwesome et html2pdf.
  - Pour offline/airgapped déployer assets localement.
  - Évaluer la politique CSP (Content-Security-Policy) pour bloquer
    contenu externe non autorisé.

7) Accessibilité & validation
- Beaucoup de labels et attributs `required` sont présents — bon point.
- Recommandation : ajouter `aria-*` attributes et vérifier contraste et
  tab navigation pour conformité minimale (WCAG AA).

---

Priorité immédiate (3 actions):
1. Ajouter contrôle serveur pour taille `shap_b64` et transformer image en
   thumbnail avant envoi si besoin.
2. Documenter `feature_cols` list dans template ou exposer via endpoint
   `/api/metadata` pour synchronisation client/serveur.
3. Examiner templates pour s'assurer qu'aucune variable n'est insérée
   sans échappement et ajouter CSP header.
