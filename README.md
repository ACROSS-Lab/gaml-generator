
Le docker qui sert à faire de la génération contrainte en streaming. Il fonctionne spécifiquement pour faire de la génération contrainte (avec la grammaire gbnf), mais le py 9.3 n'a pas besoin de génération contrainte (donc ça peut être remplacer par un llm géré par ollama et avec une grammaire lark).

Le 9.3 se lance normalement, il faut changer les paramètres à l'intérieur du fichier. Il est sensé boucler sur plusieurs prompts et faire des requêtes pour sortir le code de ces prompts.

les jsonl sont respectivement les codes gaml récupérés de github et ces mêmes codes avec un prompt généré par IA qui aurait pu les donner

les ipynb sont là pour créer une bdd sur neo4j. Pour les faire fonctionner il faut avoir les fichiers BuiltInArchitecture.md, BuiltInSkills.md etc

La 7.1 a une interface graphique et se lance avec streamlit run 7.1.incremental.py

La commande utilisée pour lancer le docker llama.cpp est ci dessous

docker run --rm   -p 8000:8080   -v ~/models:/models   ghcr.io/ggerganov/llama.cpp:server   -m /models/mistral-7b-instruct-v0.2.Q4_K_M.gguf   -c 4096 --host 0.0.0.0 --port 8080
