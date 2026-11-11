# LLM_test

Esperimento di riproducibilità sul paper **EMO** (Wang, Bhagia, Min, arXiv 2605.06663): una Mixture-of-Experts addestrata perché la modularità emerga durante il pretraining. Il notebook confronta i checkpoint rilasciati "allenai/Emo_1b14b_1T" e "allenai/StdMoE_1b14b_1T" (stessa architettura, cambia solo l'obiettivo di training).

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MDiCapua/LLM_test/blob/main/EMO_vs_StdMoE_GSM8K_inferenza_4bit.ipynb)

## Cosa fa il notebook

`EMO_vs_StdMoE_GSM8K_inferenza_4bit.ipynb` usa i due modelli **solo in inferenza**, quantizzati a 4-bit, su una GPU T4 di Colab:

1. misura la perplexity sulle soluzioni di GSM8K con tutti gli esperti;
2. tiene solo 32 o 16 esperti su 128 per layer, scelti dal router (probabilità media di routing su 40 problemi di calibrazione) oppure a caso (controllo);
3. restringe il modello mascherando i logit del router e rimisura la perplexity su 80 problemi di test.

## Risultati

Rapporto tra perplexity "ristretta" e "completa" (1.0 = nessun peggioramento):

| Esperti tenuti   | EMO, router   | EMO, random   | StdMoE, router   | StdMoE, random |
| 32/128 (25%)     | 0.98          | 2.23          | 1.53             | 9.99           |
| 16/128 (12.5%)   | 0.96          | 3.40          | 3.31             | 149            |

A modello pieno la perplexity è 4.03 per EMO e 3.64 per StdMoE. Con lo stesso taglio EMO resta quasi invariato, la MoE standard peggiora, e per entrambi la scelta del router batte quella casuale. SI conferma quindi la stessa direzione descritta nel paper.

## Come eseguirlo

1. Aprire il notebook in Colab, runtime GPU T4.
2. Eseguire le celle in ordine. Servono circa 70 GB di disco liberi per modello e circa 20 minuti per ogni download da 54 GB.
3. Prima dei download c'è un test rapido su un modello giocattolo.

## Limiti

- Misura la **perplexity**, non l'accuracy di GSM8K quindi non riproduce gli effettivi risultati del paper.
- I modelli quantizzati a 4-bit quindi contano i rapporti, non i valori assoluti ottenuti.
- Selezione one-shot, non la selezione greedy layer per layer (`layerwise`) del repository ufficiale.
- Pochi esempi (40 per la calibrazione, 80 per il test) e una sola esecuzione: i valori sono indicativi e differenze piccole (per esempio 0.96 contro 0.98) sono entro il rumore.
