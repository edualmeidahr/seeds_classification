```

RESUMO DOS RESULTADOS
=====================
Problema: classificação multiclasse de variedades de sementes (Kama, Rosa, Canadian),
210 amostras balanceadas (70/70/70), 7 atributos numéricos contínuos.

Metodologia: holdout estratificado de 20% para teste final; GridSearchCV com CV
estratificada k=10 sobre o treino para tuning; RepeatedStratifiedKFold (10x10) para
estimativas robustas; mesmas divisoes para os 3 modelos (comparacao justa).

Melhor modelo (F1-macro no teste)     : Árvore de Decisão (0.9521)
Melhor modelo (F1-macro na CV 10x10)  : MLP (0.9644 ± 0.0485)

Desempenho no conjunto de TESTE (holdout):
  -               k-NN: acc=0.8810 | F1-macro=0.8731 | AUC=0.9847 | t_treino=0.0022s
  -  Árvore de Decisão: acc=0.9524 | F1-macro=0.9521 | AUC=0.9634 | t_treino=0.0018s
  -                MLP: acc=0.9048 | F1-macro=0.9007 | AUC=0.9371 | t_treino=0.0124s

Desempenho na CV repetida (media ± desvio):
  -               k-NN: acc=0.9293±0.0560 | F1=0.9284±0.0575
  -  Árvore de Decisão: acc=0.9263±0.0690 | F1=0.9253±0.0700
  -                MLP: acc=0.9650±0.0473 | F1=0.9644±0.0485

Teste estatistico (Friedman sobre F1-macro por fold):
  estatistica = 39.2763 | p-valor = 2.960e-09
  -> As diferencas entre os modelos SAO estatisticamente significativas (p<0.05).
  Ranks medios (Nemenyi, menor=melhor): MLP=1.59, k-NN=2.19, Árvore de Decisão=2.21
  Diferenca critica (CD, alpha=0.05) = 0.331

Estabilidade: o desvio-padrao na CV repetida indica a estabilidade de cada modelo
(menor desvio = mais estavel). O MLP tende a ter maior variabilidade devido a
inicializacao aleatoria dos pesos.

Limitacoes: dataset pequeno (210 amostras) -> estimativas de teste tem variancia; as
classes Kama e Canadian apresentam leve sobreposicao (ver PCA/matrizes de confusao),
concentrando ali a maioria dos erros.

```
