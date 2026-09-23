# Auditoria de dados e execução

Execução de referência: **23/09/2026**. Ambiente e comandos no [README](../README.md).

## Fontes

| Fonte | Uso e proveniência |
| --- | --- |
| Competição | Observações de 1940–2022 e origens do teste de 2023–2024. Treze arquivos fornecidos pelo participante, importados e verificados por tamanho e SHA-256; o download Kaggle não foi repetido. |
| ECMWF SEAS5, DWD e Météo-France | Nova aquisição pelo CDS. Previsões mensais e ensembles; precipitação convertida de m/s para mm/dia, lead 2. SEAS5 inclui campos atmosféricos. Inicializações, sistemas e períodos exatos constam no inventário e nos recibos. |
| NOAA GEFS | Nova aquisição dos arquivos públicos operacionais e de reforecast; agregados de 2007–2024. Precipitação acumulada em kg/m² convertida para mm/dia pela duração válida. URLs, hashes e `Last-Modified` registrados. |

[Inventário](../src/worcap_forecast/config/acquisition_inventory.json) · [Originais esperados](../src/worcap_forecast/config/competition_expected.json) · [Licenças e fontes oficiais](../src/worcap_forecast/config/source_terms.json).

## Verificações

- **Integridade:** 13 originais, 22.720 arquivos de origem e 8.815 arquivos no manifesto conferidos. Consultas, versões, unidades, períodos e hashes estão em `data/provenance.json`, `source_receipts.json` e `manifest.json`.
- **Corte temporal:** para prever M, observações até M−1; rótulos de treinamento até 2022-12. Normalização e calibração usam apenas o prefixo disponível. A linha seguinte do teste não altera os atributos de M.
- **OOF:** 13 ajustes U-Net e nove ajustes tabulares por prefixo geram 192 meses fora da amostra, de 2007-01 a 2022-12.
- **Inferência:** executa os cinco modelos, sem reutilizar previsões de teste. Dez testes passaram. CSV com 1.885.464 linhas, IDs e ordem oficiais, valores finitos e não negativos. `work/submission.json` vincula seu hash ao código, dados, modelos e ambiente.

## Resultados

| Avaliação | RMSE |
| --- | ---: |
| Primeiro corretor LightGBM, OOF 2007–2022 | 1,754724 |
| Primeiro corretor LightGBM, OOF 2015–2020 | 1,751633 |
| Receita original, envio manual ao Kaggle | 1,53551 |
| CSV desta execução, envio manual ao Kaggle | 1,53208 |

As notas Kaggle foram **informadas pelo participante**. As métricas OOF medem apenas o primeiro corretor, não a combinação final. Um novo treinamento pode produzir previsões diferentes.

## Disponibilidade e distribuição

Um organizador confirmou que dados públicos do Copernicus podem ser utilizados. A comprovação de publicação dos hindcasts CDS e reforecasts GEFS em cada origem histórica permanece pendente: inicialização não comprova disponibilidade. Por isso, `verify --strict-origin` falha e a reconstrução usa `prepare --technical-replay`. A interpretação desses produtos históricos cabe aos organizadores.

Para os 24 meses de teste, os 144 objetos GEFS operacionais possuem `Last-Modified` anterior ao respectivo mês-alvo. Essa evidência não se estende aos demais produtos.

A licença MIT do código não altera os direitos dos dados e pesos. Antes da publicação do pacote preparado, confira os termos de redistribuição indicados em `source_terms.json`.

O pacote preparado preserva dados numéricos, pesos e CSV. A adaptação para execução local e a remoção de metadados pessoais estão registradas em `work/distribution.json`, com hashes anteriores e atuais; não representam novo treinamento ou inferência.
