# Auditoria dos dados e da execução

Execução de referência: **23/09/2026**. O [README](../README.md) contém o ambiente e os comandos.

## Proveniência

| Fonte | Conteúdo utilizado |
| --- | --- |
| Competição | Observações até 2022 e campos atmosféricos de origem do teste de 2023–2024. Os 13 arquivos originais foram importados localmente e verificados por tamanho e SHA-256. |
| ECMWF SEAS5, DWD e Météo-France | Previsões sazonais do CDS, inicializadas no mês anterior ao alvo (`leadtime_month=2`). A precipitação foi convertida de m/s para mm/dia. O SEAS5 também fornece atributos atmosféricos. |
| NOAA GEFS | Precipitação de cinco membros e média operacional, agregada em intervalos semanais. Os acumulados em kg/m² foram convertidos para mm/dia conforme a duração válida. |

O [inventário](../src/worcap_forecast/config/acquisition_inventory.json), os [originais esperados](../src/worcap_forecast/config/competition_expected.json) e os [termos das fontes](../src/worcap_forecast/config/source_terms.json) identificam os produtos. `data/provenance.json`, `data/source_receipts.json` e `data/manifest.json` registram consultas, URLs, versões, períodos e hashes. Os fornecedores classificam parte dos produtos históricos como hindcasts ou reforecasts; os recibos preservam essas classificações. O calendário dos sistemas CDS documenta os períodos operacionais, e os objetos GEFS do teste têm `Last-Modified` anterior ao mês-alvo. As respostas CDS não incluem horário individual de primeira publicação.

## Cortes e verificações

- Para o alvo M, somente observações até M−1 entram nos atributos. A precipitação observada termina em **2022-12**; nenhum rótulo de 2023–2024 integra o treinamento.
- Normalização e calibração usam os prefixos anteriores a cada bloco. Foram executados 13 ajustes U-Net e nove ajustes do primeiro LightGBM; os 192 contextos fora da amostra cobrem **2007-01–2022-12**.
- A inferência executa os cinco modelos para **2023-01–2024-12**, sem reutilizar previsões de teste. O CSV contém **1.885.464** linhas, com IDs e ordem oficiais, valores finitos e não negativos.
- Os 13 originais, 22.720 arquivos de origem e 8.815 arquivos do manifesto foram verificados. Dez testes de integridade, reconstrução e corte temporal passaram. `work/submission.json` vincula CSV, dados, código e pesos.

## Resultado

| Avaliação | Modelo | Período | RMSE |
| --- | --- | --- | --- |
| Kaggle | Combinação final | Teste de 2023–2024 | **1,53208** |
| Validação histórica local, fora da amostra | Primeiro LightGBM | 2007–2022 | 1,754724 |
| Validação histórica local, fora da amostra | Primeiro LightGBM | 2015–2020 | 1,751633 |

As métricas históricas avaliam o primeiro LightGBM isoladamente.

A licença do código não substitui os termos de redistribuição dos dados e pesos. `work/distribution.json` registra a adaptação de empacotamento sem novo treinamento ou inferência.
