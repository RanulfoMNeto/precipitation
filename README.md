# WORCAP 2026 — previsão mensal de precipitação

## Problema

O objetivo é prever a precipitação média diária de cada mês, em mm/dia, sobre a grade da América do Sul utilizada pela competição. O período de teste contém 24 meses, de janeiro de 2023 a dezembro de 2024. A saída é um `submission.csv` com uma previsão por célula e mês, na ordem do arquivo oficial de exemplo; a avaliação usa RMSE.

Para prever o mês M, o modelo utiliza observações atmosféricas até M−1 e previsões meteorológicas iniciadas em M−1. A precipitação observada usada no treinamento termina em dezembro de 2022.

## Solução

O pipeline ajusta cinco modelos: uma U-Net espacial, um corretor LightGBM, dois corretores PoET (sementes 42 e 2026) e um corretor LightGBM com GEFS. As entradas externas vêm do ECMWF SEAS5, DWD e Météo-France pelo Copernicus CDS, e do NOAA GEFS. A preparação reconstrói 192 contextos cronológicos fora da amostra, de 2007 a 2022, para treinar os corretores finais.

A previsão final combina a média dos dois PoET (peso **0,6337257586287943**) com o corretor GEFS (peso **0,3662742413712057**). **RMSE da submissão no Kaggle: 1,53208.** A arquitetura e os atributos estão no [resumo técnico](docs/MODEL_SUMMARY_EN.md).

## Ambiente e recursos

| Recurso | Requisito ou referência |
| --- | --- |
| Sistema | Linux x86-64, Python 3.12, GPU NVIDIA com CUDA 12.8 |
| Memória | 64 GiB de RAM, 16 GiB de VRAM e 8 GiB livres em `/dev/shm` recomendados |
| Armazenamento | 8 GiB para ambiente, 20 GiB para dados e trabalho, 24 GiB para cache de aquisição; reserve espaço adicional para extração |
| Tempo observado | Preparação OOF: 17min37s; treino final: 8min18s; inferência: 32s |
| Aquisição | 4–24 h ou mais, conforme as filas do CDS e a conexão |

Instale o ambiente na raiz do repositório:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-acquisition.lock
python -m pip install --no-deps -e .
```

## Executar com artefatos preparados

Extraia o pacote de artefatos fornecido separadamente na raiz do repositório. Ele contém `data/` (entradas preparadas e recibos) e `work/` (contextos OOF, estatísticas e pesos). Para conferir o pacote e gerar uma nova inferência:

```bash
python -m worcap_forecast verify --submission work/submission.csv
python -m worcap_forecast predict --output submission.csv
python -m worcap_forecast verify --submission submission.csv
```

O comando `predict` executa os cinco modelos. O arquivo `work/submission.csv` é uma referência de integridade do pacote e não é usado como entrada da inferência.

## Reconstruir a partir das fontes

Obtenha os 13 arquivos originais na [página de dados da competição](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/data) e salve-os em `competition-originals/`. Configure a conta e a API do [Copernicus CDS](https://cds.climate.copernicus.eu/how-to-api) em `~/.cdsapirc`, aceitando os termos dos conjuntos de [níveis únicos](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels) e [níveis de pressão](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-pressure-levels). O GEFS é obtido dos arquivos públicos da NOAA. Os caminhos abaixo criam uma aquisição independente do pacote preparado.

```bash
python -m worcap_forecast download --data data/rebuild --cache source-cache/rebuild --competition-dir competition-originals > download.log 2>&1
python -m worcap_forecast verify --data data/rebuild --work work/rebuild --cache source-cache/rebuild --competition-dir competition-originals
python -m worcap_forecast prepare --data data/rebuild --work work/rebuild > prepare.log 2>&1
python -m worcap_forecast train --data data/rebuild --work work/rebuild > train.log 2>&1
python -m worcap_forecast predict --data data/rebuild --work work/rebuild --output work/rebuild/submission.csv > predict.log 2>&1
python -m worcap_forecast verify --data data/rebuild --work work/rebuild --submission work/rebuild/submission.csv
```

A aquisição e os blocos concluídos da preparação podem ser reaproveitados após uma interrupção. Acompanhe etapas longas com `tail -F download.log prepare.log train.log predict.log`. Um novo treinamento pode produzir valores diferentes dos do pacote preparado.

## Proveniência e distribuição

As consultas, versões, unidades, hashes e cortes temporais estão nos recibos em `data/` e `work/`, descritos na [auditoria](docs/DATA_AUDIT.md). A aquisição combina a importação local dos originais da competição, verificados por SHA-256, com o download dos produtos CDS e NOAA. O código está sob [licença MIT](LICENSE). Dados e pesos são distribuídos separadamente e conservam seus próprios termos de uso.
