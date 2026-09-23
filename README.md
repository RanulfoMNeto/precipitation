# Previsão de precipitação — WORCAP 2026

Previsão mensal de precipitação na América do Sul, em mm/dia, para 2023–2024. A solução combina uma U-Net, um corretor LightGBM, dois PoET (sementes 42 e 2026) e um corretor LightGBM com GEFS.

**Com dados e pesos preparados, basta verificar ou gerar o CSV.** A reconstrução completa é opcional. O pacote de artefatos contém `data/` e `work/`; o endereço do dataset Kaggle será incluído após a publicação.

## Ambiente local

| Recurso | Requisito ou estimativa |
| --- | --- |
| Sistema | Linux x86-64, Python 3.12, GPU NVIDIA compatível com CUDA 12.8 |
| Memória recomendada | 64 GiB RAM, 16 GiB VRAM; 8 GiB livres em `/dev/shm` |
| Disco | Ambiente: 8 GiB; dados e trabalho: 20 GiB; cache de aquisição: 24 GiB |
| Tempos de referência | Preparo OOF: 17min37s; treino final: 8min18s; inferência: 32s |
| Aquisição estimada | 4–24h ou mais, conforme as filas CDS |

Reserve espaço adicional para extrair o pacote. Os tempos dependem do hardware; a execução numérica requer GPU CUDA.

## Instalação

Na raiz do repositório:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-acquisition.lock
python -m pip install --no-deps -e .
```

## Usar os artefatos preparados

Extraia o pacote na raiz, mantendo as pastas `data/` e `work/`. Verifique o CSV pronto:

```bash
python -m worcap_forecast verify --submission work/submission.csv
```

Para gerar `submission.csv` por nova inferência:

```bash
python -m worcap_forecast predict --output submission.csv
python -m worcap_forecast verify --submission submission.csv
```

## Reconstruir do zero

### 1. Obter os dados

Baixe os 13 originais na [página de dados da competição](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/data) e coloque-os em `competition-originals/`. Configure `~/.cdsapirc` pela [documentação CDS](https://cds.climate.copernicus.eu/how-to-api) e aceite os termos dos conjuntos sazonais de [níveis únicos](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels) e [pressão](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-pressure-levels).

```bash
python -m worcap_forecast download > download.log 2>&1
python -m worcap_forecast verify --cache source-cache --competition-dir competition-originals
```

`download` importa e verifica os originais, consulta CDS e NOAA e registra consultas e hashes. Retoma arquivos completos após interrupções. Acompanhe em outro terminal com `tail -f download.log`.

### 2. Preparar, treinar e prever

Use uma pasta de trabalho vazia; neste exemplo, `work/rebuild`:

```bash
python -m worcap_forecast prepare --work work/rebuild --technical-replay > prepare.log 2>&1
python -m worcap_forecast train --work work/rebuild > train.log 2>&1
python -m worcap_forecast predict --work work/rebuild --output work/rebuild/submission.csv > predict.log 2>&1
python -m worcap_forecast verify --work work/rebuild --submission work/rebuild/submission.csv
```

Acompanhe com `tail -F prepare.log train.log predict.log`. `--technical-replay` registra a pendência de comprovação da publicação dos produtos históricos, detalhada na [auditoria](docs/DATA_AUDIT.md); não certifica sua admissibilidade. Sem essa opção, o preparo exige disponibilidade temporal comprovada.

## Referências

- [Resumo do modelo](docs/MODEL_SUMMARY_EN.pdf) · [fonte editável](docs/MODEL_SUMMARY_EN.md): arquitetura, atributos e resultados.
- [Auditoria](docs/DATA_AUDIT.md): fontes, cortes temporais e verificações.
- [Comandos](entry_points.md), [caminhos padrão](SETTINGS.json) e [estrutura](directory_structure.txt).

Os recibos em `data/` e `work/` vinculam dados, código, pesos e CSV. Preserve a versão do código associada aos artefatos. A licença [MIT](LICENSE) cobre código e documentação; dados e pesos têm termos próprios. Mantenha artefatos e credenciais fora do Git.
