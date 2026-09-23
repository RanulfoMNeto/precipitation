# Comandos

Execute `python -m worcap_forecast <comando>` na raiz do repositório. [SETTINGS.json](SETTINGS.json) define os caminhos padrão; argumentos explícitos os substituem. Use `<comando> --help` para consultar as opções.

| Comando | Entrada → saída |
| --- | --- |
| `download` | Originais da competição e fontes CDS/NOAA → dados, proveniência e manifesto |
| `prepare` | Dados verificados → ajustes cronológicos e contextos OOF de 2007–2022 |
| `train` | Contextos OOF → dois PoET e corretor GEFS finais |
| `predict` | Dados e cinco modelos → CSV e recibo de inferência |
| `verify` | Dados, modelos e CSV → verificação de integridade, cortes e formato |

Etapas de `download --stage`: `all`, `competition`, `seasonal`, `atmosphere`, `gefs`, `finalize`. Instalação, caminhos e exemplos completos: [README](README.md).
