# 🧭 Nudge Lab Alimentar: trocas explicáveis, não persuasão

Notebook interativo em **marimo** para avaliar trocas alimentares (*nudges* / "cutucadas") de forma calculada, auditável e explicável, 
em vez de delegar a persuasão a um LLM.

## Ideia central

Dado um produto atual **A** e um candidato **B** da mesma categoria, o sistema calcula uma pontuação explicável, aplica guardas de veto (nutricionais e de preço) e apresenta a recomendação em três camadas:

* **Por quê? (evidência):** identifica o fator que mais pesa na melhora por 100 g e sua participação percentual na queda da carga.
* **Quanto? (magnitude, porção, preço e confiança):** mostra a redução da carga, o ganho efetivo (incluindo porção quando disponível), a variação de preço, a aceitabilidade estimada e a confiança dos dados.
* **E se? (contrafactual):** simula como ficaria a carga por 100 g se apenas o fator principal mudasse como no substituto.
* **Baseline persuasivo:** exibe um texto genérico apenas como termo de comparação.

O sistema também pode concluir **"Sem cutucada"** quando nenhum candidato passa nas guardas ou nos limiares mínimos de ganho e confiança.

## Como o nudge é calculado

1. **Carga nutricional `H` (0–100, menor = melhor):** média ponderada de seis termos em `[0, 1]` definidos por *sliders* do usuário:
   * Gordura total, gordura saturada, açúcares e sal (escalonados em relação aos limiares "vermelhos" da FSA/UK por 100 g).
   * Ultraprocessamento (`NOVA == 4` -> `1.0`; `NOVA` em `{1, 2, 3}` -> `0.0`; ausente -> `0.5`).
   * Pouca fibra (referência de `6 g/100 g`; ausente -> `0.5`).
2. **Ganho `G`:** melhora relativa de carga entre o produto atual **A** e o candidato **B**, com três modos selecionáveis na interface:
   * **Combinado (padrão):** `min(G_100g, G_porcao)`, exigindo melhora tanto por 100 g quanto na porção (`H_porcao = H * (porcao_g / 100)`).
   * **Por porção:** ganho relativo sobre a carga da porção (`(H_porcao_A - H_porcao_B) / max(H_porcao_A, 1.0)`).
   * **Por 100 g (clássico):** `(H_A - H_B) / max(H_A, 1.0)`.
3. **Aceitabilidade `Acc` (0–1):** proxy composto ajustável por *sliders*:
   * **Ingredientes (`ing`):** similaridade de Jaccard entre conjuntos de ingredientes.
   * **Energia (`kcal`):** proximidade calórica por 100 g.
   * **Porção (`porcao`):** proximidade de `porcao_g`.
   * **Preço (`preco`):** afinidade assimétrica — mantém `1.0` se `preco_b <= preco_a` e cai linearmente até `0.0` quando o substituto custa o dobro ou mais.
   * Quando `porcao_g` ou `preco` estão ausentes (`NaN`), os pesos dos componentes disponíveis são renormalizados automaticamente.
4. **Confiança `C`:** reflete a completude dos dados dos dois produtos (NOVA, fibra e, quando a porção é usada, `porcao_g`), combinada no par por `C = C_A * C_B`.
5. **Força do nudge:** `N = G * Acc * C`, classificada pelos limiares `t1` e `t2` em `nenhum`, `informativo` ou `sugestão`.
6. **Guardas de veto:** a troca é marcada como `vetado` (e exibida na tabela de transparência com o motivo) se piorar acima da tolerância `tol` um fator nutricional com peso maior que zero ou se ultrapassar `max_aumento_preco`.

## Fontes de dados

* **Demonstração (dados fictícios):** gera produtos sintéticos com `porcao_g`, `preco` e `nutriscore` simulados para testar sem internet.
* **Open Food Facts (país):** consulta o Parquet remoto via DuckDB, salva um cache local (`off_pais_v2.parquet`) e extrai nutrientes e `serving_size`.
* **Arquivo local (`.csv` ou `.parquet`):** aceita tabelas próprias com normalização de nomes de colunas (ex.: `nome`/`produto`, `gordura`/`fat`, `sat`/`saturated_fat`, `acucar`/`sugars`, `porcao_g`/`porcao`, `preco`/`price`/`valor`, `ingredientes`, `nutriscore`) e conversão de `sodio` para `sal` (`sodio * 2.5`).

## Como rodar

```bash
pip install marimo duckdb pandas numpy plotly
marimo edit nudge_lab_alimentar_v4.py
