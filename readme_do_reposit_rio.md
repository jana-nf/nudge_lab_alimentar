# 🧭 Nudge Lab Alimentar

> **Trocas alimentares explicáveis, não persuasão.**  
> Um laboratório interativo em [marimo](https://marimo.io/) para calcular, filtrar e auditar *nudges* ("cutucadas") nutricionais entre produtos da mesma categoria — com guardas éticas explícitas e explicações em três camadas.

---

## 🎯 Visão Geral

Em vez de delegar a persuasão a um LLM que gera textos genéricos, o **Nudge Lab Alimentar** calcula de forma determinística e auditável quando vale a pena sugerir a troca de um produto **A** por um substituto **B** da mesma categoria.

Cada sugestão é acompanhada por uma explicação estruturada em três camadas:

1. **Por quê? (evidência):** qual fator nutricional mais pesou na melhora e qual sua participação percentual exata na queda da carga.
2. **Quanto? (magnitude, porção, preço e confiança):** quanto a carga cai por 100 g, qual o ganho efetivo considerando a porção, o impacto financeiro e o nível de confiança dos dados.
3. **E se? (contrafactual):** qual seria a redução de carga se apenas o fator principal mudasse, mantendo o restante constante.

Um texto persuasivo genérico é exibido apenas como *baseline* de comparação.

---

## 🧮 Como o *Nudge* é Calculado

Para cada par de produtos **(A, B)** na mesma categoria:

### 1. Carga Nutricional $H$ ($0\text{–}100$, menor = melhor)
Cada produto é avaliado em seis termos normalizados entre $0$ e $1$:
* **Gordura total, gordura saturada, açúcares e sal** (por 100 g): escalonados pelos limiares "vermelhos" da FSA/UK ($17{,}5\text{ g}$, $5{,}0\text{ g}$, $22{,}5\text{ g}$ e $1{,}5\text{ g}$), truncados em $2\times$ o limiar.
* **Ultraprocessamento (`proc`):** $1{,}0$ para NOVA 4, $0{,}0$ para NOVA 1–3 e $0{,}5$ (neutro) quando ausente.
* **Fibra (`fibra`):** $1 - \min(\text{fibra}/6, 1)$ por 100 g, ou $0{,}5$ quando ausente.

A carga $H$ é a média ponderada desses seis termos pelos **pesos definidos pelo usuário** nos sliders da interface.

### 2. Ganho Efetivo $G$ (por 100 g, por porção ou combinado)
* **Ganho por 100 g:**
  $$G_{100\text{g}} = \frac{H_A - H_B}{\max(H_A, 1{,}0)}$$
* **Ganho por porção:** quando `porcao_g` está disponível em ambos os itens, calcula-se $H_{\text{porção}} = H \times (\text{porção}_g / 100)$ e o ganho relativo $G_{\text{porção}}$.
* **Modo combinado (padrão):**
  $$G = \min(G_{100\text{g}}, G_{\text{porção}})$$
  Evita "falsos ganhos" em que o substituto é mais leve por 100 g, mas vem em uma porção maior que piora a carga total consumida.

### 3. Aceitabilidade Composta $\text{Acc}$ ($0\text{ a }1$)
Mede o atrito prático da troca combinando até quatro critérios com pesos ajustáveis (redistribuídos automaticamente caso porção ou preço estejam ausentes):
* **Ingredientes (`ing`):** similaridade de Jaccard entre os conjuntos de ingredientes ($0{,}3$ como fallback).
* **Energia (`kcal/100 g`):** proximidade calórica relativa.
* **Tamanho da porção (`porcao_g`):** proximidade relativa entre as porções.
* **Afinidade de preço (`preco`):** assimétrica — vale $1{,}0$ se o substituto custa igual ou menos que o original, e decai linearmente até $0{,}0$ caso custe o dobro ou mais.

### 4. Confiança $C$ e Força do *Nudge* $N$
* **Confiança $C = C_A \times C_B$:** reflete a completude dos metadados (NOVA, fibra e `porcao_g` quando utilizada).
* **Força final:**
  $$N = G \times \text{Acc} \times C$$
* Classificada pelos limiares configuráveis $t_1$ e $t_2$ em: **`nenhum`** ($N < t_1$), **`informativo`** ($t_1 \le N < t_2$) ou **`sugestão`** ($N \ge t_2$).

---

## 🛡️ Guardas Éticas e Direito a "Não Cutucar"

O sistema **veta** qualquer candidato que:
1. Piore acima da tolerância (`tol`) algum fator nutricional com peso maior que zero;
2. Ultrapasse o aumento percentual máximo de preço configurado (`max_aumento_preco`), quando ambos os produtos possuem preço.

Candidatos vetados permanecem visíveis na tabela de transparência com o motivo explícito do veto. Se nenhum candidato passar nas guardas e nos patamares mínimos de ganho e confiança, o sistema conclui **"Sem cutucada"**.

---

## 📦 Fontes de Dados Suportadas

1. **Demonstração (dados fictícios):** dataset sintético gerado localmente (com porção e preço simulados) para testar o notebook sem acesso à internet.
2. **Open Food Facts (por país):** consulta direta via DuckDB ao snapshot Parquet no Hugging Face, filtrando por tag de país (ex.: `en:brazil`), extraindo `serving_size` e salvando cache local em `off_pais_v2.parquet`.
3. **Arquivo local (`.csv` ou `.parquet`):** upload direto pela interface para avaliar catálogos próprios (ex.: coletas de supermercado com preços reais).

### Esquema esperado para arquivos locais (CSV / Parquet)

| Coluna | Obrigatória | Descrição / Aliases aceitos |
| :--- | :---: | :--- |
| `nome` | Sim | Nome do produto (`produto`, `product_name`) |
| `categoria` | Sim | Categoria para agrupamento (`category`, `categories`) |
| `gordura` | Sim | Gordura total em g/100 g (`fat`, `gorduras_totais`) |
| `sat` | Sim | Gordura saturada em g/100 g (`saturated_fat`, `gordura_saturada`) |
| `acucar` | Sim | Açúcares em g/100 g (`sugars`, `sugar`, `acucares`) |
| `sal` | Sim* | Sal em g/100 g (`salt`). *Ou `sodio` (convertido via $\text{sal} = \text{sódio} \times 2{,}5$) |
| `marca` | Não | Marca do produto (`brand`, `brands`) |
| `ingredientes` | Não | Lista separada por `,`, `;` ou `\|` |
| `kcal` | Não | Energia em kcal/100 g (`energy_kcal`, `calorias`) |
| `fibra` | Não | Fibra alimentar em g/100 g (`fiber`, `fibras`) |
| `nova` | Não | Classificação NOVA de 1 a 4 (`nova_group`) |
| `porcao_g` | Não | Tamanho da porção em gramas (`serving_g`, `porcao`) |
| `preco` | Não | Preço unitário (`price`, `valor`) |
| `nutriscore` | Não | Grau oficial `a`–`e` para avaliação offline (`nutriscore_grade`) |

---

## 🚀 Como Executar

### 1. Instalar dependências
```bash
pip install marimo duckdb pandas numpy plotly
```

### 2. Abrir o notebook no marimo
```bash
marimo edit nudge_lab_alimentar_v4.py
```
*(Ou use `marimo run nudge_lab_alimentar_v4.py` para abrir apenas no modo aplicativo.)*

> **Nota sobre o Open Food Facts:** na primeira execução com dados remotos, o notebook cria o cache local `off_pais_v2.parquet` para acelerar consultas futuras. Caso a leitura remota falhe por rede ou memória, baixe o `food.parquet` manualmente.

---

## 🏗️ Estrutura do Código

* **Células de UI e Dados:** seleção de fonte, upload de arquivos, limpeza de plausibilidade física e controles interativos (`sliders` e `dropdowns`).
* **Núcleo Puro (`# <<CORE-BEGIN>>` a `# <<CORE-END>>`):** funções puras em `numpy` e `pandas` (`termos`, `indice`, `confianca`, `aceitabilidade`, `avaliar_trocas`, `avaliar_categoria`, `textos`, `gerar_demo`), desacopladas do marimo para facilitar testes unitários.
* **Avaliação Offline da Categoria (Seção 3):** calcula cobertura de *nudges* viáveis, ganho efetivo médio, tabela completa (`df_avaliacao`), mapa de dispersão (`Acc × Ganho`) e concordância externa com o Nutri-Score.

---

## ⚠️ Limitações e Escopo

* **Heurísticas não calibradas:** pesos, limiares da FSA/UK, `Acc`, `C`, $t_1$ e $t_2$ são parâmetros heurísticos. Sem experimentos empíricos com pessoas, o notebook mede **potencial técnico e fidelidade explicativa**, não mudança real de comportamento.
* **Agrupamento de categorias:** no Open Food Facts, utiliza-se a última tag de `categories_tags`; verifique se o recorte agrupa itens realmente substituíveis entre si.
* **Caráter informativo:** a carga nutricional serve como régua comparativa e **não** constitui recomendação clínica ou nutricional individualizada.
* **Atribuição de dados:** ao publicar análises usando dados do Open Food Facts, cite a base sob licença **ODbL** e a data do snapshot utilizado.