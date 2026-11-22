# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "pandas",
#     "numpy",
#     "plotly==7.1.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    return mo, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧭 Nudge Lab Alimentar: trocas explicáveis, não persuasão

    **Ideia.** Em vez de um LLM escrevendo textos para convencer, a "cutucada" (*nudge*) é
    **calculada** a partir de dados abertos e acompanhada de uma explicação em três camadas:
    **Por quê?** (evidência), **Quanto?** (magnitude e confiança) e **E se?** (contrafactual).

    **Como a força do nudge é calculada** (troca do produto A pelo produto B, mesma categoria):

    1. **Carga** `H` (0–100, menor = melhor): média ponderada de seis termos entre 0 e 1:
       gordura, gordura saturada, açúcares e sal (relativos aos limiares "vermelhos" da FSA/UK, por 100 g),
       ultraprocessamento (NOVA 4) e pouca fibra. **Os pesos são escolhidos por você.**
    2. **Ganho** `G`: melhora relativa de carga. Pode ser avaliada **por 100 g** ou **por porção**
       (`H_porcao = H × porcao_g / 100`), evitando trocas que parecem melhores em 100 g mas aumentam a carga na porção real.
    3. **Aceitabilidade** `Acc`: mede o quão viável é a troca combinando **ingredientes** (Jaccard),
       **densidade energética**, **tamanho da porção** e **afinidade de preço** (quando disponível).
    4. **Confiança** `C`: completude dos dados dos dois produtos (NOVA, fibra e porção quando usada).
    5. **Força do nudge** `N = G × Acc × C`.
    6. **Guardas:** a troca é **vetada** se piorar um fator que você priorizou ou ultrapassar a **tolerância de preço** (`Δpreço %`),
       e **não há nudge** se o ganho ou a confiança forem baixos. O sistema pode (e deve) concluir "não cutucar".

    > Os limiares e níveis são **heurísticas não calibradas**. Sem experimento com pessoas,
    > o notebook mede *potencial* e *fidelidade*, não mudança de comportamento.
    """)
    return


@app.cell
def _(mo):
    modo = mo.ui.radio(
        options=[
            "Demonstração (dados fictícios)",
            "Open Food Facts (país)",
            "Arquivo local (CSV / Parquet)",
        ],
        value="Demonstração (dados fictícios)",
        label="**Fonte de dados**",
    )
    pais = mo.ui.text(value="en:brazil", label="Tag de país (Open Food Facts)")
    min_prod = mo.ui.number(start=2, stop=2000, step=5, value=30, label="Mín. de produtos por categoria")
    upload_arquivo = mo.ui.file(
        filetypes=[".csv", ".parquet"],
        multiple=False,
        label="Enviar CSV ou Parquet (opcional)",
    )
    mo.vstack([
        modo,
        mo.hstack([pais, min_prod], justify="start"),
        upload_arquivo,
    ])
    return min_prod, modo, pais, upload_arquivo


@app.cell
def _(np, pd):
    # <<CORE-BEGIN>>
    # Limiares "vermelhos" FSA/UK (g por 100 g), usados como escala de referência.
    ALTO = {"gordura": 17.5, "sat": 5.0, "acucar": 22.5, "sal": 1.5}
    TERMOS = ["gordura", "sat", "acucar", "sal", "proc", "fibra"]
    ROTULO = {
        "gordura": "gordura total", "sat": "gordura saturada", "acucar": "açúcares",
        "sal": "sal", "proc": "ultraprocessamento (NOVA 4)", "fibra": "pouca fibra",
    }
    ROTULO_CURTO = {
        "gordura": "a gordura total", "sat": "a gordura saturada", "acucar": "o açúcar",
        "sal": "o sal", "proc": "o grau de processamento", "fibra": "a fibra",
    }
    GRAU = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    def termos(df):
        """Carga 0-1 por termo (0 = melhor). Desconhecido em NOVA/fibra vira 0,5 (neutro)."""
        t = pd.DataFrame(index=df.index)
        for k in ["gordura", "sat", "acucar", "sal"]:
            t[k] = (df[k] / ALTO[k]).clip(upper=2) / 2
        t["proc"] = np.select([df["nova"] == 4, df["nova"].isin([1, 2, 3])], [1.0, 0.0], default=0.5)
        t["fibra"] = np.where(df["fibra"].notna(), 1 - (df["fibra"] / 6).clip(upper=1), 0.5)
        return t[TERMOS]

    def indice(t, pesos):
        w = pd.Series(pesos)[TERMOS].astype(float)
        if w.sum() <= 0:
            return pd.Series(np.nan, index=t.index)
        return 100 * (t * w).sum(axis=1) / w.sum()

    def confianca(df, considerar_porcao=True):
        """4 nutrientes FSA são obrigatórios; falta de NOVA, fibra ou porção reduz a confiança."""
        base = 0.65 if considerar_porcao else 0.70
        c = (
            base
            + 0.15 * df["nova"].notna().astype(float)
            + 0.15 * df["fibra"].notna().astype(float)
        )
        if considerar_porcao:
            has_p = df["porcao_g"].notna() if "porcao_g" in df.columns else pd.Series(False, index=df.index)
            c = c + 0.05 * has_p.astype(float)
        return c.clip(upper=1.0)

    def preparar(sub, pesos, considerar_porcao=True):
        t = termos(sub)
        return (
            t.to_numpy(float),
            indice(t, pesos).to_numpy(float),
            confianca(sub, considerar_porcao=considerar_porcao).to_numpy(float),
        )

    def _sim_relativa(va, vb):
        """Similaridade simétrica em [0, 1] baseada na diferença relativa; NaN -> None."""
        if va is None or vb is None or np.isnan(va) or np.isnan(vb):
            return None
        denom = max(abs(va), abs(vb), 1e-6)
        return 1.0 - min(abs(va - vb) / denom, 1.0)

    def aceitabilidade(
        ing_a,
        ing_b,
        kcal_a,
        kcal_b,
        porcao_a=np.nan,
        porcao_b=np.nan,
        preco_a=np.nan,
        preco_b=np.nan,
        pesos_acc=None,
    ):
        """
        Proxy composto de aceitabilidade (0 a 1):
        - ing: Jaccard de ingredientes
        - kcal: proximidade calórica por 100 g
        - porcao: proximidade de tamanho de porção (g)
        - preco: penaliza apenas aumento de preço (preço menor ou igual mantém 1.0)
        Pesos dos termos ausentes (NaN) são redistribuídos proporcionalmente.
        """
        w_map = dict(ing=0.45, kcal=0.25, porcao=0.15, preco=0.15)
        if pesos_acc:
            w_map.update(pesos_acc)

        j = (len(ing_a & ing_b) / len(ing_a | ing_b)) if (ing_a and ing_b) else 0.3
        e = _sim_relativa(kcal_a, kcal_b)
        p = _sim_relativa(porcao_a, porcao_b)

        pr = None
        if not (np.isnan(preco_a) or np.isnan(preco_b)) and preco_a > 0:
            aumento = max(0.0, (preco_b - preco_a) / preco_a)
            pr = max(0.0, 1.0 - min(aumento, 1.0))

        componentes = {"ing": j, "kcal": e, "porcao": p, "preco": pr}
        ativos = {k: v for k, v in componentes.items() if v is not None and w_map.get(k, 0) > 0}
        if not ativos:
            return j
        w_tot = sum(w_map[k] for k in ativos)
        return sum(w_map[k] * v for k, v in ativos.items()) / w_tot

    COLS_TAB = [
        "j", "ganho_100g", "ganho_porcao", "ganho", "acc", "conf", "N",
        "delta_preco_pct", "vetado", "motivo",
    ]

    def avaliar_trocas(
        sub,
        T,
        H,
        C,
        i,
        pesos,
        tol,
        modo_ganho="combinado",
        max_aumento_preco=0.25,
        pesos_acc=None,
    ):
        """
        Todos os candidatos com ganho > 0 na mesma categoria, com guardas de veto:
        - Piora em fator nutricional com peso > 0 acima de `tol`
        - Aumento de preço acima de `max_aumento_preco` (quando ambos têm preço)
        - Piora na carga por porção quando `modo_ganho` exige ganho por porção
        """
        w = np.array([float(pesos[k]) for k in TERMOS])
        ing = sub["ingredientes"].tolist()
        kcal = sub["kcal"].to_numpy(float)
        porcao = sub["porcao_g"].to_numpy(float) if "porcao_g" in sub.columns else np.full(len(sub), np.nan)
        preco = sub["preco"].to_numpy(float) if "preco" in sub.columns else np.full(len(sub), np.nan)

        linhas = []
        for j in range(len(sub)):
            if j == i:
                continue
            ganho_100g = (H[i] - H[j]) / max(H[i], 1.0)

            if not (np.isnan(porcao[i]) or np.isnan(porcao[j])) and porcao[i] > 0 and porcao[j] > 0:
                hi_p = H[i] * (porcao[i] / 100.0)
                hj_p = H[j] * (porcao[j] / 100.0)
                ganho_porcao = (hi_p - hj_p) / max(hi_p, 1.0)
            else:
                ganho_porcao = ganho_100g

            if modo_ganho == "porcao":
                ganho = ganho_porcao
            elif modo_ganho == "combinado":
                ganho = min(ganho_100g, ganho_porcao)
            else:
                ganho = ganho_100g

            if not ganho > 0:
                continue

            motivos = []
            piora = (T[j] - T[i]) > tol
            mask = piora & (w > 0)
            if mask.any():
                motivos.extend(ROTULO[TERMOS[m]] for m in np.where(mask)[0])

            delta_preco_pct = np.nan
            veto_preco = False
            if not (np.isnan(preco[i]) or np.isnan(preco[j])) and preco[i] > 0:
                delta_preco_pct = (preco[j] - preco[i]) / preco[i]
                if delta_preco_pct > max_aumento_preco:
                    veto_preco = True
                    motivos.append(f"preço (+{delta_preco_pct:.0%})")

            acc = aceitabilidade(
                ing[i], ing[j], kcal[i], kcal[j],
                porcao_a=porcao[i], porcao_b=porcao[j],
                preco_a=preco[i], preco_b=preco[j],
                pesos_acc=pesos_acc,
            )
            conf = C[i] * C[j]
            vetado = bool(mask.any() or veto_preco)
            linhas.append(dict(
                j=j,
                ganho_100g=ganho_100g,
                ganho_porcao=ganho_porcao,
                ganho=ganho,
                acc=acc,
                conf=conf,
                N=ganho * acc * conf,
                delta_preco_pct=delta_preco_pct,
                vetado=vetado,
                motivo=", ".join(motivos),
            ))
        return pd.DataFrame(linhas, columns=COLS_TAB)

    def escolher_troca(tab, ganho_min, conf_min):
        if tab.empty:
            return tab
        ok = tab[(~tab["vetado"]) & (tab["ganho"] >= ganho_min) & (tab["conf"] >= conf_min)]
        return ok.sort_values("N", ascending=False)

    def nivel(N, t1, t2):
        if N < t1:
            return "nenhum"
        return "informativo" if N < t2 else "sugestão"

    def contribuicoes(T, i, j, pesos):
        """Quanto cada termo contribui para a queda de H por 100 g (soma = H_A - H_B)."""
        w = np.array([float(pesos[k]) for k in TERMOS])
        c = 100 * (T[i] - T[j]) * w / w.sum()
        return pd.Series(c, index=TERMOS).sort_values(ascending=False)

    def _v(x, nd=1):
        return "n/d" if pd.isna(x) else f"{x:.{nd}f}"

    def descricao_termo(k, a, b):
        if k in ("gordura", "sat", "acucar", "sal"):
            return f"{ROTULO[k]} ({_v(a[k])} g → {_v(b[k])} g por 100 g)"
        if k == "proc":
            return f"ultraprocessamento (NOVA {_v(a['nova'], 0)} → {_v(b['nova'], 0)})"
        return f"fibra ({_v(a['fibra'])} g → {_v(b['fibra'])} g por 100 g)"

    def textos(a, b, c, HA, HB, row_b):
        k = c.index[0]
        total = HA - HB
        share = c.iloc[0] / total if total else 0.0

        detalhes_extra = []
        if "porcao_g" in a and not (pd.isna(a["porcao_g"]) or pd.isna(b["porcao_g"])):
            detalhes_extra.append(
                f"Porção: {_v(a['porcao_g'], 0)} g → {_v(b['porcao_g'], 0)} g "
                f"(ganho por porção: {row_b['ganho_porcao']:.0%})"
            )
        if "preco" in a and not (pd.isna(a["preco"]) or pd.isna(b["preco"])):
            dp = row_b["delta_preco_pct"]
            sinal = "+" if dp >= 0 else ""
            detalhes_extra.append(
                f"Preço: R$ {_v(a['preco'], 2)} → R$ {_v(b['preco'], 2)} ({sinal}{dp:.0%})"
            )
        linha_extra = (" " + " · ".join(detalhes_extra) + ".") if detalhes_extra else ""

        return {
            "Por quê? (evidência)":
                f"O fator que mais pesa é {descricao_termo(k, a, b)}. Ele responde por {share:.0%} da melhora em 100 g.",
            "Quanto? (magnitude, porção, preço e confiança)":
                f"A carga por 100 g cai de {HA:.0f} para {HB:.0f} (ganho efetivo: {row_b['ganho']:.0%}). "
                f"Confiança dos dados: {row_b['conf']:.0%}. Aceitabilidade estimada: {row_b['acc']:.0%}.{linha_extra}",
            "E se? (contrafactual)":
                f"Se só {ROTULO_CURTO[k]} mudasse como no substituto (mantendo a mesma porção), a carga por 100 g iria de "
                f"{HA:.0f} para {HA - c.iloc[0]:.0f}; os demais fatores e o ajuste de porção respondem pelo restante.",
            "Baseline persuasivo (sem evidência, só para comparação)":
                "✨ Uma escolha mais leve e saudável para você!",
        }

    def avaliar_categoria(
        sub,
        pesos,
        tol,
        ganho_min,
        conf_min,
        t1,
        t2,
        modo_ganho="combinado",
        max_aumento_preco=0.25,
        pesos_acc=None,
        max_n=300,
    ):
        """Melhor troca (se houver) para cada produto da categoria, retornando DataFrame."""
        sub = sub.sample(min(len(sub), max_n), random_state=0).reset_index(drop=True)
        T, H, C = preparar(sub, pesos, considerar_porcao=(modo_ganho != "100g"))
        linhas = []
        for i in range(len(sub)):
            tab = avaliar_trocas(
                sub, T, H, C, i, pesos, tol,
                modo_ganho=modo_ganho,
                max_aumento_preco=max_aumento_preco,
                pesos_acc=pesos_acc,
            )
            ok = escolher_troca(tab, ganho_min, conf_min)
            if ok.empty:
                linhas.append(dict(
                    produto=sub["nome"][i], N=0.0, nivel="nenhum", ganho=np.nan,
                    ganho_100g=np.nan, ganho_porcao=np.nan, delta_preco_pct=np.nan,
                    acc=np.nan, conf=np.nan, driver=None, grau_a=np.nan, grau_b=np.nan,
                ))
                continue
            b = ok.iloc[0]
            j = int(b["j"])
            c = contribuicoes(T, i, j, pesos)
            linhas.append(dict(
                produto=sub["nome"][i], N=b["N"], nivel=nivel(b["N"], t1, t2),
                ganho=b["ganho"], ganho_100g=b["ganho_100g"], ganho_porcao=b["ganho_porcao"],
                delta_preco_pct=b["delta_preco_pct"], acc=b["acc"], conf=b["conf"],
                driver=c.index[0],
                grau_a=GRAU.get(sub["nutriscore"][i], np.nan),
                grau_b=GRAU.get(sub["nutriscore"][j], np.nan),
            ))
        return pd.DataFrame(linhas)

    def gerar_demo(n_por_cat=50, seed=7):
        """Dados FICTÍCIOS (incluindo porção e preço simulado) para testar sem internet."""
        rng = np.random.default_rng(seed)
        perfis = {
            "biscoitos-demo": dict(g=(14, 6), s=(6, 3), a=(24, 9), sal=(0.7, 0.35), f=(3, 2), kcal=(470, 40), porcao=(30, 8), preco=(6.5, 2.0)),
            "iogurtes-demo": dict(g=(3, 2), s=(2, 1.2), a=(11, 4.5), sal=(0.15, 0.08), f=(0.4, 0.4), kcal=(95, 20), porcao=(170, 30), preco=(4.8, 1.5)),
            "cereais-demo": dict(g=(6, 3), s=(1.8, 1.2), a=(18, 8), sal=(0.6, 0.3), f=(6, 3), kcal=(380, 35), porcao=(40, 10), preco=(14.0, 4.0)),
        }
        simples = ["farinha", "aveia", "leite", "fruta", "mel", "sal", "integral"]
        aditivos = ["aroma", "emulsificante", "corante", "xarope-glicose", "gordura-vegetal"]
        linhas = []
        for cat, p in perfis.items():
            for k in range(n_por_cat):
                q = rng.uniform()
                esc = 1.25 - 0.5 * q
                gord = max(0.0, rng.normal(p["g"][0] * esc, p["g"][1] * 0.5))
                sat = min(max(0.0, rng.normal(p["s"][0] * esc, p["s"][1] * 0.5)), gord)
                ac = max(0.0, rng.normal(p["a"][0] * esc, p["a"][1] * 0.5))
                sal = max(0.0, rng.normal(p["sal"][0] * esc, p["sal"][1] * 0.5))
                fib = max(0.0, rng.normal(p["f"][0] * (0.6 + 0.8 * q), p["f"][1] * 0.5))
                kcal = max(20.0, rng.normal(p["kcal"][0], p["kcal"][1]))
                porcao_g = round(max(10.0, rng.normal(p["porcao"][0], p["porcao"][1])), 0)
                preco = round(max(1.5, rng.normal(p["preco"][0] * (0.9 + 0.25 * q), p["preco"][1] * 0.5)), 2)
                nova = 4 if q < 0.35 else (3 if q < 0.7 else int(rng.choice([1, 2, 3])))
                ing = [f"{cat}:{x}" for x in rng.choice(simples, size=min(7, int(rng.binomial(7, 0.3 + 0.5 * q)) + 1), replace=False)]
                ing += [f"{cat}:{x}" for x in rng.choice(aditivos, size=int(rng.binomial(4, 0.8 * (1 - q))), replace=False)]
                grau = "abcde"[int(np.digitize(5 * (1 - q) + rng.normal(0, 0.6), [1, 2, 3, 4]))]
                linhas.append(dict(
                    codigo=f"DEMO-{cat[:3]}-{k:03d}",
                    nome=f"{cat.split('-')[0].capitalize()} fictício {k + 1:02d}",
                    marca=f"Marca {'ABCD'[k % 4]} (fictícia)",
                    categoria=cat,
                    ingredientes=frozenset(ing),
                    kcal=kcal,
                    gordura=gord,
                    sat=sat,
                    acucar=ac,
                    sal=sal,
                    fibra=np.nan if rng.uniform() < 0.25 else fib,
                    nova=np.nan if rng.uniform() < 0.10 else float(nova),
                    porcao_g=np.nan if rng.uniform() < 0.10 else porcao_g,
                    preco=np.nan if rng.uniform() < 0.15 else preco,
                    nutriscore=grau,
                ))
        return pd.DataFrame(linhas)
    # <<CORE-END>>
    return (
        avaliar_categoria,
        avaliar_trocas,
        contribuicoes,
        escolher_troca,
        gerar_demo,
        nivel,
        preparar,
        textos,
    )


@app.cell
def _(gerar_demo, mo, modo, np, pais, pd, upload_arquivo):
    import io
    import re
    from pathlib import Path

    _OFF_URL = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
    _CACHE = Path("off_pais_v2.parquet")
    diag = ""

    def _parse_porcao_g(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val) if 0 < val <= 2000 else np.nan
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(g|gr|ml)\b", str(val).lower())
        if m:
            v = float(m.group(1).replace(",", "."))
            return v if 0 < v <= 2000 else np.nan
        return np.nan

    def _parse_ingredientes_local(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return frozenset()
        if isinstance(val, (set, frozenset, list, tuple)):
            return frozenset(str(x).strip().lower() for x in val if str(x).strip())
        partes = re.split(r"[,;|]", str(val).lower())
        return frozenset(p.strip() for p in partes if p.strip())

    def _normalizar_tabela_local(df_raw):
        df = df_raw.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        aliases = {
            "produto": "nome", "product_name": "nome",
            "brand": "marca", "brands": "marca",
            "category": "categoria", "categories": "categoria",
            "fat": "gordura", "gorduras_totais": "gordura",
            "saturated_fat": "sat", "gordura_saturada": "sat", "gorduras_saturadas": "sat",
            "sugars": "acucar", "sugar": "acucar", "acucares": "acucar",
            "salt": "sal",
            "fiber": "fibra", "fibras": "fibra",
            "energy_kcal": "kcal", "calorias": "kcal",
            "nova_group": "nova",
            "serving_g": "porcao_g", "porcao": "porcao_g",
            "price": "preco", "valor": "preco",
            "nutriscore_grade": "nutriscore",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
        if "sal" not in df.columns and "sodio" in df.columns:
            df["sal"] = pd.to_numeric(df["sodio"], errors="coerce") * 2.5
        if "marca" not in df.columns:
            df["marca"] = "Marca local"
        if "codigo" not in df.columns:
            df["codigo"] = [f"LOCAL-{i:04d}" for i in range(len(df))]
        if "ingredientes" in df.columns:
            df["ingredientes"] = df["ingredientes"].apply(_parse_ingredientes_local)
        else:
            df["ingredientes"] = [frozenset() for _ in range(len(df))]
        for col in ["kcal", "gordura", "sat", "acucar", "sal", "fibra", "nova", "porcao_g", "preco"]:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "nutriscore" not in df.columns:
            df["nutriscore"] = None
        else:
            df["nutriscore"] = df["nutriscore"].astype(str).str.lower().where(
                df["nutriscore"].astype(str).str.lower().isin(list("abcde")), None
            )
        return df

    if modo.value.startswith("Demonstração"):
        bruto = gerar_demo()
        fonte_txt = "⚠️ **Dados FICTÍCIOS de demonstração** (com porção e preço simulados). Não representam produtos reais."
    elif modo.value.startswith("Arquivo local"):
        if not upload_arquivo.value:
            bruto = gerar_demo()
            fonte_txt = (
                "📂 **Aguardando arquivo CSV ou Parquet** (usando demonstração temporariamente). "
                "Colunas esperadas: `nome`, `categoria`, `gordura`, `sat`, `acucar`, `sal` "
                "(opcionais: `marca`, `ingredientes`, `kcal`, `fibra`, `nova`, `porcao_g`, `preco`, `nutriscore`)."
            )
        else:
            _arq = upload_arquivo.value[0]
            _buf = io.BytesIO(_arq.contents)
            _raw = pd.read_parquet(_buf) if _arq.name.lower().endswith(".parquet") else pd.read_csv(_buf)
            bruto = _normalizar_tabela_local(_raw)
            fonte_txt = f"📂 **Arquivo local carregado:** `{_arq.name}` ({len(bruto)} linhas)."
    else:
        import duckdb

        if not re.fullmatch(r"[a-z]{2}:[a-z-]+", pais.value):
            raise ValueError("Tag de país inválida. Exemplo: en:brazil")
        _con = duckdb.connect()
        if not _CACHE.exists():
            _con.execute("INSTALL httpfs")
            _con.execute("LOAD httpfs")
            _con.execute(f"""
                COPY (
                    SELECT code, product_name, brands, categories_tags, ingredients_tags,
                           nutriments, serving_size, nutriscore_grade, nova_group
                    FROM read_parquet('{_OFF_URL}')
                    WHERE list_contains(countries_tags, '{pais.value}')
                ) TO '{_CACHE}' (FORMAT parquet, COMPRESSION zstd)
            """)
        _df = _con.execute(f"SELECT * FROM read_parquet('{_CACHE}')").df()

        def _nutr_dict(x):
            if x is None:
                return {}
            if isinstance(x, dict):
                return x
            try:
                return {i.get("name"): (i.get("100g") if i.get("100g") is not None else i.get("value"))
                        for i in x if isinstance(i, dict)}
            except TypeError:
                return {}

        def _num(d, k):
            try:
                return float(d.get(k))
            except (TypeError, ValueError):
                return np.nan

        def _nome(x):
            try:
                itens = [i for i in x if isinstance(i, dict)]
                for i in itens:
                    if i.get("lang") == "main":
                        return i.get("text")
                return itens[0].get("text") if itens else None
            except TypeError:
                return None

        _linhas = []
        for _r in _df.itertuples(index=False):
            _d = _nutr_dict(_r.nutriments)
            _sal = _num(_d, "salt")
            if np.isnan(_sal):
                _sod = _num(_d, "sodium")
                _sal = _sod * 2.5 if not np.isnan(_sod) else np.nan
            _cats = list(_r.categories_tags) if _r.categories_tags is not None else []
            _ing = frozenset(str(x) for x in _r.ingredients_tags) if _r.ingredients_tags is not None else frozenset()
            _g = str(_r.nutriscore_grade).lower() if _r.nutriscore_grade is not None else None
            _linhas.append(dict(
                codigo=_r.code,
                nome=_nome(_r.product_name),
                marca=_r.brands,
                categoria=_cats[-1] if _cats else None,
                ingredientes=_ing,
                kcal=_num(_d, "energy-kcal"),
                gordura=_num(_d, "fat"),
                sat=_num(_d, "saturated-fat"),
                acucar=_num(_d, "sugars"),
                sal=_sal,
                fibra=_num(_d, "fiber"),
                nova=float(_r.nova_group) if pd.notna(_r.nova_group) else np.nan,
                porcao_g=_parse_porcao_g(getattr(_r, "serving_size", None)),
                preco=np.nan,
                nutriscore=_g if _g in list("abcde") else None,
            ))
        bruto = pd.DataFrame(_linhas)
        _ex = _df["nutriments"].iloc[0] if len(_df) else None
        diag = f"Tipo de `nutriments`: `{type(_ex).__name__}`. Preço ausente no OFF recebe fallback neutro."
        fonte_txt = (f"Dados: **Open Food Facts** (licença ODbL), país `{pais.value}`; "
                     f"snapshot em cache: {pd.Timestamp(_CACHE.stat().st_mtime, unit='s'):%Y-%m-%d}. {diag}")

    mo.md(fonte_txt)
    return (bruto,)


@app.cell
def _(bruto, min_prod, np):
    _b = bruto.dropna(subset=["nome", "categoria", "gordura", "sat", "acucar", "sal"]).copy()
    _lim = ["gordura", "sat", "acucar", "sal"]
    _ok = (_b[_lim].ge(0).all(axis=1) & _b[_lim].le(100).all(axis=1)
           & (_b["sat"] <= _b["gordura"] + 0.5) & ((_b["gordura"] + _b["acucar"] + _b["sal"]) <= 105))
    base = _b[_ok].copy()
    base.loc[(base["kcal"] < 0) | (base["kcal"] > 900), "kcal"] = np.nan
    base.loc[(base["fibra"] < 0) | (base["fibra"] > 100), "fibra"] = np.nan
    if "porcao_g" in base.columns:
        base.loc[(base["porcao_g"] <= 0) | (base["porcao_g"] > 2000), "porcao_g"] = np.nan
    if "preco" in base.columns:
        base.loc[base["preco"] <= 0, "preco"] = np.nan
    _cont = base["categoria"].value_counts()
    cats = _cont[_cont >= min_prod.value].index.tolist()
    base = base[base["categoria"].isin(cats)]
    return base, cats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Perfil do usuário: *você* define o que importa (não há dado sensível)
    """)
    return


@app.cell
def _(mo):
    def _s(rotulo, v=1.0):
        return mo.ui.slider(start=0, stop=3, step=0.5, value=v, label=rotulo)

    w_gordura = _s("Gordura total")
    w_sat = _s("Gordura saturada")
    w_acucar = _s("Açúcares")
    w_sal = _s("Sal (sódio)")
    w_proc = _s("Evitar ultraprocessados (NOVA 4)")
    w_fibra = _s("Mais fibra", 0.5)

    modo_ganho = mo.ui.dropdown(
        options={
            "Combinado: min(ganho 100 g, ganho porção)": "combinado",
            "Por porção (H × porção)": "porcao",
            "Por 100 g (clássico)": "100g",
        },
        value="Combinado: min(ganho 100 g, ganho porção)",
        label="Base de cálculo do ganho",
    )
    tol = mo.ui.slider(start=0.0, stop=0.5, step=0.05, value=0.10, label="Tolerância de piora nutricional (veto)")
    max_aumento_preco = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.25, label="Aumento máx. de preço (veto)")
    ganho_min = mo.ui.slider(start=0.0, stop=0.5, step=0.05, value=0.10, label="Ganho mínimo")
    conf_min = mo.ui.slider(start=0.5, stop=1.0, step=0.05, value=0.6, label="Confiança mínima")
    t1 = mo.ui.slider(start=0.0, stop=0.3, step=0.01, value=0.05, label="Limiar: informativo (N ≥)")
    t2 = mo.ui.slider(start=0.0, stop=0.6, step=0.01, value=0.15, label="Limiar: sugestão (N ≥)")

    wa_ing = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.45, label="Acc: ingredientes (Jaccard)")
    wa_kcal = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.25, label="Acc: energia (kcal/100 g)")
    wa_porcao = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.15, label="Acc: tamanho da porção")
    wa_preco = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.15, label="Acc: afinidade de preço")

    mo.hstack(
        [
            mo.vstack([mo.md("**Pesos nutricionais**"), w_gordura, w_sat, w_acucar, w_sal, w_proc, w_fibra]),
            mo.vstack([mo.md("**Porção, preço e guardas**"), modo_ganho, tol, max_aumento_preco, ganho_min, conf_min, t1, t2]),
            mo.vstack([mo.md("**Composição de Acc**"), wa_ing, wa_kcal, wa_porcao, wa_preco]),
        ],
        justify="space-around",
    )
    return (
        conf_min,
        ganho_min,
        max_aumento_preco,
        modo_ganho,
        t1,
        t2,
        tol,
        w_acucar,
        w_fibra,
        w_gordura,
        w_proc,
        w_sal,
        w_sat,
        wa_ing,
        wa_kcal,
        wa_porcao,
        wa_preco,
    )


@app.cell
def _(
    w_acucar,
    w_fibra,
    w_gordura,
    w_proc,
    w_sal,
    w_sat,
    wa_ing,
    wa_kcal,
    wa_porcao,
    wa_preco,
):
    pesos = {
        "gordura": w_gordura.value, "sat": w_sat.value, "acucar": w_acucar.value,
        "sal": w_sal.value, "proc": w_proc.value, "fibra": w_fibra.value,
    }
    pesos_acc = {
        "ing": wa_ing.value, "kcal": wa_kcal.value,
        "porcao": wa_porcao.value, "preco": wa_preco.value,
    }
    return pesos, pesos_acc


@app.cell
def _(cats, mo):
    categoria_sel = mo.ui.dropdown(options=cats or ["(nenhuma categoria)"], value=(cats or ["(nenhuma categoria)"])[0],
                                   label="**Categoria**")
    categoria_sel
    return (categoria_sel,)


@app.cell
def _(base, categoria_sel, mo, pd):
    produtos_cat = base[base["categoria"] == categoria_sel.value].reset_index(drop=True)

    def _label_prod(i, r):
        extras = []
        if hasattr(r, "porcao_g") and pd.notna(r.porcao_g):
            extras.append(f"{r.porcao_g:.0f}g")
        if hasattr(r, "preco") and pd.notna(r.preco):
            extras.append(f"R${r.preco:.2f}")
        suf = f" ({', '.join(extras)})" if extras else ""
        return f"{r.nome} — {r.marca}{suf} [{i}]"

    _opts = {_label_prod(i, r): i for i, r in enumerate(produtos_cat.itertuples())}
    _primeiro = next(iter(_opts), None)
    produto_sel = mo.ui.dropdown(options=_opts or {"(vazio)": 0}, value=_primeiro or "(vazio)", label="**Produto atual**")
    produto_sel
    return produto_sel, produtos_cat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) A cutucada para o produto escolhido
    """)
    return


@app.cell
def _(
    avaliar_trocas,
    conf_min,
    contribuicoes,
    escolher_troca,
    ganho_min,
    max_aumento_preco,
    mo,
    modo_ganho,
    nivel,
    pd,
    pesos,
    pesos_acc,
    preparar,
    produto_sel,
    produtos_cat,
    t1,
    t2,
    textos,
    tol,
):
    if len(produtos_cat) < 2:
        _saida = mo.callout(mo.md("Categoria com poucos produtos. Reduza o mínimo por categoria."), kind="warn")
    else:
        _i = produto_sel.value
        _T, _H, _C = preparar(produtos_cat, pesos, considerar_porcao=(modo_ganho.value != "100g"))
        _a = produtos_cat.iloc[_i]
        if pd.isna(_H[_i]):
            _saida = mo.callout(mo.md("Defina ao menos um peso nutricional maior que zero."), kind="warn")
        else:
            _tab = avaliar_trocas(
                produtos_cat, _T, _H, _C, _i, pesos, tol.value,
                modo_ganho=modo_ganho.value,
                max_aumento_preco=max_aumento_preco.value,
                pesos_acc=pesos_acc,
            )
            _ok = escolher_troca(_tab, ganho_min.value, conf_min.value)
            _p_txt = f" · porção {_a['porcao_g']:.0f} g" if pd.notna(_a.get("porcao_g")) else ""
            _pr_txt = f" · R$ {_a['preco']:.2f}" if pd.notna(_a.get("preco")) else ""
            _cab = mo.md(
                f"**{_a['nome']}** · {_a['marca']}{_p_txt}{_pr_txt} · "
                f"carga **{_H[_i]:.0f}/100** · confiança {_C[_i]:.0%}"
            )
            if _ok.empty:
                _n_pos = len(_tab)
                _n_veto = int(_tab["vetado"].sum()) if _n_pos else 0
                _card = mo.callout(mo.md(
                    f"**Sem cutucada.** Candidatos com ganho positivo: {_n_pos} (vetados por piorar nutriente "
                    f"ou exceder aumento de preço: {_n_veto}). Nenhum passa nas guardas atuais. "
                    "Não cutucar também é um resultado válido."), kind="neutral")
                _saida = mo.vstack([_cab, _card])
            else:
                _b = _ok.iloc[0]
                _j = int(_b["j"])
                _bp = produtos_cat.iloc[_j]
                _c = contribuicoes(_T, _i, _j, pesos)
                _tx = textos(_a, _bp, _c, _H[_i], _H[_j], _b)
                _nv = nivel(_b["N"], t1.value, t2.value)
                _corpo = "\n\n".join(f"**{k}**  \n{v}" for k, v in _tx.items())
                _card = mo.callout(mo.md(
                    f"### Troque por: {_bp['nome']} · {_bp['marca']}\n"
                    f"**Força do nudge N = {_b['N']:.2f}** (ganho {_b['ganho']:.0%} × aceitabilidade {_b['acc']:.0%} × "
                    f"confiança {_b['conf']:.0%}) → nível: **{_nv}**\n\n{_corpo}"),
                    kind="success" if _nv == "sugestão" else "info")
                _top = _tab.sort_values("N", ascending=False).head(5).copy()
                _top["produto"] = produtos_cat["nome"].iloc[_top["j"]].to_numpy()
                _top["status"] = _top["vetado"].map({True: "vetado", False: "ok"})
                _top = _top[[
                    "produto", "ganho_100g", "ganho_porcao", "ganho",
                    "delta_preco_pct", "acc", "conf", "N", "status", "motivo",
                ]].round(2)
                _saida = mo.vstack([_cab, _card, mo.md("**Melhores candidatos (inclui vetados, para transparência):**"),
                                    mo.ui.table(_top, selection=None)])
    _saida
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Avaliação offline da categoria (sem participantes)

    * **Potencial:** em quantos produtos existe uma troca que passa nas guardas?
    * **Concordância externa:** a troca também melhora (ou mantém) o Nutri-Score oficial?
    * **Fidelidade:** como a carga por 100 g é aditiva, a contribuição de cada termo fecha exatamente com `H_A − H_B`.
    """)
    return


@app.cell
def _(
    avaliar_categoria,
    conf_min,
    ganho_min,
    max_aumento_preco,
    mo,
    modo_ganho,
    np,
    pesos,
    pesos_acc,
    produtos_cat,
    px,
    t1,
    t2,
    tol,
):
    if len(produtos_cat) < 2:
        df_avaliacao = None
        _out = mo.md("Sem dados suficientes.")
    else:
        df_avaliacao = avaliar_categoria(
            produtos_cat, pesos, tol.value, ganho_min.value, conf_min.value, t1.value, t2.value,
            modo_ganho=modo_ganho.value,
            max_aumento_preco=max_aumento_preco.value,
            pesos_acc=pesos_acc,
        )
        _com = df_avaliacao[df_avaliacao["nivel"] != "nenhum"]
        _par = df_avaliacao[df_avaliacao["nivel"] != "nenhum"].dropna(subset=["grau_a", "grau_b"])
        _melhora = float((_par["grau_b"] < _par["grau_a"]).mean()) if len(_par) else np.nan
        _mantem = float((_par["grau_b"] <= _par["grau_a"]).mean()) if len(_par) else np.nan
        _txt = mo.md(
            f"Produtos avaliados: **{len(df_avaliacao)}** · com nudge viável: **{len(_com) / len(df_avaliacao):.0%}** "
            f"· ganho médio: **{(_com['ganho'].mean() if len(_com) else float('nan')):.0%}** · "
            f"Nutri-Score melhora: **{_melhora:.0%}** / melhora ou mantém: **{_mantem:.0%}** "
            f"(n com grau nos dois = {len(_par)})")
        _fig = px.scatter(_com, x="acc", y="ganho", color="N", hover_name="produto",
                          labels={"acc": "Aceitabilidade composta", "ganho": "Ganho efetivo", "N": "Força N"},
                          title="Mapa de cutucadas: quanto maior o ganho e a aceitabilidade, melhor a troca") if len(_com) else None
        _out = mo.vstack([
            _txt,
            mo.ui.table(df_avaliacao.round(2), selection=None),
            *([_fig] if _fig is not None else []),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitações (leia antes de citar qualquer número)

    * Pesos, limiares, `Acc` e `C` são **heurísticas**; a calibração exige dados de comportamento.
    * O Open Food Facts geralmente não traz preço padronizado; sem preço, o termo financeiro recebe fallback neutro sem zerar a confiança nutricional.
    * Categoria = **última tag** de `categories_tags` (suposição simples); confira se agrupa produtos comparáveis.
    * A carga usa limiares da FSA/UK como régua, não é um diagnóstico nem recomendação clínica.
    * Sem experimento, **não há afirmação sobre mudança de comportamento**.
    """)
    return


if __name__ == "__main__":
    app.run()