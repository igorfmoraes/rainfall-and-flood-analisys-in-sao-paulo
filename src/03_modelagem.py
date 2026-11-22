#!/usr/bin/env python3
"""Modelagem: previsão de dia com registro de enxurrada/inundação/alagamento na RMSP a partir da chuva.

Uso:
    python 03_modelagem.py --dataset dataset_rmsp.csv
    python 03_modelagem.py --dataset dataset_rmsp.csv --ano-max 2025     # sensibilidade: período completo
    python 03_modelagem.py --dataset dataset_rmsp.csv --sem-lag           # sem a chuva do dia anterior

Protocolo (definido antes de ver os resultados):
    - Janela padrão 1991-2013: depois de 2013 o Atlas quase não tem registros hidrológicos (mudança
      de registro, não necessariamente de ocorrência). Use --ano-max 2025 para ver o efeito.
    - Validação: leave-one-year-out (cada ano é teste uma vez; treino nos demais). As previsões
      fora da amostra são reunidas e avaliadas juntas.
    - Métrica principal: PR-AUC (average precision). Referência: a prevalência de positivos.
      Também ROC-AUC e precisão nos k dias de maior escore, com k = número de positivos.
    - Incerteza: bootstrap em blocos de ano (os anos são reamostrados, não os dias).
    - Modelos: (1) baseline de uma variável (a de melhor PR-AUC no treino), (2) regressão logística
      implementada com gradiente descendente em numpy, (3) gradient boosting (scikit-learn).
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

NAO_FEATURES = {"data", "y", "n_municipios_evento", "ano", "periodo", "mes"}


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class RegressaoLogisticaGD:
    """Regressão logística: perda de entropia cruzada ponderada + penalidade L2.

    Minimiza  (1/n) * sum_i s_i * [-y_i log p_i - (1-y_i) log(1-p_i)] + (l2/2) * ||w||^2
    por gradiente descendente em lote. O gradiente em w é (1/n) * Z^T [s * (p - y)] + l2 * w.
    s_i são pesos por classe (balanceamento) quando balancear=True.
    """

    def __init__(self, taxa=0.1, iteracoes=3000, l2=1e-2, balancear=True):
        self.taxa, self.iteracoes, self.l2, self.balancear = taxa, iteracoes, l2, balancear

    def pesos(self, y):
        if not self.balancear:
            return np.ones(len(y))
        n, n_pos = len(y), y.sum()
        return np.where(y == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))

    def fit(self, X, y):
        self.media_, self.desvio_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.media_) / self.desvio_
        n, d = Z.shape
        s = self.pesos(y)
        self.w_, self.b_ = np.zeros(d), 0.0
        for _ in range(self.iteracoes):
            erro = s * (sigmoide(Z @ self.w_ + self.b_) - y)
            self.w_ -= self.taxa * (Z.T @ erro / n + self.l2 * self.w_)
            self.b_ -= self.taxa * erro.mean()
        return self

    def decision_function(self, X):
        return ((X - self.media_) / self.desvio_) @ self.w_ + self.b_

    def predict_proba(self, X):
        return sigmoide(self.decision_function(X))


def verificar_logistica(X, y, l2):
    """Compara a implementação própria com a do scikit-learn (mesma função objetivo)."""
    meu = RegressaoLogisticaGD(l2=l2).fit(X, y)
    Z = (X - meu.media_) / meu.desvio_
    ref = LogisticRegression(C=1.0 / (l2 * len(y)), max_iter=5000).fit(Z, y, sample_weight=meu.pesos(y))
    corr = np.corrcoef(meu.decision_function(X), ref.decision_function(Z))[0, 1]
    dif = np.abs(meu.w_ - ref.coef_[0]).max()
    print(f"Own logistic regression vs scikit-learn: score correlation {corr:.4f} | largest coefficient difference {dif:.4f}")
    return meu


def melhor_variavel(X, y):
    return int(np.argmax([average_precision_score(y, X[:, j]) for j in range(X.shape[1])]))


def precisao_top_k(y, escore):
    k = int(y.sum())
    return y[np.argsort(-escore)[:k]].mean()


def avaliar(y, s):
    return {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s), "Precision@k": precisao_top_k(y, s)}


def bootstrap_anos(y, escores, anos, repeticoes, rng):
    """Reamostra anos inteiros. Retorna IC95% da PR-AUC de cada modelo e da diferença para a baseline."""
    unicos = np.unique(anos)
    indices = {a: np.where(anos == a)[0] for a in unicos}
    nomes = list(escores)
    valores = {n: [] for n in nomes}
    for _ in range(repeticoes):
        idx = np.concatenate([indices[a] for a in rng.choice(unicos, len(unicos))])
        if y[idx].sum() == 0:
            continue
        for n in nomes:
            valores[n].append(average_precision_score(y[idx], escores[n][idx]))
    ic = {n: np.percentile(v, [2.5, 97.5]) for n, v in valores.items()}
    base = np.array(valores[nomes[0]])
    dif = {n: np.percentile(np.array(valores[n]) - base, [2.5, 97.5]) for n in nomes[1:]}
    return ic, dif


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ano-min", "--year-min", type=int, default=1991)
    ap.add_argument("--ano-max", "--year-max", type=int, default=2013)
    ap.add_argument("--sem-lag", "--no-lag", action="store_true", help="remove as features da chuva do dia anterior")
    ap.add_argument("--excluir", "--exclude", default="", help="remove features cujo nome contém algum destes trechos (separados por vírgula), ex.: doy ou 7d,15d,30d")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--repeticoes", "--repetitions", type=int, default=1000)
    ap.add_argument("--semente", "--seed", type=int, default=0)
    ap.add_argument("--prefixo", "--prefix", default="resultados")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, parse_dates=["data"])
    df = df[(df["ano"] >= args.ano_min) & (df["ano"] <= args.ano_max)].reset_index(drop=True)
    excluir = [t.strip() for t in args.excluir.split(",") if t.strip()]
    colunas = [
        c for c in df.columns
        if c not in NAO_FEATURES
        and not (args.sem_lag and "dia_anterior" in c)
        and not any(t in c for t in excluir)
    ]
    df = df.dropna(subset=colunas).reset_index(drop=True)
    X, y, anos = df[colunas].to_numpy(float), df["y"].to_numpy(int), df["ano"].to_numpy()
    rng = np.random.default_rng(args.semente)

    print(f"Window {args.ano_min}-{args.ano_max}: {len(df)} days | positives {y.sum()} ({y.mean():.2%}) | {len(np.unique(anos))} years | {len(colunas)} features")
    verificar_logistica(X, y, args.l2)

    modelos = ["baseline", "logistic", "boosting"]
    oof = {m: np.zeros(len(df)) for m in modelos}
    escolhidas, importancias = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=anos):
        j = melhor_variavel(X[tr], y[tr])
        escolhidas.append(colunas[j])
        oof["baseline"][te] = X[te, j]
        oof["logistic"][te] = RegressaoLogisticaGD(l2=args.l2).fit(X[tr], y[tr]).decision_function(X[te])
        gb = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
            class_weight="balanced", random_state=args.semente,
        ).fit(X[tr], y[tr])
        oof["boosting"][te] = gb.predict_proba(X[te])[:, 1]
        if y[te].sum() > 0:
            pi = permutation_importance(gb, X[te], y[te], scoring="average_precision", n_repeats=5, random_state=args.semente)
            importancias.append(pi.importances_mean)

    ic, dif = bootstrap_anos(y, oof, anos, args.repeticoes, rng)
    linhas = []
    for m in modelos:
        met = avaliar(y, oof[m])
        linhas.append({"model": m, **met, "PR-AUC 95% CI low": ic[m][0], "PR-AUC 95% CI high": ic[m][1],
                       "diff vs baseline low": dif[m][0] if m in dif else np.nan,
                       "diff vs baseline high": dif[m][1] if m in dif else np.nan})
    tabela = pd.DataFrame(linhas)
    print(f"\nPrevalence (PR-AUC of a random score): {y.mean():.4f}")
    print(tabela.round(3).to_string(index=False))
    print("\nFeature chosen by the baseline (count across folds):")
    print(pd.Series(escolhidas).value_counts().to_string())
    tabela.to_csv(f"{args.prefixo}_metricas.csv", index=False)

    # importância das variáveis
    imp = pd.DataFrame({"feature": colunas, "boosting_perm_importance": np.mean(importancias, axis=0)})
    final = RegressaoLogisticaGD(l2=args.l2).fit(X, y)
    imp["logistic_std_coef"] = final.w_
    imp = imp.sort_values("boosting_perm_importance", ascending=False)
    print("\nFeature importance:")
    print(imp.round(4).to_string(index=False))
    imp.to_csv(f"{args.prefixo}_importancia.csv", index=False)

    # curvas precisão-recall
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nomes = {"baseline": "single-variable baseline", "logistic": "logistic regression", "boosting": "gradient boosting"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in modelos:
        p, r, _ = precision_recall_curve(y, oof[m])
        ax.plot(r, p, label=f"{nomes[m]} (PR-AUC {average_precision_score(y, oof[m]):.2f})")
    ax.axhline(y.mean(), color="gray", linestyle="--", label=f"prevalence ({y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curves, leave-one-year-out validation ({args.ano_min}-{args.ano_max})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.prefixo}_pr.png", dpi=200)
    print(f"\nWritten: {args.prefixo}_metricas.csv, {args.prefixo}_importancia.csv, {args.prefixo}_pr.png")


if __name__ == "__main__":
    main()
