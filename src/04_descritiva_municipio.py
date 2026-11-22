#!/usr/bin/env python3
"""Análise descritiva por município e figuras para o artigo.

Uso:
    python 04_descritiva_municipio.py --atlas Atlas_Apenas_SP.xlsx --chuva chirps_rmsp_diario.csv --dataset dataset_rmsp.csv

Saídas (pasta descritiva/):
    municipios_resumo.csv     registros hidrológicos por município e período, tipos, exposição à chuva forte
    fig_serie_anual.png       registros por ano: hidrológicos vs. demais tipologias (mudança de 2013/2014)
    fig_sazonalidade.png      registros por mês
    fig_municipios.png        registros por município, 1991-2013 e 2014-2025
    fig_chuva_dias.png        distribuição da chuva em dias com e sem registro (ECDF)
    fig_registros_vs_chuva.png  registros vs. dias de chuva forte, por município

Observações de interpretação:
    - "Registro" é o que consta no Atlas (depende da Defesa Civil municipal registrar), não a ocorrência real.
    - A chuva é do CHIRPS (pixel ~5 km). Municípios vizinhos e pequenos compartilham pixels, então os
      dias de chuva forte são muito parecidos entre eles; diferenças de registros entre municípios
      vizinhos refletem mais registro, relevo e urbanização do que diferença de chuva medida.
    - A análise de 1991-2013 usa a mesma janela da modelagem; 2014-2025 aparece separada.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rmsp import RMSP

ABA = "Atlas Valores Corrigidos"
COLUNAS_ATLAS = [
    "protocolo", "municipio", "uf", "regiao", "data_registro", "data_evento", "cobrade",
    "tipologia", "descricao", "grupo", "ibge", "setores", "status",
]
HIDRO = ["Enxurradas", "Inundações", "Alagamentos"]  # values of the Atlas typology field
HIDRO_EN = {"Enxurradas": "flash_floods", "Inundações": "river_floods", "Alagamentos": "urban_flooding"}
ANO_CORTE = 2013


def ler_atlas_rmsp(caminho):
    bruto = pd.read_excel(caminho, sheet_name=ABA, header=None).iloc[:, : len(COLUNAS_ATLAS)]
    bruto.columns = COLUNAS_ATLAS
    if isinstance(bruto.iloc[0]["data_evento"], str):
        bruto = bruto.iloc[1:]
    bruto["data_evento"] = pd.to_datetime(bruto["data_evento"], errors="coerce").dt.normalize()
    bruto["ibge"] = pd.to_numeric(bruto["ibge"], errors="coerce")
    ev = bruto[bruto["ibge"].isin(list(RMSP))].dropna(subset=["data_evento"]).copy()
    ev["ibge"] = ev["ibge"].astype(int)
    ev["ano"] = ev["data_evento"].dt.year
    return ev[["ibge", "data_evento", "ano", "descricao"]]


def ler_chuva(caminho):
    t = pd.read_csv(caminho, parse_dates=["data"])
    return t.pivot(index="data", columns="ibge", values="precip_mm").sort_index()


def tabela_municipios(ev_h, chuva, limiar):
    ate = ev_h[ev_h["ano"] <= ANO_CORTE]
    depois = ev_h[ev_h["ano"] > ANO_CORTE]
    tab = pd.DataFrame({"ibge": list(RMSP), "municipality": list(RMSP.values())}).set_index("ibge")
    tab["records_1991_2013"] = ate.groupby("ibge").size().reindex(tab.index).fillna(0).astype(int)
    tab["records_2014_2025"] = depois.groupby("ibge").size().reindex(tab.index).fillna(0).astype(int)
    tab["records_total"] = tab["records_1991_2013"] + tab["records_2014_2025"]
    tab["record_days_1991_2013"] = ate.groupby("ibge")["data_evento"].nunique().reindex(tab.index).fillna(0).astype(int)
    for tipo in HIDRO:
        tab[f"{HIDRO_EN[tipo]}_1991_2013"] = ate[ate["descricao"] == tipo].groupby("ibge").size().reindex(tab.index).fillna(0).astype(int)

    anos_completos = chuva.index.year.isin(range(chuva.index.year.min(), chuva.index.year.max() + 1))
    anual = chuva[anos_completos].groupby(chuva.index[anos_completos].year).sum()
    tab["mean_annual_rain_mm"] = anual.mean().reindex(tab.index).round(0)
    janela = chuva[chuva.index.year <= ANO_CORTE]
    tab[f"heavy_rain_days_{limiar:g}mm_1991_2013"] = (janela >= limiar).sum().reindex(tab.index).astype(int)
    col = f"heavy_rain_days_{limiar:g}mm_1991_2013"
    tab["records_per_100_heavy_rain_days"] = (100 * tab["records_1991_2013"] / tab[col].replace(0, np.nan)).round(1)
    return tab.reset_index(), col


def fig_serie_anual(ev, ev_h, destino):
    anos = range(1991, 2026)
    h = ev_h.groupby("ano").size().reindex(anos, fill_value=0)
    chuvas = ev[ev["descricao"] == "Chuvas Intensas"].groupby("ano").size().reindex(anos, fill_value=0)
    outros = ev[~ev["descricao"].isin(HIDRO + ["Chuvas Intensas"])].groupby("ano").size().reindex(anos, fill_value=0)
    x = np.array(list(anos))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.27, h.values, width=0.27, label="Flash flood, river flood, urban flooding")
    ax.bar(x, chuvas.values, width=0.27, label="Heavy rain")
    ax.bar(x + 0.27, outros.values, width=0.27, label="Other typologies")
    ax.axvline(ANO_CORTE + 0.5, color="gray", linestyle="--")
    ax.set_xlabel("Event year")
    ax.set_ylabel("Atlas records (RMSP)")
    ax.set_title("Atlas records per year, RMSP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def fig_sazonalidade(ev_h, destino):
    m = ev_h["data_evento"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(m.index, m.values)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Hydrological records")
    ax.set_title("Seasonality of records, RMSP (1991-2025)")
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def fig_municipios(tab, destino):
    t = tab[tab["records_total"] > 0].sort_values("records_total")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 1.5))
    ax.barh(t["municipality"], t["records_1991_2013"], label="1991-2013")
    ax.barh(t["municipality"], t["records_2014_2025"], left=t["records_1991_2013"], label="2014-2025")
    ax.set_xlabel("Hydrological records in the Atlas")
    ax.set_title("Records per municipality, RMSP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def ecdf(valores):
    v = np.sort(valores)
    return v, np.arange(1, len(v) + 1) / len(v)


def fig_chuva_dias(ds, destino):
    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, rotulo in zip(eixos, ["media_3d", "max_3d"], ["RMSP mean, 3-day total (mm)", "Maximum across municipalities, 3-day total (mm)"]):
        for y, nome in [(0, "days without record"), (1, "days with record")]:
            x, f = ecdf(ds.loc[ds["y"] == y, col].to_numpy())
            ax.plot(x, f, label=f"{nome} (n={int((ds['y'] == y).sum())})")
        ax.set_xlabel(rotulo)
        ax.set_ylabel("Cumulative proportion")
        ax.legend()
    fig.suptitle("Rainfall on days with and without a hydrological record (1991-2013)")
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def fig_dispersao(tab, col, destino):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(tab[col], tab["records_1991_2013"])
    for _, r in tab.nlargest(8, "records_1991_2013").iterrows():
        ax.annotate(r["municipality"], (r[col], r["records_1991_2013"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Heavy-rain days (1991-2013)")
    ax.set_ylabel("Hydrological records (1991-2013)")
    ax.set_title("Records vs. heavy-rain exposure, by municipality")
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--chuva", "--rain", required=True)
    ap.add_argument("--dataset", required=True, help="dataset_rmsp.csv gerado pelo script 02")
    ap.add_argument("--limiar-mm", "--threshold-mm", type=float, default=50.0, help="limiar de chuva forte diária (mm); escolha convencional, não calibrada")
    ap.add_argument("--saida", "--output", default="descritiva")
    args = ap.parse_args()

    pasta = Path(args.saida)
    pasta.mkdir(exist_ok=True)

    ev = ler_atlas_rmsp(args.atlas)
    ev_h = ev[ev["descricao"].isin(HIDRO)]
    chuva = ler_chuva(args.chuva)
    ds = pd.read_csv(args.dataset, parse_dates=["data"])
    ds = ds[ds["ano"] <= ANO_CORTE]

    tab, col = tabela_municipios(ev_h, chuva, args.limiar_mm)
    tab.sort_values("records_total", ascending=False).to_csv(pasta / "municipios_resumo.csv", index=False)

    print(f"Hydrological records in the RMSP: {len(ev_h)} | 1991-2013: {int((ev_h['ano'] <= ANO_CORTE).sum())} | 2014-2025: {int((ev_h['ano'] > ANO_CORTE).sum())}")
    top = tab.sort_values("records_1991_2013", ascending=False)
    print("\nMunicipalities with the most records (1991-2013):")
    print(top[["municipality", "records_1991_2013", "records_2014_2025", "record_days_1991_2013", "mean_annual_rain_mm", col]].head(12).to_string(index=False))
    n_zero = int((tab["records_1991_2013"] == 0).sum())
    cinco = top["records_1991_2013"].head(5).sum() / max(top["records_1991_2013"].sum(), 1)
    print(f"\nMunicipalities with no record in 1991-2013: {n_zero} of {len(tab)} | the top 5 hold {cinco:.0%} of the records")
    rho_chuva = tab["records_1991_2013"].corr(tab[col], method="spearman")
    rho_anual = tab["records_1991_2013"].corr(tab["mean_annual_rain_mm"], method="spearman")
    print(f"Spearman correlation across municipalities (n={len(tab)}): records x heavy-rain days = {rho_chuva:.2f} | records x mean annual rain = {rho_anual:.2f}")

    ds1, ds0 = ds[ds["y"] == 1], ds[ds["y"] == 0]
    print("\n3-day accumulated rainfall (RMSP mean, mm), 1991-2013:")
    for nome, d in [("days with record", ds1), ("days without record", ds0)]:
        q = d["media_3d"].quantile([0.25, 0.5, 0.75, 0.95]).round(1).tolist()
        print(f"  {nome}: n={len(d)} | quartiles 25/50/75 and p95 = {q}")

    mensal = ev_h["data_evento"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    mensal.rename_axis("month").rename("records").to_csv(pasta / "records_by_month.csv")
    dm = int(mensal.loc[[12, 1, 2, 3]].sum())
    print(f"\nRecords by month, 1991-2025: December-March = {dm} of {int(mensal.sum())} ({dm / mensal.sum():.0%}); see records_by_month.csv")

    fig_serie_anual(ev, ev_h, pasta / "fig_serie_anual.png")
    fig_sazonalidade(ev_h, pasta / "fig_sazonalidade.png")
    fig_municipios(tab, pasta / "fig_municipios.png")
    fig_chuva_dias(ds, pasta / "fig_chuva_dias.png")
    fig_dispersao(tab, col, pasta / "fig_registros_vs_chuva.png")
    print(f"\nWritten to {pasta}/: municipios_resumo.csv, records_by_month.csv and 5 figures")


if __name__ == "__main__":
    main()
