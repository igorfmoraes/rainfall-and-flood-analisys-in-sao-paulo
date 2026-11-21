#!/usr/bin/env python3
"""Monta o dataset de modelagem: rótulos do Atlas Digital de Desastres + features de chuva (CHIRPS).

Uso:
    python 02_montar_dataset.py --atlas Atlas_Apenas_SP.xlsx --chuva chirps_rmsp_diario.csv
    python 02_montar_dataset.py ... --nivel municipio
    python 02_montar_dataset.py ... --tipos "Enxurradas,Inundações,Alagamentos,Chuvas Intensas"

Níveis:
    rmsp       (padrão) 1 linha por dia; y = 1 se algum município da RMSP tem registro naquele dia.
    municipio  1 linha por município-dia; y = 1 se aquele município tem registro naquele dia.

Features (acumulados terminando no dia, janelas de 1, 2, 3, 7, 15 e 30 dias):
    rmsp: média e máximo entre municípios do acumulado, mais a chuva do dia anterior.
    municipio: acumulado do próprio município, mais a chuva do dia anterior.
    Calendário: mês e sazonalidade (seno/cosseno do dia do ano).

Atenção aos rótulos: o Atlas tem poucos registros hidrológicos depois de 2013 (ver o resumo
impresso no final). A coluna "periodo" marca o regime de registro para uso na divisão treino/teste.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from rmsp import RMSP

ABA = "Atlas Valores Corrigidos"
COLUNAS_ATLAS = [
    "protocolo", "municipio", "uf", "regiao", "data_registro", "data_evento", "cobrade",
    "tipologia", "descricao", "grupo", "ibge", "setores", "status",
]
TIPOS_PADRAO = ["Enxurradas", "Inundações", "Alagamentos"]
JANELAS = [1, 2, 3, 7, 15, 30]


def ler_eventos(caminho, tipos):
    bruto = pd.read_excel(caminho, sheet_name=ABA, header=None).iloc[:, : len(COLUNAS_ATLAS)]
    bruto.columns = COLUNAS_ATLAS
    if isinstance(bruto.iloc[0]["data_evento"], str):  # arquivo com linha de cabeçalho
        bruto = bruto.iloc[1:]
    bruto["data_evento"] = pd.to_datetime(bruto["data_evento"], errors="coerce").dt.normalize()
    bruto["ibge"] = pd.to_numeric(bruto["ibge"], errors="coerce")
    ev = bruto[bruto["ibge"].isin(list(RMSP)) & bruto["descricao"].isin(tipos)].dropna(subset=["data_evento"])
    ev = ev.assign(ibge=ev["ibge"].astype(int))
    return ev[["ibge", "data_evento", "descricao", "protocolo"]].reset_index(drop=True)


def ler_chuva(caminho):
    tabela = pd.read_csv(caminho, parse_dates=["data"])
    p = tabela.pivot(index="data", columns="ibge", values="precip_mm").sort_index()
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="D"))
    p.index.name, p.columns.name = "data", "ibge"
    ausentes = set(RMSP) - set(p.columns)
    if ausentes:
        sys.exit(f"Municípios sem chuva no arquivo: {sorted(ausentes)}")
    return p[sorted(RMSP)]


def calendario(indice):
    doy = indice.dayofyear.to_numpy()
    return pd.DataFrame(
        {"mes": indice.month, "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25)},
        index=indice,
    )


def periodo(ano):
    return np.select([ano <= 2004, ano <= 2013], ["1991-2004", "2005-2013"], default="2014-2025")


def montar_rmsp(p, ev):
    X = pd.DataFrame(index=p.index)
    for w in JANELAS:
        acum = p.rolling(w, min_periods=w).sum()
        X[f"media_{w}d"] = acum.mean(axis=1)
        X[f"max_{w}d"] = acum.max(axis=1)
    X["media_1d_dia_anterior"] = X["media_1d"].shift(1)
    X["max_1d_dia_anterior"] = X["max_1d"].shift(1)
    X = X.join(calendario(p.index))

    por_dia = ev.groupby("data_evento")["ibge"].nunique()
    X["y"] = X.index.isin(por_dia.index).astype(int)
    X["n_municipios_evento"] = por_dia.reindex(X.index).fillna(0).astype(int)
    X["ano"] = X.index.year
    X["periodo"] = periodo(X["ano"].to_numpy())
    return X.dropna(subset=[c for c in X.columns if c.startswith(("media_", "max_"))]).reset_index()


def montar_municipio(p, ev):
    def longo(df, nome):
        return df.reset_index().melt(id_vars="data", var_name="ibge", value_name=nome)

    # todos os melt têm a mesma ordem de linhas (mesmo índice e colunas), então a atribuição por posição é segura
    X = longo(p.rolling(1, min_periods=1).sum(), "p_1d")
    for w in JANELAS[1:]:
        X[f"p_{w}d"] = longo(p.rolling(w, min_periods=w).sum(), "v")["v"].to_numpy()
    X["p_1d_dia_anterior"] = longo(p.shift(1), "v")["v"].to_numpy()
    X["ibge"] = X["ibge"].astype(int)
    X = X.dropna(subset=[f"p_{w}d" for w in JANELAS]).reset_index(drop=True)

    chaves = set(zip(ev["ibge"], ev["data_evento"]))
    X["y"] = [int((i, d) in chaves) for i, d in zip(X["ibge"], X["data"])]
    cal = calendario(pd.DatetimeIndex(X["data"]))
    X["mes"], X["doy_sin"], X["doy_cos"] = cal["mes"].to_numpy(), cal["doy_sin"].to_numpy(), cal["doy_cos"].to_numpy()
    X["ano"] = X["data"].dt.year
    X["periodo"] = periodo(X["ano"].to_numpy())
    return X


def resumo(ds, ev, nivel):
    print(f"\nDataset ({nivel}): {len(ds)} linhas | {ds['data'].min().date()} a {ds['data'].max().date()}")
    print(f"Positivos: {int(ds['y'].sum())} ({ds['y'].mean():.3%})")
    g = ds.groupby("periodo")["y"].agg(linhas="size", positivos="sum")
    g["taxa"] = (g["positivos"] / g["linhas"]).map("{:.2%}".format)
    print(g.to_string())
    print("\nRegistros do Atlas por ano (tipos selecionados):")
    print(ev["data_evento"].dt.year.value_counts().sort_index().to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--chuva", required=True)
    ap.add_argument("--nivel", choices=["rmsp", "municipio"], default="rmsp")
    ap.add_argument("--tipos", default=",".join(TIPOS_PADRAO), help="descrições de tipologia, separadas por vírgula")
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    tipos = [t.strip() for t in args.tipos.split(",") if t.strip()]
    ev = ler_eventos(args.atlas, tipos)
    p = ler_chuva(args.chuva)
    ev = ev[(ev["data_evento"] >= p.index.min()) & (ev["data_evento"] <= p.index.max())]

    if args.nivel == "rmsp":
        ds = montar_rmsp(p, ev)
    else:
        ds = montar_municipio(p, ev)

    saida = args.saida or f"dataset_{args.nivel}.csv"
    ds.to_csv(saida, index=False)
    resumo(ds, ev, args.nivel)
    print(f"\nGravado: {saida}")


if __name__ == "__main__":
    main()
