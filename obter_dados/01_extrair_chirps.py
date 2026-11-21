#!/usr/bin/env python3
"""Extrai a chuva diária do CHIRPS (média por município) para a RMSP, via Google Earth Engine.

Pré-requisitos:
    pip install -r requirements.txt
    - Conta no Google Earth Engine (uso não comercial) e um projeto do Google Cloud com a
      API do Earth Engine habilitada. O nome do projeto vai em --projeto.
    - Malha municipal: por padrão usa o pacote geobr (baixa a malha de SP). Se o geobr falhar,
      baixe a malha municipal de SP no site do IBGE e passe --shapefile (coluna CD_MUN).

Uso:
    python 01_extrair_chirps.py --projeto SEU_PROJETO
    python 01_extrair_chirps.py --projeto SEU_PROJETO --shapefile SP_Municipios_2022.shp

Saída:
    chirps_rmsp_diario.csv   colunas: ibge, data, precip_mm, buffer
    chirps_anos/chirps_AAAA.csv   um arquivo por ano (o script retoma de onde parou)

Notas metodológicas:
    - Coleção UCSB-CHG/CHIRPS/DAILY (v2.0 final), banda "precipitation" (mm/dia), pixel ~0,05° (~5,6 km).
    - Valor do município = média dos pixels (ee.Reducer.mean) dentro do polígono.
    - Municípios menores que um pixel podem ficar sem valor; nesses casos o script repete a
      redução com o polígono ampliado em 5 km (coluna buffer = True).
    - O dia do CHIRPS é definido em UTC (a confirmar na descrição do produto); o dia do evento no
      Atlas é a data local. Por isso o script 02 inclui a chuva do dia anterior como feature.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from rmsp import RMSP

COLECAO = "UCSB-CHG/CHIRPS/DAILY"
ESCALA_M = 5566  # ~0,05 grau
BUFFER_M = 5000
PADRAO_BANDA = re.compile(r"^(\d{8})(?:_precipitation)?$")


# --------------------------------------------------------------------------- geometrias
def carregar_malha(shapefile=None):
    """Retorna GeoDataFrame (EPSG:4326) com colunas ibge e geometry para os 39 municípios."""
    import geopandas as gpd

    if shapefile:
        gdf = gpd.read_file(shapefile)
        coluna = next((c for c in ("CD_MUN", "code_muni", "CD_GEOCMU") if c in gdf.columns), None)
        if coluna is None:
            sys.exit("O shapefile não tem coluna de código IBGE (esperado: CD_MUN ou code_muni).")
    else:
        try:
            import geobr

            gdf = geobr.read_municipality(code_muni="SP", year=2020)
            coluna = "code_muni"
        except Exception as erro:  # a API do geobr muda entre versões
            sys.exit(f"Falha ao obter a malha com o geobr ({erro}). Use --shapefile com a malha do IBGE.")

    gdf = gdf.copy()
    gdf["ibge"] = pd.to_numeric(gdf[coluna], errors="coerce").astype("Int64")
    gdf = gdf[gdf["ibge"].isin(list(RMSP))].to_crs(4326)

    faltando = sorted(set(RMSP) - set(gdf["ibge"].astype(int)))
    if faltando:
        sys.exit(f"Municípios ausentes na malha: {[(c, RMSP[c]) for c in faltando]}")

    gdf["ibge"] = gdf["ibge"].astype(int)
    # ~100 m de tolerância: reduz o tamanho da requisição sem efeito relevante em pixels de 5 km
    gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)
    return gdf[["ibge", "geometry"]].reset_index(drop=True)


def malha_para_ee(ee, gdf, buffer_m=0):
    from shapely.geometry import mapping

    feicoes = []
    for ibge, geom in zip(gdf["ibge"], gdf["geometry"]):
        geojson = json.loads(json.dumps(mapping(geom)))  # tuplas -> listas
        feicao = ee.Feature(ee.Geometry(geojson, None, False), {"ibge": int(ibge)})
        if buffer_m:
            feicao = feicao.buffer(buffer_m)
        feicoes.append(feicao)
    return ee.FeatureCollection(feicoes)


# --------------------------------------------------------------------------- Earth Engine
def iniciar_ee(projeto):
    import ee

    try:
        ee.Initialize(project=projeto)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=projeto)
    return ee


def com_retentativa(funcao, tentativas=4):
    for i in range(tentativas):
        try:
            return funcao()
        except Exception as erro:
            if i == tentativas - 1:
                raise
            espera = 5 * 2**i
            print(f"  falha ({erro}); nova tentativa em {espera}s", file=sys.stderr)
            time.sleep(espera)


def reduzir_periodo(ee, fc, inicio, fim_exclusivo):
    """Uma chamada: média por município para todos os dias do período (uma banda por dia)."""
    colecao = ee.ImageCollection(COLECAO).select("precipitation").filterDate(inicio, fim_exclusivo)
    imagem = colecao.toBands()
    resultado = imagem.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=ESCALA_M,
        crs=colecao.first().projection(),
        tileScale=2,
    )
    return resultado.getInfo()["features"]


def feicoes_para_tabela(feicoes):
    """Converte as feições do Earth Engine em tabela longa (ibge, data, precip_mm)."""
    linhas = []
    exemplos = []
    for feicao in feicoes:
        props = feicao["properties"]
        ibge = int(props["ibge"])
        exemplos = exemplos or list(props)[:5]
        for chave, valor in props.items():
            m = PADRAO_BANDA.match(chave)
            if m:
                linhas.append((ibge, m.group(1), valor))
    if not linhas:
        raise RuntimeError(f"Nenhuma banda diária reconhecida. Propriedades de exemplo: {exemplos}")
    tabela = pd.DataFrame(linhas, columns=["ibge", "data", "precip_mm"])
    tabela["data"] = pd.to_datetime(tabela["data"], format="%Y%m%d")
    tabela["precip_mm"] = pd.to_numeric(tabela["precip_mm"], errors="coerce")
    return tabela


def processar_ano(ee, gdf, fc, ano, inicio, fim):
    ini = max(pd.Timestamp(ano, 1, 1), inicio)
    fim_exc = min(pd.Timestamp(ano + 1, 1, 1), fim + pd.Timedelta(days=1))
    f_ini, f_fim = ini.strftime("%Y-%m-%d"), fim_exc.strftime("%Y-%m-%d")

    tabela = feicoes_para_tabela(com_retentativa(lambda: reduzir_periodo(ee, fc, f_ini, f_fim)))
    tabela["buffer"] = False

    # municípios pequenos: sem nenhum valor no período -> repetir com polígono ampliado
    sem_valor = tabela.groupby("ibge")["precip_mm"].apply(lambda s: s.notna().sum() == 0)
    sem_valor = sem_valor[sem_valor].index.tolist()
    if sem_valor:
        print(f"  {ano}: {len(sem_valor)} município(s) sem pixel; repetindo com buffer de {BUFFER_M} m")
        fc_buf = malha_para_ee(ee, gdf[gdf["ibge"].isin(sem_valor)], buffer_m=BUFFER_M)
        extra = feicoes_para_tabela(com_retentativa(lambda: reduzir_periodo(ee, fc_buf, f_ini, f_fim)))
        extra["buffer"] = True
        tabela = pd.concat([tabela[~tabela["ibge"].isin(sem_valor)], extra], ignore_index=True)
    return tabela.sort_values(["ibge", "data"]).reset_index(drop=True)


# --------------------------------------------------------------------------- verificação
def verificar(tabela, inicio, fim):
    dias = (fim - inicio).days + 1
    esperado = len(RMSP) * dias
    print("\nVerificação")
    print(f"  linhas: {len(tabela)} (esperado {esperado})")
    print(f"  municípios: {tabela['ibge'].nunique()} (esperado {len(RMSP)})")
    print(f"  valores ausentes: {tabela['precip_mm'].isna().sum()}")
    print(f"  valores negativos: {(tabela['precip_mm'] < 0).sum()}")
    print(f"  municípios com buffer: {sorted(tabela.loc[tabela['buffer'], 'ibge'].unique().tolist())}")
    anual = tabela.assign(ano=tabela["data"].dt.year).groupby(["ibge", "ano"])["precip_mm"].sum()
    media = anual.groupby("ibge").mean()
    print(f"  chuva anual média por município (mm): mín {media.min():.0f} | mediana {media.median():.0f} | máx {media.max():.0f}")
    print("  (confira se a ordem de grandeza é plausível para a RMSP; anos incompletos puxam a média para baixo)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projeto", required=True, help="projeto do Google Cloud com Earth Engine habilitado")
    ap.add_argument("--inicio", default="1991-01-01")
    ap.add_argument("--fim", default="2025-12-31")
    ap.add_argument("--shapefile", help="malha municipal do IBGE (usa geobr se omitido)")
    ap.add_argument("--saida", default="chirps_rmsp_diario.csv")
    ap.add_argument("--pasta-anos", default="chirps_anos")
    args = ap.parse_args()

    inicio, fim = pd.Timestamp(args.inicio), pd.Timestamp(args.fim)
    pasta = Path(args.pasta_anos)
    pasta.mkdir(exist_ok=True)

    print("Carregando malha municipal...")
    gdf = carregar_malha(args.shapefile)
    ee = iniciar_ee(args.projeto)
    fc = malha_para_ee(ee, gdf)

    for ano in range(inicio.year, fim.year + 1):
        arquivo = pasta / f"chirps_{ano}.csv"
        if arquivo.exists():
            print(f"{ano}: já existe, pulando")
            continue
        print(f"{ano}: extraindo...")
        processar_ano(ee, gdf, fc, ano, inicio, fim).to_csv(arquivo, index=False)

    partes = [pd.read_csv(a, parse_dates=["data"]) for a in sorted(pasta.glob("chirps_*.csv"))]
    tabela = pd.concat(partes, ignore_index=True).sort_values(["ibge", "data"])
    tabela.to_csv(args.saida, index=False)
    print(f"\nGravado: {args.saida}")
    verificar(tabela, inicio, fim)


if __name__ == "__main__":
    main()
