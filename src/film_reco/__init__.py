"""Film recommendations from a Letterboxd export, enriched with TMDB."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="film-reco")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="read the Letterboxd export and fetch TMDB metadata")
    commands.add_parser("catalog", help="build the catalog of candidate films")
    commands.add_parser("evaluate", help="cross-validate every scorer on the user's ratings")
    args = parser.parse_args()

    from film_reco.config import Settings

    settings = Settings()  # the API key comes from .env
    if args.command == "ingest":
        from film_reco.ingest import ingest

        ingest(settings)
    elif args.command == "catalog":
        from film_reco.catalog import build_catalog

        build_catalog(settings)
    elif args.command == "evaluate":
        import polars as pl

        from film_reco import models
        from film_reco.collaborative import collaborative_embeddings
        from film_reco.content import content_embeddings
        from film_reco.evaluate import evaluate

        catalog = pl.read_parquet(settings.data_dir / "catalog.parquet")
        ids = catalog["tmdb_id"].to_numpy()
        content, collaborative = (
            content_embeddings(settings, catalog),
            collaborative_embeddings(settings, catalog),
        )

        def make_scorers() -> list[models.Scorer]:
            return [
                models.RandomScorer(),
                models.PopularityScorer(catalog),
                models.QualityScorer(catalog),
                models.SimilarityScorer("Content (TMDB description)", ids, content),
                models.SimilarityScorer("Collaborative (MovieLens)", ids, collaborative),
            ]

        with pl.Config(tbl_rows=-1, float_precision=3, tbl_hide_dataframe_shape=True):
            print(evaluate(settings, make_scorers))
