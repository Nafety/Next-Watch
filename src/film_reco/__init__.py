"""Film recommendations from a Letterboxd export, enriched with TMDB."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="film-reco")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="read the Letterboxd export and fetch TMDB metadata")
    args = parser.parse_args()

    from film_reco.config import Settings

    settings = Settings()  # the API key comes from .env
    if args.command == "ingest":
        from film_reco.ingest import ingest

        ingest(settings)
