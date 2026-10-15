// One visitor's Letterboxd export -> scored catalog. Everything runs in the browser: ratings never leave it.

import type { AppData } from './data'
import { embedFilms } from './embed'
import { readExport } from './letterboxd'
import { score, type ExtraFilm, type Scored } from './scoring'
import { findFilm, movie, pool } from './tmdb'

export type Step =
  | { step: 'reading' }
  | { step: 'matching'; done: number; total: number }
  | { step: 'embedding'; films: number; download: number | null }
  | { step: 'scoring' }

export interface Analysis {
  scored: Scored
  rated: number // rated films in the export
  found: number // of which found on TMDB
  miniseries: number
  watchlist: Set<number>
}

export async function analyse(file: Blob, data: AppData, onStep: (step: Step) => void): Promise<Analysis> {
  onStep({ step: 'reading' })
  const { rated, watchlist } = await readExport(file)

  const films = [...new Map([...rated, ...watchlist].map((f) => [f.uri, f])).values()]
  onStep({ step: 'matching', done: 0, total: films.length })
  const links = await pool(films, findFilm, 8, (done) => onStep({ step: 'matching', done, total: films.length }))
  const tmdbId = new Map(films.map((f, i) => [f.uri, links[i]]))

  // One rating per TMDB film (the first one, as in the Python package)
  const ratings = new Map<number, number>()
  for (const f of rated) {
    const id = tmdbId.get(f.uri)?.id
    if (id != null && !ratings.has(id)) ratings.set(id, f.rating)
  }
  const ratedFilms = [...ratings].map(([id, rating]) => ({ id, rating }))

  // Rated films outside the catalog: content vector computed here, collaborative one looked up
  const known = new Set(data.catalog.ids)
  const outside = ratedFilms.filter((r) => !known.has(r.id)).map((r) => r.id)
  let extras: ExtraFilm[] = []
  if (outside.length) {
    onStep({ step: 'embedding', films: outside.length, download: null })
    const movies = await pool(outside, movie)
    const vectors = await embedFilms(movies, data.embedder, (download) =>
      onStep({ step: 'embedding', films: outside.length, download }),
    )
    const { index, matrix } = data.movielens
    extras = movies.map((m, i) => {
      const row = index.get(m.id)
      const collaborative = new Float32Array(matrix.dim)
      if (row !== undefined) {
        const factor = matrix.scales[row] / 127
        for (let k = 0; k < matrix.dim; k++) collaborative[k] = matrix.values[row * matrix.dim + k] * factor
      }
      return { id: m.id, voteAverage: m.vote_average, voteCount: m.vote_count, content: vectors[i], collaborative }
    })
  }

  onStep({ step: 'scoring' })
  return {
    scored: score(data.catalog, ratedFilms, extras),
    rated: rated.length,
    found: ratedFilms.length,
    miniseries: links.filter((l) => l.series).length,
    watchlist: new Set(watchlist.map((f) => tmdbId.get(f.uri)?.id).filter((id): id is number => id != null)),
  }
}
