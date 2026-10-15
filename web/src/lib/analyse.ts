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
  username: string // the Letterboxd account: favourites are kept per account
  scored: Scored
  rated: number // rated films in the export
  found: number // of which found on TMDB
  miniseries: number
  watchlist: Set<number>
}

export async function analyse(file: Blob, data: AppData, onStep: (step: Step) => void): Promise<Analysis> {
  onStep({ step: 'reading' })
  const { username, rated, watchlist } = await readExport(file)

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
    username,
    scored: score(data.catalog, ratedFilms, extras),
    rated: rated.length,
    found: ratedFilms.length,
    miniseries: links.filter((l) => l.series).length,
    watchlist: new Set(watchlist.map((f) => tmdbId.get(f.uri)?.id).filter((id): id is number => id != null)),
  }
}

// The last analysis is kept in the browser (scores only, about 200 KB), so a reload goes straight back to it
const SAVED = 'next-watch:analysis'

function toBase64(array: ArrayBufferView): string {
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength)
  let text = ''
  for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192))
  return btoa(text)
}
const fromBase64 = (text: string) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0)).buffer

/** Identifies the catalogue version: scores saved with another catalogue are not reused. */
function fingerprint(data: AppData): string {
  const ids = data.catalog.ids
  return `${ids.length}:${ids[0]}:${ids[ids.length - 1]}`
}

export function saveAnalysis(analysis: Analysis, data: AppData): void {
  const { scored } = analysis
  const saved = {
    catalog: fingerprint(data),
    username: analysis.username,
    rated: analysis.rated,
    found: analysis.found,
    miniseries: analysis.miniseries,
    watchlist: [...analysis.watchlist],
    ids: toBase64(scored.ids),
    quality: toBase64(scored.quality),
    content: toBase64(scored.content),
    collaborative: toBase64(scored.collaborative),
  }
  try {
    localStorage.setItem(SAVED, JSON.stringify(saved))
  } catch {
    // storage full or unavailable: the visitor will import again after a reload
  }
}

export function loadAnalysis(data: AppData): Analysis | null {
  try {
    const saved = JSON.parse(localStorage.getItem(SAVED) ?? 'null')
    if (!saved || saved.catalog !== fingerprint(data)) return null
    return {
      username: saved.username,
      rated: saved.rated,
      found: saved.found,
      miniseries: saved.miniseries,
      watchlist: new Set(saved.watchlist),
      scored: {
        ids: new Int32Array(fromBase64(saved.ids)),
        quality: new Float32Array(fromBase64(saved.quality)),
        content: new Float32Array(fromBase64(saved.content)),
        collaborative: new Float32Array(fromBase64(saved.collaborative)),
      },
    }
  } catch {
    return null
  }
}

export function forgetAnalysis(): void {
  try {
    localStorage.removeItem(SAVED)
  } catch {
    // nothing saved
  }
}
