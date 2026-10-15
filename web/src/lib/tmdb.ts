// TMDB, through the site's own relay (/api/tmdb/...) which keeps the API key secret.

import { bestMatch, type SearchResult } from './matching'
import type { LetterboxdFilm } from './letterboxd'

export interface Movie extends SearchResult {
  overview: string
  poster_path: string | null
  vote_average: number
  vote_count: number
  genres: { name: string }[]
  credits: { cast: { name: string }[]; crew: { name: string; job: string }[] }
  keywords: { keywords: { name: string }[] }
}

async function get<T>(path: string, params: Record<string, string> = {}, attempt = 0): Promise<T> {
  const response = await fetch(`/api/tmdb/${path}?${new URLSearchParams(params)}`)
  if ((response.status === 429 || response.status >= 500) && attempt < 5) {
    await new Promise((resolve) => setTimeout(resolve, 2 ** attempt * 500))
    return get(path, params, attempt + 1)
  }
  if (!response.ok) throw new Error(`TMDB ${path}: ${response.status}`)
  return response.json() as Promise<T>
}

const movies = new Map<number, Promise<Movie>>()

export function movie(id: number): Promise<Movie> {
  if (!movies.has(id)) movies.set(id, get<Movie>(`movie/${id}`, { append_to_response: 'credits,keywords' }))
  return movies.get(id)!
}

async function search(kind: 'movie' | 'tv', title: string, year: number | null): Promise<SearchResult[]> {
  const params: Record<string, string> = { query: title }
  if (year) params[kind === 'movie' ? 'primary_release_year' : 'first_air_date_year'] = String(year)
  const { results } = await get<{ results: Record<string, unknown>[] }>(`search/${kind}`, params)
  return results.map((r) => ({
    id: r.id as number,
    title: (r.title ?? r.name) as string,
    original_title: (r.original_title ?? r.original_name ?? '') as string,
    release_date: (r.release_date ?? r.first_air_date ?? '') as string,
    popularity: (r.popularity ?? 0) as number,
  }))
}

/** TMDB id of a Letterboxd film, or null (not found, or a miniseries, which Letterboxd lists among films). */
export async function findFilm(film: LetterboxdFilm): Promise<{ id: number | null; series: boolean }> {
  // The year filter is strict (release dates differ between countries): retry without it
  const match =
    bestMatch(film.title, film.year, await search('movie', film.title, film.year)) ??
    bestMatch(film.title, film.year, await search('movie', film.title, null))
  if (!match || match.score < 100) {
    const show = bestMatch(film.title, film.year, await search('tv', film.title, film.year))
    if (show && show.score === 100) return { id: null, series: true }
  }
  return { id: match?.result.id ?? null, series: false }
}

/** Runs tasks with at most `limit` at a time, reporting progress. */
export async function pool<T, R>(
  items: T[],
  task: (item: T) => Promise<R>,
  limit = 8,
  onProgress?: (done: number) => void,
): Promise<R[]> {
  const results = new Array<R>(items.length)
  let next = 0
  let done = 0
  const worker = async () => {
    while (next < items.length) {
      const i = next++
      results[i] = await task(items[i])
      onProgress?.(++done)
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
  return results
}
