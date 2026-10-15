// The recommendation scores, as in the Python package (film_reco/models.py and hybrid.py):
// quality (Bayesian-averaged TMDB rating), content and collaborative (each rated film votes for similar
// films, weighted by rating minus the user's mean), each standardized over the catalog, then summed.

/** Vectors stored as int8, with one scale per row: value = q * scale / 127. A zero scale means no vector. */
export interface Matrix {
  values: Int8Array
  scales: Float32Array
  dim: number
}

export interface Catalog {
  ids: Int32Array
  years: (number | null)[]
  genres: string[][]
  voteAverage: Float32Array
  voteCount: Float32Array
  content: Matrix
  collaborative: Matrix
}

/** A rated film outside the catalog: its vectors are computed or looked up by the caller. */
export interface ExtraFilm {
  id: number
  voteAverage: number
  voteCount: number
  content: Float32Array
  collaborative: Float32Array
}

export type Mode = 'mix' | 'taste' | 'public'

export const MODES: Record<Mode, { quality: number; content: number; collaborative: number }> = {
  mix: { quality: 1, content: 1, collaborative: 1 },
  taste: { quality: 0, content: 1, collaborative: 1 }, // close to the user's ratings only
  public: { quality: 1, content: 0, collaborative: 0 }, // public opinion only
}

export interface Scored {
  ids: Int32Array // unrated catalog films
  quality: Float32Array // standardized signals, same order
  content: Float32Array
  collaborative: Float32Array
}

export function row(matrix: Matrix, i: number): Float32Array {
  const out = new Float32Array(matrix.dim)
  const factor = matrix.scales[i] / 127
  for (let k = 0; k < matrix.dim; k++) out[k] = matrix.values[i * matrix.dim + k] * factor
  return out
}

function dotRow(matrix: Matrix, i: number, vector: Float32Array): number {
  let sum = 0
  const offset = i * matrix.dim
  for (let k = 0; k < matrix.dim; k++) sum += matrix.values[offset + k] * vector[k]
  return (sum * matrix.scales[i]) / 127
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0
  for (let k = 0; k < a.length; k++) sum += a[k] * b[k]
  return sum
}

/** (x - mean) / std over all values, as numpy's population standard deviation. */
function standardize(values: Float64Array): Float64Array {
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length
  const std = Math.sqrt(variance) + 1e-12
  return values.map((v) => (v - mean) / std)
}

export function score(catalog: Catalog, rated: { id: number; rating: number }[], extras: ExtraFilm[]): Scored {
  const n = catalog.ids.length
  const index = new Map<number, number>()
  catalog.ids.forEach((id, i) => index.set(id, i))
  const extraById = new Map(extras.map((e) => [e.id, e]))

  // Quality: TMDB average rating, shrunk towards the mean when there are few votes (1,000 prior votes)
  const total = n + extras.length
  let sum = 0
  catalog.voteAverage.forEach((v) => (sum += v))
  extras.forEach((e) => (sum += e.voteAverage))
  const mean = sum / total
  const bayes = (average: number, count: number) => (average * count + 1000 * mean) / (count + 1000)

  // Content and collaborative: profile = sum of the rated films' vectors, weighted by rating - mean rating
  const ratingMean = rated.reduce((a, r) => a + r.rating, 0) / rated.length
  const profile = (matrix: Matrix, extra: (e: ExtraFilm) => Float32Array) => {
    const p = new Float32Array(matrix.dim)
    for (const r of rated) {
      const i = index.get(r.id)
      const vector = i !== undefined ? row(matrix, i) : extra(extraById.get(r.id)!)
      const weight = r.rating - ratingMean
      for (let k = 0; k < matrix.dim; k++) p[k] += weight * vector[k]
    }
    return p
  }
  const contentProfile = profile(catalog.content, (e) => e.content)
  const collaborativeProfile = profile(catalog.collaborative, (e) => e.collaborative)

  // Every film of the catalog (plus the extras) is scored, so that standardization matches Python's
  const quality = new Float64Array(total)
  const content = new Float64Array(total)
  const collaborative = new Float64Array(total)
  for (let i = 0; i < n; i++) {
    quality[i] = bayes(catalog.voteAverage[i], catalog.voteCount[i])
    content[i] = dotRow(catalog.content, i, contentProfile) / rated.length
    collaborative[i] = dotRow(catalog.collaborative, i, collaborativeProfile) / rated.length
  }
  extras.forEach((e, j) => {
    quality[n + j] = bayes(e.voteAverage, e.voteCount)
    content[n + j] = dot(e.content, contentProfile) / rated.length
    collaborative[n + j] = dot(e.collaborative, collaborativeProfile) / rated.length
  })
  const [q, c, f] = [quality, content, collaborative].map(standardize)

  const ratedIds = new Set(rated.map((r) => r.id))
  const keep = [...catalog.ids.keys()].filter((i) => !ratedIds.has(catalog.ids[i]))
  return {
    ids: Int32Array.from(keep, (i) => catalog.ids[i]),
    quality: Float32Array.from(keep, (i) => q[i]),
    content: Float32Array.from(keep, (i) => c[i]),
    collaborative: Float32Array.from(keep, (i) => f[i]),
  }
}

/** Unrated catalog films, best first, for one mode. */
export function rank(scored: Scored, mode: Mode): { id: number; score: number; quality: number; content: number; collaborative: number }[] {
  const w = MODES[mode]
  return [...scored.ids.keys()]
    .map((i) => ({
      id: scored.ids[i],
      score: w.quality * scored.quality[i] + w.content * scored.content[i] + w.collaborative * scored.collaborative[i],
      quality: scored.quality[i],
      content: scored.content[i],
      collaborative: scored.collaborative[i],
    }))
    .sort((a, b) => b.score - a.score)
}
