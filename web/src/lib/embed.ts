// Content vectors of rated films outside the catalog, computed in the browser with Transformers.js.
// The model is only downloaded when a visitor has such films (the library itself is loaded lazily too).

import type { Movie } from './tmdb'

export interface EmbedderInfo {
  model: string // Transformers.js (ONNX) name of the model used for the catalog vectors
  prefix: string
  pooling: 'mean' | 'cls'
}

/** What the embedding model reads for one film: same text as film_text() in the Python package. */
export function filmText(movie: Movie): string {
  const names = (values: { name: string }[], n: number) => values.slice(0, n).map((v) => v.name).join(', ')
  const directors = movie.credits.crew.filter((c) => c.job === 'Director')
  return (
    `${movie.title}. ${names(movie.genres, 5)}. Keywords: ${names(movie.keywords.keywords, 15)}. ` +
    `Directed by ${names(directors, 2)}. Starring ${names(movie.credits.cast, 5)}. ${movie.overview}`
  )
}

type Extractor = (texts: string[], options: object) => Promise<{ tolist(): number[][] }>
let extractor: Promise<Extractor> | null = null

// Loaded from a CDN, not bundled: its WebAssembly runtime (25 MB) is over Cloudflare Pages' file size limit,
// and only visitors with films outside the catalogue need it. The version is pinned.
const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1'
interface TransformersModule {
  pipeline: (task: string, model: string, options: object) => Promise<Extractor>
}

export async function embedFilms(
  movies: Movie[],
  info: EmbedderInfo,
  onDownload?: (percent: number) => void,
): Promise<Float32Array[]> {
  if (!movies.length) return []
  extractor ??= (import(/* @vite-ignore */ TRANSFORMERS) as Promise<TransformersModule>).then(({ pipeline }) =>
    pipeline('feature-extraction', info.model, {
      dtype: 'q8',
      progress_callback: (event: { status: string; progress?: number }) => {
        if (event.status === 'progress' && event.progress !== undefined) onDownload?.(event.progress)
      },
    }),
  )
  const output = await (await extractor)(
    movies.map((m) => info.prefix + filmText(m)),
    { pooling: info.pooling, normalize: true },
  )
  return output.tolist().map((vector) => Float32Array.from(vector))
}
