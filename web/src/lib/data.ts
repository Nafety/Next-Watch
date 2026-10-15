// Loading the data exported by the Python package (film-reco export-app) into the shapes scoring.ts uses.

import type { EmbedderInfo } from './embed'
import type { Catalog, Matrix } from './scoring'

export interface AppData {
  catalog: Catalog
  genreNames: string[]
  movielens: { index: Map<number, number>; matrix: Matrix }
  embedder: EmbedderInfo
}

type Load = (file: string) => Promise<ArrayBuffer>

/** In the browser: files served from /data/. */
export const fetchFile: Load = async (file) => {
  const response = await fetch(`/data/${file}`)
  if (!response.ok) throw new Error(`Could not load ${file}`)
  return response.arrayBuffer()
}

export async function loadData(load: Load = fetchFile): Promise<AppData> {
  const json = async <T>(file: string) => JSON.parse(new TextDecoder().decode(await load(file))) as T
  const meta = await json<{ embedder: EmbedderInfo; dims: Record<string, number> }>('meta.json')
  const columns = await json<{
    ids: number[]
    years: (number | null)[]
    genreNames: string[]
    genres: number[][]
    voteAverage: number[]
    voteCount: number[]
  }>('catalog.json')
  const matrix = async (name: string): Promise<Matrix> => ({
    values: new Int8Array(await load(`${name}.i8`)),
    scales: new Float32Array(await load(`${name}.scales.f32`)),
    dim: meta.dims[name],
  })
  const [content, collaborative, movielens, movielensIds] = await Promise.all([
    matrix('content'),
    matrix('collaborative'),
    matrix('movielens'),
    load('movielens.ids.i32').then((b) => new Int32Array(b)),
  ])
  return {
    catalog: {
      ids: Int32Array.from(columns.ids),
      years: columns.years,
      genres: columns.genres.map((g) => g.map((i) => columns.genreNames[i])),
      voteAverage: Float32Array.from(columns.voteAverage),
      voteCount: Float32Array.from(columns.voteCount),
      content,
      collaborative,
    },
    genreNames: columns.genreNames,
    movielens: { index: new Map([...movielensIds].map((id, i) => [id, i])), matrix: movielens },
    embedder: meta.embedder,
  }
}
