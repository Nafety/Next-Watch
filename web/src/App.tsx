import { useEffect, useState } from 'react'
import { Results } from './components/Results'
import { Upload } from './components/Upload'
import { analyse, forgetAnalysis, loadAnalysis, saveAnalysis, type Analysis, type Step } from './lib/analyse'
import { loadData, type AppData } from './lib/data'
import { useFavorites } from './lib/favorites'

function Progress({ step }: { step: Step }) {
  const [title, detail, share] =
    step.step === 'reading'
      ? ['Reading your export', 'Opening the ZIP file.', 0.05]
      : step.step === 'matching'
        ? ['Finding your films', `${step.done} of ${step.total} films looked up on TMDB.`, step.done / step.total]
        : step.step === 'embedding'
          ? [
              'Reading rare films',
              step.download === null
                ? `${step.films} of your films are outside the catalogue: analysing their descriptions.`
                : `Downloading the language model, once (${Math.round(step.download)} %).`,
              (step.download ?? 50) / 100,
            ]
          : ['Scoring 10,000 films', 'Almost there.', 1]
  return (
    <section className="progress" aria-live="polite">
      <h2>{title}</h2>
      <p>{detail}</p>
      <div className="track">
        <div style={{ width: `${Math.max(3, share * 100)}%` }} />
      </div>
    </section>
  )
}

export default function App() {
  const [data, setData] = useState<AppData | null>(null)
  const [step, setStep] = useState<Step | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<'films' | 'favourites'>('films')
  const { favorites, isFavorite, toggle } = useFavorites(analysis?.username ?? '')

  // The catalogue (about 9 MB) loads while the visitor exports their data
  useEffect(() => {
    loadData().then(
      (loaded) => {
        setData(loaded)
        setAnalysis(loadAnalysis(loaded)) // back to the last recommendations after a reload
      },
      () => setError('Could not load the film catalogue. Please reload the page.'),
    )
  }, [])

  const start = async (file: File) => {
    setError(null)
    try {
      const catalogue = data ?? (await loadData())
      setData(catalogue)
      const result = await analyse(file, catalogue, setStep)
      if (result.found < 10) throw new Error('Rate at least 10 films on Letterboxd for meaningful recommendations.')
      saveAnalysis(result, catalogue)
      setTab('films')
      setAnalysis(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong with this file.')
    } finally {
      setStep(null)
    }
  }

  return (
    <div className="page">
      <header className="header">
        <button className="brand" onClick={() => setTab('films')}>
          next watch
        </button>
        {analysis && (
          <nav>
            <button className="tab" aria-pressed={tab === 'films'} onClick={() => setTab('films')}>
              Films
            </button>
            <button className="tab" aria-pressed={tab === 'favourites'} onClick={() => setTab('favourites')}>
              Favourites{favorites.length ? ` (${favorites.length})` : ''}
            </button>
            <button
              className="tab"
              title={`Recommendations for ${analysis.username}`}
              onClick={() => {
                forgetAnalysis()
                setAnalysis(null)
              }}
            >
              Change export
            </button>
          </nav>
        )}
      </header>
      <main>
        {step ? (
          <Progress step={step} />
        ) : analysis && data ? (
          <Results
            analysis={analysis}
            data={data}
            tab={tab}
            favorites={favorites}
            isFavorite={isFavorite}
            toggle={toggle}
          />
        ) : (
          <Upload onFile={start} error={error} />
        )}
      </main>
      <footer className="footer">
        <span>
          Film data and posters from{' '}
          <a href="https://www.themoviedb.org" target="_blank" rel="noreferrer">
            TMDB
          </a>
          . This product uses the TMDB API but is not endorsed or certified by TMDB.
        </span>
        <span>
          Taste signal learned from{' '}
          <a href="https://grouplens.org/datasets/movielens/32m/" target="_blank" rel="noreferrer">
            MovieLens
          </a>
          {' · '}
          <a href="https://github.com/Nafety/movies-recommandations" target="_blank" rel="noreferrer">
            Source code
          </a>
        </span>
      </footer>
    </div>
  )
}
