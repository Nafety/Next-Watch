// Cloudflare Worker: serves the static site (dist/) and relays a few read-only TMDB requests,
// adding the API key server-side. The key (TMDB_API_KEY) is a secret of the Worker and never reaches the browser.

interface Env {
  TMDB_API_KEY: string
  ASSETS: Fetcher
}

// Only what the app needs: film search, TV search (to spot miniseries), and one film's details
const ALLOWED = [/^search\/movie$/, /^search\/tv$/, /^movie\/\d+$/]
const DAY = 24 * 60 * 60

async function tmdb(url: URL, env: Env): Promise<Response> {
  const path = url.pathname.slice('/api/tmdb/'.length)
  if (!ALLOWED.some((pattern) => pattern.test(path))) {
    return new Response('Not found', { status: 404 })
  }
  const query = new URLSearchParams(url.search)
  query.set('api_key', env.TMDB_API_KEY)
  const response = await fetch(`https://api.themoviedb.org/3/${path}?${query}`, {
    // TMDB data changes slowly: Cloudflare caches each answer for a week
    cf: { cacheTtl: 7 * DAY, cacheEverything: true },
  })
  return new Response(response.body, {
    status: response.status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': response.ok ? `public, max-age=${DAY}` : 'no-store',
    },
  })
}

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url)
    if (request.method === 'GET' && url.pathname.startsWith('/api/tmdb/')) return tmdb(url, env)
    if (url.pathname.startsWith('/api/')) return new Response('Not found', { status: 404 })
    return env.ASSETS.fetch(request)
  },
} satisfies ExportedHandler<Env>
