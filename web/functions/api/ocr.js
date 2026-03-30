// Cloudflare Pages Function — CORS proxy for OCR API
export async function onRequestPost(context) {
  const { request } = context;

  try {
    const body = await request.json();
    const { base_url, api_key, payload } = body;

    if (!base_url || !api_key || !payload) {
      return new Response(JSON.stringify({ error: 'Missing base_url, api_key, or payload' }), {
        status: 400,
        headers: corsHeaders('application/json'),
      });
    }

    const resp = await fetch(`${base_url}/ocr`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${api_key}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await resp.text();
    return new Response(data, {
      status: resp.status,
      headers: corsHeaders(resp.headers.get('content-type') || 'application/json'),
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: e.message }), {
      status: 502,
      headers: corsHeaders('application/json'),
    });
  }
}

// Handle CORS preflight
export async function onRequestOptions() {
  return new Response(null, { status: 204, headers: corsHeaders() });
}

function corsHeaders(contentType) {
  const h = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
  if (contentType) h['Content-Type'] = contentType;
  return h;
}
