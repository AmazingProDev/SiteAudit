import { handleUpload } from '@vercel/blob/client';

function jsonHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Content-Type': 'application/json; charset=utf-8',
  };
}

async function readJsonBody(request) {
  if (typeof request?.json === 'function') {
    return await request.json();
  }

  const rawBody = await new Promise((resolve, reject) => {
    let chunks = '';
    request.setEncoding?.('utf8');
    request.on('data', (chunk) => {
      chunks += chunk;
    });
    request.on('end', () => resolve(chunks));
    request.on('error', reject);
  });

  return JSON.parse(rawBody || '{}');
}

function sendJson(response, status, payload) {
  if (response && typeof response.status === 'function' && typeof response.json === 'function') {
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    response.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    return response.status(status).json(payload);
  }

  return new Response(JSON.stringify(payload), {
    status,
    headers: jsonHeaders(),
  });
}

export default async function handler(request, response) {
  console.log('blob-upload request start', {
    method: request.method,
    url: request.url,
  });

  if (request.method === 'OPTIONS') {
    if (response) {
      response.setHeader('Access-Control-Allow-Origin', '*');
      response.setHeader('Access-Control-Allow-Headers', 'Content-Type');
      response.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
      response.statusCode = 204;
      response.end();
      return;
    }
    return new Response(null, { status: 204, headers: jsonHeaders() });
  }

  if (request.method !== 'POST') {
    return sendJson(response, 405, { error: 'Method not allowed.' });
  }

  let body;
  try {
    body = await readJsonBody(request);
    console.log('blob-upload request parsed', {
      keys: body && typeof body === 'object' ? Object.keys(body) : [],
    });
  } catch (error) {
    return sendJson(response, 400, { error: 'Invalid upload payload.' });
  }

  try {
    console.log('blob-upload handleUpload start');
    const uploadResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname) => {
        console.log('blob-upload onBeforeGenerateToken', { pathname });
        const safePathname = String(pathname || 'upload.xlsx').replace(/[^a-zA-Z0-9._/-]+/g, '-');
        return {
          allowedContentTypes: [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/octet-stream',
          ],
          addRandomSuffix: true,
          tokenPayload: JSON.stringify({ kind: 'ssv-workbook' }),
          pathname: `ssv-uploads/${safePathname}`,
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        console.log('blob-upload onUploadCompleted', {
          url: blob?.url,
          tokenPayload,
        });
      },
    });
    console.log('blob-upload handleUpload resolved');

    return sendJson(response, 200, uploadResponse);
  } catch (error) {
    console.error('blob-upload failed', error);
    return sendJson(response, 500, {
      error: error instanceof Error ? error.message : 'Blob upload failed.',
    });
  }
}
